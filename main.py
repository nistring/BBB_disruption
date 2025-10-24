import os
import numpy as np
from multiprocessing import cpu_count, Pool, Manager
import pandas as pd
import nibabel as nib
import cv2
import cc3d
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import RocCurveDisplay
from pathlib import Path


def analyze_data(results_path, qa_type):
    results_dir = os.path.dirname(results_path)

    # Load the data
    data = pd.read_csv(results_path)
    flair = "volume(%)"
    # Define the independent variable (X) and dependent variable (y)
    y = data["outcome"]
    X = data[[flair, qa_type]]

    # Plot decision boundary
    scatter = plt.scatter(X[flair], X[qa_type], c=y, cmap=plt.cm.Paired, edgecolors="k", alpha=0.6)
    for i, txt in enumerate(data["ID"]):
        if isinstance(txt, str):
            txt = int(txt.split("/")[0])
        plt.annotate(txt, (X[flair].iloc[i], X[qa_type].iloc[i]), fontsize=5, alpha=0.7)
    plt.xlabel(f"{flair}")
    plt.ylabel(f"{qa_type}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(handles=scatter.legend_elements()[0], labels=["good", "poor"])
    plt.title(f"Scatter plot of {flair} and {qa_type}")
    plt.savefig(os.path.join(results_dir, f"scatter_plot_{flair.replace(' ', '_')}_and_{qa_type}.svg"))
    plt.close()

    # Calculate the ROC curves
    fig, ax = plt.subplots()
    
    RocCurveDisplay.from_predictions(y, data[[qa_type]].values * data[[flair]].values, ax=ax, name=f"{qa_type} x {flair}")
    RocCurveDisplay.from_predictions(y, data[[qa_type]], ax=ax, name=f"{qa_type}")
    RocCurveDisplay.from_predictions(y, data[[flair]], ax=ax, name=f"{flair}")
    
    plt.title(f"ROC Curves for prognosis on {qa_type} and {flair}")
    plt.savefig(os.path.join(results_dir, f"roc_curves_{flair.replace(' ', '_')}_and_{qa_type}.svg"))
    plt.close()


def quantify_vol(subject, sigma=6.0, plot=False):
    # Masking non-parynchmal region
    subject_dir = list(Path(f"data/nii").glob(subject + "*"))[0]
    subject_file = subject_dir.parent.name + "_" + subject_dir.name
    bid_path = subject.split("/")[0] + ("first" if subject.split("/")[1][0] == "1" else "second")

    csf = nib.load(f"bids-out/smriprep/sub-{bid_path}/anat/sub-{bid_path}_dseg.nii.gz").get_fdata()
    ne_image = nib.load(f"{subject_dir}/ne/{subject_file}_ne.nii").get_fdata().transpose(1, 0, 2)[::-1]
    e_image = nib.load(f"{subject_dir}/e_align/{subject_file}_e_align.nii.gz").get_fdata().transpose(1, 0, 2)[::-1]

    parenchyma = (csf > 0) * (csf < 3)
    csf = csf == 3
    for z in range(parenchyma.shape[2]):
        if parenchyma[..., z].sum() < csf[..., z].sum():
            parenchyma[..., z] = False

    parenchyma = cc3d.largest_k(parenchyma, 1, connectivity=26, binary_image=True).astype(bool)
    parenchyma = parenchyma.transpose(1, 0, 2)[::-1, ::-1]

    z_map = e_image / e_image[parenchyma].mean() - ne_image / ne_image[parenchyma].mean()
    z_map_std = z_map[parenchyma].std()
    z_map = z_map / z_map_std
    th = z_map[parenchyma].mean() + sigma

    if z_map.shape[:2] != (512, 512):
        z_map = np.stack([cv2.resize(z_map[:, :, z], (512, 512), interpolation=cv2.INTER_CUBIC) for z in range(z_map.shape[2])], axis=2)
        e_image = np.stack(
            [cv2.resize(e_image[:, :, z], (512, 512), interpolation=cv2.INTER_CUBIC) for z in range(e_image.shape[2])], axis=2
        )
        ne_image = np.stack(
            [cv2.resize(ne_image[:, :, z], (512, 512), interpolation=cv2.INTER_CUBIC) for z in range(ne_image.shape[2])], axis=2
        )

    parenchyma = np.stack(
        [
            cv2.resize(parenchyma[:, :, z].astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) >= 0.5
            for z in range(parenchyma.shape[2])
        ],
        axis=2,
        dtype=bool,
    )

    parenchyma_pixel_count = parenchyma.sum()
    mask = (z_map > th) * (e_image > e_image[parenchyma].mean()) * parenchyma

    # Remove superficial parenchymal enhancement
    for z in range(mask.shape[2]):
        parenchyma_mask_removed = (parenchyma[..., z] * (1 - mask[..., z])).astype(np.uint8)
        parenchyma_mask_removed = cv2.floodFill(parenchyma_mask_removed, None, (0, 0), 1)[1]
        mask[..., z] = mask[..., z] * (1 - parenchyma_mask_removed)

    mean_pv = mask.sum() / parenchyma_pixel_count * 100
    avg_z = (z_map[mask] - th).sum() / parenchyma_pixel_count

    if plot:
        e_image = cv2.normalize(e_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ne_image = cv2.normalize(ne_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        os.makedirs(f"results/images/{subject_file}", exist_ok=True)

        for z in range(z_map.shape[2]):
            overlay_img = deepcopy(e_image[:, :, z])
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)
            parenchyma_mask_slice = np.tile(parenchyma[:, :, z : z + 1], (1, 1, 3))
            overlay_img[parenchyma_mask_slice] = (
                overlay_img[parenchyma_mask_slice].reshape(-1, 3) * 0.7 + np.array([255, 0, 0], dtype=np.uint8) * 0.3
            ).flatten()
            if np.any(mask[:, :, z]):
                overlay_img[np.tile(mask[..., z : z + 1], (1, 1, 3))] = cv2.applyColorMap(
                    cv2.normalize(z_map[:, :, z][mask[:, :, z]], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_AUTUMN
                ).flatten()
            combined_image = np.concatenate(
                (overlay_img, cv2.cvtColor(e_image[:, :, z], cv2.COLOR_GRAY2BGR), cv2.cvtColor(ne_image[:, :, z], cv2.COLOR_GRAY2BGR)),
                axis=1,
            )
            cv2.imwrite(f"results/images/{subject_file}/{subject_file}_slice_{z}.png", combined_image)

        # Plot histogram
        plt.hist(z_map[mask], weights=z_map[mask] - th, bins=30, edgecolor="black", density=True, alpha=0.6, color="b", log=False)
        plt.xlabel("SD units")
        plt.ylabel("frequency x value")
        plt.savefig(f"results/images/{subject_file}/z_map_histogram.png")
        plt.close()

    return mean_pv, avg_z, z_map_std


def obj_func(sigma, plot):

    processes = cpu_count()

    subjects, qa, outcome = dataset
    subjects = np.array_split(subjects, processes)
    subjects = [list(s) for s in subjects]

    qa = np.array_split(qa, processes)
    qa = [list(q) for q in qa]

    outcome = np.array_split(outcome, processes)
    outcome = [list(o) for o in outcome]

    manager = Manager()
    results = manager.list()

    global do

    def do(sub, qa, out):
        for s, q, o in zip(sub, qa, out):
            percentage_vol, avg_z_score, z_map_std = quantify_vol(s, sigma, plot=plot)
            results.append([s, q, o, percentage_vol, avg_z_score, z_map_std])

    with Pool(processes=processes) as pool:
        pool.starmap(do, zip(subjects, qa, outcome))

    return list(results)


if __name__ == "__main__":

    subjects1, subjects2 = [], []
    for subject in os.listdir("data/nii"):
        for sub in os.listdir(f"data/nii/{subject}"):
            if "1st" in sub:
                subjects1.append(subject)
            else:
                subjects2.append(subject)

    df = pd.read_excel("data/outcome.xlsx", header=0, usecols="A,C,H,I").iloc[:75]
    df["outcome"] -= 1

    df1 = df.loc[df["ID"].isin([int(subject) for subject in subjects1])]
    df1 = df1.loc[~df["Qa1"].isna()]
    df1["ID"] = df1["ID"].astype(int).astype(str).str.zfill(8) + "/1st"

    df2 = df.loc[df["ID"].isin([int(subject) for subject in subjects2])]
    df2 = df2.loc[~df["Qa2"].isna()]
    df2["ID"] = df2["ID"].astype(int).astype(str).str.zfill(8) + "/2nd"

    for th in np.arange(0.5, 5.1, 0.5):
        os.makedirs(f"results/{th}", exist_ok=True)

        dataset = (df1["ID"].values, df1["Qa1"].values, df1["outcome"].values)
        res1 = pd.DataFrame(obj_func(th, False), columns=["ID", "Qa1", "outcome", "volume(%)", "z score", "z_map_std"])
        res1.to_csv(f"results/{th}/final_results_qa1.csv", index=False)
        analyze_data(f"results/{th}/final_results_qa1.csv", "Qa1")

        dataset = (df2["ID"].values, df2["Qa2"].values, df2["outcome"].values)
        res2 = pd.DataFrame(obj_func(th, False), columns=["ID", "Qa2", "outcome", "volume(%)", "z score", "z_map_std"])
        res2.to_csv(f"results/{th}/final_results_qa2.csv", index=False)
        analyze_data(f"results/{th}/final_results_qa2.csv", "Qa2")

        res1["ID"] = res1["ID"].str.replace("/1st", "")
        res2["ID"] = res2["ID"].str.replace("/2nd", "")
        df = pd.merge(res1, res2, on="ID", suffixes=("_1", "_2"))
        # Divide the results because the values are in log scale
        df["volume(%)"] = df["volume(%)_1"] # / df["volume(%)_2"]
        df["z score"] = df["z score_1"] / df["z score_2"]
        df["Qa_diff"] = df["Qa2"] / df["Qa1"]
        df["outcome"] = df["outcome_1"]
        df = df[["ID", "Qa_diff", "outcome", "volume(%)", "z score"]].dropna()
        df.to_csv(f"results/{th}/final_results_diff.csv", index=False)
        analyze_data(f"results/{th}/final_results_diff.csv", "Qa_diff")

    # df = pd.concat([df1, df2])
    # df["Qa"] = df["Qa1"].fillna(df["Qa2"])
    # df = df[["ID", "Qa", "outcome"]].dropna()
    # obj_func(2.0, True)