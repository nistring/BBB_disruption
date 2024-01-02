import matplotlib.pyplot as plt
import os
import numpy as np
from multiprocessing import cpu_count, Pool, Manager
from fsl.wrappers import flirt
import nibabel as nib
import subprocess
from scipy.stats import norm
from scipy.ndimage import binary_erosion
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
from nilearn.image import new_img_like, smooth_img
from nilearn.plotting import plot_stat_map
import cv2
import cc3d

def FLIRT(subject):
    if not os.path.exists(f"data/e_aligned/{subject}0001.dcm_dicom_FLAIR_AX_CE_SENSE.nii"):
        flirt(
            f"data/e/{subject}0001.dcm_dicom_FLAIR_AX_CE_SENSE.nii",
            f"data/ne/{subject}0001.dcm_dicom_T2_FLAIR_AX_SENSE.nii",
            out=f"data/e_aligned/{subject}0001.dcm_dicom_FLAIR_AX_CE_SENSE.nii",
            twod=True,
        )


def quantify_vol(subject, prognosis):
    # mask = nib.load(os.path.join("data/ne", subject, "_seg.nii")).get_fdata()
    # gm = mask > 2.5
    mask = nib.load(
        os.path.join("data/spm", f"c1{subject}0001.dcm_dicom_T2_FLAIR_AX_SENSE.nii")
    ).get_fdata()
    gm = mask > 0

    ne_ori = nib.load(f"data/ne/{subject}0001.dcm_dicom_T2_FLAIR_AX_SENSE.nii")
    e_ori = nib.load(f"data/e_aligned/{subject}0001.dcm_dicom_FLAIR_AX_CE_SENSE.nii")

    # ne = smooth_img(ne_ori, fwhm=(1,1,0)).get_fdata()
    # e = smooth_img(e_ori, fwhm=(1,1,0)).get_fdata()
    ne = ne_ori.get_fdata()
    e = e_ori.get_fdata()

    z_map = e - ne / ne[gm].mean() * e[gm].mean()
    z_map /= z_map[gm].std()
    p_values = norm.sf(z_map[gm])
    contrast_mask, _ = fdrcorrection(p_values, alpha=0.05)

    threshold = z_map[gm][contrast_mask].min()
    z_map[gm == False] = 0
    contrast_mask = z_map >= threshold
    
    labels_out = cc3d.connected_components(contrast_mask)
    stats = cc3d.statistics(labels_out)
    sizes = stats["voxel_counts"][1:]
    cc3d.dust(contrast_mask, threshold=sizes.mean() + norm.ppf(0.95) * sizes.std(), connectivity=26, in_place=True)
    z_map[contrast_mask == False] = 0

    z_map_img = new_img_like(e_ori, z_map)
    mean_pv = contrast_mask.sum() / gm.sum() * 100
    avg_z = z_map.sum() / gm.sum()
    
    plot_stat_map(
        z_map_img,
        e_ori,
        output_file=f"results/e/{prognosis}/{subject}.png",
        threshold=threshold,
        title=f"{subject}/{prognosis}/e/{mean_pv:.2f}%/z={avg_z:.4f}",
        radiological=True,
        display_mode="z",
    )
    plot_stat_map(
        z_map_img,
        ne_ori,
        output_file=f"results/ne/{prognosis}/{subject}.png",
        threshold=threshold,
        title=f"{subject}/{prognosis}/ne/{mean_pv:.2f}%/z={avg_z:.4f}",
        radiological=True,
        display_mode="z",
    )
    
    return mean_pv, avg_z


if __name__ == "__main__":
    plt.rcParams["figure.dpi"] = 1200
    outcome = pd.read_excel("data/outcome.xlsx", header=1, index_col=0, usecols="A,C")
    processes = cpu_count()

    manager = Manager()
    good, poor = manager.list(), manager.list()
    
    subjects = np.array_split(outcome.index.values, processes)
    subjects = [list(s) for s in subjects]
    prognosis = np.array_split(outcome["1, good 2, poor"].values, processes)
    prognosis = [list(p) for p in prognosis]

    def do(sub, prog):
        for s, p in zip(sub, prog):
            FLIRT(s)
            percentage_vol, avg_z_score = quantify_vol(s, "good" if p == 1 else "poor")
            print(s, p, percentage_vol, avg_z_score)
            # if(percentage_vol.shape[0] == 24):
            #     if outcome.loc[int(subject), "1, good 2, poor"] == 1:
            #         good.append(percentage_vol)
            #     else:
            #         poor.append(percentage_vol)

    with Pool(processes=processes) as pool:
        pool.starmap(do, zip(subjects, prognosis))

    # histogram = pd.DataFrame(np.concatenate((good + poor)))
    # histogram["layer"] = np.tile(np.arange(24), len(good + poor))
    # histogram["prognosis"] = ["good"] * (len(good) * 24) + ["poor"] * (len(poor) * 24)
    # histogram.to_excel("histogram.xlsx")
