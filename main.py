import matplotlib.pyplot as plt
import os
import numpy as np
from multiprocessing import cpu_count, Pool, Manager
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from utils import quantify_vol
from sklearn.model_selection import KFold

datetime_boundary = "2021-01-01"
init_points=1
n_iter=1
seed=0

def obj_func(vmin, kernel):

    dv = float("inf")
    size_max = 0
    processes = cpu_count()
    
    subjects, prognosis, qa = get_meta_info(dataset)
    subjects = np.array_split(subjects, processes)
    subjects = [list(s) for s in subjects]
    prognosis = np.array_split(prognosis, processes)
    prognosis = [list(p) for p in prognosis]
    qa = np.array_split(np.log10(qa), processes)
    # qa = np.array_split(qa, processes)
    qa = [list(q) for q in qa]

    manager = Manager()
    good_list = manager.list()
    poor_list = manager.list()

    global do
    def do(sub, prog, qa):
        plot = False if phase == "train" else True
        for s, p, q in zip(sub, prog, qa):
            if p == 1:
                percentage_vol, avg_z_score = quantify_vol(s, "good", vmin, vmin + dv, kernel, plot)
                good_list.append([p, q, percentage_vol, avg_z_score])
            else:
                percentage_vol, avg_z_score = quantify_vol(s, "poor", vmin, vmin + dv, kernel, plot)
                poor_list.append([p, q, percentage_vol, avg_z_score])

    with Pool(processes=processes) as pool:
        pool.starmap(do, zip(subjects, prognosis, qa))

    good = np.array(good_list)[:, [1, 3]]  
    poor = np.array(poor_list)[:, [1, 3]]

    # good[:, 1] = np.log10(good[:, 1])
    # poor[:, 1] = np.log10(poor[:, 1])

    if phase == "train":
        # return pearsonr(np.concatenate((good[:, 0], poor[:, 0])), np.concatenate((good[:, 1], poor[:, 1])))[0]
        return spearmanr(np.concatenate((good[:, 0], poor[:, 0])), np.concatenate((good[:, 1], poor[:, 1])))[0]
    else:
        return list(good_list) + list(poor_list)

def get_meta_info(dset):
    qa1 = dset[~dset['Qa1'].isna()]
    id1 = qa1["ID"].astype(int).astype(str).str.zfill(8).values
    out1 = qa1["outcome"].astype(int).values
    qa1 = qa1["Qa1"].values

    qa2 = dset[~dset['Qa2'].isna()]
    id2 = qa2["ID"].astype(int).astype(str).values
    out2 = qa2["outcome"].astype(int).values
    qa2 = qa2["Qa1"].values

    qa = np.concatenate((qa1, qa2))
    out = np.concatenate((out1, out2))
    id = np.concatenate((id1, id2))

    valid = []
    for subject in id:
        subject = f"c1{subject}_T2_FLAIR_AX_SENSE.nii" if subject[0] == "0" else f"c1{subject}0001.dcm_dicom_T2_FLAIR_AX_SENSE.nii"
        valid.append(True if subject in os.listdir("data/spm_seg/gm") else False)

    return id[valid], out[valid], qa[valid]

if __name__ == "__main__":

    df = pd.read_excel("data/outcome.xlsx", header=0, usecols="A,C,G,H,I").iloc[:75]
    two_samples = df[(~df['Qa1'].isna()) & (~df['Qa2'].isna())]
    one_sample = df[(~df['Qa1'].isna()) ^ (~df['Qa2'].isna())]
    df[(~df['Qa1'].isna()) | (~df['Qa2'].isna())].to_excel("outcome.xlsx")
    kf = KFold(3, shuffle=True, random_state=seed)
    plt.rcParams["figure.dpi"] = 1200
    df_list = []
    for two_samples_index, one_sample_index in zip(kf.split(two_samples), kf.split(one_sample)):
        phase = 'train'
        pbounds = {"vmin": (0.0, 0.5), "kernel": (0, 0.999)}
        dataset = pd.concat([two_samples.iloc[two_samples_index[0]], one_sample.iloc[one_sample_index[0]]])

        optimizer = BayesianOptimization(f=obj_func, pbounds=pbounds, random_state=seed, allow_duplicate_points=True)
        logger = JSONLogger(path="./logs.log")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        print(optimizer.max)
        
        phase = 'test'
        dataset = pd.concat([two_samples.iloc[two_samples_index[1]], one_sample.iloc[one_sample_index[1]]])

        params = optimizer.max['params']
        outcome = np.array(obj_func(params['vmin'], params['kernel']))
        outcome = np.hstack((outcome, np.tile(np.array([params['vmin']]), (outcome.shape[0], 1))))
        df_list.append(pd.DataFrame(outcome, columns=["1, good 2, poor", "Qa", "volume(%)", "z score", 'vmin']))

    with pd.ExcelWriter("results/table.xlsx") as writer:
        for i, table in enumerate(df_list):
            table.to_excel(writer, sheet_name=f"cv{i}", index=False)