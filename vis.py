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
from nilearn.plotting import plot_anat, plot_stat_map
import cv2
import cc3d

example_file = "306860001.dcm_dicom"

e = (nib.load(os.path.join("data", f"e/{example_file}_FLAIR_AX_CE_SENSE.nii")), "e.png")
e_aligned = (nib.load(os.path.join("data", f"e_aligned/{example_file}_FLAIR_AX_CE_SENSE.nii")), "e_aligned.png")
ne = (nib.load(os.path.join("data", f"ne/{example_file}_T2_FLAIR_AX_SENSE.nii")), "ne.png")
mask = (nib.load(os.path.join("data", f"spm/c1{example_file}_T2_FLAIR_AX_SENSE.nii")), "mask.png")

for brain_image, name in [e, e_aligned, ne, mask]:
    plot_anat(brain_image, output_file=os.path.join("figure", name), cut_coords=[-45], title=name.replace(".png", ""), display_mode="z", radiological=True, draw_cross=False)

gm = mask[0].get_fdata() > 0
ne = ne[0].get_fdata()
e = e_aligned[0].get_fdata()


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

z_map_img = new_img_like(e_aligned[0], z_map)
mean_pv = contrast_mask.sum() / gm.sum() * 100
avg_z = z_map.sum() / gm.sum()

plot_stat_map(
    z_map_img,
    e_aligned[0],
    output_file=f"figure/difference.png",
    threshold=threshold,
    title=f"{mean_pv:.2f}%/z={avg_z:.4f}",
    radiological=True,
    display_mode="z",
    cut_coords=1
)