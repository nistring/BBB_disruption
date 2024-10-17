import matplotlib as mpl
import os
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, smooth_img
from nilearn.plotting import plot_stat_map
import cv2
import copy


def plot_wrapper(subject, prognosis, ne, e, z_map, mean_pv, avg_z, vmin, vmax=1.0):

    z_map_img = new_img_like(e, z_map)
    os.makedirs(f"results/e/{prognosis}/", exist_ok=True)
    os.makedirs(f"results/ne/{prognosis}/", exist_ok=True)

    subject = subject[:-4] + "****"
    plot_stat_map(
        z_map_img,
        e,
        output_file=f"results/e/{prognosis}/{subject}.png",
        threshold=vmin,
        title=f"{subject}/{prognosis}/e/{mean_pv:.2f}%/z={avg_z:.4f}",
        radiological=True,
        display_mode="z",
        cmap=mpl.colormaps["autumn"],
        vmin=vmin,
        vmax=vmax,
    )
    plot_stat_map(
        z_map_img,
        ne,
        output_file=f"results/ne/{prognosis}/{subject}.png",
        threshold=vmin,
        title=f"{subject}/{prognosis}/ne/{mean_pv:.2f}%/z={avg_z:.4f}",
        radiological=True,
        display_mode="z",
        cmap=mpl.colormaps["autumn"],
        vmin=vmin,
        vmax=vmax,
    )


def quantify_vol(subject, prognosis, vmin, vmax, kernel, plot=False):
    # Masking non-parynchmal region
    file_name = f"{subject}_T2_FLAIR_AX_SENSE" if subject[0] == "0" else f"{subject}0001.dcm_dicom_T2_FLAIR_AX_SENSE"
    parenchyma = np.load(f"data/mask/{file_name}.npy").astype(np.uint8)

    parenchyma_sum = parenchyma.sum()
    kernel = int(kernel) * 2 + 1
    for i in range(parenchyma.shape[2]):
        parenchyma[:, :, i] = cv2.erode(
            parenchyma[:, :, i],
            kernel=np.ones((kernel, kernel), np.uint8),
            iterations=1,
        )

    parenchyma = parenchyma.astype(bool)
    
    ne = nib.load(f"data/ne/{file_name}.nii")
    e = nib.load(f"data/e_aligned/{file_name}.nii".replace("T2_FLAIR_AX", "FLAIR_AX_CE"))

    z_map = make_zmap(ne, e, parenchyma)
    contrast_mask = (z_map >= vmin) * (z_map < vmax)

    # contrast_mask = cc3d.dust(contrast_mask, threshold=size_min * parenchyma_sum, connectivity=26) * (
    #     ~cc3d.dust(contrast_mask, threshold=size_max * parenchyma_sum, connectivity=26)
    # )
    z_map[contrast_mask == False] = 0

    mean_pv = contrast_mask.sum() / parenchyma_sum * 100
    avg_z = (z_map[contrast_mask] - vmin).sum() / parenchyma_sum

    z_map = parenchyma.astype(float)

    if plot:
        plot_wrapper(subject, prognosis, ne, e, z_map, mean_pv, avg_z, vmin)

    return mean_pv, avg_z


def make_zmap(ne, e, parenchyma, sigma=0.4296875 * 3):
    # sigma = size of voxel x 3(empirical)
    # Masking non-parynchmal region

    ne = smooth_img(ne, fwhm=(sigma, sigma, 0)).get_fdata()
    e = smooth_img(e, fwhm=(sigma, sigma, 0)).get_fdata()
    z_map = e / e[parenchyma].mean() - ne / ne[parenchyma].mean()
    # z_map[parenchyma] /= ne[parenchyma].mean()
    # print(z_map[parenchyma].max())
    # z_map[parenchyma] = z_map[parenchyma] - z_map[parenchyma].mean()
    z_map[z_map < 0] = 0
    z_map[parenchyma == False] = 0

    return z_map
