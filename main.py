from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import *
from multiprocessing import Pool, cpu_count
from fsl.wrappers import bet, fast, fslmaths
import nibabel as nib
import subprocess
from scipy.ndimage import zoom
from skimage.filters import threshold_otsu
import pandas as pd


duration = 500


def make_gif(subject, ne, e, combined):
    results_dir = os.path.join("results", subject)
    os.makedirs(results_dir, exist_ok=True)

    ne = np.transpose(ne, (1, 0, 2))[:, :, ::-1]

    e = np.transpose(e, (1, 0, 2))[:, :, ::-1]
    combined = np.transpose(combined, (1, 0, 2, 3))[:, :, ::-1]
    img_shape = ne.shape
    cor_aspect, sag_aspect, ax_aspect = [aspect / img_shape[2] for aspect in img_shape]
    # plot 3 orthogonal slices
    for name, img in zip(["non-enhance", "enhance"], [ne, e]):
        a1 = plt.subplot(2, 2, 1)
        plt.imshow(img[:, :, img_shape[2] // 2], cmap="gray", vmin=0, vmax=255)
        a1.set_aspect(ax_aspect)
        a1.set_title("Axial view")

        a2 = plt.subplot(2, 2, 2)
        plt.imshow(img[:, img_shape[1] // 2, :].T, cmap="gray", vmin=0, vmax=255)
        a2.set_aspect(sag_aspect)
        a2.set_title("Sagittal view")

        a3 = plt.subplot(2, 2, 3)
        plt.imshow(img[img_shape[0] // 2, :, :].T, cmap="gray", vmin=0, vmax=255)
        a3.set_aspect(cor_aspect)
        a3.set_title("Coronal view")

        plt.savefig(os.path.join(results_dir, f"{name}.png"))

    for name, img in zip(["combined"], [combined]):
        a1 = plt.subplot(2, 2, 1)
        plt.imshow(img[:, :, img_shape[2] // 2, :])
        a1.set_aspect(ax_aspect)
        a1.set_title("Axial view")

        a2 = plt.subplot(2, 2, 2)
        transpose = np.stack(
            [
                img[:, img_shape[1] // 2, :, 0].T,
                img[:, img_shape[1] // 2, :, 1].T,
                img[:, img_shape[1] // 2, :, 2].T,
            ],
            axis=-1,
        )
        plt.imshow(transpose)
        a2.set_aspect(sag_aspect)
        a2.set_title("Sagittal view")

        a3 = plt.subplot(2, 2, 3)
        transpose = np.stack(
            [
                img[img_shape[0] // 2, :, :, 0].T,
                img[img_shape[0] // 2, :, :, 1].T,
                img[img_shape[0] // 2, :, :, 2].T,
            ],
            axis=-1,
        )
        plt.imshow(transpose)
        a3.set_aspect(cor_aspect)
        a3.set_title("Coronal view")

        plt.savefig(os.path.join(results_dir, f"{name}.png"))

    # make short gif video clips
    img_list = []
    for i in range(img_shape[2]):
        b_img = ne[:, :, i]
        b_img = np.stack([b_img, b_img, b_img], axis=-1)
        a_img = e[:, :, i]
        a_img = np.stack([a_img, a_img, a_img], axis=-1)
        img = np.concatenate([b_img, a_img, combined[:, :, i, :]], axis=1)
        img_list.append(Image.fromarray(img, "RGB"))

    img_list[0].save(
        os.path.join(results_dir, "plain-contrast-subtracted.gif"),
        save_all=True,
        append_images=img_list[1:],
        duration=duration,
    )

    dst_dir = os.path.join("results", subject, name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i, img in enumerate(img_list):
        img.save(os.path.join(dst_dir, str(i) + ".png"), "PNG")


def show_image_with_control_bar(before_img, after_img, subtracted):
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(bottom=0.25)

    ax[0].imshow(before_img)
    ax[1].imshow(after_img)

    axcolor = "lightgoldenrodyellow"
    axmin = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)

    smin = plt.Slider(axmin, "Min", np.min(subtracted), np.max(subtracted), valinit=np.min(subtracted))
    smax = plt.Slider(axmax, "Max", np.min(subtracted), np.max(subtracted), valinit=np.max(subtracted))

    def update(val):
        vmin = smin.val
        vmax = smax.val

        mask = (subtracted > vmax) * (subtracted < vmin)
        mask = np.stack([mask, mask, mask], axis=-1)

        combined = np.clip((subtracted - vmin) / (vmax - vmin), 0, 1)
        combined = (combined * 255.0).astype(np.uint8)
        combined = apply_colormap(combined)
        combined[mask] = np.stack([after_img, after_img, after_img], axis=-1)[mask]

        ax[2].imshow(combined)
        ax.set_clim([vmin, vmax])
        fig.canvas.draw_idle()

    smin.on_changed(update)
    smax.on_changed(update)

    plt.show()


def dcm2niix(subject):
    output_dir = os.path.join("data/nifti", subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        for modality in ["dw", "ne", "e"]:
            input_dir = os.path.join("data/dicom", subject, modality)
            subprocess.run([f"dcm2niix -o {output_dir} -f {modality} {input_dir}"], shell=True)


def BET(subject):
    os.makedirs(os.path.join("data/dw", subject), exist_ok=True)
    if not os.path.exists(os.path.join("data/dw", subject, "bet.nii")):
        bet(os.path.join("data/nifti", subject, "dw.nii"), os.path.join("data/dw", subject, "bet.nii"), mask=True)
        mask = nib.load(os.path.join("data/dw", subject, "bet_mask.nii")).get_fdata()
        ne = nib.load(os.path.join("data/nifti", subject, "ne.nii")).get_fdata()
        mask = zoom(mask, [ne_shape / m_shape for ne_shape, m_shape in zip(ne.shape, mask.shape)])
        mask = (mask >= 0.5).astype(np.float64)
        mask = nib.Nifti1Image(mask, affine=np.eye(4))
        nib.save(mask, f"data/dw/{subject}/bet_mask.nii")


def FAST(subject):
    os.makedirs(os.path.join("data/ne", subject), exist_ok=True)
    if not os.path.exists(os.path.join("data/ne", subject, "_seg.nii")):
        fslmaths(os.path.join("data/nifti", subject, "ne.nii")).mas(f"data/dw/{subject}/bet_mask.nii").run(f"data/ne/{subject}/bet.nii")
        fast(
            f"data/ne/{subject}/bet.nii",
            out=f"data/ne/{subject}/",
            n_classes=3,
            nopve=True,
        )


def quantify_vol(subject):
    mask = nib.load(os.path.join("data/ne", subject, "_seg.nii")).get_fdata()
    ne = nib.load(os.path.join("data/nifti", subject, "ne.nii")).get_fdata()
    e = nib.load(os.path.join("data/nifti", subject, "e.nii")).get_fdata()
    ne[mask == 0] = 0
    e[mask == 0] = 0

    e_mean = e[mask > 0].mean()
    ne_mean = ne[mask > 0].mean()
    contrast = (e - e_mean) / e[mask > 0].std() - (ne - ne_mean) / ne[mask > 0].std()
    contrast_th = 1.96
    contrast_mask = np.logical_and(contrast >= contrast_th, mask >= 1.5)

    percentage_vol = contrast_mask.sum() / mask.sum()

    max_signal = e.max()
    e = (e / max_signal * 255).astype(np.uint8)
    ne = (ne / ne_mean * e_mean / max_signal * 255).astype(np.uint8)
    contrast = np.clip((contrast - contrast_th) * 128, 0, 255)
    contrast = apply_colormap(contrast)
    combined = np.stack([e, e, e], axis=-1)
    combined[contrast_mask] = contrast[contrast_mask]

    make_gif(subject, ne, e, combined)

    return percentage_vol


if __name__ == "__main__":
    outcome = pd.read_excel("data/outcome.xlsx", header=1, index_col=0, usecols="A,C")
    for subject in os.listdir("data/dicom"):
        # Load
        dcm2niix(subject)
        BET(subject)
        FAST(subject)
        percentage_vol = quantify_vol(subject)
        print(subject, percentage_vol)
        outcome.loc[int(subject), "volume(%)"] = percentage_vol * 100
        print(outcome.loc[int(subject), "volume(%)"])

    outcome.to_excel("results.xlsx")
