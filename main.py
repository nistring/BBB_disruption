from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import *

if __name__ == "__main__":
    for subject in ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5"]:
        # load the DICOM files
        before_dir = os.path.join("data", subject, "1")
        after_dir = os.path.join("data", subject, "2")
        results_dir = os.path.join("results", subject)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # convert into numpy array
        before_img, aspects = load_3d(before_dir)
        after_img, _ = load_3d(after_dir)
        ax_aspect, sag_aspect, cor_aspect = aspects
        img_shape = before_img.shape

        # calculate structural similarity score
        score, grad, diff = ssim(
            before_img,
            after_img,
            full=True,
            data_range=after_img.max(),
            win_size=3,
            gradient=True,
        )

        # normalization
        diff = np.clip(1 - diff, 0, 1)
        diff = (diff * 255).astype("uint8")
        diff = apply_colormap(diff)
        grad -= grad.min()
        grad = np.clip(grad / grad.max() * 255, 0, 255).astype("uint8")
        grad = apply_colormap(grad)

        norm = 255.0 / after_img.max()
        subtracted = after_img - before_img
        subtracted[subtracted < 0] = 0
        subtracted = np.array(
            np.clip(subtracted * 255 / subtracted.max(), 0, 255), dtype=np.uint8
        )
        subtracted = apply_colormap(subtracted)
        before_img = np.array(before_img * norm, dtype=np.uint8)
        after_img = np.array(after_img * norm, dtype=np.uint8)

        # plot 3 orthogonal slices
        for name, img in zip(["plain", "with_contrast"], [before_img, after_img]):
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

        for name, img in zip(
            ["difference", "gradient", "subtracted"], [diff, grad, subtracted]
        ):
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
        img1 = []
        img2 = []
        img2_1 = []
        grads = []
        subs = []
        for i in range(img_shape[2]):
            img1.append(Image.fromarray(before_img[:, :, i], "L"))
            img2.append(Image.fromarray(after_img[:, :, i], "L"))
            img2_1.append(Image.fromarray(diff[:, :, i, :], "RGB"))
            grads.append(Image.fromarray(grad[:, :, i, :], "RGB"))
            subs.append(Image.fromarray(subtracted[:, :, i, :], "RGB"))

        img1[0].save(
            os.path.join(results_dir, "plain.gif"),
            save_all=True,
            append_images=img1[1:],
            duration=42,
        )
        img2[0].save(
            os.path.join(results_dir, "with_contrast.gif"),
            save_all=True,
            append_images=img2[1:],
            duration=42,
        )
        img2_1[0].save(
            os.path.join(results_dir, "difference.gif"),
            save_all=True,
            append_images=img2_1[1:],
            duration=42,
        )
        grads[0].save(
            os.path.join(results_dir, "gradient.gif"),
            save_all=True,
            append_images=grads[1:],
            duration=42,
        )
        subs[0].save(
            os.path.join(results_dir, "subtracted.gif"),
            save_all=True,
            append_images=subs[1:],
            duration=42,
        )
