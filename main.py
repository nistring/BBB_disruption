from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import *
from multiprocessing import Pool, cpu_count
import os
import keyboard

duration = 500
before_dir = "BrainPrep/data/FLAIRBrain/1/"
after_dir = "BrainPrep/data/FLAIR/2/"


def loader(subject):
    # convert into numpy array
    before_img, aspects = load_3d(before_dir, subject)
    after_img, _ = load_3d(after_dir, subject)
    after_img[before_img == 0] = 0

    # # calculate structural similarity score
    # score, diff = ssim(
    #     before_img,
    #     after_img,
    #     full=True,
    #     data_range=after_img.max(),
    #     win_size=3,
    #     gradient=False,
    # )

    # # normalization
    # diff = np.clip(1 - diff, 0, 1)
    # diff = (diff * 255).astype("uint8")
    # diff = apply_colormap(diff)

    
    subtracted = after_img - before_img
    subtracted_max = 512

    counts, bins = np.histogram(subtracted, range=(1, subtracted_max), bins = subtracted_max)
    plt.stairs(counts, bins)
    plt.savefig(os.path.join('results', subject, "Histogram_of_subtracted.png"))

    norm = 255.0 / after_img.max()
    before_img = np.array(before_img * norm, dtype=np.uint8)
    after_img = np.array(after_img * norm, dtype=np.uint8)

    vmin, vmax = 48, 304
    combined = np.clip((subtracted - vmin) / (vmax - vmin), 0, 1)
    mask = np.logical_or(combined >= 1, combined <= 0)
    mask = np.stack([mask, mask, mask], axis=-1)
    combined = (combined * 255.0).astype(np.uint8)
    combined = apply_colormap(combined)
    combined[mask] = np.stack([after_img, after_img, after_img], axis=-1)[mask]

    return before_img, after_img, combined, aspects


def make_gif(subject):
    results_dir = os.path.join("results", subject)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    before_img, after_img, combined, aspects = loader(
        subject
    )

    ax_aspect, sag_aspect, cor_aspect = aspects
    img_shape = before_img.shape
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
        ["combined"], [combined]
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
    img_list = []
    for i in range(img_shape[2]):
        b_img = before_img[:, :, i]
        b_img = np.stack([b_img, b_img, b_img], axis=-1)
        a_img = after_img[:, :, i]
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
        img.save(os.path.join(dst_dir, str(i)+'.png'), 'PNG')


def show_image_with_control_bar(before_img, after_img, subtracted):
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(bottom=0.25)

    ax[0].imshow(before_img)
    ax[1].imshow(after_img)

    axcolor = "lightgoldenrodyellow"
    axmin = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)

    smin = plt.Slider(
        axmin, "Min", np.min(subtracted), np.max(subtracted), valinit=np.min(subtracted)
    )
    smax = plt.Slider(
        axmax, "Max", np.min(subtracted), np.max(subtracted), valinit=np.max(subtracted)
    )

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


if __name__ == "__main__":

    # for subject in os.listdir(before_dir):
    #     before_img, after_img, subtracted_color, diff, combined, subtracted = loader(
    #         subject
    #     )
    #     current_image = 0
    #     image_len = before_img.shape[2]

    #     not_quit = True
    #     while True:
    #         key = keyboard.read_key()
    #         if key == 'q':
    #             not_quit = False
    #         elif key == 'a':
    #             current_image -= 1
    #             if current_image < 0:
    #                 current_image = image_len - 1
    #         elif key == 'd':
    #             current_image += 1
    #             if current_image >= image_len:
    #                 current_image = 0
    #         else:
    #             pass

    #     show_image_with_control_bar(before_img[0], after_img[0], subtracted[0])

    pool = Pool(processes=cpu_count())
    pool.map(make_gif, os.listdir(before_dir))
