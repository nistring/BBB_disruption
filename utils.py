import pydicom
import os
import numpy as np
import cv2
from scipy.ndimage import zoom


def load_3d(dir):
    """Reads a MRI DICOM sequence and returns 3-dimensionally stacked array
    Adopted from https://pydicom.github.io/pydicom/stable/auto_examples/image_processing/reslice.html#sphx-glr-auto-examples-image-processing-reslice-py

    Args:
        dir (str): Path of directory containing DICOM files

    Returns:
        np.ndarray: 3D stacked MRI sequence
        (float, float, float): Aspect ratios
    """

    files = []
    for file in sorted(os.listdir(dir)):
        files.append(pydicom.dcmread(os.path.join(dir, file)))

    print("file count: {}".format(len(files)))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation, reverse=True)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness

    ax_aspect = ps[1] / ps[0]
    sag_aspect = ss / ps[1]
    cor_aspect = ss / ps[0]
    zoom_ratio = sag_aspect
    sag_aspect /= zoom_ratio
    cor_aspect /= zoom_ratio

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    # applay interpolation in z-direction
    img3d = zoom(img3d, (1, 1, zoom_ratio), mode="nearest")
    img3d[img3d < 0] = 0

    return img3d, (ax_aspect, sag_aspect, cor_aspect)


def apply_colormap(gray3d):
    """Convert 3D gray-scale array into 3D color array

    Args:
        gray3d (np.ndarray):  A gray-scale array to be converted

    Returns:
        np.ndarray: A 3D color array
    """
    gray3d = np.stack([gray3d, gray3d, gray3d], axis=-1)
    color3d = np.zeros_like(gray3d)
    for i in range(gray3d.shape[2]):
        color3d[:, :, i, :] = cv2.cvtColor(
            cv2.applyColorMap(gray3d[:, :, i, :], cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
        )
    return color3d