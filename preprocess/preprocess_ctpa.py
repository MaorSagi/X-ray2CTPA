import hashlib
import itertools
import logging
import traceback
import math
import os
import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from PIL import Image
from numpy.ma.extras import average
from torch.utils.data import Dataset
import re

from params import *
from utils import *
import numpy as np
import SimpleITK as sitk
# import pandas as pd
# import random
import matplotlib.pyplot as plt
import scipy.ndimage
import warnings
import scipy
from scipy.ndimage import zoom

# import vtk
# from vtk.util import numpy_support
import cv2
import glob
from scipy import ndimage
import pydicom as dicom
from lungmask import mask, LMInferer
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)

# Hounsfield Units for Air
AIR_HU_VAL = -1024.




# CONTRAST_HU_MEAN = 0.15897  # Mean voxel value after normalization and clipping
# CONTRAST_HU_STD = 0.19974   # Standard deviation of voxel values after normalization and clipping
LOGS_DIR_PATH = 'logs'
MASKS_DIR_PATH = 'masks'
ct_xray_manual_accession = ['4015010182925']

FILLIN_VALUE_SCAN = -1000  # Air Hounsfield Units, as placeholder for failed segmentation


def window(img, WL=50, WW=350):
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X * 255.0).astype('uint8')
    return X


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def get_series_info(dicom_file):
    ds = dicom.dcmread(dicom_file)

    # #TODO Test if it is not breaking anything
    try:
        SeriesDescription = ds.SeriesDescription
    except:
        SeriesDescription = 'NA'

    series_info = {
        'SeriesInstanceUID': ds.SeriesInstanceUID,
        'SeriesDescription': SeriesDescription
    }

    try:
        series_info['SeriesOrientation'] = ds.ImageOrientationPatient
    except:
        # TODO : add logger error/warning
        series_info['SeriesOrientation'] = 'NA'
    return series_info


def get_first_of_dicom_field_as_int(x):
    if type(x) == dicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def make_rgb(volume):
    """Tile a NumPy array to make sure it has 3 channels."""
    z, c, h, w = volume.shape

    tiling_shape = [1] * (len(volume.shape))
    tiling_shape[1] = 3
    np_vol = torch.tile(volume, tiling_shape)
    return np_vol


def encode_ctpa(ct, vae):
    device = "cuda"
    weight_dtype = torch.float32
    ct = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0)

    print("ct before vae = ", ct.shape)
    vae.eval()

    latents = []
    with torch.no_grad():
        for i in range(ct.shape[2]):
            slice = make_rgb(ct[:, :, i, :, :])
            latent = vae.encode(slice.to(device, dtype=weight_dtype)).latent_dist.sample()
            latents.append(latent)
        latents = torch.stack(latents, dim=4).squeeze()  # .unsqueeze(0)#.permute(0,2,3,4,1)

        latents = latents.detach().cpu().numpy()
        print("latents shape", latents.shape)
        # The scale_factor ensures that the initial latent space on which the diffusion model is operating has approximately unit variance.
        # Used to scale the latents so it can be decoded back into an image
        latents = latents * 0.18215

    return latents


def normalize(img):
    img = img.astype(np.float32)
    img = (img - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN)
    img = np.clip(img, 0., 1.) * 2 - 1
    return img


def resample_volume(volume, current_spacing, new_spacing):
    resize_factor = np.array(current_spacing) / new_spacing
    new_real_shape = volume.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / volume.shape
    new_spacing = np.array(current_spacing) / real_resize_factor

    print("new sampling = ", new_spacing)
    resampled_volume = scipy.ndimage.interpolation.zoom(volume, real_resize_factor)
    return resampled_volume


def resize_volume(tensor, output_size):
    z, h, w = tensor.shape
    resized_scan = np.zeros((output_size[0], output_size[1], output_size[2]))

    volume = tensor[:, :, :].squeeze()

    real_resize_factor = np.array(output_size) / np.shape(volume)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized_scan[:, :, :] = scipy.ndimage.interpolation.zoom(volume, real_resize_factor, mode='nearest').astype(
            np.int16)

    return resized_scan


def find_largest_connected_components(image):
    # Find connected components in the binary image
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image)

    # Sort the connected components by area in descending order
    sorted_stats = sorted(stats, key=lambda x: x[4], reverse=True)

    # Get the labels of the two largest connected components (excluding background)
    largest_labels = [sorted_stats[i][0] for i in range(1, min(3, len(sorted_stats)))]

    # Create a mask with the two largest connected components
    mask = np.isin(labels, largest_labels).astype(np.uint8)

    return mask


def get_lungs_voi(scan, scan_dst):
    '''
    This funtion finds the lungs VOI from the given 3D scan.
    '''
    # Segment lungs and create lung mask with one label
    # lungs = mask.apply(scan)
    modelpath = "weights/R231/unet_r231-d5d2fc3d.pth"
    lungs = LMInferer(modelpath=modelpath).apply(scan)
    # plt.imsave(scan_dst + '22.png', lungs[int(len(lungs) / 2)], cmap='gray')
    # tmp = scan[int(len(scan) / 2)] + lungs[int(len(lungs) / 2)]
    # plt.imsave(scan_dst + '222.png', tmp, cmap='gray')
    # rgb_image = create_rgb_image(scan[int(len(scan) / 2)], lungs[int(len(lungs) / 2)])
    # plt.imsave(scan_dst + '222.png', rgb_image)
    # os.makedirs(scan_dst, exist_ok=True)
    # create_video(scan, lungs, scan_dst + "/ct_video.mp4", vmin=-150, vmax=300, alpha=0.5, fps=20)
    lung_pixels = np.nonzero(lungs)
    if len(lung_pixels[0]) == 0:
        # Padding if z axis lung mask is empty
        return np.full(scan.shape, FILLIN_VALUE_SCAN)

    # Get values from lung segmentation
    min_z = np.amin(lung_pixels[0])
    max_z = np.amax(lung_pixels[0])
    min_x = np.amin(lung_pixels[1])
    max_x = np.amax(lung_pixels[1])
    min_y = np.amin(lung_pixels[2])
    max_y = np.amax(lung_pixels[2])

    # Crop scan to lung VOI
    cropped_scan = scan[min_z:max_z, min_x:max_x, min_y:max_y]
    cropped_lungs = lungs[min_z:max_z, min_x:max_x, min_y:max_y]
    return cropped_scan, cropped_lungs


def create_rgb_image(ct_slice, mask, vmin=-150, vmax=300, alpha=0.2):
    """
    Create an RGB image by overlaying the mask (orange for gray, green for white) with adjustable opacity
    on a CT slice windowed to the specified [vmin, vmax].

    Parameters:
        ct_slice (numpy.ndarray): Grayscale CT slice.
        mask (numpy.ndarray): Mask for lungs with labeled regions (0=black, 1=gray, 2=white).
        vmin (int): Minimum intensity value for windowing.
        vmax (int): Maximum intensity value for windowing.
        alpha (float): Opacity for the overlay colors (0=fully transparent, 1=fully opaque).

    Returns:
        numpy.ndarray: RGB image (3-channel array) with the overlay applied.
    """
    # Apply windowing only to grayscale CT slice (clip values in CT slice range [vmin, vmax])
    ct_windowed = np.clip(ct_slice, vmin, vmax)

    # Normalize the windowed CT slice to [0, 1] for visualization purposes
    normalized_ct = (ct_windowed - vmin) / (vmax - vmin)

    # Convert the normalized grayscale CT slice to RGB format (repeat grayscale in R, G, B channels)
    ct_rgb = np.stack([normalized_ct] * 3, axis=-1)  # Shape = (H, W, 3)

    # Create RGB colors for the mask overlays
    mask_colors = np.zeros_like(ct_rgb)  # Create an empty RGB array matching CT shape
    mask_colors[mask == 1] = [1, 0.7, 0.4]  # Orange for label 1 (gray region in mask)
    mask_colors[mask == 2] = [0, 0.8, 0.4]  # Green for label 2 (white region in mask)
    print("Hello!!")
    # Blend the mask colors with the CT slice using the specified alpha (opacity)
    overlay_image = np.where(mask[..., np.newaxis] != 0,  # Mask is nonzero
                             alpha * mask_colors + (1 - alpha) * ct_rgb,  # Blend mask colors with CT RGB
                             ct_rgb)  # Default to CT grayscale when mask is zero

    return overlay_image


def create_video(ct_slices, masks, output_path, vmin=-150, vmax=300, alpha=0.5, fps=10):
    """
    Create a video from a series of CT slices with mask overlays.

    Parameters:
        ct_slices (numpy.ndarray): 3D array of grayscale CT slices (num_slices, height, width).
        masks (numpy.ndarray): 3D array of masks (num_slices, height, width).
        output_path (str): Path where the video will be saved.
        vmin (int): Minimum intensity value for windowing.
        vmax (int): Maximum intensity value for windowing.
        alpha (float): Opacity for the overlay colors (0=fully transparent, 1=fully opaque).
        fps (int): Frames per second for the video.
    """
    # Get dimensions of CT slices
    num_slices, height, width = ct_slices.shape

    # Create a temporary folder to store frames
    temp_folder = "temp_frames"
    os.makedirs(temp_folder, exist_ok=True)

    # Generate RGB frames for each slice
    print("Generating frames...")
    for i in range(num_slices):
        ct_slice = ct_slices[i]
        mask_slice = masks[i]

        # Create RGB image with mask overlay for each slice
        rgb_image = create_rgb_image(ct_slice, mask_slice, vmin, vmax, alpha)

        # Convert RGB image from [0, 1] to [0, 255] for saving as image
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

        # Write frame as an image
        frame_path = os.path.join(temp_folder, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2BGR))

    # Get frame dimensions
    frame_shape = rgb_image_uint8.shape[:2]  # (height, width)

    # Create video writer
    print(f"Writing video to {output_path}...")
    video_writer = cv2.VideoWriter(output_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),  # Codec for MP4 format
                                   fps,
                                   (frame_shape[1], frame_shape[0]))  # (width, height)

    # Read all frames and add them to the video
    for i in range(num_slices):
        frame_path = os.path.join(temp_folder, f"frame_{i:04d}.png")
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release video writer
    video_writer.release()

    # Clean up temporary frames
    print(f"Cleaning up temporary frames...")
    for i in range(num_slices):
        frame_path = os.path.join(temp_folder, f"frame_{i:04d}.png")
        os.remove(frame_path)
    os.rmdir(temp_folder)

    print(f"Video saved to {output_path}!")


import numpy as np
import nibabel as nib  # For saving in NIfTI format

import os
import numpy as np
import nibabel as nib  # For saving in NIfTI format


def save_3d_lungs_mask(mask_3d, filename, dir_path=MASKS_DIR_PATH, save_format='npy'):
    """
    Save a 3D lungs mask to disk in the specified format and directory.

    Parameters:
        mask_3d (numpy.ndarray): 3D NumPy array representing the lungs mask (shape: [depth, height, width]).
        dir_path (str): Directory path where the mask file will be saved.
        filename (str): The name of the saved file (without extension).
        save_format (str): Format to save the file ('npy' or 'nii.gz').

    Raises:
        ValueError: If `save_format` is unsupported.
    """
    # Make sure the directory exists, or create it
    os.makedirs(dir_path, exist_ok=True)

    # Construct the full output path by combining directory, filename, and extension
    if save_format.lower() == 'npy':
        output_path = os.path.join(dir_path, f"{filename}.npy")
        np.save(output_path, mask_3d)
        print(f"3D lungs mask saved as .npy file at {output_path}")

    elif save_format.lower() == 'nii.gz':
        output_path = os.path.join(dir_path, f"{filename}.nii.gz")
        mask_affine = np.eye(4)  # Use an identity matrix for affine unless specified otherwise
        nii_mask = nib.Nifti1Image(mask_3d.astype(np.uint8), mask_affine)
        nib.save(nii_mask, output_path)
        print(f"3D lungs mask saved as .nii.gz file at {output_path}")

    else:
        raise ValueError(f"Unsupported save format: {save_format}. Use 'npy' or 'nii.gz'.")


def preprocess_ctpa(img, attr, scan_dst):
    """reshape, normalize image and convert to tensor"""
    img_cropped, lungs = get_lungs_voi(img, scan_dst)
    ## TODO: save 3D lungs masks?
    print("Cropping scan to lung VOI:   ", img_cropped.shape)
    z, x, y = img_cropped.shape
    if (z and x and y) == 0:
        print(" failed to compute voi : compute manually")
        return 0, 0

    # Resample
    # Maor - TODO: try to remove this function from the pipeline - Ayelet E. said it probably redundant
    spacing = attr['Spacing']
    img_resampled = resample_volume(img_cropped, spacing, [1, 1, 1])

    # Rescale

    img_resize = resize_volume(img_resampled, [128, 256, 256])

    # noramlize and window Hounsfield Units 
    img_normalized = normalize(img_resize)
    return img_normalized


def calculate_current_spacing(slices):
    spacings = []

    for slice in slices:
        pixel_spacing = slice.PixelSpacing
        slice_thickness = slice.SliceThickness

        # Convert the spacing values to floats
        pixel_spacing = [float(spacing) for spacing in pixel_spacing]
        slice_thickness = float(slice_thickness)

        spacings.append((slice_thickness, pixel_spacing[0], pixel_spacing[1]))

    spacing = list(np.quantile(np.array(spacings), 0.5, axis=0))

    spacing_CHECK = [float(slices[0].SliceThickness),
                     float(slices[0].PixelSpacing[0]),
                     float(slices[0].PixelSpacing[0])]

    if not all(np.isclose(spacing, spacing_CHECK, rtol=0, atol=1e-5)):
        # spacing = spacing_CHECK
        # Spacing is relevant only for resampling interpolation
        print("WARNING: Spacing checking failed!!!")

    return spacing


# Either path or slices should be given
def load_scan(path=None, slices=None):
    # Two Sorting options: 'InstanceNumber', 'SliceLocation'
    attr = {}
    if slices == None:
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    else:
        slices = [dicom.dcmread(s) for s in slices]
    # For missing ImagePositionPatient
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    slices2 = []
    prev = -1000000
    # remove redundant slices
    for slice in slices:
        # For missing ImagePositionPatient
        cur = slice.ImagePositionPatient[2]

        if cur == prev:
            continue
        prev = cur
        slices2.append(slice)
    slices = slices2

    for i in range(len(slices) - 1):
        try:
            slice_thickness = np.abs(slices[i].ImagePositionPatient[2] - slices[i + 1].ImagePositionPatient[2])
        except:
            # TODO : add logger error/warning
            slice_thickness = np.abs(slices[i].SliceLocation - slices[i + 1].SliceLocation)
        if slice_thickness != 0:
            break

    spacing = calculate_current_spacing(slices)

    for s in slices:
        s.SliceThickness = slice_thickness

    x, y = slices[0].PixelSpacing

    if slice_thickness == 0:
        attr['Spacing'] = spacing[0]
    else:
        attr['Spacing'] = (slice_thickness, x, y)

    attr['Position'] = slices[0].ImagePositionPatient
    attr['Orientation'] = slices[0].ImageOrientationPatient

    return (slices, attr)


def dicom_load_scan(paths):
    attr = {}
    slices = [dicom.dcmread(path) for path in paths]
    slices.sort(key=lambda x: int(x.InstanceNumber), reverse=True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        # TODO : add logger error/warning
        if len(slices) == 0:
            return None, None
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    spacing = calculate_current_spacing(slices)

    for s in slices:
        s.SliceThickness = slice_thickness

    # Old code - new spacing check and calc
    # if not all(np.isclose(slices[0].PixelSpacing, (spacing[1], spacing[2]), rtol=0, atol=1e-5)):
    #     x, y = slices[0].PixelSpacing
    # else:
    x, y = spacing[1], spacing[2]

    if slice_thickness == 0:
        attr['Spacing'] = spacing[0]
    else:
        attr['Spacing'] = (slice_thickness, x, y)

    attr['Position'] = slices[0].ImagePositionPatient
    attr['Orientation'] = slices[0].ImageOrientationPatient

    window_center, window_width, intercept, slope = get_windowing(slices[0])
    attr['window_center'] = window_center
    attr['window_width'] = window_width
    attr['intercept'] = intercept
    attr['slope'] = slope

    return (slices, attr)


def get_pixels_hu(slices):
    # print([s.pixel_array.shape for s in slices])
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0  # -1024

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def dicom_get_pixels_hu(scans):
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    # Stack all slices into a 3D numpy array
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16, should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels.
    # Common DICOM intercept is -1024, thus raw pixel value 0 becomes air (~ -1024 HU).
    # image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    # HU formula = slope * ARRAY + intercept
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_high_resolusion_dicom_series(folder_path):
    series_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            dicom_file = os.path.join(root, file)
            if not dicom.dcmread(dicom_file).get('SeriesInstanceUID'):
                continue

            series_info = get_series_info(dicom_file)
            series_uid = series_info['SeriesInstanceUID']

            if series_uid not in series_dict:
                series_dict[series_uid] = []

            if series_info['SeriesOrientation'] != 'NA':
                # Choosing CT series that are standard axial scans (not tilted or reformatted), oriented straight along the patient’s body axes.
                if np.allclose(np.array(series_info['SeriesOrientation']).astype(float), [1., 0., 0., 0., 1., 0.],
                               rtol=1e-3):
                    series_dict[series_uid].append(dicom_file)

    longest_series = max(series_dict.values(), key=len)
    print(series_info['SeriesOrientation'])
    print(len(longest_series))
    return longest_series


def preprocess_ctpa_directory(src_path, dst_path):
    device = DEVICE
    weight_dtype = torch.float32
    longest_series_len_list = []
    hu_min, hu_max = float('inf'), -float('inf')

    local_path = "weights/vae-ft-mse-840000-ema-pruned.safetensors"  # url="https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    config_path = "weights/config.json"
    vae = AutoencoderKL.from_single_file(local_path, local_files_only=True, config=config_path)
    vae.to(device, dtype=weight_dtype)

    src_paths = os.listdir(src_path)
    for i, accession_number in enumerate(src_paths):  # [int(len(os.listdir(src_path))/2):-1]:#[int(len(os.listdir(src_path))/2):]:#[::-1]:#[int(len(os.listdir(src_path))/2):-1]:
        scan = src_path + accession_number

        if not os.path.isdir(scan):
            continue
        scan_dst = dst_path + accession_number

        if os.path.isfile(scan_dst + '.npy'):
            # print("File exists - ", scan_dst + '.npy')
            pass
        else:
            print(f"processing scan - {scan}, {i}/{len(src_paths)}")
            if accession_number in ct_xray_manual_accession:
                continue
            try:
                highers_series = get_high_resolusion_dicom_series(scan)
                longest_series_len_list.append(len(highers_series))

                slices, attr = dicom_load_scan(highers_series)
                if slices is None:
                    continue
                # sitk.WriteImage(sitk.GetImageFromArray(slices[int(len(slices)/2)].pixel_array), scan_dst + '.png')

                ct = dicom_get_pixels_hu(slices)
                hu_min = min(hu_min, np.min(ct))
                hu_max = max(hu_max, np.max(ct))
                # sitk.WriteImage(sitk.GetImageFromArray(ct[int(len(ct)/2)]), scan_dst + '.png')
                # plt.imsave(scan_dst + '1.png',ct[int(len(ct)/2)], cmap='gray', vmin=-150, vmax=300)
                # plt.imsave(scan_dst + '11.png', ct[int(len(ct) / 2)], cmap='gray')

                print(
                    f"percentile hu ct scan [0,5,25,50,75,95,98,99,100]: {np.percentile(ct, [0, 5, 25, 50, 75, 95, 98, 99, 100])}")
                print(
                    f"AVG series length: {sum(longest_series_len_list) / len(longest_series_len_list) if sum(longest_series_len_list) > 0 else None}")
                print(
                    f"MAX series length: {max(longest_series_len_list)}")
                print(
                    f"MIN series length: {min(longest_series_len_list)}")
                print("Observed HU range:", hu_min, hu_max)

                print("after loading exam = ", ct.shape)
                ct = preprocess_ctpa(ct, attr, scan_dst)
                print("after preprocessing = ", ct.shape)
                # sitk.WriteImage(sitk.GetImageFromArray(ct[int(len(ct)/2)]), scan_dst + '.png')
                # plt.imsave(scan_dst + '3.png', ct[int(len(ct) / 2)], cmap='gray')

                latents = encode_ctpa(ct, vae)
                print("scan_dst = ", scan_dst)
                # sitk.WriteImage(sitk.GetImageFromArray(latents), scan_dst + '.nii')
                # plt.imsave(scan_dst + '4.png', ct[int(len(ct) / 2)], cmap='gray')
                # plt.imsave(scan_dst + '4.png', ct[int(len(ct) / 2)], cmap='gray')

                np.save(scan_dst, latents)

            except:
                # TODO : add logger error
                print(f"Skip: {accession_number}, Error: {traceback.format_exc()}")
        # print(f"AVG series length: {sum(longest_series_len_list)/len(longest_series_len_list) if sum(longest_series_len_list) > 0 else None}")
        # print("Observed HU range:", hu_min, hu_max)
    return


if __name__ == "__main__":
    src_path = r"data/train/"  # r"\\sbmdcl_nas\dicomer$\orig_data\PE_CT\\"
    dst_path = "preprocessed_data/XRayCTPA/CTPA_256/"
    os.makedirs(dst_path, exist_ok=True)
    preprocess_ctpa_directory(src_path, dst_path)
