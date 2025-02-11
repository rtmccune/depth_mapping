# Functions used for depth mapping processes. Including both pre and post processing.
### Imports ###
import os
import re
import pytz
import cv2
import shutil
import random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageColor
import os
import re
import zarr
import cv2

import numpy as np
import pandas as pd

from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator as reg_interp

### Pre-processing ###

def create_labels_from_preds(preds_folder, labels_destination, color_map=None):
    # Check if the directory exists
    os.makedirs(labels_destination, exist_ok=True)

    # Set default color map corresponding to plotly qualitative G10
    if color_map is None:
        color_map = {
            '#3366CC': '#3366CC',
            '#DC3912': '#DC3912',
            '#FF9900': '#FF9900',
            '#109618': '#1BB392',
            '#990099': '#5B1982',
            '#0099C6': '#C1C4C9',
            '#DD4477': '#FA9BDA',
            '#66AA00': '#A2DDF2',
            '#B82E2E': '#047511',
            '#316395': '#755304'
        }

    # Precompute RGB values from keys in color map.
    orig_rgb = [ImageColor.getrgb(hex_color_in) for hex_color_in in color_map.keys()]

    # Create new dictionary mapping of integers to RGB values.
    orig_rgb_dict = {0: (0, 0, 0)}
    for i, rgb in enumerate(orig_rgb, start=1):
        orig_rgb_dict[i] = rgb

    preds_list = os.listdir(preds_folder)

    # Process each prediction image in parallel
    def process_image(preds):
        preds_path = os.path.join(preds_folder, preds)
        preds_image = cv2.imread(preds_path)

        # Convert image to RGB
        preds_image_rgb = cv2.cvtColor(preds_image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image into integer labels
        gray_img = orig_rgb_to_gray_labels(preds_image_rgb, orig_rgb_dict)

        file_root, file_ext = os.path.splitext(preds)

        # Create the new filename with the label appended
        label_image_name = f"{file_root}_labels.png"
        label_image_path = os.path.join(labels_destination, label_image_name)

        cv2.imwrite(label_image_path, gray_img)

    with tqdm(total=len(preds_list), desc='Generating labels from predictions') as pbar:
        with ThreadPoolExecutor() as executor:
            for _ in executor.map(process_image, preds_list):
                pbar.update(1)
                

def orig_rgb_to_gray_labels(rgb_image, color_map):
    """
    Converts an RGB image to a grayscale array representing labels based on the given color map.
    :param rgb_image: numpy array
        RGB image array.
    :param color_map: dict
        Color map where RGB colors are mapped to labels.
    :return: numpy array
        Grayscale array representing labels.
    """

    # Initialize labels array with zeros
    labels_image = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    # Create masks for each color in the color map
    masks = [(rgb_image == np.array(color)).all(axis=2) for color in color_map.values()]

    # Combine masks to find the label for each pixel
    for label, mask in enumerate(masks):
        labels_image[mask] = label

    return labels_image

def gen_color_labels(image, color_map=None):
    """
    This function generates a colored label representation using the grayscale labels from Doodler.
    :param image: array
        An image array from reading the Doodler label in with opencv.
    :param color_map: dict
        An optional dictionary of hex color mappings. Defaults to colors used for segmap overlays.
    :return: array
        A color image (BGR) representing class labels.
    """
    # Set default color map.
    if color_map is None:
        color_map = {
            0: '#3366CC',
            1: '#DC3912',
            2: '#FF9900',
            3: '#109618',
            4: '#990099',
            5: '#0099C6',
            6: '#DD4477',
            7: '#66AA00',
            8: '#B82E2E',
            9: '#316395',
            10: '#000000'
        }

    # Convert default BGR image format to grayscale.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Precompute RGB colors from hex values.
    color_labels = sorted(color_map.keys())
    bgr_colors = []
    for label in color_labels:
        hex_color = color_map[label]
        rgb_color = np.array(ImageColor.getrgb(hex_color), dtype=np.uint8)
        bgr_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2BGR)[0][0]
        bgr_colors.append(bgr_color)
    bgr_colors = np.array(bgr_colors)

    # Map grayscale pixels to BGR colors using color_map.
    bgr_image = bgr_colors[gray_image]

    return bgr_image

def recolor_image(image, color_map=None):
    """
        This function takes a read image from opencv and recolors it according
        to the desired color mapping with a programmed plotly G10 default.
    :param image: numpy array
        Image for recoloring already read in using opencv.
    :param color_map: dict
        Hex color to hex color dictionary for color mapping.
    :return: numpy array
        This function returns an image as an array in BGR format.
    """
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set default color map corresponding to plotly qualitative G10
    if color_map is None:
        color_map = {
            '#3366CC': '#3366CC',
            '#DC3912': '#DC3912',
            '#FF9900': '#FF9900',
            '#109618': '#1BB392',
            '#990099': '#5B1982',
            '#0099C6': '#C1C4C9',
            '#DD4477': '#FA9BDA',
            '#66AA00': '#A2DDF2',
            '#B82E2E': '#047511',
            '#316395': '#755304'
        }

    # Precompute RGB values from keys in color map.
    orig_rgb = [ImageColor.getrgb(hex_color_in) for hex_color_in in
                color_map.keys()]

    # Create new dictionary mapping of integers to RGB values.
    orig_rgb_dict = {0: (0, 0, 0)}
    for i, rgb in enumerate(orig_rgb, start=1):
        orig_rgb_dict[i] = rgb

    # Create new dictionary relating new integers to the desired new hex colors.
    # We accelerated the processing using numpy arrays and dictionaries, so we
    # are converting the provided image into grayscale integer labels and then
    # converting back to the desired hex colors.
    new_int_to_hex_dict = {0: '#000000'}
    for i, (key, value) in enumerate(color_map.items(), start=1):
        new_int_to_hex_dict[i] = value

    # Convert the RGB image into integer labels.
    gray_img = orig_rgb_to_gray_labels(image_rgb, orig_rgb_dict)
    # Reshape to three channels to work with gen_color_labels function
    gray_img = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)

    # Create a color image from grayscale integers (same function used for Doodler
    # labels).
    bgr_image = gen_color_labels(gray_img, new_int_to_hex_dict)

    return bgr_image

def gen_overlay(background_img, foreground_img, alpha=None, gamma=None, color_map=None):
    """
    This function takes two images and creates an overlay.
    :param background_img: array
        The desired background image already read in using opencv (cv2.imread()).
    :param foreground_img: array
        The desired foreground image already read in using opencv (cv2.imread()).
    :param alpha: int
        Integer from 0 to 1 representing level of transparency for the foreground image.
        Default value is 0.5.
    :param gamma: int
        Additional weight for the cv2.addWeighted() function defaults to 0.0.
    :param color_map: dict
        Color map dictionary if using color map other than the default ten colors from
        plotly G10 palette.
    :return: array
        Overlay image array in opencv format (BGR).
    """
    # Set default alpha value.
    if alpha is None:
        alpha = 0.5

    # Calculate beta value. Transparency of background image in cv2.addWeighted().
    beta = 1 - alpha

    # Set default gamma value.
    if gamma is None:
        gamma = 0

    # Call recolor function for segmentation map if using default color map.
    if color_map is None:
        recolored_foreground = recolor_image(foreground_img)
    # Call recolor function with provided color map.
    else:
        recolored_foreground = recolor_image(foreground_img, color_map)

    # Create overlay image.
    overlay = cv2.addWeighted(recolored_foreground, alpha, background_img, beta, gamma)

    return overlay

def gen_segmap_overlays(image_folder, segs_folder, destination_folder=None, alpha=None, gamma=None,
                        color_map=None):
    """
        This function generates a folder of segmentation map overlays given a folder
        of original images and folder of predicted segmentations.
    :param image_folder: str
        Path of image folder.
    :param segs_folder: str
        Path of predicted segmentation folder.
    :param destination_folder: str
        Path to create folder for overlays.
    :param alpha: int
        Integer from 0 to 1 representing level of transparency for the foreground image.
        Default value is 0.5.
    :param gamma: int
        Additional weight for the cv2.addWeighted() function defaults to 0.0.
    :param color_map: dict
        Color map dictionary if using color map other than the default ten colors from
        plotly G10 palette.
    :return:
        The function will print a message after the task has completed but there are no
        variables returned.
    """
    # Create folder to store segmap overlays.
    if destination_folder is None:
        destination_folder = 'segmentation_overlays'
    if os.path.exists(destination_folder):
        os.makedirs(os.path.join(destination_folder,'/segmentation_overlays'))
    else:
        print("Segmentation overlays folder already exists.")

    # Read in folders of images and segmentation maps.
    if not os.path.exists(image_folder):
        raise FileNotFoundError("Provided image folder path does not exist.")
    if not os.path.exists(segs_folder):
        raise FileNotFoundError("Provided segmentation map folder path does not exist.")

    # Collect image files
    image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
                   if not filename.startswith('.') and os.path.isfile(os.path.join(image_folder, filename))]

    # Collect segmentation map files
    segmap_files = [os.path.join(segs_folder, filename) for filename in os.listdir(segs_folder)
                    if not filename.startswith('.') and os.path.isfile(os.path.join(segs_folder, filename))]

    # Sort the files so that they match.
    image_files.sort()
    segmap_files.sort()

    # Confirm that the number of files match before running. Will exit otherwise.
    num_images = len(image_files)
    num_segmaps = len(segmap_files)
    if num_images == num_segmaps:
        # Initialize tqdm with the total number of files
        progress_bar = tqdm(total=num_images, desc="Generating segmentation overlays...")

        # Load images from each folder and run the overlays function.
        for i, (image_file, segmap_file) in enumerate(zip(image_files, segmap_files)):
            image = cv2.imread(image_file)
            segmap = cv2.imread(segmap_file)
            overlay_img = gen_overlay(image, segmap, alpha, gamma, color_map)
            cv2.imwrite(
                os.path.join(destination_folder, f'segmap_overlay_{os.path.basename(image_file)}'),
                overlay_img)

            # Clear memory
            image = None
            segmap = None

            # Update the progress bar
            progress_bar.update(1)
    else:
        print("Number of image files and segmentation maps did not match.")
        return

    # Close the progress bar
    progress_bar.close()

    return print("Folder of segmentation overlays generated.")


def gen_grid(xMin, xMax, yMin, yMax, resolution, z=None, dir='generated_grids'):

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory to store grids created: {dir}')
    else:
        print(f'Directory to store grids already exists: {dir}')

    grid_x, grid_y = np.mgrid[xMin:xMax:resolution, yMin:yMax:resolution]

    if z is None:
        z = 0
    
    if isinstance(z, int):
        grid_z = np.full_like(grid_x, z)

    if isinstance(z, np.ndarray):
        x = z[0]
        y = z[1]
        z = z[2]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
    # Save the grid arrays to a compressed Zarr files
    zarr.save(os.path.join(dir, 'grid_x.zarr'), grid_x)
    zarr.save(os.path.join(dir, 'grid_y.zarr'), grid_y)
    zarr.save(os.path.join(dir, 'grid_z.zarr'), grid_z)

    return grid_x, grid_y, grid_z

def mergeRectifyFolder(folder_path, intrinsics, extrinsics, grid_x, grid_y, grid_z):
    
    s = grid_x.shape

    # Calculate Ud, Vd once since they are the same for all images
    Ud, Vd = xyz2DistUV(intrinsics, extrinsics, grid_x, grid_y, grid_z)
    Ud = np.round(Ud).astype(int)
    Vd = np.round(Vd).astype(int)
    
    data = []
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        I = cv2.imread(image_path)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        
        ir = getPixels(I, Ud, Vd, s)
        ir = np.array(ir, dtype=np.uint8)
        
        # Append the image name and ir array to the list
        data.append({'image_name': image_name, 'ir': ir})

    results = pd.DataFrame(data)
    
    return results