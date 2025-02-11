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