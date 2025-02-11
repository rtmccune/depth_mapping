import os
import re
import cv2
import pytz
import shutil
import numpy as np
from datetime import datetime
from PIL import ImageColor
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_hex_colors(image_path):
    """
    This function provides a list of hex colors present in an image.
    :param image_path: str
        A string representing the path to an image.
    :return: dict
        Returns hex_colors dictionary containing all hex color codes present in an image.
    """

    # Read in image
    image = cv2.imread(image_path)

    # Return error message if image is empty
    if image is None:
        print("Error: Unable to load image.")
        return []

    # Initialize dictionary for hex colors
    hex_colors = set()

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image into a 2D array of RGB values
    pixels = np.reshape(image_rgb, (-1, 3))

    # Convert RGB to hex
    for pixel in pixels:
        hex_color = '#{0:02x}{1:02x}{2:02x}'.format(*pixel)
        hex_colors.add(hex_color)

    # Return dictionary of present hex colors
    return hex_colors


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


def extract_utc_str(filename):
    """
    This function extracts the UTC timestamps from Sunny Day Flooding Project image file names.
    :param filename: str
        Image file name.
    :return: str
        Returns a string UTC timestamp.
    """
    # Define expression pattern to match UTC timestamps.
    # Assuming the timestamp format is YYYYMMDDHHMMSS (e.g., 20240313154500)
    pattern = r'(\d{14})'

    # Search for the pattern in the filename.
    match = re.search(pattern, filename)

    if match:
        # Extract the matched timestamp.
        timestamp = match.group(1)
        return timestamp
    else:
        return None


def utc_str_to_datetime(utc_timestamp_str):
    """
    This function converts a string UTC timestamp to a datetime.
    :param utc_timestamp_str: str
        UTC timestamp string extracted from filename.
    :return: datetime
        Datetime object of UTC timestamp.
    """
    # Define the format of the UTC timestamp string.
    timestamp_format = "%Y%m%d%H%M%S"

    # Convert the UTC timestamp string to a datetime object.
    timestamp_datetime = datetime.strptime(utc_timestamp_str, timestamp_format)

    return timestamp_datetime


def convert_utc_to_eastern(utc_datetime):
    """
    This function converts the UTC timestamp to Eastern time (either EDT or EST).
    :param utc_datetime: datetime
        The UTC timestamp as a datetime object.
    :return: datetime
        The datetime object in Eastern time.
    """
    # Define the UTC timezone.
    utc_timezone = pytz.timezone('UTC')

    # Define the eastern timezone (shifts between EDT and EST).
    est_timezone = pytz.timezone('US/Eastern')

    # Localize the UTC datetime to UTC timezone.
    utc_datetime = utc_timezone.localize(utc_datetime)

    # Convert the localized datetime to EST timezone.
    est_datetime = utc_datetime.astimezone(est_timezone)

    return est_datetime


def is_daytime(timestamp, start=None, end=None):
    """
    This function checks if the timestamp provided falls within the bounds of a
    specified daytime. Daytime defaults to 5AM to 9PM.
    :param timestamp:
    :param start: int
        Start time (hour).
    :param end: int
        End time (hour).
    :return: boolean
        Returns a True or False value.
    """
    # Define daylight hours.
    if start is None:
        start = 5  # 5 AM
    if end is None:
        end = 21   # 9 PM

    # Check if the hour of the timestamp falls within the daylight hours.
    return start <= timestamp.hour < end


def filter_files_by_daylight_hours(folder_path, start=None, end=None):
    """
    This function filters a folder of files to return a list of those files with
    timestamps in the specified daytime hours.
    :param folder_path: str
        A string representing the folder path.
    :param start: int
        Integer representing daylight start time (hour). (e.g. 5 is 5AM)
    :param end: int
        Integer representing daylight end time (hour). (e.g. 19 is 7PM)
    :return: list
        A list of the filtered file names.
    """
    # Initialize file list.
    filtered_files = []

    # Remove hidden files from the folder list.
    files = [filename for filename
             in os.listdir(folder_path) if not filename.startswith('.')]

    # Iterate through all files in the folder.
    for filename in files:
        # Extract the UTC timestamp from the filename.
        utc_timestamp_str = extract_utc_str(filename)

        if utc_timestamp_str:
            utc_timestamp = utc_str_to_datetime(utc_timestamp_str)
            east_timestamp = convert_utc_to_eastern(utc_timestamp)

            # Check if the timestamp is during daylight hours.
            if is_daytime(east_timestamp, start, end):
                # Add the filename to the list of filtered files.
                filtered_files.append(filename)

    return filtered_files


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


def gen_label_overlays(image_folder, labels_folder, destination_folder=None, alpha=None, gamma=None,
                       color_map=None):
    """
        This function generates a folder of label overlays given a folder
        of original images and folder of Doodler labels.
    :param image_folder: str
        Path of image folder.
    :param labels_folder: str
        Path of labels folder.
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
    # Create folder to store label overlays.
    if destination_folder is None:
        destination_folder = 'label_overlays'

    if os.path.exists(destination_folder):
        os.makedirs(os.path.join(destination_folder,'/label_overlays'))
    else:
        print("Label overlays folder already exists.")

    # Read in folders of images and segmentation maps.
    if not os.path.exists(image_folder):
        raise FileNotFoundError("Provided image folder path does not exist.")
    if not os.path.exists(labels_folder):
        raise FileNotFoundError("Provided labels folder path does not exist.")

    # Collect image files
    image_files = [os.path.join(image_folder, filename) for filename in
                   os.listdir(image_folder) if not filename.startswith('.')
                   and os.path.isfile(os.path.join(image_folder, filename))]

    # Collect segmentation map files
    label_files = [os.path.join(labels_folder, filename) for filename in
                   os.listdir(labels_folder) if not filename.startswith('.')
                   and os.path.isfile(os.path.join(labels_folder, filename))]

    # Sort the files so that they match.
    image_files.sort()
    label_files.sort()

    # Confirm that the number of files match before running. Will exit otherwise.
    num_images = len(image_files)
    num_segmaps = len(label_files)
    if num_images == num_segmaps:
        # Initialize tqdm with the total number of files
        progress_bar = tqdm(total=num_images, desc="Generating label overlays...")

        # Load images from each folder and run the overlays function.
        for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
            image = cv2.imread(image_file)
            label = cv2.imread(label_file)

            # Generate color labels from grayscale Doodler labels
            color_label = gen_color_labels(label, color_map)

            overlay_img = gen_overlay(image, color_label, alpha, gamma, color_map)

            cv2.imwrite(
                os.path.join(destination_folder, f'label_overlay_{os.path.basename(image_file)}'),
                overlay_img)

            # Clear memory
            image = None
            segmap = None

            # Update the progress bar
            progress_bar.update(1)
    else:
        print("Number of image files and labels did not match.")
        return

    # Close the progress bar
    progress_bar.close()

    return print("Folder of label overlays generated.")


def process_image_folders(target_folder_path):
    # Create the "daytime" folder in the top folder
    daytime_images_folder = os.path.join(target_folder_path, "daytime_images")
    os.makedirs(daytime_images_folder, exist_ok=True)

    # List all folders (excluding hidden folders)
    folders = [f for f in os.listdir(target_folder_path)
               if os.path.isdir(os.path.join(target_folder_path, f)) and
               not f.startswith(".")]

    # Initialize tqdm with the total number of files
    progress_bar = tqdm(total=len(folders), desc="Filtering image folders...")

    # Iterate through each folder
    for folder in folders:
        folder_path = os.path.join(target_folder_path, folder)
        # Filter files in the current folder
        daytime_files = filter_files_by_daylight_hours(folder_path)

        # Copy filtered files to the "daytime" folder
        for file in daytime_files:
            src = os.path.join(folder_path, file)
            dst = os.path.join(daytime_images_folder, file)

            if src != dst:
                shutil.copy(src, dst)

                # Update the progress bar
                progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    return print(f'Daytime files copied to {daytime_images_folder}.')


def plot_images_side_by_side(images_folder, overlays_folder, output_folder, dpi=250):
    """
    This function plots original images and overlay images from their respective folders side by side.
    :param images_folder: str
        Path to the folder containing the original images.
    :param overlays_folder: str
        Path to the folder containing the segmentation overlays.
    :param output_folder: str
        Path to place the side-by-side images.
    :param dpi: int
        The dpi for side by side plots. Default value of 250.
    :return: none
        There is no return from this function.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of files in each folder
    images = [os.path.join(images_folder, filename) for filename in
                   os.listdir(images_folder) if not filename.startswith('.')
                   and os.path.isfile(os.path.join(images_folder, filename))]
    overlays = [os.path.join(overlays_folder, filename) for filename in
                   os.listdir(overlays_folder) if not filename.startswith('.')
                   and os.path.isfile(os.path.join(overlays_folder, filename))]

    # Sort the files so that they match.
    images.sort()
    overlays.sort()

    # Ensure both folders have the same number of images
    if len(images) != len(overlays):
        print("Error: The two folders must contain the same number of images.")
        return

    progress_bar = tqdm(total=len(images), desc="Generating side-by-sides...")

    for image, overlay in zip(images, overlays):
        # Open images from both folders
        img1 = Image.open(image)
        img2 = Image.open(overlay)

        # Create a new figure
        fig, axes = plt.subplots(1, 2)

        # Plot images side by side
        axes[0].imshow(img1)
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        axes[1].imshow(img2)
        axes[1].axis('off')
        axes[1].set_title('Segmentation Overlay')

        # Save the figure to the output folder
        output_file = os.path.join(output_folder, f'side_by_side_{os.path.basename(image)}')
        plt.savefig(output_file, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    return print(f"Side by side images saved successfully.")

