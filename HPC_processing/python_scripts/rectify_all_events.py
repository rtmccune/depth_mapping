import os
import numpy as np
import image_processing

intrinsics = np.array(
    [
        3040,  # number of pixel columns
        4056,  # number of pixel rows
        1503.0136,  # U component of principal point
        2163.4301,  # V component of principal point
        2330.4972,  # U component of focal length
        2334.0017,  # V component of focal length
        -0.3587,  # radial distortion
        0.1388,  # radial distortion
        -0.0266,  # radial distortion
        -0.0046,  # tangential distortion
        0.0003,  # tangential distortion
    ]
)

extrinsics = np.array(
    [
        712159.597863065,  # camera x in world
        33136.9994153273,  # camera y in world
        3.72446811607855,  # camera elev in world
        1.30039127961854,  # azimuth
        1.02781393967485,  # tilt
        -0.160877893129538,  # roll/swing
    ]
)

file_path = "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/lidar/Job1051007_34077_04_88.laz"

min_x_extent = 712160
max_x_extent = 712230
min_y_extent = 33100
max_y_extent = 33170

grid_gen = image_processing.GridGenerator(
    file_path, min_x_extent, max_x_extent, min_y_extent, max_y_extent
)

resolution = 0.05  # meters

pts_array = grid_gen.create_point_array()
grid_x, grid_y, grid_z = grid_gen.gen_grid(resolution, z=pts_array)

rectifier = image_processing.ImageRectifier(
    intrinsics, extrinsics, grid_x, grid_y, grid_z, use_gpu=True
)

# Path to the main directory containing subfolders
main_directory = "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/flood_events"

# Iterate through each subfolder in the main directory


for subfolder in os.listdir(main_directory):
    subfolder_path = os.path.join(main_directory, subfolder)

    # Check if it is a directory
    if os.path.isdir(subfolder_path):
        orig_images_folder = os.path.join(subfolder_path, "orig_images")
        labels_folder = os.path.join(subfolder_path, "labels")

        # Check if the orig_images and labels folders exist
        if os.path.exists(orig_images_folder) and os.path.exists(labels_folder):
            # Define paths for saving rectified images
            zarr_store_orig = os.path.join(subfolder_path, "zarr", "orig_image_rects")
            zarr_store_labels = os.path.join(subfolder_path, "zarr", "labels_rects")

            # Merge and rectify the orig_images folder
            rectifier.merge_rectify_folder(orig_images_folder, zarr_store_orig)
            # Merge and rectify the labels folder
            rectifier.merge_rectify_folder(
                labels_folder, zarr_store_labels, labels=True
            )

            # print(f'Merged and rectified images in {orig_images_folder} and labels in {labels_folder}')
