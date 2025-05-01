import os
import cupy as cp
from tqdm import tqdm
import image_processing

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
grid_z_gpu = cp.asarray(grid_z)

# Path to the main directory containing subfolders
main_directory = "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/flood_events"

subfolders = [
    f
    for f in os.listdir(main_directory)
    if os.path.isdir(os.path.join(main_directory, f))
]

mapper = image_processing.depth_mapper.DepthMapper(grid_z_gpu)

# Iterate through each subfolder with a progress bar
for subfolder in tqdm(subfolders, desc="Processing flood events", unit="event"):
    subfolder_path = os.path.join(main_directory, subfolder)
    labels_rects_zarr_folder = os.path.join(subfolder_path, "zarr", "labels_rects")

    # Check if the labels_rects folder exists
    if os.path.exists(labels_rects_zarr_folder):
        # print(f"Processing folder {subfolder}.")
        depth_map_zarr_save_dir = os.path.join(
            subfolder_path, "zarr", "depth_maps_95th_ponding"
        )
        mapper.process_depth_maps(labels_rects_zarr_folder, depth_map_zarr_save_dir)
