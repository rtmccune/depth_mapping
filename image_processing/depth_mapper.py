import os
import zarr
import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.ndimage import label
from cupyx.scipy.ndimage import binary_closing
from skimage.measure import find_contours

class DepthMapper:
    
    def __init__(self, elevation_grid):
        
        self.elev_grid = elevation_grid

    def label_ponds(self, gpu_label_array):
        labels_squeezed = gpu_label_array.squeeze()
        mask = (labels_squeezed == 1)  # Boolean mask

        # Create binary mask directly as uint8 (avoids redundant cp.where)
        masked_labels = mask.astype(cp.uint8)

        # Apply binary closing (morphological operation)
        closed_data = binary_closing(masked_labels, structure=cp.ones((3, 3), dtype=cp.uint8))

        # Label connected components
        labeled_data, num_features = label(closed_data)

        # Remove small ponds
        min_size = 100
        unique, counts = cp.unique(labeled_data, return_counts=True)
        small_ponds = unique[counts < min_size]

        # In-place update (avoid unnecessary memory allocation)
        mask_small_ponds = cp.isin(labeled_data, small_ponds)
        labeled_data[mask_small_ponds] = 0

        # Relabel remaining ponds
        labeled_data, num_features = label(labeled_data)

        return labeled_data

    def extract_contours(self, labeled_data, gpu_label_array):
        unique_ponds = cp.unique(labeled_data)
        unique_ponds = unique_ponds[unique_ponds != 0]  # Exclude background label

        pond_contours = {}
        arr = cp.where(gpu_label_array.squeeze() == 1, self.elev_grid, 0)

        # Convert to NumPy only once per loop iteration
        labeled_data_np = cp.asnumpy(labeled_data)
        arr_np = cp.asnumpy(arr)

        for pond_id in unique_ponds:
            pond_id_int = int(pond_id.get())  # Convert cupy scalar to Python int
            pond_mask = (labeled_data_np == pond_id_int)
            pond_arr = np.where(pond_mask, arr_np, 0)

            contours = find_contours(pond_arr, level=0.5)
            pond_contours[pond_id_int] = contours

        contour_pixels_per_pond = {}
        contour_values_per_pond = {}

        for pond_id, contours in pond_contours.items():
            if not contours:
                continue  # Skip empty contours

            contour_pixels = np.vstack([
                np.round(contour).astype(int) for contour in contours
            ])

            # Ensure indices are within bounds
            valid_mask = (0 <= contour_pixels[:, 1]) & (contour_pixels[:, 1] < arr_np.shape[1]) & \
                        (0 <= contour_pixels[:, 0]) & (contour_pixels[:, 0] < arr_np.shape[0])

            contour_pixels = contour_pixels[valid_mask]
            contour_values = arr_np[contour_pixels[:, 0], contour_pixels[:, 1]]

            contour_pixels_per_pond[pond_id] = contour_pixels
            contour_values_per_pond[pond_id] = contour_values

        return contour_pixels_per_pond, contour_values_per_pond

    def calculate_depths(self, labeled_data, contour_values_per_pond):
        pond_depths = {}

        unique_pond_ids = cp.unique(labeled_data)
        
        if len(unique_pond_ids) == 1 and unique_pond_ids[0] == 0:
            pond_depths[1] = cp.full_like(self.elev_grid, cp.nan)
            return pond_depths
        
        for pond_id in unique_pond_ids:
            if pond_id == 0:  # Skip background
                continue

            pond_mask = (labeled_data == pond_id)
            masked_elevations = cp.where(pond_mask, self.elev_grid, cp.nan)

            max_elevation = cp.percentile(cp.array(contour_values_per_pond[pond_id.item()]), 95)
            depth_map = masked_elevations - max_elevation
            depth_map[depth_map > 0] = 0
            depth_map = cp.abs(depth_map)

            pond_depths[pond_id.item()] = depth_map

        return pond_depths

    def combine_depth_maps(self, pond_depths):

        combined_depth_map = pond_depths[1]

        for i in range(2, len(pond_depths) + 1):
            combined_depth_map = cp.where(cp.isnan(combined_depth_map), pond_depths[i], 
                                        cp.where(cp.isnan(pond_depths[i]), combined_depth_map, combined_depth_map + pond_depths[i]))
        
        return combined_depth_map
    
    def process_file(self, zarr_store_path, file_name):
        depth_data = []
        img_store = zarr.open(zarr_store_path)
        array = img_store[:]
        
        # print(f"Processing array: {file_name}")

        gpu_label_array = cp.array(array)
        labeled_data = self.label_ponds(gpu_label_array)

        contour_pixels_per_pond, contour_values_per_pond = self.extract_contours(labeled_data, gpu_label_array)

        pond_depths = self.calculate_depths(labeled_data, contour_values_per_pond)

        combined_depth_map = self.combine_depth_maps(pond_depths)
        depth_data.append({
            'image_name': f"{file_name}_depth_map_95_perc_edge_ponding",
            'depth_map': combined_depth_map
        })
        return depth_data
    
    def process_depth_maps(self, labels_zarr_dir, depth_map_zarr_dir):
        for file_name in os.listdir(labels_zarr_dir):
            if file_name.endswith('_rectified'):
                rectified_label_array = os.path.join(labels_zarr_dir, file_name)
                depth_data = self.process_file(rectified_label_array, file_name)
                
                depth_maps = pd.DataFrame(depth_data)
                
                self.save_depth_maps(depth_maps, depth_map_zarr_dir)

    def save_depth_maps(self, depth_maps_dataframe, depth_map_zarr_dir):
        
        for _, row in depth_maps_dataframe.iterrows():
            store = zarr.open_group(depth_map_zarr_dir, mode='a')
            
            image_name = row['image_name']
            
            depth_map = row['depth_map']

            store[image_name] = depth_map.get()