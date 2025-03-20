import os
import zarr
import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.ndimage import label
from cupyx.scipy.ndimage import binary_closing
from skimage.measure import find_contours

class DepthMapper:
    """_summary_
    """    
    def __init__(self, elevation_grid):
        """Initialize the object with with an elevation grid.

        Args:
            elevation_grid (numpy.ndarray): A 2D array representing elevation values,
            where each element corresponds to an elevation at a specific grid point.
        """        
        
        self.elev_grid = elevation_grid

    def label_ponds(self, gpu_label_array):
        """Labels and processes pond regions in a binary mask.

        This function takes a binary array indicating pond regions, applies morphological operations 
        to clean up the mask, labels connected pond regions, and removes small ponds below a size threshold.

        Args:
            gpu_label_array (cp.ndarray): A CuPy array where pond regions are marked as 1.

        Returns:
            cp.ndarray: A labeled CuPy array where each connected pond has a unique integer ID.
        """                
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
        """Extracts contour pixels and their label values for each pond.

        This function identifies contours around labeled pond regions, extracts their pixel coordinates, 
        and retrieves corresponding label values.

        Args:
            labeled_data (cp.ndarray): A CuPy array where each pond is labeled with a unique integer ID.
            gpu_label_array (cp.ndarray): A CuPy array representing the original labeled dataset, 
                                        where 1 indicates flooded regions.

        Returns:
            tuple[dict[int, np.ndarray], dict[int, np.ndarray]]: 
                - A dictionary mapping each pond ID to an array of its contour pixel coordinates (Nx2).
                - A dictionary mapping each pond ID to an array of label values at contour pixels.`
        """        
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
        """Calculates the depth of each pond based on elevation differences.

        Args:
            labeled_data (cupy.ndarray): A 2D array where each pond is assigned a unique label.
            contour_values_per_pond (dict of int -> cupy.ndarray): A dictionary mapping 
                pond IDs to an array of elevation values along the pond contour.

        Returns:
            dict of int -> cupy.ndarray: A dictionary where each key is a pond ID, and 
                the value is a 2D CuPy array representing the depth map of that pond.
                Depth values are computed as the difference between the pond's elevation 
                and the 95th percentile of its contour elevations. Areas above this 
                threshold are set to zero, ensuring only ponding regions remain.
        """        
        pond_depths = {} # initialize pond depth dictionary

        unique_pond_ids = cp.unique(labeled_data) # array of unique pond labels
        
        if len(unique_pond_ids) == 1 and unique_pond_ids[0] == 0: # If no ponds present
            pond_depths[1] = cp.full_like(self.elev_grid, cp.nan) # Fill pond depths with NaN (background, no water)
            return pond_depths
        
        for pond_id in unique_pond_ids:
            if pond_id == 0:  # Skip background
                continue

            pond_mask = (labeled_data == pond_id) # mask to current pond only
            masked_elevations = cp.where(pond_mask, self.elev_grid, cp.nan) # replace any other values not belonging to this pond with NaN

            max_elevation = cp.percentile(cp.array(contour_values_per_pond[pond_id.item()]), 95) # calculate 95th percentile of edges
            depth_map = masked_elevations - max_elevation # calculate depth across pond
            depth_map[depth_map > 0] = 0 # set depths greater than 0 to 0 to handle edges above 95th percentile
            depth_map = cp.abs(depth_map) # take the absolute value of the depths

            pond_depths[pond_id.item()] = depth_map

        return pond_depths

    def combine_depth_maps(self, pond_depths):
        """Combines multiple pond depth maps into a single depth map.

        Args:
            pond_depths (list of cupy.ndarray): A list of depth maps where each element 
                represents the depth values for a specific pond.

        Returns:
            cupy.ndarray: A combined depth map where overlapping pond depths are summed,
            and NaN values are handled appropriately.
        """
        combined_depth_map = pond_depths[1]

        for i in range(2, len(pond_depths) + 1):
            combined_depth_map = cp.where(
                cp.isnan(combined_depth_map), pond_depths[i], # If combined_depth_map has NaN, use pond_depths[i]
                cp.where(
                    cp.isnan(pond_depths[i]), combined_depth_map, # If pond_depths[i] has NaN, keep combined_depth_map
                    combined_depth_map + pond_depths[i])) # Otherwise, sum the values
        
        return combined_depth_map
    
    def process_file(self, zarr_store_path, file_name):
        """Processes a Zarr store containing rectified labeled image data and computes depth maps.

        Args:
            zarr_store_path (str): Path to the Zarr store containing the rectified image label array.
            file_name (str): Name of the file being processed.

        Returns:
            list[dict]: A list containing a dictionary with:
                - 'image_name' (str): The name of the processed depth map.
                - 'depth_map' (cupy.ndarray or similar): The computed depth map.
        """        
        depth_data = [] # intialize depth data list
        img_store = zarr.open(zarr_store_path)
        array = img_store[:] # open image array
        
        gpu_label_array = cp.array(array) # convert to cupy array for GPU processing
        labeled_data = self.label_ponds(gpu_label_array) # separate ponds

        contour_pixels_per_pond, contour_values_per_pond = self.extract_contours(labeled_data, gpu_label_array) # extract elevations of pond edges

        pond_depths = self.calculate_depths(labeled_data, contour_values_per_pond) # calculate depths based on extracted edge elevations

        combined_depth_map = self.combine_depth_maps(pond_depths) # combine separate ponds depth maps into one map
        depth_data.append({
            'image_name': f"{file_name}_depth_map_95_perc_edge_ponding",
            'depth_map': combined_depth_map
        })
        return depth_data
    
    def process_depth_maps(self, labels_zarr_dir, depth_map_zarr_dir):
        """Creates and saves depth maps as zarr arrays at the provided destination,
            given the zarr direcotry containing rectified labels.

        Args:
            labels_zarr_dir (str): Path to zarr directory containing rectified labels.
            depth_map_zarr_dir (str): Path to the directory where processed depth maps will be saved.
        """        
        for file_name in os.listdir(labels_zarr_dir): # for each rectified label array
            if file_name.endswith('_rectified'): # confirm that it has been rectified
                rectified_label_array = os.path.join(labels_zarr_dir, file_name) # combine file path
                depth_data = self.process_file(rectified_label_array, file_name) # generate depth map
                
                depth_maps = pd.DataFrame(depth_data) # create dataframe from dictionary output
                
                self.save_depth_maps(depth_maps, depth_map_zarr_dir) # save to zarr

    def save_depth_maps(self, depth_maps_dataframe, depth_map_zarr_dir):
        """Saves depth maps from a DataFrame into a Zarr store.

        Args:
            depth_maps_dataframe (pd.DataFrame): A DataFrame containing depth maps,
                where each row includes an 'image_name' and a 'depth_map'.
            depth_map_zarr_dir (str): Path to the directory where the depth maps will be stored in a Zarr group.
        """         
        for _, row in depth_maps_dataframe.iterrows():
            store = zarr.open_group(depth_map_zarr_dir, mode='a')
            
            image_name = row['image_name']
            
            depth_map = row['depth_map']

            store[image_name] = depth_map.get()