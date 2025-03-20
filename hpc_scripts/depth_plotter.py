import os
import gc
import numpy as np
import pandas as pd
import pytz
import zarr
from tqdm import tqdm
from datetime import datetime
import cmocean
import matplotlib.pyplot as plt
from mpi4py import MPI
import image_processing

class DepthPlotter:
    
    def __init__(self, main_dir, virtual_sensor_locations):
        
        self.main_dir = main_dir
        self.virtual_sensor_loc = virtual_sensor_locations
        
        self.water_level_color = cmocean.cm.balance(0.2)
        self.max_depth_color = cmocean.cm.balance(0.9)
        self.avg_depth_color = cmocean.cm.balance(0.6)
        self.sensor_1_color = cmocean.cm.phase(0.1)
        self.sensor_2_color = cmocean.cm.phase(0.3)
        self.sensor_3_color = cmocean.cm.phase(0.5)
        
    def list_flood_event_folders(self):
        
        flood_event_folders = [flood_event for flood_event in os.listdir(self.main_dir) if os.path.isdir(os.path.join(self.main_dir, flood_event))]

        return flood_event_folders
    
    def match_measurements_to_images(self, flood_event, flood_event_path):
        
        sunnyd_data = pd.read_csv(os.path.join(flood_event_path, flood_event + '.csv'))
        sunnyd_data['time_UTC'] = pd.to_datetime(sunnyd_data['time_UTC'])
        
        orig_images_path = os.path.join(flood_event_path, 'orig_images')
        image_list = sorted(os.listdir(orig_images_path))
        match = []

        # Iterate over image filenames
        for filename in image_list:
            # Extract the sensor id and timestamp
            sensor_id = image_processing.image_utils.extract_camera_name(filename)[4:]
            timestamp = image_processing.image_utils.extract_timestamp(filename)

            timestamp = pytz.utc.localize(datetime.strptime(timestamp, "%Y%m%d%H%M%S"))
            
            # Filter the dataframe by sensor id
            filtered_df = sunnyd_data[sunnyd_data['sensor_ID'] == sensor_id]
            
            # Find the closest timestamp
            closest_row = filtered_df.iloc[(filtered_df['time_UTC'] - timestamp).abs().argsort()[:1]]
            
            # Append the result
            if not closest_row.empty:
                result = {
                    'image_filename': filename,
                    'closest_utc_time': closest_row['time_UTC'].values[0],
                    'water_level': closest_row['water_level'].values[0] * 0.3048
                    # 'sensor_water_level': (closest_row['sensor_water_level_adj'].values[0] - 3.05) * 0.3048
                }
                match.append(result)

        # Convert the results to a dataframe
        obs_to_image_matches = pd.DataFrame(match)
        obs_to_image_matches.to_csv(os.path.join(flood_event_path, 'wtr_lvl_obs_to_image_matches.csv'))
        
    def gen_virtual_sensor_depths(self, flood_event_path):
        
        depth_maps_zarr_dir = os.path.join(flood_event_path, 'zarr', 'depth_maps_95th_ponding')
        output_zarr_store = os.path.join(flood_event_path, 'zarr', 'virtual_sensor_depths')
        
        timestamp_list = []
        
        if os.path.exists(depth_maps_zarr_dir):
            file_names = [f for f in os.listdir(depth_maps_zarr_dir) if f.endswith('_ponding')]
            num_files = len(file_names)

            # Preallocate NumPy arrays for better performance
            max_depth_array = np.empty(num_files, dtype=np.float32)
            avg_depth_array = np.empty(num_files, dtype=np.float32)
            vs_depth_array = np.empty((num_files, len(self.virtual_sensor_loc)), dtype=np.float32)

            for idx, file_name in enumerate(file_names):
                timestamp = image_processing.image_utils.extract_timestamp(file_name)
                timestamp_list.append(timestamp)

                file_zarr_store = os.path.join(depth_maps_zarr_dir, file_name)
                img_store = zarr.open(file_zarr_store, mode='r')
                depth_map = img_store[:]

                max_depth_array[idx] = np.nanmax(depth_map)
                avg_depth_array[idx] = np.nanmean(depth_map)

                for i, (x, y) in enumerate(self.virtual_sensor_loc):
                    vs_depth_array[idx, i] = depth_map[y, x]

            # datetimes = pd.to_datetime(timestamp_list, utc=True)
            # Convert timestamps to a NumPy array of strings
            datetimes = np.array(pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U")
            
            # Save to a Zarr store
            root = zarr.open_group(output_zarr_store, mode="w")  # Overwrite existing store

            root.create_array("timestamps", shape=datetimes.shape, dtype="U")
            root["timestamps"][:] = datetimes  # Assign data
            
            root.create_array("max_depths", shape=max_depth_array.shape, dtype=np.float32)
            root["max_depths"][:] = max_depth_array

            root.create_array("avg_depths", shape=avg_depth_array.shape, dtype=np.float32)
            root["avg_depths"][:] = avg_depth_array

            root.create_array("vs_depths", shape=vs_depth_array.shape, dtype=np.float32)
            root["vs_depths"][:] = vs_depth_array
    

    def preprocess_flood_events(self):
        
        flood_event_folders = self.list_flood_event_folders()
        
        for flood_event in tqdm(flood_event_folders, desc="Preprocessing flood events for plotting...", unit="event"):
            
            flood_event_path = os.path.join(self.main_dir, flood_event)
            self.match_measurements_to_images(flood_event, flood_event_path)
            self.gen_virtual_sensor_depths(flood_event_path)
            
            
    def load_virtual_sensor_depths(self, flood_event_path):
        zarr_store_path = os.path.join(flood_event_path, 'zarr', 'virtual_sensor_depths')
        
        if os.path.exists(zarr_store_path):
            root = zarr.open(zarr_store_path, mode='r')

            timestamps = root["timestamps"][:]  # Load as an array of strings
            max_depths = root["max_depths"][:]
            avg_depths = root["avg_depths"][:]
            vs_depths = root["vs_depths"][:]

            # Convert timestamps back to pandas datetime
            datetimes = pd.to_datetime(timestamps, utc=True)

            return datetimes, max_depths, avg_depths, vs_depths
        else:
            raise FileNotFoundError(f"Zarr store not found: {zarr_store_path}")

    # def plot_depth_map_and_wtr_levels(self, depth_map_zarr_dir, orig_image_rects_zarr_dir, datetimes, 
    #                                    obs_to_img_matches, vs_depths, plotting_folder, depth_min=0, depth_max=0.25):

    #     if os.path.exists(depth_map_zarr_dir):
    #         for file_name in sorted(os.listdir(depth_map_zarr_dir)):
    #             if file_name.endswith('_ponding'):
                    
    #                 timestamp = image_processing.image_utils.extract_timestamp(file_name)
    #                 date = pd.to_datetime(timestamp, utc=True)
    #                 orig_file_name = None
    #                 for f in os.listdir(orig_image_rects_zarr_dir):
    #                     if image_processing.image_utils.extract_timestamp(f) == timestamp:
    #                         orig_file_name = f
    #                         break
                    
    #                 if orig_file_name is None:
    #                     print(f"Warning: No matching original image found for {file_name}")
    #                     continue
                    
    #                 orig_zarr_store_path = os.path.join(orig_image_rects_zarr_dir, orig_file_name)
    #                 orig_img_store = zarr.open(orig_zarr_store_path, mode='r')
    #                 orig_image = orig_img_store[:]
                    
    #                 zarr_store_path = os.path.join(depth_map_zarr_dir, file_name)
    #                 img_store = zarr.open(zarr_store_path)
                    
    #                 depth_map = img_store[:]
                    
    #                 print(f"Processing depth map: {file_name}")

    #                 fig = plt.figure(figsize=(12, 12))

    #                 gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])

    #                 ax1 = fig.add_subplot(gs[0])
    #                 ax1.imshow(orig_image, cmap='gray')  # Assuming ir_array is your grayscale image
    #                 im = ax1.imshow(depth_map, cmap=cmocean.cm.deep, vmin=depth_min, vmax=depth_max)  # Adjust alpha for transparency
    #                 ax1.scatter(self.virtual_sensor_loc[0][0], self.virtual_sensor_loc[0][1], color=self.sensor_1_color, s=15, marker='v')
    #                 ax1.scatter(self.virtual_sensor_loc[1][0], self.virtual_sensor_loc[1][1], color=self.sensor_2_color, s=15, marker='s')
    #                 ax1.scatter(self.virtual_sensor_loc[2][0], self.virtual_sensor_loc[2][1], color=self.sensor_3_color, s=15, marker='d')
    #                 cbar = plt.colorbar(im, label='Depth')
    #                 cbar.set_label('Depth (m)')
    #                 ax1.invert_yaxis()
    #                 ax1.set_xlabel('X (cm)')
    #                 ax1.set_ylabel('Y (cm)')

    #                 ax1.text(0.05, 0.95, f'Spatial Extent ($m^2$): {round((np.sum(~np.isnan(depth_map))) * 0.0001 * 10 * 10, 2)}', 
    #                         transform=ax1.transAxes, 
    #                         fontsize=12, 
    #                         verticalalignment='top', 
    #                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    #                 ax3 = fig.add_subplot(gs[1])
                    
    #                 ax3.plot(obs_to_img_matches['closest_utc_time'], obs_to_img_matches['water_level'], label='Observed Water Level', color=self.water_level_color)

    #                 ax3.scatter(datetimes, vs_depths[:,2], label='Sensor 1 Depth', marker='d', color=self.sensor_3_color, s=10)
    #                 ax3.scatter(datetimes, vs_depths[:,0], label='Sensor 2 Depth',  marker='v', color=self.sensor_1_color, s=10)
    #                 ax3.scatter(datetimes, vs_depths[:,1], label='Sensor 3 Depth', marker='s', color=self.sensor_2_color, s=10)
                    
    #                 ax3.axvline(x=date, color='k', linestyle='-', zorder=1)
    #                 ax3.set_ylim(-0.25,0.75)

    #                 # Axis labels and title
    #                 ax3.tick_params(axis='x', rotation=45)
    #                 ax3.set_title('Water Level From Virtual Sensor Locations Over Time')
    #                 ax3.set_ylabel('Water Depth (m)')

    #                 # Adding gridlines
    #                 ax3.grid(True)

    #                 # Adding legend outside of the figure bounds
    #                 ax3.legend(loc='upper right')

    #                 plt.tight_layout()
                    
    #                 # Save the figure
    #                 plt.savefig(os.path.join(plotting_folder, file_name), 
    #                             bbox_inches='tight', pad_inches=0.1, dpi=300)
                    
    #                 plt.close()
    

    def plot_depth_map_and_wtr_levels(self, depth_map_zarr_dir, orig_image_rects_zarr_dir, datetimes, 
                                    obs_to_img_matches, vs_depths, plotting_folder, depth_min=0, depth_max=0.25):

        if os.path.exists(depth_map_zarr_dir):
            for file_name in sorted(os.listdir(depth_map_zarr_dir)):
                if file_name.endswith('_ponding'):
                    timestamp = image_processing.image_utils.extract_timestamp(file_name)
                    date = pd.to_datetime(timestamp, utc=True)

                    orig_file_name = next(
                        (f for f in os.listdir(orig_image_rects_zarr_dir) 
                        if image_processing.image_utils.extract_timestamp(f) == timestamp),
                        None
                    )

                    if orig_file_name is None:
                        print(f"Warning: No matching original image found for {file_name}")
                        continue

                    orig_zarr_store_path = os.path.join(orig_image_rects_zarr_dir, orig_file_name)
                    orig_img_store = zarr.open(orig_zarr_store_path, mode='r')
                    orig_image = orig_img_store[:]  # Consider downsampling if necessary

                    zarr_store_path = os.path.join(depth_map_zarr_dir, file_name)
                    img_store = zarr.open(zarr_store_path, mode='r')
                    depth_map = img_store[:]  # Again, consider loading only necessary slices

                    # print(f"Processing depth map: {file_name}")

                    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1.5, 1]})

                    # Plot depth map
                    im = ax1.imshow(orig_image, cmap='gray')  # Assuming orig_image is grayscale
                    im = ax1.imshow(depth_map, cmap=cmocean.cm.deep, vmin=depth_min, vmax=depth_max)
                    ax1.scatter(self.virtual_sensor_loc[0][0], self.virtual_sensor_loc[0][1], color=self.sensor_1_color, s=15, marker='v')
                    ax1.scatter(self.virtual_sensor_loc[1][0], self.virtual_sensor_loc[1][1], color=self.sensor_2_color, s=15, marker='s')
                    ax1.scatter(self.virtual_sensor_loc[2][0], self.virtual_sensor_loc[2][1], color=self.sensor_3_color, s=15, marker='d')
                    cbar = plt.colorbar(im, ax=ax1, label='Depth')
                    cbar.set_label('Depth (m)')
                    ax1.invert_yaxis()
                    ax1.set_xlabel('X (cm)')
                    ax1.set_ylabel('Y (cm)')
                    ax1.text(0.05, 0.95, f'Spatial Extent ($m^2$): {round((np.sum(~np.isnan(depth_map))) * 0.0001 * 10 * 10, 2)}',
                            transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

                    # Plot water levels
                    ax3.plot(obs_to_img_matches['closest_utc_time'], obs_to_img_matches['water_level'], label='Observed Water Level', color=self.water_level_color)
                    ax3.scatter(datetimes, vs_depths[:, 2], label='Sensor 1 Depth', marker='d', color=self.sensor_3_color, s=10)
                    ax3.scatter(datetimes, vs_depths[:, 0], label='Sensor 2 Depth', marker='v', color=self.sensor_1_color, s=10)
                    ax3.scatter(datetimes, vs_depths[:, 1], label='Sensor 3 Depth', marker='s', color=self.sensor_2_color, s=10)
                    ax3.axvline(x=date, color='k', linestyle='-', zorder=1)
                    ax3.set_ylim(-0.25, 0.75)
                    ax3.tick_params(axis='x', rotation=45)
                    ax3.set_title('Water Level From Virtual Sensor Locations Over Time')
                    ax3.set_ylabel('Water Depth (m)')
                    ax3.grid(True)
                    ax3.legend(loc='upper right')

                    plt.tight_layout()

                    # Save the figure
                    plt.savefig(os.path.join(plotting_folder, file_name), bbox_inches='tight', pad_inches=0.1, dpi=300)

                    plt.close(fig)  # Close the figure to free memory
                    del orig_image, depth_map  # Delete large variables
                    gc.collect()  # Force garbage collection



    def process_flood_events(self, plotting_dir):
        
        # self.preprocess_flood_events()
        
        flood_event_folders = self.list_flood_event_folders()
        
        for flood_event in tqdm(flood_event_folders, desc="Plotting flood events...", unit="event"):
            
            flood_event_path = os.path.join(self.main_dir, flood_event)
            
            datetimes, max_depths, avg_depths, vs_depths = self.load_virtual_sensor_depths(flood_event_path)
            
            plotting_folder = os.path.join(flood_event_path, 'plots', plotting_dir)
            os.makedirs(plotting_folder, exist_ok=True)
            
            depth_maps_zarr_dir = os.path.join(flood_event_path, 'zarr', 'depth_maps_95th_ponding')
            orig_image_rects_zarr_dir = os.path.join(flood_event_path, 'zarr', 'orig_image_rects')
            
            obs_to_img_matches = pd.read_csv(os.path.join(flood_event_path, 'wtr_lvl_obs_to_image_matches.csv'))
            obs_to_img_matches['closest_utc_time'] = pd.to_datetime(obs_to_img_matches['closest_utc_time'], utc=True)
            
            self.plot_depth_map_and_wtr_levels(depth_maps_zarr_dir, orig_image_rects_zarr_dir, datetimes, obs_to_img_matches, vs_depths, plotting_folder)
            
            del datetimes, max_depths, avg_depths, vs_depths, obs_to_img_matches  # Delete large variables
            gc.collect()  # Force garbage collection
            
            
    def process_flood_events_HPC(self, plotting_dir):
        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Only the master process will list flood event folders
        if rank == 0:
            flood_event_folders = self.list_flood_event_folders()
            self.preprocess_flood_events()
        else:
            flood_event_folders = None

        # Broadcast the flood event folders to all processes
        flood_event_folders = comm.bcast(flood_event_folders, root=0)

        # Split the work among processes
        n_folders = len(flood_event_folders)
        chunk_size = n_folders // size
        start_index = rank * chunk_size
        end_index = start_index + chunk_size if rank != size - 1 else n_folders

        # Process only the assigned folders
        for flood_event in tqdm(flood_event_folders[start_index:end_index], desc="Plotting flood events...", unit="event"):
            self.process_single_flood_event(flood_event, plotting_dir)

    def process_single_flood_event(self, flood_event, plotting_dir):
        flood_event_path = os.path.join(self.main_dir, flood_event)
        
        datetimes, max_depths, avg_depths, vs_depths = self.load_virtual_sensor_depths(flood_event_path)
        
        plotting_folder = os.path.join(flood_event_path, 'plots', plotting_dir)
        os.makedirs(plotting_folder, exist_ok=True)
        
        depth_maps_zarr_dir = os.path.join(flood_event_path, 'zarr', 'depth_maps_95th_ponding')
        orig_image_rects_zarr_dir = os.path.join(flood_event_path, 'zarr', 'orig_image_rects')
        
        obs_to_img_matches = pd.read_csv(os.path.join(flood_event_path, 'wtr_lvl_obs_to_image_matches.csv'))
        obs_to_img_matches['closest_utc_time'] = pd.to_datetime(obs_to_img_matches['closest_utc_time'], utc=True)
        
        self.plot_depth_map_and_wtr_levels(depth_maps_zarr_dir, orig_image_rects_zarr_dir, datetimes, obs_to_img_matches, vs_depths, plotting_folder)

