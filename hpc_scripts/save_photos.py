from photo_utils import *
import pandas as pd

mount_dir = '/rsstu/users/k/kanarde/Sunnyverse-Images'
storage_dir = '/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/all_events/orig_images'

flood_record = pd.read_csv('/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/all_events/abbr_flood_events.csv')

flood_record['start_time_UTC'] = pd.to_datetime(flood_record['start_time_UTC'])
flood_record['end_time_UTC'] = pd.to_datetime(flood_record['end_time_UTC'])

flood_record['camera_ID'] = 'CAM_' + flood_record['sensor_ID']

pull_files(flood_record, mount_dir, storage_dir)