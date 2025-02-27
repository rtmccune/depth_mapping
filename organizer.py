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

class ImageOrganizer:

    def __init__(self, research_storage_dir):
        
        self.parent_dir = research_storage_dir


    def pull_and_save_images(self, flood_event_record, destination_dir):
        # Create destination folder if it does not already exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        tasks = []

        # Progress bar for processing rows
        with tqdm(total=df.shape[0], desc='Processing rows') as pbar:
            # For each row in the dataframe
            for _, row in df.iterrows():
                # set the start time, end time, and camera ID
                start_time = row['start_time_UTC']
                end_time = row['end_time_UTC']
                camera_id = row['camera_ID']

                # set the current time to begin iterating through folders
                current_date = start_time.date()
                #  set the end date (possible that the flood event occurred over more than one date)
                end_date = end_time.date()
                
                # While the current iteration of the date is earlier than the end date
                while current_date <= end_date:
                    # list the files in the current date iteration's corresponding folder
                    file_list = list_files_in_date_directory(base_dir, current_date, camera_id)

                    # filter the file list to between the start and end time 
                    # (using end time rather than end date accounts for multiday events)
                    filtered_files = filter_files(file_list, start_time, end_time)

                    # add the file paths and destination folder to the task list to be copied
                    for file_path in filtered_files:
                        tasks.append((file_path, destination_folder))

                    # continue to the next day
                    current_date += timedelta(days=1)

                pbar.update(1)

        # Progress bar for copying files
        with tqdm(total=len(tasks), desc='Copying files') as pbar:
            # copy the files in the task list using the copy function
            for file_path, destination_folder in tasks:
                copy_file(file_path, destination_folder)
                pbar.update(1)