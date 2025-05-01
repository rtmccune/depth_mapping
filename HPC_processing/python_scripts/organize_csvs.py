import os
import shutil
import pandas as pd

# Load the abbreviated flood events and full data records
abbr_flood_events = pd.read_csv(
    "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/abbr_flood_events.csv"
)
flood_events = pd.read_csv(
    "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/flood_events.csv"
)

# Ensure the output directory exists
output_dir = "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/output_flood_event_csvs"
os.makedirs(output_dir, exist_ok=True)

# Convert the start and end time columns to datetime
abbr_flood_events["start_time_UTC"] = pd.to_datetime(
    abbr_flood_events["start_time_UTC"], utc=True
)
abbr_flood_events["end_time_UTC"] = pd.to_datetime(
    abbr_flood_events["end_time_UTC"], utc=True
)

# Format datetime to string for folder naming
abbr_flood_events["start_time_str"] = abbr_flood_events["start_time_UTC"].dt.strftime(
    "%Y%m%d%H%M%S"
)
abbr_flood_events["end_time_str"] = abbr_flood_events["end_time_UTC"].dt.strftime(
    "%Y%m%d%H%M%S"
)

# Iterate over each unique flood event in the abbreviated events
for _, row in abbr_flood_events.iterrows():
    flood_event_number = row["flood_event"]
    start_time_utc = row["start_time_str"]
    end_time_utc = row["end_time_str"]

    sensor_id = row["sensor_ID"]

    # Filter the full flood events DataFrame for the current flood event number
    filtered_events = flood_events[flood_events["flood_event"] == flood_event_number]

    # Generate a filename based on start_time_UTC and end_time_UTC
    filename = f"{sensor_id}_{start_time_utc}_{end_time_utc}.csv"

    # Save the filtered DataFrame to a new CSV file
    filtered_events.to_csv(os.path.join(output_dir, filename), index=False)

print(f"Successfully created CSV files in '{output_dir}' directory.")

# Define the main directory where the output CSV files are located
main_directory = output_dir
# Define the parent directory where you want to create subfolders
parent_directory = "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/flood_events"  # Change this to your actual parent directory

# Iterate over the files in the main directory
for filename in os.listdir(main_directory):
    if filename.endswith(".csv"):
        # Extract the folder name from the filename (without the .csv extension)
        folder_name = filename[:-4]  # Remove '.csv' from the filename

        # Create the full path for the subfolder
        subfolder_path = os.path.join(parent_directory, folder_name)

        # Create the subfolder if it doesn't exist
        os.makedirs(subfolder_path, exist_ok=True)

        # Move the CSV file to the corresponding subfolder
        source_path = os.path.join(main_directory, filename)
        destination_path = os.path.join(subfolder_path, filename)
        shutil.move(source_path, destination_path)

print("CSV files have been moved to their corresponding subfolders.")
