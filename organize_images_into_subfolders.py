import image_processing


orig_images_folder = '/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/orig_images'
labels_folder = '/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/labels'

abbr_flood_events_csv = '/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/abbr_flood_events.csv'
flood_events_folder = '/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/flood_events'

image_processing.image_utils.organize_images_by_flood_events(orig_images_folder, abbr_flood_events_csv, flood_events_folder, 'orig_images')
image_processing.image_utils.organize_images_by_flood_events(labels_folder, abbr_flood_events_csv, flood_events_folder, 'labels')