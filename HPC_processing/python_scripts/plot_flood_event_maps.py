import numpy as np
import image_processing

main_directory = "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/depth_mapping/data/CB_03_flood_events/flood_events"
virtual_sensor_locs = np.array([[125, 390], [75, 440], [5, 370], [9, 355]]) * 2

plotter = image_processing.DepthPlotter(main_directory, virtual_sensor_locs)

plotter.process_flood_events_HPC("depth_maps_test_plotting")
