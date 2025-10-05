import os
import zarr
import laspy
import numpy as np
from scipy.interpolate import griddata


class GridGenerator:
    """A class for generating and processing LiDAR data grids.

    This class allows for loading LiDAR data from a specified file, creating point arrays
    based on classification values, and generating grids of points in specified spatial extents
    and resolutions. The class also provides methods for saving the generated grids as
    compressed Zarr files.

    Attributes:
        filepath (str): The path to the LiDAR data file.
        filename (str): The name of the LiDAR data file.
        lidar (laspy.LasData): The loaded LiDAR data.
        min_x_extent (float): The minimum x-coordinate extent for grid generation.
        max_x_extent (float): The maximum x-coordinate extent for grid generation.
        min_y_extent (float): The minimum y-coordinate extent for grid generation.
        max_y_extent (float): The maximum y-coordinate extent for grid generation.
        point_mask_val (int): The classification value used to filter points.

    Methods:
        load_lidar(): Load LiDAR data from the specified file.
        create_point_array(point_mask_value=2): Create an array of filtered LiDAR points based on classification.
        gen_grid(resolution, z=0, dir='generated_grids'): Generate a grid of points in the specified extent and resolution.
    """

    def __init__(
        self,
        file_path,
        min_x_extent=None,
        max_x_extent=None,
        min_y_extent=None,
        max_y_extent=None,
    ):
        """Initialize the LiDAR processing object.

        This constructor initializes the LiDAR processing object by loading the LiDAR data
        from the specified file path. It sets the spatial extents for the LiDAR data, which
        can be specified or automatically determined from the loaded data.

        Args:
            file_path (str): The path to the LiDAR data file to be loaded.
            min_x_extent (float, optional): The minimum x-coordinate extent for the grid.
                                            Defaults to None, which will set it to the minimum
                                            x-coordinate of the LiDAR data converted to meters.
            max_x_extent (float, optional): The maximum x-coordinate extent for the grid.
                                            Defaults to None, which will set it to the maximum
                                            x-coordinate of the LiDAR data converted to meters.
            min_y_extent (float, optional): The minimum y-coordinate extent for the grid.
                                            Defaults to None, which will set it to the minimum
                                            y-coordinate of the LiDAR data converted to meters.
            max_y_extent (float, optional): The maximum y-coordinate extent for the grid.
                                            Defaults to None, which will set it to the maximum
                                            y-coordinate of the LiDAR data converted to meters.
        """
        self.filepath = file_path
        self.filename = os.path.basename(file_path)
        self.lidar = self.load_lidar()
        self.min_x_extent = (
            min_x_extent if min_x_extent is not None else np.min(self.lidar.x) * 0.3048
        )
        self.max_x_extent = (
            max_x_extent if max_x_extent is not None else np.max(self.lidar.x) * 0.3048
        )
        self.min_y_extent = (
            min_y_extent if min_y_extent is not None else np.min(self.lidar.y) * 0.3048
        )
        self.max_y_extent = (
            max_y_extent if max_y_extent is not None else np.max(self.lidar.y) * 0.3048
        )

    def load_lidar(self):
        """Load LiDAR data from the specified file.

        This function reads the LiDAR data from the file located at `self.filepath`
        and returns the LiDAR object for further processing.

        Returns:
            laspy.LasData: The LiDAR data loaded from the file.
        """
        lidar = laspy.read(self.filepath)

        return lidar

    def create_point_array(self, point_mask_value=2):
        """Create an array of LiDAR points based on classification.

        This function filters the LiDAR data to select points that match the
        specified classification value, converts their coordinates from feet to meters,
        and applies extent filtering based on predefined minimum and maximum extents.

        Args:
            point_mask_value (int, optional): The classification value to filter points. Defaults to 2.

        Returns:
            np.ndarray: An array of filtered LiDAR points in meters, within the specified extents. Returns all LiDAR
                        points if no extent is specified.
        """
        self.point_mask_val = point_mask_value

        point_mask = (
            self.lidar.classification == point_mask_value
        )  # Mask points based on classification

        # Convert feet to meters and stack into an array
        xyz_m = np.vstack(
            [
                self.lidar.x[point_mask] * 0.3048,
                self.lidar.y[point_mask] * 0.3048,
                self.lidar.z[point_mask] * 0.3048,
            ]
        )

        # Apply extent filtering
        extent_mask = (
            (xyz_m[0] >= self.min_x_extent)
            & (xyz_m[0] <= self.max_x_extent)
            & (xyz_m[1] >= self.min_y_extent)
            & (xyz_m[1] <= self.max_y_extent)
        )

        return xyz_m[:, extent_mask]  # Keep only the points within the extents

    def gen_grid(self, resolution, z=0, dir="generated_grids"):
        """Generate a grid of points in the specified extent and resolution.

        This function creates a grid based on the specified resolution and z-value. It can
        generate a uniform grid at a specified elevation (z) or interpolate a set of points
        provided in a 2D array. The generated grid arrays are saved as compressed Zarr files
        in the specified directory.

        Args:
            resolution (float): The spacing between grid points in the x and y dimensions.
            z (int, float, np.ndarray, optional): The elevation value to use for the z dimension. Defaults to 0.
            dir (str, optional): The directory where the generated grid files will be saved. Defaults to 'generated_grids'.

        Returns:
            tuple: A tuple containing three numpy arrays:
                - grid_x (np.ndarray): The x-coordinates of the grid points.
                - grid_y (np.ndarray): The y-coordinates of the grid points.
                - grid_z (np.ndarray): The z-coordinates of the grid points.
        """
        # Check that the path exists
        if not os.path.exists(dir):
            os.makedirs(dir)  # Create the directory if not
            print(f"Directory to store grids created: {dir}")
        else:
            print(f"Directory to store grids already exists: {dir}")

        # Generate structured grid for extents and resolution
        grid_x, grid_y = np.mgrid[
            self.min_x_extent : self.max_x_extent : resolution,
            self.min_y_extent : self.max_y_extent : resolution,
        ]

        # If it or float provided for z, create set elevation grid
        if isinstance(z, int) or isinstance(z, float):
            grid_z = np.full_like(grid_x, z)

        # If points array is provided, fit points to structured grid
        if isinstance(z, np.ndarray):
            x = z[0]
            y = z[1]
            z = z[2]
            grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

        grid_x = grid_x.T
        grid_y = grid_y.T
        grid_z = grid_z.T

        # Save the grid arrays to a compressed Zarr files
        zarr.save(os.path.join(dir, f"{self.filename}_grid_x_{resolution}m.zarr"), grid_x)
        zarr.save(os.path.join(dir, f"{self.filename}_grid_y_{resolution}m.zarr"), grid_y)
        zarr.save(os.path.join(dir, f"{self.filename}_grid_z_{resolution}m.zarr"), grid_z)

        return grid_x, grid_y, grid_z
