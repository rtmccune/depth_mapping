import cmocean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_elev_grid(grid_z, save_fig=False, fig_name='pixel_DEM.png'):

    # Get the topo colormap from cmocean
    cmap = cmocean.cm.topo

    # Truncate the colormap to get only the above-land portion
    # Assuming "above land" is the upper half of the colormap
    above_land_cmap = LinearSegmentedColormap.from_list(
        'above_land_cmap', cmap(np.linspace(0.5, 1, 256))
    )

    plt.imshow(grid_z, origin='lower', cmap=above_land_cmap)
    plt.colorbar(label='Elevation (meters)')
    plt.title('Pixel DEM')
    plt.xlabel('Easting')
    plt.ylabel('Northing')

    if save_fig:
        # Save the figure before showing it
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.show()