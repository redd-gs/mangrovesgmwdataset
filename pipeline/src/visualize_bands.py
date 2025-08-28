import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# This mapping is for Sentinel-2 bands
sentinel2_mapping = {
    'blue': 'B02',
    'green': 'B03',
    'red': 'B04',
    "nir": "B5",
    "swir1": "B6",
    "swir2": "B7",
}

def process(band_data, xmin, xmax, ymin, ymax):
    """Crops the band data to the given extent."""
    return band_data[0, ymin:ymax, xmin:xmax]

def create_rgb_image(base_path: Path, product_id: str, xmin: int, xmax: int, ymin: int, ymax: int):
    """
    Creates and displays an RGB image from Sentinel-2 bands.

    Args:
        base_path (Path): The base directory where band files are stored (e.g., 'data/temp').
        product_id (str): The Sentinel-2 product ID (the directory name for the bands).
        xmin (int): Minimum x-coordinate for cropping.
        xmax (int): Maximum x-coordinate for cropping.
        ymin (int): Minimum y-coordinate for cropping.
        ymax (int): Maximum y-coordinate for cropping.
    """
    data = {}
    product_path = base_path / product_id
    
    if not product_path.exists():
        print(f"Error: Directory not found at {product_path}")
        return

    print(f"Reading bands from: {product_path}")

    for band_name, band_code in sentinel2_mapping.items():
        # Construct the file path for each band
        band_file = product_path / f'{band_code}.tif'
        
        if not band_file.exists():
            print(f"Error: Band file not found at {band_file}")
            return
            
        print(f"Reading band: {band_file.name}")
        with rasterio.open(band_file) as src:
            data[band_name] = process(src.read(), xmin, xmax, ymin, ymax)

    if len(data) != 3:
        print("Error: Could not read all required bands (Red, Green, Blue).")
        return

    rgb = np.dstack((data['red'], data['green'], data['blue']))

    # Normalize the image for display
    rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    plt.imshow(rgb_normalized)
    plt.title(f'RGB Composite for {product_id}')
    plt.show()

if __name__ == '__main__':
    # --- Example Usage ---
    # IMPORTANT: Replace with the actual product ID and desired coordinates
    
    # The base path to the temporary data
    temp_data_path = Path(__file__).parent.parent / 'data' / 'temp'

    # The product ID (directory name) of the downloaded Sentinel-2 data
    # You will need to check the 'data/temp' folder for the correct name after running the download.
    example_product_id = 'S2A_MSIL1C_20220101T032141_N0301_R018_T48NUG_20220101T050552' #<-- REPLACE THIS

    # Define the pixel coordinates for the area of interest to crop
    xmin, xmax, ymin, ymax = 3000, 3300, 3900, 4100

    create_rgb_image(temp_data_path, example_product_id, xmin, xmax, ymin, ymax)
