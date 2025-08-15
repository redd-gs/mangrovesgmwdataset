def tile_image(image, tile_size):
    """
    Splits the input image into smaller tiles of specified size.

    Parameters:
    - image: numpy array representing the image to be tiled.
    - tile_size: size of each tile (width, height).

    Returns:
    - List of image tiles.
    """
    tiles = []
    img_height, img_width, _ = image.shape
    tile_width, tile_height = tile_size

    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            tile = image[y:y + tile_height, x:x + tile_width]
            tiles.append(tile)

    return tiles

def process_tiles(tiles, enhancement_function):
    """
    Applies the specified enhancement function to each tile.

    Parameters:
    - tiles: List of image tiles.
    - enhancement_function: Function to enhance the image tiles.

    Returns:
    - List of enhanced image tiles.
    """
    enhanced_tiles = [enhancement_function(tile) for tile in tiles]
    return enhanced_tiles

def reconstruct_image(tiles, image_shape, tile_size):
    """
    Reconstructs the original image from its tiles.

    Parameters:
    - tiles: List of image tiles.
    - image_shape: Shape of the original image (height, width, channels).
    - tile_size: Size of each tile (width, height).

    Returns:
    - Numpy array representing the reconstructed image.
    """
    img_height, img_width, _ = image_shape
    tile_width, tile_height = tile_size
    reconstructed_image = np.zeros((img_height, img_width, tiles[0].shape[2]), dtype=tiles[0].dtype)

    tile_index = 0
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            reconstructed_image[y:y + tile_height, x:x + tile_width] = tiles[tile_index]
            tile_index += 1

    return reconstructed_image

def tile_and_process_image(image, tile_size, enhancement_function):
    """
    Tiles the input image, processes each tile with the enhancement function, 
    and reconstructs the enhanced image.

    Parameters:
    - image: numpy array representing the image to be processed.
    - tile_size: size of each tile (width, height).
    - enhancement_function: Function to enhance the image tiles.

    Returns:
    - Numpy array representing the enhanced image.
    """
    tiles = tile_image(image, tile_size)
    enhanced_tiles = process_tiles(tiles, enhancement_function)
    enhanced_image = reconstruct_image(enhanced_tiles, image.shape, tile_size)
    return enhanced_image