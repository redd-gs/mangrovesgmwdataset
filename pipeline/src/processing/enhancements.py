import numpy as np
from PIL import Image, ImageEnhance

def enhance_image(image_array, brightness_factor=1.2, contrast_factor=1.2, gamma=None):
    """
    Enhance the image quality by adjusting brightness and contrast.

    Parameters:
    - image_array: numpy array of the image to enhance.
    - brightness_factor: Factor by which to adjust brightness (1.0 means no change).
    - contrast_factor: Factor by which to adjust contrast (1.0 means no change).

    Returns:
    - Enhanced image as a numpy array.
    """
    # Validation basique
    if image_array is None:
        raise ValueError("image_array is None")
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image_array must be a 3D numpy array with 3 channels")

    # Convert the numpy array (assumed float 0-1) to a PIL Image
    image = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8))

    # Enhance brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Convert back to numpy array
    enhanced_image_array = np.array(image) / 255.0
    # Gamma correction (appliquée en float)
    if gamma is not None:
        enhanced_image_array = np.power(np.clip(enhanced_image_array, 1e-6, 1), gamma)
    return enhanced_image_array

def batch_process_images(image_arrays, brightness_factor=1.2, contrast_factor=1.2):
    """
    Process a batch of images to enhance their quality.

    Parameters:
    - image_arrays: List of numpy arrays representing images.
    - brightness_factor: Factor by which to adjust brightness.
    - contrast_factor: Factor by which to adjust contrast.

    Returns:
    - List of enhanced images as numpy arrays.
    """
    enhanced_images = []
    for img_array in image_arrays:
        enhanced_img = enhance_image(img_array, brightness_factor, contrast_factor)
        enhanced_images.append(enhanced_img)
    return enhanced_images

def save_enhanced_images(enhanced_images, output_paths):
    """
    Save enhanced images to specified paths.

    Parameters:
    - enhanced_images: List of enhanced images as numpy arrays.
    - output_paths: List of file paths to save the images.
    """
    for img, path in zip(enhanced_images, output_paths):
        image_to_save = Image.fromarray((img * 255).astype(np.uint8))
        image_to_save.save(path)
        print(f"Saved enhanced image to {path}")