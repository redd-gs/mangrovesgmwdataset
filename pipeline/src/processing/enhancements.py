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

def create_rgb_from_bands(red_band_path, green_band_path, blue_band_path, output_path, enhanced=False):
    """
    Crée une image RGB à partir de 3 bandes de Sentinel-2.
    
    Parameters:
    - red_band_path: Chemin vers la bande rouge (B04)
    - green_band_path: Chemin vers la bande verte (B03)  
    - blue_band_path: Chemin vers la bande bleue (B02)
    - output_path: Chemin de sortie pour l'image RGB
    - enhanced: Si True, applique des améliorations d'image
    
    Returns:
    - bool: True si la création a réussi, False sinon
    """
    try:
        import rasterio
        from pathlib import Path
        
        # Lire les trois bandes
        with rasterio.open(red_band_path) as red_src:
            red = red_src.read(1).astype(np.float32)
            
        with rasterio.open(green_band_path) as green_src:
            green = green_src.read(1).astype(np.float32)
            
        with rasterio.open(blue_band_path) as blue_src:
            blue = blue_src.read(1).astype(np.float32)
        
        # Normaliser les valeurs (Sentinel-2 utilise des valeurs uint16)
        red = np.clip(red / 3000.0, 0, 1)
        green = np.clip(green / 3000.0, 0, 1)
        blue = np.clip(blue / 3000.0, 0, 1)
        
        # Combiner en image RGB
        rgb_array = np.stack([red, green, blue], axis=2)
        
        # Appliquer des améliorations si demandé
        if enhanced:
            rgb_array = enhance_image(rgb_array, brightness_factor=1.3, contrast_factor=1.2)
        
        # Convertir en image PIL et sauvegarder
        rgb_uint8 = (rgb_array * 255).astype(np.uint8)
        image = Image.fromarray(rgb_uint8)
        
        # Créer le dossier parent si nécessaire
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        image.save(output_path)
        print(f"[SUCCESS] Image RGB créée et sauvegardée dans {output_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Erreur lors de la création de l'image RGB: {e}")
        return False