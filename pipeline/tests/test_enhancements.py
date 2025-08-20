import pytest
import numpy as np
from src.processing.enhancements import enhance_image

def test_enhance_image_brightness():
    # Create a dummy image (3x3 pixels with 3 color channels)
    image = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    ])
    
    enhanced_image = enhance_image(image, brightness_factor=1.5, contrast_factor=1.0)
    # On vérifie que la moyenne a augmenté
    assert enhanced_image.mean() > image.mean()

def test_enhance_image_contrast():
    # Create a dummy image (3x3 pixels with 3 color channels)
    image = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    ])
    
    enhanced_image = enhance_image(image, brightness_factor=1.0, contrast_factor=2.0)
    # Le contraste augmente: écart-type plus grand (ou égal si clipping massif)
    assert enhanced_image.std() >= image.std()

def test_enhance_image_gamma():
    # Create a dummy image (3x3 pixels with 3 color channels)
    image = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    ])
    
    enhanced_image = enhance_image(image, brightness_factor=1.0, contrast_factor=1.0, gamma=0.5)
    expected_image = np.power(np.clip(image, 1e-6, 1), 0.5)
    assert np.allclose(enhanced_image, expected_image, atol=1e-2)

def test_enhance_image_invalid_input():
    # Test with an invalid image input
    with pytest.raises(ValueError):
        enhance_image(None)

    with pytest.raises(ValueError):
        enhance_image(np.array([[1, 2], [3, 4]]))  # Not a 3-channel image (2D)

    with pytest.raises(ValueError):
        enhance_image(np.array([[1, 2, 3], [4, 5, 6]]))  # Not a 3D array