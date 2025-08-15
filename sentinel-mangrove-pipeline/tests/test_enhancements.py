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
    
    enhanced_image = enhance_image(image, brightness_factor=1.5)
    
    # Check if the brightness is enhanced correctly
    expected_image = np.clip(image * 1.5, 0, 1)
    assert np.array_equal(enhanced_image, expected_image)

def test_enhance_image_contrast():
    # Create a dummy image (3x3 pixels with 3 color channels)
    image = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    ])
    
    enhanced_image = enhance_image(image, contrast_factor=2.0)
    
    # Check if the contrast is enhanced correctly
    expected_image = np.clip((image - 0.5) * 2.0 + 0.5, 0, 1)
    assert np.array_equal(enhanced_image, expected_image)

def test_enhance_image_gamma():
    # Create a dummy image (3x3 pixels with 3 color channels)
    image = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    ])
    
    enhanced_image = enhance_image(image, gamma=0.5)
    
    # Check if the gamma correction is applied correctly
    expected_image = np.power(image, 0.5)
    assert np.allclose(enhanced_image, expected_image, atol=1e-2)

def test_enhance_image_invalid_input():
    # Test with an invalid image input
    with pytest.raises(ValueError):
        enhance_image(None)

    with pytest.raises(ValueError):
        enhance_image(np.array([[1, 2], [3, 4]]))  # Not a 3-channel image

    with pytest.raises(ValueError):
        enhance_image(np.array([[1, 2, 3], [4, 5, 6]]))  # Not a 3D array