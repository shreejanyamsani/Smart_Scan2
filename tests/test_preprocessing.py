import pytest
import cv2
import numpy as np
from src.core.preprocessing import ImagePreprocessor
import os

@pytest.fixture
def sample_image(tmp_path):
    """Create a sample grayscale image for testing."""
    img = np.zeros((512, 512), dtype=np.uint8)
    img[100:400, 100:400] = 255
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    return str(img_path)

def test_image_preprocessor(sample_image):
    """Test the ImagePreprocessor class."""
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    
    # Test loading and preprocessing
    result = preprocessor.load_and_preprocess(sample_image)
    
    assert 'original' in result
    assert 'resized' in result
    assert 'equalized' in result
    assert 'preprocessed' in result
    
    assert result['resized'].shape == (256, 256)
    assert result['preprocessed'].shape == (256, 256)
    assert result['original'].dtype == np.uint8
    assert result['preprocessed'].dtype == np.uint8

def test_invalid_image_path():
    """Test handling of invalid image path."""
    preprocessor = ImagePreprocessor()
    with pytest.raises(ValueError):
        preprocessor.load_and_preprocess("non_existent.jpg")