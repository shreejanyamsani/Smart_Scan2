import pytest
import numpy as np
from src.core.feature_extraction import FeatureExtractor
from src.core.preprocessing import ImagePreprocessor
import cv2

@pytest.fixture
def sample_image():
    """Create a sample grayscale image for testing."""
    img = np.zeros((256, 256), dtype=np.uint8)
    img[50:200, 50:200] = 255
    return img

def test_feature_extractor(sample_image):
    """Test the FeatureExtractor class."""
    extractor = FeatureExtractor()
    
    # Test edge detection
    edges = extractor.detect_edges(sample_image)
    assert edges.shape == sample_image.shape
    assert edges.dtype == np.uint8
    
    # Test lung segmentation
    segmented, mask = extractor.segment_lungs(sample_image)
    assert segmented.shape == sample_image.shape
    assert mask.shape == sample_image.shape
    assert mask.dtype == np.uint8
    
    # Test feature extraction
    feature_vector, all_features, processed_images = extractor.extract_all_features(sample_image)
    assert isinstance(feature_vector, np.ndarray)
    assert isinstance(all_features, dict)
    assert isinstance(processed_images, dict)
    assert len(feature_vector) == len(all_features)
    assert set(processed_images.keys()) == {'edges', 'segmented', 'mask'}