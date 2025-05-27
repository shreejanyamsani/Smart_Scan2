import pytest
import numpy as np
from src.core.detector import PneumoniaDetector
from src.core.preprocessing import ImagePreprocessor
import os
import cv2

@pytest.fixture
def sample_dataset(tmp_path):
    """Create a mock dataset structure for testing."""
    train_normal = tmp_path / "train" / "NORMAL"
    train_pneumonia = tmp_path / "train" / "PNEUMONIA"
    test_normal = tmp_path / "test" / "NORMAL"
    test_pneumonia = tmp_path / "test" / "PNEUMONIA"
    
    train_normal.mkdir(parents=True)
    train_pneumonia.mkdir(parents=True)
    test_normal.mkdir(parents=True)
    test_pneumonia.mkdir(parents=True)
    
    # Create sample images
    img = np.zeros((256, 256), dtype=np.uint8)
    img[50:200, 50:200] = 255
    
    cv2.imwrite(str(train_normal / "normal1.jpg"), img)
    cv2.imwrite(str(train_pneumonia / "pneumonia1.jpg"), img)
    cv2.imwrite(str(test_normal / "normal2.jpg"), img)
    cv2.imwrite(str(test_pneumonia / "pneumonia2.jpg"), img)
    
    return str(tmp_path)

def test_pneumonia_detector(sample_dataset):
    """Test the PneumoniaDetector class."""
    detector = PneumoniaDetector(model_type='random_forest')
    
    # Test dataset loading
    files, labels = detector.load_dataset(sample_dataset, 'train')
    assert len(files) == 2
    assert len(labels) == 2
    assert set(labels) == {0, 1}
    
    # Test training
    detector.train(sample_dataset, max_train_samples=2)
    assert detector.is_trained
    
    # Test evaluation
    metrics = detector.evaluate(sample_dataset, max_test_samples=2, db_collection=None)
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'}
    
    # Test prediction
    prediction, confidence = detector.predict(str(sample_dataset / "test" / "NORMAL" / "normal2.jpg"), visualize=False)
    assert prediction in [0, 1]
    assert 0 <= confidence <= 1