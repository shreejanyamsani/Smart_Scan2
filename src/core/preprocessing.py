import cv2
import numpy as np

class ImagePreprocessor:
    """Class for preprocessing chest X-ray images."""
    
    def __init__(self, target_size=(256, 256)):
        """
        Initialize the preprocessor with target image size.
        
        Args:
            target_size (tuple): Target dimensions for resizing images
        """
        self.target_size = target_size
        
    def load_and_preprocess(self, image_path):
        """
        Load image from path and apply preprocessing steps.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing original and preprocessed images
        """
        # Load image in grayscale
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Resize to target dimensions
        resized = cv2.resize(original, self.target_size)
        
        # Apply histogram equalization for contrast enhancement
        equalized = cv2.equalizeHist(resized)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        return {
            'original': original,
            'resized': resized,
            'equalized': equalized,
            'preprocessed': blurred
        }