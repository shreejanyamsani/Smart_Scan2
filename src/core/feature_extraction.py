import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
import streamlit as st

class FeatureExtractor:
    """Class for extracting features from preprocessed chest X-ray images."""
    
    def detect_edges(self, image):
        """
        Apply Canny edge detection.
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            numpy.ndarray: Edge-detected image
        """
        edges = cv2.Canny(image, 50, 150)
        return edges
    
    def segment_lungs(self, image):
        """
        Segment the lung region using Otsu's thresholding and morphological operations.
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            tuple: Segmented image and binary mask
        """
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean the binary image
        kernel = np.ones((7, 7), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Dilate to fill holes
        dilated = cv2.dilate(opening, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for lung regions
        mask = np.zeros_like(image)
        for contour in contours:
            # Filter small contours (noise)
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply mask to original image
        segmented = cv2.bitwise_and(image, mask)
        
        return segmented, mask
    
    def extract_texture_features(self, image, mask):
        """
        Extract GLCM texture features from the segmented lung region.
        
        Args:
            image (numpy.ndarray): Preprocessed image
            mask (numpy.ndarray): Binary mask of lung region
            
        Returns:
            dict: Dictionary of texture features
        """
        # Ensure there are masked pixels to analyze
        if np.count_nonzero(mask) == 0:
            return {
                'contrast': 0, 
                'dissimilarity': 0, 
                'homogeneity': 0, 
                'energy': 0, 
                'correlation': 0,
                'ASM': 0
            }
        
        # Apply mask to focus on lung region
        masked_img = cv2.bitwise_and(image, mask)
        
        # Reduce to 16 gray levels for computational efficiency
        bins = 16
        masked_img = masked_img[mask > 0]  # Only consider pixels in the mask
        if len(masked_img) == 0:
            return {
                'contrast': 0, 
                'dissimilarity': 0, 
                'homogeneity': 0, 
                'energy': 0, 
                'correlation': 0,
                'ASM': 0
            }
            
        img_norm = np.uint8(np.floor(masked_img / 256 * bins))
        img_norm = img_norm.reshape((-1, 1))  # Reshape for GLCM computation
        
        # Calculate GLCM
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            glcm = graycomatrix(img_norm, distances, angles, levels=bins, symmetric=True, normed=True)
            
            # Extract GLCM properties
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            ASM = graycoprops(glcm, 'ASM').mean()
            
            return {
                'contrast': contrast, 
                'dissimilarity': dissimilarity, 
                'homogeneity': homogeneity, 
                'energy': energy, 
                'correlation': correlation,
                'ASM': ASM
            }
        except Exception as e:
            st.error(f"Error calculating GLCM: {e}")
            return {
                'contrast': 0, 
                'dissimilarity': 0, 
                'homogeneity': 0, 
                'energy': 0, 
                'correlation': 0,
                'ASM': 0
            }
    
    def extract_shape_features(self, mask):
        """
        Extract shape features from the lung mask.
        
        Args:
            mask (numpy.ndarray): Binary mask of lung region
            
        Returns:
            dict: Dictionary of shape features
        """
        # Label connected regions
        labeled_mask, num_labels = measure.label(mask, connectivity=2, return_num=True)
        
        if num_labels == 0:
            return {
                'area': 0,
                'perimeter': 0,
                'eccentricity': 0,
                'extent': 0,
                'solidity': 0
            }
        
        # Calculate region properties
        regions = measure.regionprops(labeled_mask)
        
        # Extract shape features
        total_area = sum(region.area for region in regions)
        avg_perimeter = np.mean([region.perimeter for region in regions]) if regions else 0
        avg_eccentricity = np.mean([region.eccentricity for region in regions]) if regions else 0
        avg_extent = np.mean([region.extent for region in regions]) if regions else 0
        avg_solidity = np.mean([region.solidity for region in regions]) if regions else 0
        
        return {
            'area': total_area,
            'perimeter': avg_perimeter,
            'eccentricity': avg_eccentricity,
            'extent': avg_extent,
            'solidity': avg_solidity
        }
    
    def extract_edge_features(self, edges, mask):
        """
        Extract features from the edge-detected image.
        
        Args:
            edges (numpy.ndarray): Edge-detected image
            mask (numpy.ndarray): Binary mask of lung region
            
        Returns:
            dict: Dictionary of edge features
        """
        # Apply mask to focus on lung region edges
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Calculate edge density
        if np.count_nonzero(mask) == 0:
            edge_density = 0
        else:
            edge_density = np.count_nonzero(masked_edges) / np.count_nonzero(mask)
        
        # Divide into quadrants
        h, w = edges.shape
        h_mid, w_mid = h // 2, w // 2
        
        # Get quadrant masks
        q1_mask = mask[:h_mid, :w_mid]
        q2_mask = mask[:h_mid, w_mid:]
        q3_mask = mask[h_mid:, :w_mid]
        q4_mask = mask[h_mid:, w_mid:]
        
        # Get quadrant edges
        q1_edges = masked_edges[:h_mid, :w_mid]
        q2_edges = masked_edges[:h_mid, w_mid:]
        q3_edges = masked_edges[h_mid:, :w_mid]
        q4_edges = masked_edges[h_mid:, w_mid:]
        
        # Calculate quadrant edge density
        q1_density = np.count_nonzero(q1_edges) / max(np.count_nonzero(q1_mask), 1)
        q2_density = np.count_nonzero(q2_edges) / max(np.count_nonzero(q2_mask), 1)
        q3_density = np.count_nonzero(q3_edges) / max(np.count_nonzero(q3_mask), 1)
        q4_density = np.count_nonzero(q4_edges) / max(np.count_nonzero(q4_mask), 1)
        
        return {
            'edge_density': edge_density,
            'q1_density': q1_density,
            'q2_density': q2_density,
            'q3_density': q3_density,
            'q4_density': q4_density
        }
    
    def extract_intensity_features(self, image, mask):
        """
        Extract statistical features from the intensity values in the lung region.
        
        Args:
            image (numpy.ndarray): Preprocessed image
            mask (numpy.ndarray): Binary mask of lung region
            
        Returns:
            dict: Dictionary of intensity features
        """
        # Apply mask to focus on lung region
        masked_img = image[mask > 0]
        
        if len(masked_img) == 0:
            return {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'median': 0
            }
        
        # Calculate statistics
        mean = np.mean(masked_img)
        std = np.std(masked_img)
        min_val = np.min(masked_img)
        max_val = np.max(masked_img)
        median = np.median(masked_img)
        
        return {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'median': median
        }
    
    def extract_all_features(self, preprocessed_image):
        """
        Extract all features from the preprocessed image.
        
        Args:
            preprocessed_image (numpy.ndarray): Preprocessed image
            
        Returns:
            tuple: Feature vector and feature dictionary
        """
        # Detect edges
        edges = self.detect_edges(preprocessed_image)
        
        # Segment lungs
        segmented, mask = self.segment_lungs(preprocessed_image)
        
        # Extract features
        texture_features = self.extract_texture_features(preprocessed_image, mask)
        shape_features = self.extract_shape_features(mask)
        edge_features = self.extract_edge_features(edges, mask)
        intensity_features = self.extract_intensity_features(preprocessed_image, mask)
        
        # Combine all features into a dictionary
        all_features = {
            **texture_features,
            **shape_features,
            **edge_features,
            **intensity_features
        }
        
        # Convert dictionary to vector for ML model
        feature_vector = np.array(list(all_features.values()))
        
        return feature_vector, all_features, {
            'edges': edges,
            'segmented': segmented,
            'mask': mask
        }