from datetime import datetime
import os
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from src.core.preprocessing import ImagePreprocessor
from src.core.feature_extraction import FeatureExtractor
from src.utils.visualization import plot_confusion_matrix, visualize_prediction, plot_feature_importance

class PneumoniaDetector:
    """Main class for pneumonia detection from chest X-ray images."""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the detector.
        
        Args:
            model_type (str): Type of classifier ('random_forest', 'svm', or 'knn')
        """
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        
        # Initialize classifier based on model_type
        if model_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.classifier = SVC(probability=True, kernel='rbf', random_state=42)
        elif model_type == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.feature_names = None
        self.is_trained = False
    
    def load_dataset(self, dataset_path, split='train'):
        """
        Load images from the dataset directory.
        
        Args:
            dataset_path (str): Path to the dataset directory
            split (str): Dataset split to load ('train', 'test', or 'val')
            
        Returns:
            tuple: X (feature vectors), y (labels), and file paths
        """
        normal_dir = os.path.join(dataset_path, split, 'NORMAL')
        pneumonia_dir = os.path.join(dataset_path, split, 'PNEUMONIA')
        
        print(f"Checking normal directory: {normal_dir}")
        print(f"Checking pneumonia directory: {pneumonia_dir}")
        
        # Get file paths (support both .jpeg and .jpg)
        normal_files = glob(os.path.join(normal_dir, '*.jpeg')) + glob(os.path.join(normal_dir, '*.jpg'))
        pneumonia_files = glob(os.path.join(pneumonia_dir, '*.jpeg')) + glob(os.path.join(pneumonia_dir, '*.jpg'))
        
        print(f"Found {len(normal_files)} normal files")
        print(f"Found {len(pneumonia_files)} pneumonia files")
        
        if not normal_files or not pneumonia_files:
            raise ValueError(f"Could not find images in {normal_dir} or {pneumonia_dir}")
        
        print(f"Found {len(normal_files)} normal and {len(pneumonia_files)} pneumonia images in {split} set")
        
        all_files = normal_files + pneumonia_files
        labels = [0] * len(normal_files) + [1] * len(pneumonia_files)  # 0: Normal, 1: Pneumonia
        
        return all_files, labels
    
    def extract_features_from_files(self, file_paths, max_files=None):
        """
        Extract features from a list of image files.
        
        Args:
            file_paths (list): List of image file paths
            max_files (int, optional): Maximum number of files to process
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        if max_files:
            file_paths = file_paths[:max_files]
        
        features = []
        valid_indices = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Load and preprocess image
                preprocessed_data = self.preprocessor.load_and_preprocess(file_path)
                
                # Extract features
                feature_vector, _, _ = self.feature_extractor.extract_all_features(preprocessed_data['preprocessed'])
                
                features.append(feature_vector)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return np.array(features), valid_indices
    
    def train(self, dataset_path, max_train_samples=None):
        """
        Train the classifier on the training set.
        
        Args:
            dataset_path (str): Path to the dataset directory
            max_train_samples (int, optional): Maximum number of training samples to use
            
        Returns:
            self: Trained detector
        """
        # Load training data
        train_files, train_labels = self.load_dataset(dataset_path, 'train')
        
        if max_train_samples:
            # Keep class balance when limiting samples
            n_normal = sum(1 for label in train_labels if label == 0)
            n_pneumonia = sum(1 for label in train_labels if label == 1)
            
            # Calculate how many of each class to keep
            keep_normal = min(n_normal, max_train_samples // 2)
            keep_pneumonia = min(n_pneumonia, max_train_samples - keep_normal)
            
            # Create balanced subset
            normal_indices = [i for i, label in enumerate(train_labels) if label == 0][:keep_normal]
            pneumonia_indices = [i for i, label in enumerate(train_labels) if label == 1][:keep_pneumonia]
            
            selected_indices = normal_indices + pneumonia_indices
            train_files = [train_files[i] for i in selected_indices]
            train_labels = [train_labels[i] for i in selected_indices]
            
            print(f"Using {len(normal_indices)} normal and {len(pneumonia_indices)} pneumonia samples for training")
        
        # Extract features
        print("Extracting features from training images...")
        X_train, valid_indices = self.extract_features_from_files(train_files)
        y_train = [train_labels[i] for i in valid_indices]
        
        # Store feature names if not already set
        if not self.feature_names:
            # Create a sample feature dictionary to get keys
            _, sample_dict, _ = self.feature_extractor.extract_all_features(
                self.preprocessor.load_and_preprocess(train_files[0])['preprocessed']
            )
            self.feature_names = list(sample_dict.keys())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the classifier
        print("Training classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        return self
    
    def evaluate(self, dataset_path, max_test_samples=None, db_collection=None):
        """
        Evaluate the classifier on the test set and store results in MongoDB.
        
        Args:
            dataset_path (str): Path to the dataset directory
            max_test_samples (int, optional): Maximum number of test samples to use
            db_collection (pymongo.collection): MongoDB collection for training results
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Load test data
        test_files, test_labels = self.load_dataset(dataset_path, 'test')
        
        if max_test_samples:
            # Keep class balance when limiting samples
            normal_indices = [i for i, label in enumerate(test_labels) if label == 0][:max_test_samples//2]
            pneumonia_indices = [i for i, label in enumerate(test_labels) if label == 1][:max_test_samples//2]
            
            selected_indices = normal_indices + pneumonia_indices
            test_files = [test_files[i] for i in selected_indices]
            test_labels = [test_labels[i] for i in selected_indices]
        
        # Extract features
        print("Extracting features from test images...")
        X_test, valid_indices = self.extract_features_from_files(test_files)
        y_test = [test_labels[i] for i in valid_indices]
        
        # Scale features using the same scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for MongoDB
        
        # Print evaluation results
        print("Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia']))
        print("Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm)
        
        # Store results in MongoDB
        if db_collection is not None:
            feature_importance = {}
            if isinstance(self.classifier, RandomForestClassifier) and self.feature_names:
                importances = self.classifier.feature_importances_
                feature_importance = {self.feature_names[i]: float(importances[i]) for i in range(len(self.feature_names))}
            
            training_result = {
                "timestamp": datetime.utcnow(),
                "model_type": "random_forest",
                "metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1)
                },
                "confusion_matrix": cm,
                "feature_importance": feature_importance
            }
            db_collection.insert_one(training_result)
        
        # Plot feature importance if using Random Forest
        if isinstance(self.classifier, RandomForestClassifier):
            plot_feature_importance(self.classifier, self.feature_names)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def predict(self, image_path, visualize=True, db_collection=None):
        """
        Predict whether a chest X-ray shows pneumonia and log to MongoDB.
        
        Args:
            image_path (str): Path to the image file
            visualize (bool): Whether to visualize the results
            db_collection (pymongo.collection): MongoDB collection for prediction logs
            
        Returns:
            tuple: Prediction (0: Normal, 1: Pneumonia) and confidence
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        try:
            # Load and preprocess image
            preprocessed_data = self.preprocessor.load_and_preprocess(image_path)
            
            # Extract features
            feature_vector, feature_dict, processed_images = self.feature_extractor.extract_all_features(
                preprocessed_data['preprocessed']
            )
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make prediction
            prediction = self.classifier.predict(feature_vector_scaled)[0]
            probabilities = self.classifier.predict_proba(feature_vector_scaled)[0]
            confidence = probabilities[prediction]
            
            # Convert numeric prediction to label
            prediction_label = 'PNEUMONIA' if prediction == 1 else 'NORMAL'
            
            print(f"Prediction: {prediction_label} (confidence: {confidence:.2f})")
            
            # Store prediction in MongoDB
            if db_collection is not None:
                prediction_log = {
                    "timestamp": datetime.utcnow(),
                    "image_path": image_path,
                    "prediction": prediction_label,
                    "confidence": float(confidence)
                }
                db_collection.insert_one(prediction_log)
            
            if visualize:
                visualize_prediction(
                    preprocessed_data['original'],
                    preprocessed_data['preprocessed'],
                    processed_images['edges'],
                    processed_images['segmented'],
                    processed_images['mask'],
                    prediction_label,
                    confidence
                )
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None, None