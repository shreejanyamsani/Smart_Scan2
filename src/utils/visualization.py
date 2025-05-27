import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import cv2

def plot_confusion_matrix(cm):
    """
    Plot the confusion matrix.
    
    Args:
        cm (list): Confusion matrix as a list
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    plt.close()

def visualize_prediction(original, preprocessed, edges, segmented, mask, prediction, confidence):
    """
    Visualize the processing steps and prediction.
    
    Args:
        original (numpy.ndarray): Original image
        preprocessed (numpy.ndarray): Preprocessed image
        edges (numpy.ndarray): Edge-detected image
        segmented (numpy.ndarray): Segmented lungs
        mask (numpy.ndarray): Lung mask
        prediction (str): Prediction label
        confidence (float): Prediction confidence
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    
    # Preprocessed image
    axes[0, 1].imshow(preprocessed, cmap='gray')
    axes[0, 1].set_title('Preprocessed Image')
    
    # Edge detection
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title('Edge Detection')
    
    # Segmented lungs
    axes[1, 0].imshow(segmented, cmap='gray')
    axes[1, 0].set_title('Segmented Lungs')
    
    # Lung mask
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('Lung Mask')
    
    # Prediction result
    prediction_img = np.ones_like(original) * 200
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Create empty array for prediction display
    h, w = original.shape if len(original.shape) == 2 else original.shape[:2]
    prediction_img = np.ones((h, w), dtype=np.uint8) * 200
    
    # Add text
    text_color = 0  # Black
    prediction_text = f"Prediction: {prediction}"
    confidence_text = f"Confidence: {confidence:.2f}"
    
    # Draw texts at different positions
    cv2.putText(prediction_img, prediction_text, (w//8, h//2 - 20), font, 0.8, text_color, 2)
    cv2.putText(prediction_img, confidence_text, (w//8, h//2 + 20), font, 0.8, text_color, 2)
    
    # Add color indicator (green for normal, red for pneumonia)
    indicator_color = 150 if prediction == 'NORMAL' else 50
    cv2.circle(prediction_img, (w//2, h//4), h//8, indicator_color, -1)
    
    axes[1, 2].imshow(prediction_img, cmap='gray')
    axes[1, 2].set_title('Classification Result')
    
    # Turn off axis for all subplots
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def plot_feature_importance(classifier, feature_names):
    """
    Plot feature importance for Random Forest classifier.
    
    Args:
        classifier: RandomForestClassifier instance
        feature_names (list): List of feature names
    """
    if not feature_names:
        print("Feature names are not available.")
        return
    
    # Get feature importances
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
    
    # Print top 10 features
    print("Top 10 most important features:")
    for i in range(min(10, len(indices))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")