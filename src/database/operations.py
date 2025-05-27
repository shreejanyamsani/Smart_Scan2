from datetime import datetime

def log_dataset_metadata(collection, dataset_path, status, error_message=None):
    """
    Log dataset download metadata to MongoDB.
    
    Args:
        collection (pymongo.collection): MongoDB collection
        dataset_path (str): Path to the dataset
        status (str): Status of the download ('success' or 'failed')
        error_message (str, optional): Error message if failed
    """
    dataset_metadata = {
        "dataset_path": dataset_path,
        "download_timestamp": datetime.utcnow(),
        "status": status
    }
    if error_message:
        dataset_metadata["error_message"] = error_message
    collection.insert_one(dataset_metadata)

def log_training_results(collection, model_type, metrics, confusion_matrix, feature_importance):
    """
    Log training results to MongoDB.
    
    Args:
        collection (pymongo.collection): MongoDB collection
        model_type (str): Type of model
        metrics (dict): Evaluation metrics
        confusion_matrix (list): Confusion matrix
        feature_importance (dict): Feature importance scores
    """
    training_result = {
        "timestamp": datetime.utcnow(),
        "model_type": model_type,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "feature_importance": feature_importance
    }
    collection.insert_one(training_result)

def log_prediction(collection, image_path, prediction, confidence):
    """
    Log prediction results to MongoDB.
    
    Args:
        collection (pymongo.collection): MongoDB collection
        image_path (str): Path to the image
        prediction (str): Prediction label
        confidence (float): Prediction confidence
    """
    prediction_log = {
        "timestamp": datetime.utcnow(),
        "image_path": image_path,
        "prediction": prediction,
        "confidence": float(confidence)
    }
    collection.insert_one(prediction_log)