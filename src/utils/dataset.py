import os
import kagglehub
import zipfile
import streamlit as st
from src.database.operations import log_dataset_metadata

def download_dataset(db_collection):
    """Download and extract the chest X-ray dataset from Kaggle and log to MongoDB."""
    try:
        print("Downloading Chest X-Ray dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"Dataset downloaded to: {dataset_path}")

        # Check if the downloaded path contains a zip file
        zip_files = [f for f in os.listdir(dataset_path) if f.endswith('.zip')]
        if zip_files:
            zip_path = os.path.join(dataset_path, zip_files[0])
            extract_path = os.path.join(dataset_path, 'extracted')
            print(f"Extracting {zip_path} to {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("Extraction complete.")
            # Update dataset_path to point to the extracted chest_xray folder
            dataset_path = os.path.join(extract_path, 'chest_xray')
        else:
            # If no zip file, check for chest_xray folder
            chest_xray_path = os.path.join(dataset_path, 'chest_xray')
            if os.path.exists(chest_xray_path):
                dataset_path = chest_xray_path
        
        # Log to MongoDB
        log_dataset_metadata(db_collection, dataset_path, "success")
        
        return dataset_path
    except Exception as e:
        st.error(f"Error downloading or extracting dataset: {e}")
        st.error("Please ensure you have kagglehub installed and properly configured.")
        st.error("You can install it with: pip install kagglehub")
        st.error("Then authenticate with your Kaggle credentials if needed.")
        # Log failure to MongoDB
        if db_collection is not None:
            log_dataset_metadata(db_collection, None, "failed", str(e))
        return None