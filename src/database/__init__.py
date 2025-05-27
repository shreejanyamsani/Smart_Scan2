# src/database/__init__.py
from .connection import get_mongo_client
from .operations import log_dataset_metadata, log_training_results, log_prediction