from pymongo import MongoClient
import streamlit as st

def get_mongo_client(uri="mongodb://localhost:27017/"):
    """
    Establish a connection to MongoDB.
    
    Args:
        uri (str): MongoDB connection URI
    
    Returns:
        MongoClient: MongoDB client instance
    """
    try:
        client = MongoClient(uri)
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        st.error("Please ensure MongoDB is running or check your connection string.")
        return None