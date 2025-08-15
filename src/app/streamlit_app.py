import os
import sys
import streamlit as st
from datetime import datetime
from contextlib import redirect_stdout
from PIL import Image


from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Replace with your actual MongoDB credentials
username = "shreejanyamsani"  # Your MongoDB username
password = "HqfQKFcO1rKL0lJr"  # Your MongoDB password
MONGODB_URI = f"mongodb+srv://{username}:{password}@cluster0.xcfxtu5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Test connection
try:
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print("❌ Connection failed:", e)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from core.detector import PneumoniaDetector
from database.connection import get_mongo_client
from utils.dataset import download_dataset
from utils.logger import StreamlitLogger


def run_app():
    """Run the Streamlit app for pneumonia detection with MongoDB."""
    
    # ─── App Title ───────────────────────────────────────────
    st.title("Pneumonia Detection from Chest X-Ray Images")
    st.write(
        "This app uses a Random Forest classifier to detect pneumonia "
        "from chest X-ray images, with data stored in MongoDB."
    )

    # ─── MongoDB Connection ─────────────────────────────────
    client = get_mongo_client(MONGODB_URI)
    if client is None:
        st.error("Could not connect to MongoDB. Please check your connection.")
        return

    db = client["pneumonia_detection"]
    dataset_collection = db["dataset_metadata"]
    training_collection = db["training_results"]
    prediction_collection = db["prediction_logs"]

    st.success("Connected to MongoDB database.")

    # ─── Create directory for uploaded images ────────────────
    os.makedirs("data/uploaded_images", exist_ok=True)

    # ─── Sidebar Controls ───────────────────────────────────
    st.sidebar.header("Model Controls")
    download_button = st.sidebar.button("Download Dataset")
    train_button = st.sidebar.button("Train Model")

    # ─── Session State Initialization ───────────────────────
    if "detector" not in st.session_state:
        st.session_state.detector = None
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = None

    # ─── Dataset Download ───────────────────────────────────
    if download_button:
        with st.spinner("Downloading dataset..."):
            st.session_state.dataset_path = download_dataset(dataset_collection)
            if st.session_state.dataset_path:
                st.success(f"Dataset downloaded to: {st.session_state.dataset_path}")
            else:
                st.error("Failed to download dataset. Please check KaggleHub configuration.")

    if st.session_state.dataset_path:
        st.write(f"**Dataset Path**: {st.session_state.dataset_path}")

    # ─── Model Training ─────────────────────────────────────
    if train_button and st.session_state.dataset_path:
        if not os.path.exists(st.session_state.dataset_path):
            st.error(f"Dataset path {st.session_state.dataset_path} does not exist.")
            return

        logger = StreamlitLogger()
        with redirect_stdout(logger):
            st.session_state.detector = PneumoniaDetector(model_type="random_forest")
            st.write("Training pneumonia detector...")
            st.session_state.detector.train(
                st.session_state.dataset_path,
                max_train_samples=500
            )

            st.write("\nEvaluating on test set...")
            metrics = st.session_state.detector.evaluate(
                st.session_state.dataset_path,
                max_test_samples=200,
                db_collection=training_collection
            )

        st.subheader("Evaluation Metrics")
        st.write(f"**Accuracy**: {metrics['accuracy']:.4f}")
        st.write(f"**Precision**: {metrics['precision']:.4f}")
        st.write(f"**Recall**: {metrics['recall']:.4f}")
        st.write(f"**F1 Score**: {metrics['f1']:.4f}")

    # ─── Image Upload & Prediction ──────────────────────────
    st.subheader("Upload Chest X-Ray Image for Prediction")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG/JPEG)", 
        type=["jpg", "jpeg"]
    )

    if uploaded_file and st.session_state.detector:
        try:
            # Save uploaded file
            image_filename = f"data/uploaded_images/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
            with open(image_filename, "wb") as f:
                f.write(uploaded_file.read())

            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            st.write("Predicting...")
            logger = StreamlitLogger()
            with redirect_stdout(logger):
                prediction, confidence = st.session_state.detector.predict(
                    image_filename,
                    visualize=True,
                    db_collection=prediction_collection
                )

            if prediction is not None:
                st.success(
                    f"Prediction: {'PNEUMONIA' if prediction == 1 else 'NORMAL'} "
                    f"(Confidence: {confidence:.2f})"
                )
            else:
                st.error("Failed to make prediction.")

        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")

    # ─── Prediction History ─────────────────────────────────
    st.subheader("Prediction History")
    try:
        predictions = list(
            prediction_collection.find().sort("timestamp", -1).limit(10)
        )
        if predictions:
            for pred in predictions:
                st.write(
                    f"**Time**: {pred['timestamp']}, "
                    f"**Image**: {pred['image_path']}, "
                    f"**Prediction**: {pred['prediction']}, "
                    f"**Confidence**: {pred['confidence']:.2f}"
                )
                if os.path.exists(pred["image_path"]):
                    st.image(pred["image_path"], caption="Predicted Image", width=100)
        else:
            st.write("No predictions found in the database.")

    except Exception as e:
        st.error(f"Error retrieving prediction history: {e}")


if __name__ == "__main__":
    run_app()





