# Pneumonia Detection from Chest X-Ray Images

This project is a modularized system for detecting pneumonia from chest X-ray images using a Random Forest classifier. It preprocesses images, extracts features, trains a machine learning model, and provides a Streamlit web interface for predictions. Results and metadata are stored in a MongoDB database.

## Project Structure

```
pneumonia_detection/
├── src/
│   ├── core/
│   │   ├── preprocessing.py
│   │   ├── feature_extraction.py
│   │   └── detector.py
│   ├── database/
│   │   ├── connection.py
│   │   └── operations.py
│   ├── utils/
│   │   ├── dataset.py
│   │   ├── logger.py
│   │   └── visualization.py
│   ├── app/
│   │   └── streamlit_app.py
├── config/
│   └── settings.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_detector.py
├── data/
│   └── uploaded_images/
├── requirements.txt
├── setup.py
├── README.md
└── main.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pneumonia_detection.git
   cd pneumonia_detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure MongoDB is running locally or provide a MongoDB Atlas URI in `config/settings.py`.

4. (Optional) Install the package:
   ```bash
   python setup.py install
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run src/app/streamlit_app.py
```

- **Download Dataset**: Click the "Download Dataset" button in the sidebar to fetch the chest X-ray dataset from Kaggle.
- **Train Model**: Click the "Train Model" button to train the Random Forest classifier.
- **Predict**: Upload a chest X-ray image (JPG/JPEG) to get a pneumonia prediction.
- **View History**: Recent predictions are displayed from the MongoDB database.

## Requirements

See `requirements.txt` for a list of dependencies.

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## License

MIT License
