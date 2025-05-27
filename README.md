# Iris Tumor Detection using CNN

This project utilizes Convolutional Neural Networks (CNN) for detecting tumors in iris images. It aims to assist in automating the classification of iris images as either tumorous or healthy, leveraging the power of deep learning.

## Features
- Model Architecture: Custom CNN or MobileNetV2 for efficient and accurate classification.
- Dataset Handling: Automated train/test split and preprocessing.
- User-Friendly Interface: Supports easy-to-run configurations with a web-based UI for image upload and prediction.

## File Structure
- train_model_v2.py: Code for training the CNN model.
- app.py: Main Flask application file for making predictions and serving the web UI.
- requirements.txt: List of required Python packages.
- Templates/: Contains HTML templates for the web UI (index.html and result.html).
- Dataset folders: Organize the dataset as per the project requirements.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Scikit-learn
- Flask 2.2.2
- Werkzeug 2.2.3

## Setup and Usage

### Training the Model
To train the CNN model from scratch, run the training script:

```
python train_model_v2.py
```

This script will:
- Load and preprocess the dataset.
- Train the CNN model (MobileNetV2 based).
- Save the trained model to `iris_tumor_cnn_model.keras`.

Make sure your dataset is organized correctly before training.

### Running the Flask App
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

4. Upload an eye image to detect the presence of tumors. The system will analyze the image and display the prediction result with confidence scores.

## License
This project is licensed under the MIT License.
