# Activity Recognition and Fall Detection Model

This project focuses on recognizing human activities and detecting falls using sensor data. The model is trained using TensorFlow and then converted to TensorFlow Lite (TFLite) for deployment on edge devices.

## Overview

The project processes accelerometer and gyroscope data to classify activities into one of 10 predefined categories, including fall detection. It uses a windowing technique to group sensor readings into meaningful chunks for training and prediction.

---

## Dataset

The dataset used includes sensor readings:
- **Features**: `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`
- **Labels**: Integer values (0-9) representing the activity classes.

---

## Workflow

1. **Data Preprocessing**:
   - Load raw data.
   - Apply windowing to group sequential data into overlapping windows.
   - Standardize the data to zero mean and unit variance.
   - Split the dataset into training and validation sets.

2. **Model Training**:
   - A Convolutional Neural Network (CNN) architecture is used.
   - Model includes:
     - **Conv1D layers** for feature extraction.
     - **Dropout layers** to prevent overfitting.
     - A **Dense layer** for classification.
   - Categorical Crossentropy loss and Adam optimizer are used for training.
   - Early stopping is implemented to monitor and avoid overfitting.

3. **Model Conversion**:
   - Convert the trained model to TFLite for deployment on devices.

4. **Model Evaluation**:
   - Evaluate model performance on validation data.
   - Calculate validation accuracy and loss.

## Installation
### Prerequisites:
- Python 3.8 or higher
- TensorFlow 2.x
- Required Python libraries

## Example Deployment

- Edge Devices: The TFLite model can be deployed on IoT devices such as Raspberry Pi or ESP32.
- Real-Time Predictions: The model can process sensor data in real time to detect activities or falls.

## Future Work
- Expand the dataset to include more activities.
- Improve the model architecture for higher accuracy.
- Optimize the TFLite model with quantization techniques.
- Deploy the model on hardware for real-world testing.


