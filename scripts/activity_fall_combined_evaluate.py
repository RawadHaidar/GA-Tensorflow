import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data/liwaa_fall_test.csv")

# Select features and labels
features = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
activity_labels = data['activity'].values  # Use the combined activity label column

# Windowing parameters (use the same as in training)
WINDOW_SIZE = 6
STEP_SIZE = 2

# Create windows of data with corresponding labels
def create_windows(features, labels, window_size, step_size):
    X, y = [], []
    for i in range(0, len(features) - window_size, step_size):
        X.append(features[i:i + window_size])
        y.append(labels[i + window_size - 1])  # Labels for the end of the window
    return np.array(X), np.array(y)

# Apply windowing on the dataset
X, y = create_windows(features, activity_labels, WINDOW_SIZE, STEP_SIZE)

# Manually assign the mean and standard deviation values
mean_values = np.array([0.19612816,0.72473526,0.10040915,76.16988267,42.41790313,16.0228941])
std_dev_values = np.array([0.29894563,0.39455784,0.41798397,48.26036759,73.88258501,36.13358661])


# Standardize the sensor data
X_flat = X.reshape(-1, X.shape[-1])  # Flatten for scaling
X_flat = (X_flat - mean_values) / std_dev_values  # Apply standardization using the provided mean and std_dev
X_scaled = X_flat.reshape(-1, WINDOW_SIZE, X.shape[-1])  # Reshape back to the original windowed shape

# Load the trained model
model = tf.keras.models.load_model('models/activity_fall_combined_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_scaled, tf.keras.utils.to_categorical(y, num_classes=10))
print(f"Evaluation Loss: {loss}, Evaluation Accuracy: {accuracy}")

# Predict on the test data
predictions = model.predict(X_scaled)

# Convert probabilities to class predictions
predicted_classes = np.argmax(predictions, axis=1)

# Get the unique classes from the test labels
unique_classes = np.unique(y)

# Update the target names to match the unique classes in the test set
target_names = [f'Activity {i}' for i in unique_classes]

# Print classification report
print("Classification Report:")
print(classification_report(y, predicted_classes, target_names=target_names, labels=unique_classes))

# Print overall accuracy of the test sample
overall_accuracy = accuracy * 100  # Convert accuracy to percentage
print(f"Overall Accuracy of the Test Sample: {overall_accuracy:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(y, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=unique_classes, yticklabels=unique_classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
