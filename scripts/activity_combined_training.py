import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load and preprocess the dataset
data = pd.read_csv("data/fp_fd_wd_data1.csv")

# Select features and the combined activity label
features = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
activity_labels = data['activity'].values  # Assumes 'activity' is now in range [0-9]

# Windowing parameters
WINDOW_SIZE = 6
STEP_SIZE = 2

# Create windows of data with corresponding labels
def create_windows(features, labels, window_size, step_size):
    X, y = [], []
    for i in range(0, len(features) - window_size, step_size):
        X.append(features[i:i + window_size])
        y.append(labels[i + window_size - 1])  # Labels for the end of the window
    return np.array(X), np.array(y)

X, y = create_windows(features, activity_labels, WINDOW_SIZE, STEP_SIZE)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the sensor data
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, X.shape[-1])  # Flatten to 2D for scaling
X_val_flat = X_val.reshape(-1, X.shape[-1])

X_train_flat = scaler.fit_transform(X_train_flat)
X_val_flat = scaler.transform(X_val_flat)

# Reshape back to the original shape
X_train_scaled = X_train_flat.reshape(-1, WINDOW_SIZE, X.shape[-1])
X_val_scaled = X_val_flat.reshape(-1, WINDOW_SIZE, X.shape[-1])

# Print mean and standard deviation for future use
mean = scaler.mean_
std_dev = np.sqrt(scaler.var_)
print("Mean of each feature:", ", ".join(map(str, mean)))
print("Standard Deviation of each feature:", ", ".join(map(str, std_dev)))


# Convert labels to categorical (one-hot encoded)
y_train_cat = to_categorical(y_train, num_classes=10)
y_val_cat = to_categorical(y_val, num_classes=10)

# Define the model architecture
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, X_train_scaled.shape[-1])),
    Dropout(0.2),
    Conv1D(32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')  # 10 output nodes for 10 activity classes
])

# Compile the model for multi-class classification
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train_cat,
    validation_data=(X_val_scaled, y_val_cat),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val_cat)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Save the model
model.save('models/activity_fall_combined_model.h5')

# Example prediction
# predictions = model.predict(new_data)  # Use on new data if needed
