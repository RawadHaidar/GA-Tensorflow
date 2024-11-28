import numpy as np
import tensorflow as tf

# Manually input mean and std_dev for each feature
mean_values = np.array([0.1970764,0.72730746,0.10252927,75.70195317,43.70116143,16.10059766])
std_dev_values = np.array([0.29751629,0.3869502,0.40807957,48.23379922,72.59531508,35.99314821])
# Define window parameters to match the training configuration
WINDOW_SIZE = 6  # Change if different
NUM_FEATURES = len(mean_values)

# Load the model
model = tf.keras.models.load_model('models/activity_fall_combined_model.h5')

# Function to standardize input sample manually
def standardize_sample(sample, mean, std_dev):
    standardized_sample = (sample - mean) / std_dev
    return standardized_sample

# Function to prepare a single windowed sample and make a prediction
def predict_single_sample(sample_data, model):
    # Ensure sample_data is in shape (WINDOW_SIZE, NUM_FEATURES)
    if sample_data.shape != (WINDOW_SIZE, NUM_FEATURES):
        raise ValueError(f"Sample data must be of shape {(WINDOW_SIZE, NUM_FEATURES)}")
    
    # Standardize the sample
    standardized_sample = standardize_sample(sample_data, mean_values, std_dev_values)
    
    # Reshape for model input (1 sample, WINDOW_SIZE, NUM_FEATURES)
    standardized_sample = standardized_sample.reshape(1, WINDOW_SIZE, NUM_FEATURES)
    
    # Make prediction
    prediction = model.predict(standardized_sample)
    
    # Convert prediction to class label
    predicted_class = np.argmax(prediction, axis=1)[0]  # Gets the class index with the highest probability
    predicted_probabilities = prediction[0]
    
    return predicted_class, predicted_probabilities

# Example: Test a single sample (replace with actual test sample values)
# Test sample data for 6 timesteps (WINDOW_SIZE) and 6 features
test_sample_data = np.array([
    [0.27,0.91,0.2,77.47,52.59,16.2],
    [0.03,0.84,0.14,80.54,12.53,2.12],
    [-0.34,0.98,-0.28,105.95,-129.29,-19.25],
    [-0.37,0.93,-0.33,109.44,-131.78,-21.55],
    [0.43,0.82,-0.05,93.81,97.25,27.65],
    [0.2,0.65,-0.56,130.94,160.85,16.76]
])  # Replace with actual test data


# Predict on the test sample
predicted_class, predicted_probabilities = predict_single_sample(test_sample_data, model)

# Output the results
print(f"Predicted Class: {predicted_class}")
print("Predicted Probabilities:", predicted_probabilities)
