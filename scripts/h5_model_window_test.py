import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from io import StringIO

# Load your trained model (update the path as necessary)
model = load_model('models/fall_detection_model.h5')

# Input data as a CSV string
data_csv = """acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
0.31,0.86,-0.09,95.71,105.38,19.98
0.18,0.9,-0.2,102.26,137.39,11.31
0.01,1.09,-0.25,102.88,178.21,0.41
0,0.85,0.05,86.85,0,0
0.29,1.3,0.19,81.82,57.03,12.49
-0.91,1.13,0.08,86.05,-85.11,-38.9
"""

# 18,0.31,0.86,-0.09,95.71,105.38,19.98,Standing still,false
# 19,0.18,0.9,-0.2,102.26,137.39,11.31,Standing still,false
# 20,0.01,1.09,-0.25,102.88,178.21,0.41,Standing still,false
# 21,0,0.85,0.05,86.85,0,0,Standing still,false
# 22,0.29,1.3,0.19,81.82,57.03,12.49,Standing still,false
# 23,-0.91,1.13,0.08,86.05,-85.11,-38.9,Standing still,false

# 10,0.38,0.97,0.27,74.67,54.69,21.16,Standing still,0
# 11,0.23,1.01,0.31,72.77,36.87,13.09,Standing still,0
# 12,0.13,0.85,0.4,64.93,17.42,8.35,Standing still,0
# 13,0.35,0.7,0.34,63.69,45.64,26.82,Standing still,0
# 14,-0.08,0.98,-0.25,104.36,-162.65,-4.57,Standing still,1
# 15,0.02,0.65,-0.48,126.31,177.18,2.07,Standing still,1

# 17,0.27,0.8,0.3,69.75,42.65,18.77,Standing still,0
# 18,0.16,0.67,0.38,60.33,23.2,13.72,Standing still,0
# 19,0.4,1.2,0.13,83.7,71.57,18.32,Standing still,0
# 20,0.16,0.23,0.05,76.43,71.57,35.91,Standing still,1
# 21,0.06,0.68,-0.6,131.51,174.07,5.25,Standing still,1
# 22,-0.03,0.55,-0.78,144.63,-177.71,-3.22,Standing still,1

# 8,0.28,0.96,0.51,62.15,28.98,16.31,Standing still,0
# 9,0.15,0.82,0.41,63.22,19.72,10.26,Standing still,0
# 10,0.38,0.97,0.27,74.67,54.69,21.16,Standing still,0
# 11,0.23,1.01,0.31,72.77,36.87,13.09,Standing still,0
# 12,0.13,0.85,0.4,64.93,17.42,8.35,Standing still,0
# 13,0.35,0.7,0.34,63.69,45.64,26.82,Standing still,1

# Read the input data into a DataFrame
data = pd.read_csv(StringIO(data_csv))

# Prepare features for the model
features = data.values  # Convert DataFrame to numpy array

# Replace these with the actual mean and std dev from your training data
mean = np.array([0.21410772,0.65118435,0.18824759,67.12458199,49.09383976,25.32327974])  # Replace with actual mean
std_dev = np.array([0.28364018,0.45101985,0.44122823,51.15351246,60.78453387,47.19293414])  # Replace with actual std dev

# Standardize the features
features_standardized = (features - mean) / std_dev

# Reshape features for the model input (adding batch size dimension)
features_reshaped = features_standardized.reshape((1, 6, 6))  # (samples, timesteps, features)

# Get fall detection probability from the model
fall_detection_prob = model.predict(features_reshaped)

# Print the fall prediction probability
print(f"Fall Detection Probability: {fall_detection_prob[0][0]:.4f}")  # Assuming the model outputs a single probability 
