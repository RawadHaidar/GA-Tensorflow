import tensorflow as tf

# Load the saved Keras model
model = tf.keras.models.load_model('models/activity_fall_combined_model.h5')

# Define the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization options (optional for better performance)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables optimization for size and latency
converter.target_spec.supported_types = [tf.float16]  # Use float16 for reduced precision

# Convert the model to TFLite format
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_path = 'models/activity_fall_combined_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model has been saved to {tflite_model_path}")
