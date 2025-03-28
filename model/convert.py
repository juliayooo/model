import tensorflow as tf

# Load model in TF 2.18
model = tf.keras.models.load_model("trained_model3.h5")

# Save the model in a format that older TensorFlow versions can load
model.save("converted_model_tf211.h5", save_format="h5")