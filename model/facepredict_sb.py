import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Load trained model
saved_model = "trained_model3.h5"
model = load_model(saved_model)
model.summary()

dataset = image_dataset_from_directory("/Users/juliayoo/Desktop/DATA/resized_dataset2", batch_size=1)
class_names = dataset.class_names  # This stores the correct class mapping

print("Model trained class names:", class_names)

print("Available GPUs:", len(tf.config.list_physical_devices('GPU')))

# Set save directory for captured faces
save_dir = "../images/yes"
os.makedirs(save_dir, exist_ok=True)

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Open video capture (change source if needed)
cap = cv2.VideoCapture(2)  # 0 for default camera, 1 for external
# webcam
faces_seen = set()
counter = 0

# Define class names manually (adjust based on your model's training labels)
# class_names = ["no", "yes"]  # Update based on your dataset classes

print("Press SPACE to capture a face, or ESC to exit.")

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the live feed
    cv2.imshow('Face Capture', img)

    # Wait for user key press
    key = cv2.waitKey(1) & 0xFF

    # If ESC is pressed, exit
    if key == 27:
        break

    # If SPACE is pressed, capture and predict face
    if key == 32 and len(faces) > 0:
        for (x, y, w, h) in faces:
            face_id = (x // 100, y // 100, w // 100, h // 100)

            # Save only if face has not been seen before
            if face_id not in faces_seen:
                faces_seen.add(face_id)
                counter += 1

                # Extract and resize the face
                face_img = img[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (224, 224))

                # Save captured face
                face_path = f"{save_dir}/face_{counter}.png"
                cv2.imwrite(face_path, face_img)
                print(f"Face saved: {face_path}")

                ####### IMAGE PREDICTION ########
                # Preprocess image
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                face_img = face_img / 255.0  # Normalize pixel values
                face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

                # Make prediction
                predictions = model.predict(face_img)
                predicted_label = np.argmax(predictions, axis=1)[0]

                # Remove saved image after prediction
                # os.remove(face_path)

                # Display prediction
                plt.figure(figsize=(5, 5))
                plt.imshow(face_img[0])  # Remove batch dimension for visualization
                plt.title(f"Prediction: {class_names[predicted_label]}")
                plt.axis("off")
                plt.show()
                break  # Stop after one prediction

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
