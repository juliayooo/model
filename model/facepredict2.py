import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load the trained model
saved_model = "trained_model4.h5"
model = tf.keras.models.load_model(saved_model)
model.summary()

print("Available GPUs:", len(tf.config.list_physical_devices('GPU')))

# Set up image save directory
save_dir = "../images/yes"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Open webcam (use 0 or 1 if needed)
cap = cv2.VideoCapture(2)
faces_seen = set()
counter = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        fid = (x // 100, y // 100, w // 100, h // 100)

        if fid not in faces_seen:  # If the face hasn't been processed
            faces_seen.add(fid)
            counter += 1  # Update counter
            faceimg = img[y:y + h, x:x + w]
            faceimg = cv2.resize(faceimg, (224, 224))  # Resize to match model input

            # Save the face image
            face_path = os.path.join(save_dir, f"face_{counter}.png")
            cv2.imwrite(face_path, faceimg)
            print(f"Saved {face_path}")

            # Predict only the most recent face image
            img = image.load_img(face_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0  # Normalize pixels
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Run model prediction
            predictions = model.predict(img_array)
            predicted_label = np.argmax(predictions)

            # Display prediction
            plt.figure(figsize=(5, 5))
            plt.imshow(img_array[0])  # Show image
            plt.title(f"Prediction: {predicted_label}")  # Show class
            plt.axis("off")
            plt.show()

    cv2.imshow('img', img)
    if cv2.waitKey(30) & 0xFF == 27:  # Press Esc to stop
        break

cap.release()
cv2.destroyAllWindows()
