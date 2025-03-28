import tensorflow as tf
from tensorflow.keras.preprocessing import \
    image_dataset_from_directory
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import serial

# Load trained model
saved_model = ("trained_model3.h5")
model = load_model(saved_model)
model.summary()

print("Available GPUs:", len(tf.config.list_physical_devices('GPU')))

# Set image directories
ref_dir = "../images"
save_dir = "../images/yes"
MARGIN = 30
# Create directories if not existing
os.makedirs(save_dir, exist_ok=True)

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_alt.xml')

# Open video capture (change source if needed)
# 0 for iphone input w OBS 2 for webcam input OBS
cap = cv2.VideoCapture(0)
faces_seen = set()
fid = (0, 0, 0, 0)
counter = 0
ser = serial.Serial('/dev/tty.usbmodem101', 9600)

def trigger_light(num):
    ser.write(str(num).encode())  # Send signal to Arduino


print("Press SPACE to capture a face, or ESC to exit.")

while True:
    ret, img = cap.read()
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

    # If SPACE is pressed, capture face
    if key == 32 and len(faces) > 0:
        for (x, y, w, h) in faces:
            counter += 1
            # Expand bounding box slightly
            x1 = max(0, x - MARGIN)
            y1 = max(0, y - MARGIN)
            x2 = min(img.shape[1], x + w + MARGIN)
            y2 = min(img.shape[0], y + h + MARGIN)

            faceimg = img[y1:y2, x1:x2]
            faceimg = cv2.resize(faceimg, (224, 224))
            face_path = f"{save_dir}/face_{counter}.png"
            cv2.imwrite(face_path, faceimg)
            print(f"Face saved: {face_path}")

            ####### IMAGE PREDICTION ########
            live_dataset = image_dataset_from_directory(
                ref_dir,
                image_size=(224, 224),
                batch_size=1,
                labels='inferred'
            )

            # Identify class names
            class_names = live_dataset.class_names

            for images, labels in live_dataset.take(1):
                predictions = model.predict(images)
                predicted_label = np.argmax(predictions, axis=1)
                predicted_class = class_names[predicted_label[0]]

                # Remove the saved image after prediction
                os.remove(face_path)

                # # Show the prediction
                # plt.figure(figsize=(5, 5))
                # ax = plt.subplot(1, 1, 1)
                # plt.imshow(images[0].numpy().astype("uint8"))
                # plt.title(
                print(f"Prediction: {class_names[predicted_label[0]]}")
                # plt.show()

                if (predicted_class == "yes"):
                    trigger_light(1)

                elif (predicted_class == "no"):
                    trigger_light(2)
                break  # Stop after one prediction




# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
