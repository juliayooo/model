import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import serial


# # Load trained model
model = tf.keras.models.load_model("trained_model3.h5")

model.summary()


print("Available GPUs:", len(tf.config.list_physical_devices('GPU')))

# ARDUINO PORT

ser = serial.Serial('/dev/tty.usbmodem1101', 9600)

def trigger_light(num):
    ser.write(str(num).encode())  # Send signal to Arduino


# Set image directories
ref_dir = "../images"
save_dir = "../images/yes"

# Create directories if not existing
os.makedirs(save_dir, exist_ok=True)

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_alt.xml')

# Open video capture (change source if needed)
# 0 for iphone input w OBS 2 for webcam input OBS
cap = cv2.VideoCapture(2)
faces_seen = set()
fid = (0, 0, 0, 0)
counter = 0
MARGIN = 30

print("Press SPACE to capture a face, or ESC to exit.")

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0,
                                                         0), 2)

    # Display the live feed
    cv2.imshow('Face Capture', img)

    # Wait for user key press
    key = cv2.waitKey(1) & 0xFF

    # If ESC is pressed, exit
    if key == 27:
        break

    # If SPACE is pressed, capture face
    if key == 32 and len(
            faces) > 0:  # Ensure a face is detected before saving
        for (x, y, w, h) in faces:
            fid = (x // 100, y // 100, w // 100, h // 100)

            # Save only if face has not been seen before
            if fid not in faces_seen:
                faces_seen.add(fid)
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

                # CONVERT
                faceimg_array = np.expand_dims(faceimg,
                                               axis=0) / 255.0

                # Make prediction
                predictions = model.predict(faceimg_array)
                predicted_label = np.argmax(predictions)

                # Get class name
                predicted_class = class_names[predicted_label]

                # Save image in respective class folder
                save_path = os.path.join(save_dir, predicted_class,
                                         f"face_{counter}.png")
                cv2.imwrite(save_path, faceimg)

                print(
                    f"Face saved: {save_path} | Predicted as: {predicted_class}")

                # for images, labels in live_dataset.take(1):
                #     predictions = model.predict(images)
                #     predicted_labels = np.argmax(predictions, axis=1)
                #
                #     # Remove the saved image after prediction
                #     os.remove(face_path)
                #     print(class_names[predicted_labels[0]])
                if (predicted_class == "yes"):
                    trigger_light(1)

                elif (predicted_class == "no"):
                    trigger_light(2)

                break  # Stop after one prediction


# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
