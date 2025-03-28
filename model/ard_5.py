import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import serial

model = tf.keras.models.load_model("trained_model3.h5")

model.summary()
print("Available GPUs:", len(tf.config.list_physical_devices('GPU')))

# ARDUINO PORT
ser = serial.Serial('/dev/tty.usbmodem101', 9600)
# SEND SERIAL SIGNALS TO ARDUINO
def trigger_light(num):
    ser.write(str(num).encode())  # Send signal to Arduino

# IMAGE DIRECTORIES
base_dir = "./images"
yes_dir = os.path.join(base_dir, "yes")
no_dir = os.path.join(base_dir, "no")

os.makedirs(yes_dir, exist_ok=True)
os.makedirs(no_dir, exist_ok=True)

# CASCADE LOAD
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# GET LIVE VIDEO INDEX
cap = cv2.VideoCapture(2)
counter = 0
MARGIN = 30
class_names = ["no", "yes"]

print("Press SPACE to capture a face, or ESC to exit.")

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the live feed
    cv2.imshow('Face Capture', img)

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

            # CONVERT
            faceimg_array = np.expand_dims(faceimg, axis=0) / 255.0

            # Make prediction
            predictions = model.predict(faceimg_array)
            predicted_label = np.argmax(predictions)

            # Get class name
            predicted_class = class_names[predicted_label]

            # Save image in respective class folder
            save_path = os.path.join(base_dir, predicted_class, f"face_{counter}.png")
            cv2.imwrite(save_path, faceimg)

            print(f"Face saved: {save_path} | Predicted as: {predicted_class}")

            # Send signal to Arduino
            if predicted_class == "yes":
                trigger_light(1)
            elif predicted_class == "no":
                trigger_light(2)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
