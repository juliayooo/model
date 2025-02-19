import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import cv2
import os

saved_model = ("trained_model3.h5")
model = load_model(saved_model)
model.summary()


print("available GPUs:", len(tf.config.list_physical_devices('GPU')))

# get images from saved directory
ref_dir = "../images"

#make dir if not existing
save_dir = "../images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# # Check if your system can detect camera and what is the source number
# cams_test = 10
# for i in range(0, cams_test):
#     cap = cv2.VideoCapture(i)
#     test, frame = cap.read()
#     print("i : "+str(i)+" /// result: "+str(test))
#

# this works only with OBS on
cap = cv2.VideoCapture(2)
faces_seen = set()
fid = (0,0,0,0)
counter = 0

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Look for faces
    for (x, y, w, h) in faces:

        fid = (x // 100, y // 100, w // 100, h // 100)
        print(fid)
        if fid not in faces_seen:  # Check if this face has been
            # saved before
            faces_seen.add(fid)  # Mark this face as saved
            faceimg = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # save face images to image folder
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            counter+=1
            faceimg = cv2.resize(faceimg, (224, 224))
            cv2.imwrite("../images/yes/face_" + str(counter) + ".png",
                        faceimg)
            print(fid)
            fid = (0, 0, 0, 0)


######
            # create the img dataset
            live_dataset = image_dataset_from_directory(ref_dir,
                                                        image_size=(
                                                        224, 224),
                                                        batch_size=1,
                                                        labels='inferred')

            # identify class names
            class_names = live_dataset.class_names

            # use pyplot to show predictions and real answers
            print(counter)

            for images, labels in live_dataset.take(
                    1):  # Take one batch
                predictions = model.predict(
                    images)  # Get model predictions
                predicted_labels = np.argmax(predictions, axis=1)
                os.remove("../images/yes/face_" + str(counter) + ".png")
                plt.figure(
                    figsize=(5,5))

                for i in range(1):
                    ax = plt.subplot(1, 1, i + 1)
                    plt.imshow(images[i].numpy().astype(
                        "uint8"))  # Convert tensor to image
                    plt.title(
                        f"Prediction:"
                        f" {class_names[predicted_labels[i]]}")
                plt.show()  # Show the figure





                break


#######
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Press Esc to stop the video
        break

cap.release()
cv2.destroyAllWindows()



