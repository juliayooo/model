import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import cv2
import os

saved_model = "trained_model3.h5"
model = load_model(saved_model)
model.summary()




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
        if id not in faces_seen:  # Check if this face has been
            # saved before
            faces_seen.add(id)  # Mark this face as saved
            faceimg = img[y:y + h, x:x + w]
            fid = (0,0,0,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # save face images to image folder
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            counter+=1
            cv2.imwrite("../images/face_" + str(fid) + ".png",
                        faceimg)
            print(fid)


    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Press Esc to stop the video
        break

cap.release()
cv2.destroyAllWindows()