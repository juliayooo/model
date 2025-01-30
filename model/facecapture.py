
import numpy as np
import cv2
import os

#make dir if not existing
save_dir = "../images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#cv files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')


# this works only with OBS on
cap = cv2.VideoCapture(2)
faces_seen = set()
#face id and eye id
fid = (0,0,0,0)
eid = (0,0,0,0)
#face counter
counter = 0

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Look for faces
    for (x, y, w, h) in faces:

        #record unique face id
        fid = (x // 100, y // 100, w // 100, h // 100)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        reye = reye_cascade.detectMultiScale(
            roi_gray)  # Use the gray face image to detect eyes

        for (ex, ey, ew, eh) in reye:
            rid = (ex // 100, ey // 100, ew // 100, eh // 100)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
                          (0, 255, 0),
                          2)  # Draw green bounding boxes around the eyes

        print(rid)
        if rid not in faces_seen:  # Check if this face has been
            # saved before
            faces_seen.add(rid)  # Mark this face as saved
            faceimg = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # save face images to image folder
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            counter+=1
            cv2.imwrite("../images/face_" + str(rid) + ".png",
                        faceimg)
            rid = (0,0,0,0)
            print(rid)


    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Press Esc to stop the video
        break

cap.release()
cv2.destroyAllWindows()