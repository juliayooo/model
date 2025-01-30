
import numpy as np
import cv2
import os
import dlib

#make dir if not existing
save_dir = "../images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#cv files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
model_path = "shape_predictor_68_face_landmarks.dat"  # Update
sp = dlib.shape_predictor(model_path)

# path if needed


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
    dlibimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detector = dlib.get_frontal_face_detector()
    detections = detector(dlibimg, 1)
    # sp = dlib.shape_predictor(
    #     "shape_predictor_68_face_landmarks.dat.bz2")
    dfaces = dlib.full_object_detections()
    for det in detections:
        dfaces.append(sp(dlibimg, det))
        # Bounding box and eyes
        bb = [i.rect for i in dfaces]
        bb = [((i.left(), i.top()),
               (i.right(), i.bottom())) for i in bb]
        imgd = cv2.cvtColor(img,
                            cv2.COLOR_RGB2BGR)  # Convert back to OpenCV
        for i in bb:
            cv2.rectangle(imgd, i[0], i[1], (255, 0, 0),
                          5)  # Bounding box

        # # Look for faces
    # for (x, y, w, h) in faces:
    #
    #     #record unique face id
    #     fid = (x // 100, y // 100, w // 100, h // 100)
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = img[y:y + h, x:x + w]
    #     reye = reye_cascade.detectMultiScale(
    #         roi_gray)  # Use the gray face image to detect eyes
    #
    #     for (ex, ey, ew, eh) in reye:
    #         rid = (ex // 100, ey // 100, ew // 100, eh // 100)
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
    #                       (0, 255, 0),
    #                       2)  # Draw green bounding boxes around the eyes
    #
    #     print(rid)
    #     if rid not in faces_seen:  # Check if this face has been
    #         # saved before
    #         faces_seen.add(rid)  # Mark this face as saved
    #         faceimg = img[y:y + h, x:x + w]
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         # save face images to image folder
    #         counter+=1
    #         cv2.imwrite("../images/face_" + str(rid) + ".png",
    #                     faceimg)
    #         rid = (0,0,0,0)
    #         print(rid)


    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Press Esc to stop the video
        break

cap.release()
cv2.destroyAllWindows()