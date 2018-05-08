# face detection using Haar Feature-based Cascade Classifiers
# Open CV is the library used for ML
# Download latest OpenCV release from sourceforge site and double-click to extract it.
# http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.6/OpenCV-2.4.6.0.exe/download
# Copy cv2.pyd to C:/Python27/lib/site-packeges.


import numpy as np
import cv2 as cv

# OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc.
# Those XML files are stored in the opencv/data/haarcascades/ folder
# First we need to load the required XML classifiers

face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# load our input image (or video) in grayscale mode
img = cv.imread('images/sampleface6.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), )
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

