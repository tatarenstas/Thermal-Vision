import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_q = frame
    frame = np.zeros((120, 160), dtype=np.uint8)
    frame = cv2.normalize(frame_q, frame, 0, 255, cv2.NORM_MINMAX)
    frame = np.uint8(frame)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = haar_cascade_face.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) % 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
