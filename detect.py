import cv2
import numpy as np
import os
from PIL import Image
import sqlite3

# Load face detector and recognizer
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

def getprofile(Id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM Students WHERE Id=?", (Id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        profile = getprofile(Id)
        
        if profile is not None:
            cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)
            cv2.putText(img, "Age: " + str(profile[2]), (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)
    cv2.imshow("Face", img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
