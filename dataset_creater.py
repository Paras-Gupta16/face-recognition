import cv2
import numpy as np
import sqlite3

# Load the cascade
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the camera
cam = cv2.VideoCapture(0)

def insertorupdate(Id, Name, Age):
    conn = sqlite3.connect("sqlite.db")
    cmd = "SELECT * FROM students WHERE Id=?"
    cursor = conn.execute(cmd, (Id,))
    isRecordExist = 0

    for row in cursor:
        isRecordExist = 1

    if isRecordExist == 1:
        conn.execute("UPDATE Students SET Name=?, Age=? WHERE Id=?", (Name, Age, Id))
    else:
        conn.execute("INSERT INTO Students(Id, Name, Age) VALUES (?, ?, ?)", (Id, Name, Age))

    conn.commit()
    conn.close()

Id = input("Enter user ID: ")
Name = input("Enter user name: ")
Age = input("Enter your age: ")

insertorupdate(Id, Name, Age)

sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite("dataset/user." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)

    cv2.imshow("Face", img)
    cv2.waitKey(1)

    if sampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()
