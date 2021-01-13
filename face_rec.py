import face_recognition
import dlib
import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect face in
img = cv2.imread('RDJ.jpg')


# Must convert to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

for (x, y, w, h) in face_coordinates:
    # Draw rectangles around faces
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)


# print(face_coordinates)

cv2.imshow('Me', img)
cv2.waitKey()


print("code completed")