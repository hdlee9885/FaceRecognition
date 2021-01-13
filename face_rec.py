import face_recognition
import dlib
import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect face in
# img = cv2.imread('RDJ.jpg')
webcam = cv2.VideoCapture(0)

while True:
    
    # read the current frame
    frame_read, frame = webcam.read()

    # Must convert to greyscale
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

    for (x, y, w, h) in face_coordinates:
        # Draw rectangles around faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)


    cv2.imshow('Me', frame)
    key = cv2.waitKey(1)   

    if key == 81 or key == 113:
        break

webcam.release()

print("code completed")