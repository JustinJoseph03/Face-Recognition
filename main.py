import cv2
from random import randrange

#Pre-trained face data
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Images to detect faces in
#img = cv2.imread("F.jpg")

#Camera read
webcam = cv2.VideoCapture(0)

#Read over frames of video
while True:

    successful_frame_read, frame = webcam.read()

    #Must do grayscale conversion
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coords = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coords:
        #Random colors
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

        #One color
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Window/ Display Image
    cv2.imshow("Face Detector By Justin Joseph", frame)
    #Wait for key press
    key = cv2.waitKey(1)

    #Quit on Q or q press
    if key == 81 or key == 113:
        break

webcam.release()

print("Code Completed")