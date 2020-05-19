import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

blush = cv2.imread("blush.png", cv2.IMREAD_UNCHANGED)
ears = cv2.imread("ears.png", cv2.IMREAD_UNCHANGED)

_, frame = cap.read()
rows, cols, _ = frame.shape
face_mask = np.zeros((rows, cols), np.uint8)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = detector(gray_frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        # Face coordinates
        left = (landmarks.part(3).x, landmarks.part(3).y)
        center = (landmarks.part(51).x, landmarks.part(51).y)
        right = (landmarks.part(13).x, landmarks.part(13).y)

        face_width = int(hypot(left[0] - right[0], left[1] - right[1]))
        face_height = int(face_width * (blush.shape[0]/blush.shape[1]))

        # New face position
        top_left = (int(center[0] - face_width / 2),
                    int(center[1] - face_height * 1.5))
        bottom_right = (int(center[0] + face_width / 2),
                        int(center[1]))

        ears_left = (landmarks.part(0).x, landmarks.part(0).y)
        ears_center = (landmarks.part(27).x, landmarks.part(27).y)
        ears_right = (landmarks.part(16).x, landmarks.part(16).y)


        ears_face_width = int(hypot(ears_left[0] - ears_right[0], ears_left[1] - ears_right[1]))
        ears_face_height = int(ears_face_width * (ears.shape[0]/ears.shape[1]))


        ears_top_left = (int(ears_center[0] - ears_face_width / 2),
                    int(ears_center[1] - ears_face_height * 2.5))
        ears_bottom_right = (int(ears_center[0] + ears_face_width / 2),
                        int(ears_center[1]))

        blush_effect = cv2.resize(blush, (face_width, face_height))
        ears_effect = cv2.resize(ears, (ears_face_width, ears_face_height))

        alpha = blush_effect[:, :, 3] / 255.0
        beta = 1.0 - alpha

        for c in range(0, 3):
            frame[top_left[1]:top_left[1] + face_height, top_left[0]:top_left[0] + face_width, c] = (
                    alpha * blush_effect[:, :, c] + beta * frame[top_left[1]:top_left[1] + face_height,
                                                                top_left[0]:top_left[0] + face_width, c])

        ears_alpha = ears_effect[:, :, 3] / 255.0
        ears_beta = 1.0 - ears_alpha

        for c in range(0, 3):
            frame[ears_top_left[1]:ears_top_left[1] + ears_face_height, ears_top_left[0]:ears_top_left[0] + ears_face_width, c] = (
                    ears_alpha * ears_effect[:, :, c] + ears_beta * frame[ears_top_left[1]:ears_top_left[1] + ears_face_height,
                                                                ears_top_left[0]:ears_top_left[0] + ears_face_width, c])

    out.write(frame)
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        out.release()
        break
