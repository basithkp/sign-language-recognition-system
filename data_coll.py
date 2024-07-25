import cv2
import mediapipe as mp
import numpy as np
import os
import uuid

IMAGES_PATH = 'collected_images(3)'
labels = ['Z']  # Add your desired labels here

cap = cv2.VideoCapture(0)



mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for label in labels:
        os.makedirs(os.path.join(IMAGES_PATH, label), exist_ok=True)
        print(label)
        n = True
        taken = 0
        while n:
            ret, frame = cap.read()
            if ret == True:
                img = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = holistic.process(frame)

                mp_drawing.draw_landmarks(
                    img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

                mp_drawing.draw_landmarks(
                    img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))

                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Frame", frame)
                cv2.imshow("Black", img)

                if taken % 1 == 0:  # Capture an image every 10 frames
                    imagename = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
                    cv2.imwrite(imagename, img)
                    taken += 1
                    print(taken)

                if cv2.waitKey(1) & 0xFF == ord('m'):
                    print('Quit')
                    n = False
                    break
            else:
                break

cap.release()
cv2.destroyAllWindows()
