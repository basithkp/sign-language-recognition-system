import cv2
import os
import uuid
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
app = Flask(__name__)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
from keras.models import load_model
model = load_model('E:\\PONDICHERRY\\SEM 4\\Project\\new\\model.h5')
labels=[' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', ' A', ' B', ' C', ' D', ' E', ' F', ' G', ' H', ' Hello', ' I', ' J', ' K', ' L', ' M', ' Men', ' N', ' No',' ',' O', ' P', ' Please', ' Q', ' R', ' S', ' T', ' Thanks', ' U', ' V', ' W', ' Women', ' X', ' Y', ' Yes', ' Z']
# reading the input using the camera



mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 
IMAGES_PATH = 'collected_images(1)'
image_folder = 'collected_images(1)\C(1)'  # Folder containing input images

label = 'C'
def get_holistic(image_file):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        os.makedirs(os.path.join(IMAGES_PATH, label), exist_ok=True)
        print(label)
        n = True
        taken = 0
        image_files = os.listdir(image_folder)
        image_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort files numerically
        
        # if not image_file.endswith('.png'):
        #     continue
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_file)
        # if frame is None:
        #     continue

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
            # cv2.imshow("Frame", frame)
            
        return img
        
# image_file = "E:\\PONDICHERRY\\SEM 4\\Project\\new\\collected_images(1)\\A(1)\\0.png"

# result = get_holistic(image_file)

# imagename ="E:\\PONDICHERRY\\SEM 4\\Project\\new\\result\\1.png"
# pred_1=[]
# cv2.imwrite(imagename, result)
# image_1=cv2.imread(imagename)
# img__=image_1.copy()

# image_1 = cv2.resize(image_1, (128, 128))
# image_1 = img_to_array(image_1)
# pred_1.append(image_1)
# pred_1 = np.array(pred_1, dtype="float32") / 255.0
#     #image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #cv2.imshow("prediction", image_1)
# result_2=model.predict(pred_1)
# si_la = labels[np.argmax(result_2)]

# print(si_la)


@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['image']
    img_path = "static/" + image_file.filename	
    image_file.save(img_path)
    print(img_path,"----------------------")
    result = get_holistic(img_path)

    imagename ="E:\\PONDICHERRY\\SEM 4\\Project\\new\\result\\1.png"
    pred_1=[]
    cv2.imwrite(imagename, result)
    image_1=cv2.imread(imagename)
    img__=image_1.copy()

    image_1 = cv2.resize(image_1, (128, 128))
    image_1 = img_to_array(image_1)
    pred_1.append(image_1)
    pred_1 = np.array(pred_1, dtype="float32") / 255.0
        #image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("prediction", image_1)
    result_2=model.predict(pred_1)
    si_la = labels[np.argmax(result_2)]
    print(si_la)
    return si_la