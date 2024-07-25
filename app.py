import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
from tensorflow.keras.models import Model
from flask import Flask, jsonify, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
model = load_model('E:\\PONDICHERRY\\SEM 4\\Project\\new\\model.h5')
label = [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', ' A', ' B', ' C', ' D', ' E', ' F', ' G', ' H', ' Hello', ' I',
         ' J', ' K', ' L', ' M', ' Men', ' N', ' No', ' O', ' P', ' Please', ' Q', ' R', ' S', ' T', ' Thanks', ' U', ' V',
         ' W', ' Women', ' X', ' Y', ' Yes', ' Z']

font = cv2.FONT_HERSHEY_SIMPLEX
n = 0
si_la_ = ''
si_la = [' ' for i in range(5)]
si_k = ''
svStr = ''
tmp = ''


@socketio.on('frame')
def process_frame(frame):
    global n, si_la_, si_la, si_k, svStr, tmp

    frame = np.frombuffer(frame, dtype=np.uint8).reshape((480, 640, 3))

    pred_1 = []
    if si_la_ != '.' and si_la_ != tmp:
        svStr = svStr + si_la_
    tmp = si_la_
    if n > 10:
        si_k = ''
        for i in range(3):
            si_la[i] = si_la[i + 1]
            si_k = si_k + si_la[i]
        si_la[3] = si_la_
        si_k = si_k + si_la[3]
        si_la_ = ''
        tmp = ''
        n = 0

    n += 1
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
    frame = cv2.putText(frame, si_k, (200, 450), font, 1, (128, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    cv2.imwrite('black.jpg', img)
    image_1 = cv2.imread("black.jpg")
    img__ = image_1.copy()
    image_1 = cv2.resize(image_1, (128, 128))
    image_1 = img_to_array(image_1)
    pred_1.append(image_1)
    pred_1 = np.array(pred_1, dtype="float32") / 255.0
    result_2 = model.predict(pred_1)
    si_la_ = label[np.argmax(result_2)]
    
    socketio.emit('prediction', si_la_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        socketio.stop()


@app.route('/', methods=['POST'])
def process_video():
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    if 'video' not in request.files:
        return jsonify({'error': 'No video found'})

    video_file = request.files['video']
    frame_rate = int(request.form.get('frame_rate', 5))

    capture = cv2.VideoCapture(video_file)
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        socketio.emit('frame', buffer.tobytes())
        time.sleep(1 / frame_rate)

    capture.release()
    return jsonify({'message': 'Processing completed'})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
