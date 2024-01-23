# inference_classifier.py

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import base64
import pickle
import numpy as np
import mediapipe as mp

app = Flask(__name__)
app.secret_key = "mysecret"
# CORS 설정
CORS(app, resources={r"/socket.io/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
socket_io = SocketIO(app, cors_allowed_origins="*", logger=True)


# 학습된 모델 파일 로드
right_hand_model_dict = pickle.load(open('./rightModel.p', 'rb'))
right_hand_model = right_hand_model_dict['model']

# mediapipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles =mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# 학습 모델에 의한 분류 라벨
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
               9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R',
               19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 14: 'nothing'}

# 클라이언트가 서버에게 이미지에 대해 지문자 분류 작업을 수행하라고 요청하는 엔드포인트 (현재는 사용되지 않음)
"""@app.route('/predict')
def index():
    print('predict')
    return render_template('predict.html')"""

# 클라이언트가 서버에게 손 인식 작업을 수행하라고 요청하는 엔드포인트 (현재는 사용되지 않음)
"""@app.route('/identify')
def identify():
    return render_template('identify.html')"""

# 웹소켓 통신이 연결되었을 때
@socket_io.on('connect')
def handle_connect():
    emit('stream', {'stream': True}, broadcast=False)

# 웹소켓 통신이 해제되었을 때
@socket_io.on('disconnect')
def handle_disconnect():
    emit('stream', {'stream': False}, broadcast=False)

# 웹소켓으로 서버에게 이미지에 대해 지문자 분류 작업을 수행하라고 요청
@socket_io.on('prediction')
def handle_image(data_uri):
    # 데이터 URI를 OpenCV 이미지로 변환
    encoded_data = data_uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 이미지 처리 및 예측 수행
    result = predict_image(img, right_hand_model)

    # 예측한 알파벳 결과를 요청을 한 클라이언트에게 전송
    emit('prediction_result', {'alphabet': result}, broadcast=False)
    
# 웹소켓으로 서버에게 손 인식 작업을 수행하라고 요청
@socket_io.on('identification')
def identify_image(data_uri):
    # Convert data URI to OpenCV image
    encoded_data = data_uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result = check_if_hand(img)
    
    # 손 인식 결과를 요청을 한 클라이언트에게 전송
    emit('identification_result', {'flag': result}, broadcast=False)
    
# 입력받은 이미지를 바탕으로 어떤 알파벳에 해당하는 지문자인지 리턴
def predict_image(frame, model):
    hand_detected = False
    predicted_character = None
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MediaPipe Hands module for hand detection
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:
            hand_detected = True

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if hand_detected:
                # Make predictions using the model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

    return predicted_character

# 현재 캠에 한 개의 손만 있는지를 확인
def check_if_hand(frame):
    hand_detected = False

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MediaPipe Hands module for hand detection
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:
            # only one hand detected!
            hand_detected = True
    
    return hand_detected
            

if __name__ == '__main__':
    socket_io.run(app, debug=True, host='localhost', port=9999, use_reloader=False)