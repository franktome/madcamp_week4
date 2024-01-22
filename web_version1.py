import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the model and other configurations
model_dict = pickle.load(open('C:/Users/yunhs/vscode_project/madcamp_week4_3_direct/model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
               9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R',
               19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 14: 'nothing'}

def main():
    st.title("Hand Gesture Recognition with Streamlit")

    video_capture = cv2.VideoCapture(0)

    while st.checkbox("Capture Video"):
        ret, frame = video_capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                data_aux = []
                x_ = []
                y_ = []

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

                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10

                x2 = int(max(x_) * frame.shape[1]) - 10
                y2 = int(max(y_) * frame.shape[0]) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        st.image(frame, channels="BGR", use_column_width=True)

    video_capture.release()

if __name__ == '__main__':
    main()
