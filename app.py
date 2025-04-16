#Streamlit app for real time testing using .h5 model
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

st.set_page_config(page_title="Gesture Recognition (CNN)", layout="centered")

# Load trained Keras model
@st.cache_resource
def load_cnn_model():
    return load_model(r"path to your .h5 model file")

model = load_cnn_model()

# MediaPipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Add Gesture labels according to your model file
gesture_labels = {
    0: "CAPACITOR",
    1: "HE",
    2: "HEART",
    3: "HUNGRY",
    4: "MORNING",
    5: "SHE",
    6: "SLEEP",
    7: "THANK YOU",
    8: "WHAT",
    9: "YOU"
}


# UI
st.title("🤖 Real-Time ISL Gesture Recognition with CNN")
st.sidebar.write("📸 Click 'Start Webcam' to begin.")
run = st.sidebar.button("Start Webcam")
FRAME_WINDOW = st.image([])

# Preprocessing
def preprocess_frame(frame):
    img_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.add(img_hsv[:, :, 2], 30)
    img_preprocessed = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_preprocessed

def extract_landmarks(image_rgb):
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                data_aux.extend([landmark.x, landmark.y])
        return data_aux
    return None

# Real-time loop
if run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows compatibility

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Unable to access webcam.")
            break

        preprocessed = preprocess_frame(frame)
        img_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        landmarks = extract_landmarks(img_rgb)

        if landmarks:
            # Get hand bounding box from landmarks
            x_coords = landmarks[::2]
            y_coords = landmarks[1::2]
            h, w, _ = img_rgb.shape

            x_min = int(min(x_coords) * w) - 20
            x_max = int(max(x_coords) * w) + 20
            y_min = int(min(y_coords) * h) - 20
            y_max = int(max(y_coords) * h) + 20

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Draw bounding box on original image
            cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Crop hand region and resize
            hand_img = img_rgb[y_min:y_max, x_min:x_max]
            hand_img = cv2.resize(hand_img, (224, 224))
            hand_img = hand_img.astype("float32") / 255.0  # Normalize
            hand_input = np.expand_dims(hand_img, axis=0)  # Shape: (1, 224, 224, 3)

            # Predict using CNN
            prediction = model.predict(hand_input, verbose=0)
            gesture = gesture_labels.get(np.argmax(prediction), str(np.argmax(prediction)))
        else:
            gesture = ""

        # Overlay prediction text
        cv2.putText(
            img_rgb,
            f"Prediction: {gesture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        FRAME_WINDOW.image(img_rgb)

    cap.release()
else:
    st.warning("🎬 Click 'Start Webcam' to begin.")
