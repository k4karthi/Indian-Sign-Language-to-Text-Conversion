import streamlit as st
st.set_page_config(page_title="Gesture Recognition", layout="centered")

import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load model once
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\karth\OneDrive\Desktop\Biriyani\real_time_processing\random_forest_model.pkl")

model = load_model()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2)

# Streamlit UI
st.title("🤖 Real-Time Hand Gesture Recognition")
st.sidebar.write("📸 Click 'Start Webcam' to begin.")
run = st.sidebar.button("Start Webcam")
FRAME_WINDOW = st.image([])

# 🔁 Updated gesture label mapping (0 to 29)
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

# Preprocessing function (CLAHE only, limited)
def preprocess_frame(frame):
    img_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    # Apply very mild CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    img_preprocessed = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
    return img_preprocessed


# Landmark extraction
def extract_landmarks(image_rgb):
    results = hands.process(image_rgb)
    coords = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                coords.extend([landmark.x, landmark.y])
    return coords, results

# Main loop
if run:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Unable to access webcam.")
            break

        preprocessed = preprocess_frame(frame)
        img_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        landmarks, hand_results = extract_landmarks(img_rgb)

        # Draw landmarks if detected
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        # Predict
        if landmarks:
            if len(landmarks) < model.n_features_in_:
                landmarks.extend([0] * (model.n_features_in_ - len(landmarks)))
            elif len(landmarks) > model.n_features_in_:
                landmarks = landmarks[:model.n_features_in_]

            prediction = model.predict(np.array(landmarks).reshape(1, -1))[0]
            gesture = gesture_labels.get(prediction, str(prediction))
        else:
            gesture = ""

        # Draw prediction in black color
        cv2.putText(
            img_rgb,
            f"Prediction: {gesture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),  # Black font color
            2,
            cv2.LINE_AA
        )

        FRAME_WINDOW.image(img_rgb)
    cap.release()
else:
    st.warning("🎬 Click 'Start Webcam' to begin.")
