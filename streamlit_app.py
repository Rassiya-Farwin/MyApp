import streamlit as st
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate lip distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Constants
EYE_AR_THRESH = 0.3
YAWN_THRESH = 20

# Streamlit UI
st.title("Drowsiness and Yawn Detection")
st.write("This app detects drowsiness and yawning using your webcam.")
run_app = st.checkbox("Start Webcam")

# Video feed
if run_app:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam. Make sure it's connected.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Calculate EAR and lip distance
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            distance = lip_distance(shape)
            
            # Display alerts
            if ear < EYE_AR_THRESH:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if distance > YAWN_THRESH:
                cv2.putText(frame, "YAWN ALERT!", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the video frame in the Streamlit app
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
