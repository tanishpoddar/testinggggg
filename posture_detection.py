import cv2
import time
import math as m
import mediapipe as mp
import streamlit as st
from PIL import Image

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = theta * (180 / m.pi)
    return degree

def sendWarning():
    st.warning("Warning: Bad posture detected for too long!")

# Streamlit UI
st.title("Real-Time Posture Detection")

# Select camera source
camera_index = st.selectbox("Select Camera Source", options=[0, 1, 2])

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize counters
good_frames = 0
bad_frames = 0

# Set up the video capture and processing
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    st.error("Error: Could not open video.")

# Display live feed and analysis
frame_window = st.image([])

while cap.isOpened():
    success, image = cap.read()
    if not success:
        st.warning("Skipping empty frame.")
        continue
    
    # Process frame for pose detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image_rgb)

    if not keypoints.pose_landmarks:
        st.info("No pose detected")
        frame_window.image(image_rgb)
        continue

    lm = keypoints.pose_landmarks.landmark
    h, w = image.shape[:2]

    # Calculate relevant keypoints
    try:
        l_shldr_x = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm[mp_pose.PoseLandmark.LEFT_EAR].x * w)
        l_ear_y = int(lm[mp_pose.PoseLandmark.LEFT_EAR].y * h)
        l_hip_x = int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w)
        l_hip_y = int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)

        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1
            color = (127, 233, 100)
        else:
            good_frames = 0
            bad_frames += 1
            color = (50, 50, 255)

        if bad_frames / cap.get(cv2.CAP_PROP_FPS) > 180:
            sendWarning()

        # Draw landmarks and lines for posture
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, (0, 255, 255), -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (255, 0, 255), -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, (0, 255, 255), -1)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 4)

        # Display feedback on posture
        angle_text_string = f'Neck: {int(neck_inclination)}°, Torso: {int(torso_inclination)}°'
        cv2.putText(image, angle_text_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    except Exception as e:
        st.error(f"Error processing frame: {e}")

    # Show the frame in Streamlit
    frame_window.image(image_rgb)

cap.release()
pose.close()
