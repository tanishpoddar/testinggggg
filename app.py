from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
import math as m
import mediapipe as mp
from datetime import datetime
import re

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class PostureTracker:
    def __init__(self):
        self.good_frames = 0
        self.bad_frames = 0
        self.start_time = datetime.now()
        self.fps = 10

def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int((180 / m.pi) * theta)
    return degree

posture_tracker = PostureTracker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get frame data from request
        data = request.json
        frame_data = data['frame']
        
        # Extract base64 data
        base64_data = re.sub('^data:image/.+;base64,', '', frame_data)
        
        # Decode base64 image
        frame_bytes = base64.b64decode(base64_data)
        frame_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(frame_rgb)
        
        posture_status = "Unknown"
        
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            lm = results.pose_landmarks.landmark
            lmPose = mp_pose.PoseLandmark
            
            # Get key points
            l_shldr_x = int(lm[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm[lmPose.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm[lmPose.RIGHT_SHOULDER].y * h)
            l_ear_x = int(lm[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm[lmPose.LEFT_EAR].y * h)
            l_hip_x = int(lm[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm[lmPose.LEFT_HIP].y * h)
            
            # Draw landmarks and connections
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Calculate angles
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            
            # Check posture
            if neck_inclination < 40 and torso_inclination < 10:
                posture_tracker.bad_frames = 0
                posture_tracker.good_frames += 1
                posture_status = "Good posture"
                cv2.putText(frame, "Good Posture", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                posture_tracker.good_frames = 0
                posture_tracker.bad_frames += 1
                posture_status = "Bad posture - Please correct your posture"
                cv2.putText(frame, "Bad Posture", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_frame = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_frame': processed_frame,
            'posture_status': posture_status
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
