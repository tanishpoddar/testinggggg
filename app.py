from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
import math as m
import mediapipe as mp
from datetime import datetime
import re
from werkzeug.serving import WSGIServer
import gc
import os

app = Flask(__name__)

# Initialize MediaPipe Pose with lower resource usage
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0  # Use simpler model to reduce memory usage
)

class PostureTracker:
    def __init__(self):
        self.good_frames = 0
        self.bad_frames = 0
        self.start_time = datetime.now()
        self.fps = 10
        
    def reset(self):
        self.good_frames = 0
        self.bad_frames = 0
        self.start_time = datetime.now()

def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    return int((180 / m.pi) * theta)

# Create posture tracker instance
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
        
        # Resize frame to reduce memory usage
        frame = cv2.resize(frame, (640, 480))
        
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
            l_ear_x = int(lm[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm[lmPose.LEFT_EAR].y * h)
            l_hip_x = int(lm[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm[lmPose.LEFT_HIP].y * h)
            
            # Draw landmarks and connections (simplified)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=1)
            )
            
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
        
        # Convert processed frame back to base64 with lower quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        processed_frame = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
        
        # Clean up to prevent memory leaks
        del frame_rgb, frame, results
        gc.collect()
        
        return jsonify({
            'processed_frame': processed_frame,
            'posture_status': posture_status
        })
        
    except Exception as e:
        # Reset tracker on error
        posture_tracker.reset()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Production WSGI server
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
