from flask import Flask, render_template, Response
import cv2
import math as m
import mediapipe as mp

app = Flask(__name__)

# Function to calculate distance between two points
def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate angle between two points
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int((180 / m.pi) * theta)
    return degree

# Function to display warning for poor posture
def sendWarning():
    print("Warning: Bad posture detected for too long!")

# Posture detection variables
good_frames = 0
bad_frames = 0

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def generate_frames(camera_index):
    cap = cv2.VideoCapture(camera_index)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(frame_rgb)

        # Check if pose landmarks are detected
        if keypoints.pose_landmarks:
            h, w = frame.shape[:2]
            lm = keypoints.pose_landmarks.landmark
            lmPose = mp_pose.PoseLandmark

            # Get keypoints
            try:
                l_shldr_x = int(lm[lmPose.LEFT_SHOULDER].x * w)
                l_shldr_y = int(lm[lmPose.LEFT_SHOULDER].y * h)
                r_shldr_x = int(lm[lmPose.RIGHT_SHOULDER].x * w)
                r_shldr_y = int(lm[lmPose.RIGHT_SHOULDER].y * h)
                l_ear_x = int(lm[lmPose.LEFT_EAR].x * w)
                l_ear_y = int(lm[lmPose.LEFT_EAR].y * h)
                l_hip_x = int(lm[lmPose.LEFT_HIP].x * w)
                l_hip_y = int(lm[lmPose.LEFT_HIP].y * h)

                # Calculate shoulder alignment and inclinations
                offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
                neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
                torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

                # Check for good or bad posture
                global good_frames, bad_frames
                if neck_inclination < 40 and torso_inclination < 10:
                    bad_frames = 0
                    good_frames += 1
                    color = (127, 233, 100)  # Light green
                else:
                    good_frames = 0
                    bad_frames += 1
                    color = (50, 50, 255)  # Red

                # Calculate time in seconds
                good_time = (1 / fps) * good_frames
                bad_time = (1 / fps) * bad_frames

                # Trigger warning if bad posture persists
                if bad_time > 180:
                    sendWarning()

                # Display angles and posture time on frame
                cv2.putText(frame, f'Neck: {int(neck_inclination)}  Torso: {int(torso_inclination)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if good_time > 0:
                    cv2.putText(frame, f'Good Posture Time: {round(good_time, 1)}s', (10, h - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)
                else:
                    cv2.putText(frame, f'Bad Posture Time: {round(bad_time, 1)}s', (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)

            except Exception as e:
                print(f"Error in calculating pose landmarks: {e}")

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    return Response(generate_frames(camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
