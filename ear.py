import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
import mediapipe as mp
import winsound  # For beep alert (Windows)

alert_start_time = None
ALERT_DURATION = 5  # seconds
ALERT_ACTIVE = False  # Track if alert is ongoing

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Landmark indices for EAR (Eyes) and MAR (Mouth)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_IDX = [78, 308, 13, 14, 87, 317]

# Thresholds
EAR_THRESH = 0.17
EYE_CLOSED_FRAMES_THRESH = 15
YAWN_THRESH = 0.6
YAWN_FRAMES = 15


YAWN_DISPLAY_TIME = 2  # seconds to keep "Yawning" text visible
last_yawn_time = 0



# State
eye_closed_counter = 0
yawn_counter = 0
yawns = 0
eye_status = "Unknown"
yawn_status = "No Yawn"

# Webcam
cap = cv2.VideoCapture(0)

def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye and mouth points
            left_eye = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in LEFT_EYE_IDX])
            right_eye = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in RIGHT_EYE_IDX])
            mouth = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in MOUTH_IDX])

            # EAR Calculation
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            EAR = (left_EAR + right_EAR) / 2.0

            # MAR Calculation
            A = dist.euclidean(mouth[2], mouth[3])
            B = dist.euclidean(mouth[4], mouth[5])
            MAR = A / B

            # Eye state
            if EAR < EAR_THRESH:
                eye_closed_counter += 1
                eye_status = "Closed"
            else:
                eye_closed_counter = 0
                eye_status = "Open"

            # Yawn state
            if MAR > YAWN_THRESH:
                yawn_counter += 1
            else:
                if yawn_counter >= YAWN_FRAMES:
                    yawns += 1
                    yawn_status = "Yawning"
                    last_yawn_time = time.time()
                    print("Yawn Detected!")
                yawn_counter = 0
                # yawn_status = "No Yawn"
            # Keep "Yawning" visible for a while
            if time.time() - last_yawn_time < YAWN_DISPLAY_TIME:
                yawn_status = "Yawning"
            else:
                yawn_status = "No Yawn"

            # Draw bounding boxes
            left_eye_rect = cv2.boundingRect(left_eye)
            right_eye_rect = cv2.boundingRect(right_eye)
            mouth_rect = cv2.boundingRect(mouth)

            cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]),
                          (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]),
                          (0, 255, 0), 2)
            cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]),
                          (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]),
                          (0, 255, 0), 2)
            cv2.rectangle(frame, (mouth_rect[0], mouth_rect[1]),
                          (mouth_rect[0] + mouth_rect[2], mouth_rect[1] + mouth_rect[3]),
                          (0, 255, 0), 2)

            # Drowsiness logic
            current_time = time.time()
            if eye_closed_counter > EYE_CLOSED_FRAMES_THRESH or yawn_counter >= YAWN_FRAMES:
                alert_start_time = current_time
                if not ALERT_ACTIVE:
                    ALERT_ACTIVE = True
                    # Beep sound alert
                    winsound.Beep(1000, 700)  # (frequency, duration in ms)
                    winsound.Beep(1000, 700)  # (frequency, duration in ms)
                    winsound.Beep(1000, 700)  # (frequency, duration in ms)

            # Stop alert after duration
            if alert_start_time and (current_time - alert_start_time) > ALERT_DURATION:
                ALERT_ACTIVE = False
                alert_start_time = None

            # Display alert
            if ALERT_ACTIVE:
                cv2.putText(frame, "DROWSINESS ALERT!", (120, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # # Info overlay
            # cv2.putText(frame, f"EAR: {EAR:.2f}", (10, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 70),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Eye: {eye_status}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
            cv2.putText(frame, f"Yawn: {yawn_status}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
            cv2.putText(frame, f"Yawn Count: {yawns}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

            break  # only process first face

    cv2.imshow("Drowsiness Detection (EAR + MAR)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
