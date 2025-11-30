import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
import mediapipe as mp
import winsound  # For beep alert (Windows)

# ----------------- Global Alert State -----------------
alert_start_time = None
ALERT_DURATION = 5  # seconds
ALERT_ACTIVE = False

# ----------------- Mediapipe FaceMesh -----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# ----------------- Landmark Indices -------------------
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_IDX = [78, 308, 13, 14, 87, 317]

# ----------------- Thresholds -------------------------
EAR_THRESH = 0.17
EYE_CLOSED_FRAMES_THRESH = 15
YAWN_THRESH = 0.6
YAWN_FRAMES = 15

# Head tilt thresholds (normalized values)
ROLL_PIXEL_THRESH = 10        # Eye height difference
PITCH_THRESH_NORM = 0.10      # 10% of face height

YAWN_DISPLAY_TIME = 2
last_yawn_time = 0

# ----------------- State Variables --------------------
eye_closed_counter = 0
yawn_counter = 0
yawns = 0
eye_status = "Unknown"
yawn_status = "No Yawn"
head_status = "Normal"

# ----------------- Webcam -----------------------------
cap = cv2.VideoCapture(0)


def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)


def get_head_tilt_improved(landmarks, w, h):
    """
    Fix pitch detection using *nose average position* relative to eyes
    Normalized by face height → stable for all faces
    """
    # Eye outer corners
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)

    eye_center_y = (ly + ry) / 2

    # Nose top, mid, bottom
    nose_top = landmarks[1]
    nose_mid = landmarks[6]
    nose_bottom = landmarks[2]

    nose_y = (nose_top.y + nose_mid.y + nose_bottom.y) / 3
    nose_y *= h  # convert to pixel coordinates

    # Face height (chin to forehead)
    chin_y = landmarks[152].y * h
    forehead_y = landmarks[10].y * h
    face_height = abs(chin_y - forehead_y)

    # NORMALIZED pitch offset
    pitch_offset_norm = (nose_y - eye_center_y) / face_height

    # ---------------- PITCH detection (looking down) ----------------
    if pitch_offset_norm > PITCH_THRESH_NORM:
        pitch_status = "Head Down"
        pitch_tilt = True
    else:
        pitch_status = "Normal"
        pitch_tilt = False

    # ---------------- ROLL detection (side tilt) --------------------
    eye_height_diff = abs(ry - ly)

    if eye_height_diff > ROLL_PIXEL_THRESH:
        roll_status = "Head Tilt Sideways"
        roll_tilt = True
    else:
        roll_status = "Normal"
        roll_tilt = False

    head_tilted = pitch_tilt or roll_tilt
    return pitch_status, roll_status, head_tilted


# ----------------- Main Loop -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # ---------- HEAD TILT DETECTION ----------
        pitch_status, roll_status, head_tilted = get_head_tilt_improved(face_landmarks, w, h)

        if head_tilted:
            head_status = pitch_status if pitch_status != "Normal" else roll_status
        else:
            head_status = "Normal"

        # ---------- EYE & MOUTH POINTS -----------
        left_eye = np.array([[int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)] for i in LEFT_EYE_IDX])
        right_eye = np.array([[int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)] for i in RIGHT_EYE_IDX])
        mouth = np.array([[int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)] for i in MOUTH_IDX])

        # ---------- EAR ----------
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        EAR = (left_EAR + right_EAR) / 2.0

        # ---------- MAR ----------
        A = dist.euclidean(mouth[2], mouth[3])
        B = dist.euclidean(mouth[4], mouth[5])
        MAR = A / B

        # ---------- Eye status ----------
        if EAR < EAR_THRESH:
            eye_closed_counter += 1
            eye_status = "Closed"
        else:
            eye_closed_counter = 0
            eye_status = "Open"

        # ---------- Yawn detection ----------
        if MAR > YAWN_THRESH:
            yawn_counter += 1
        else:
            if yawn_counter >= YAWN_FRAMES:
                yawns += 1
                yawn_status = "Yawning"
                last_yawn_time = time.time()
            yawn_counter = 0

        if time.time() - last_yawn_time < YAWN_DISPLAY_TIME:
            yawn_status = "Yawning"
        else:
            yawn_status = "No Yawn"

        # ---------- ALERT ----------
        current_time = time.time()
        if (eye_closed_counter > EYE_CLOSED_FRAMES_THRESH
            or yawn_counter >= YAWN_FRAMES
            or head_tilted):

            alert_start_time = current_time
            if not ALERT_ACTIVE:
                ALERT_ACTIVE = True
                winsound.Beep(1000, 700)
                winsound.Beep(1000, 700)
                winsound.Beep(1000, 700)

        if alert_start_time and (current_time - alert_start_time) > ALERT_DURATION:
            ALERT_ACTIVE = False
            alert_start_time = None

        # ---------- UI ----------
        if ALERT_ACTIVE:
            cv2.putText(frame, "DROWSINESS ALERT!", (150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.putText(frame, f"Eye: {eye_status}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
        cv2.putText(frame, f"Yawn: {yawn_status}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
        cv2.putText(frame, f"Head: {head_status}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, f"Yawn Count: {yawns}", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

    cv2.imshow("Drowsiness Detection (EAR + MAR + Improved HEAD TILT)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
