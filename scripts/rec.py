import cv2
import face_recognition
import pickle
import datetime
import os
import pandas as pd
import threading
import time
from playsound import playsound
from sklearn.neighbors import KDTree
import numpy as np

# === CONFIGURATION ===
ENCODINGS_FILE = "encodings.pickle"
ATTENDANCE_FILE = "attendance.csv"
IMPOSTOR_SOUND = "warning.mp3"
CONFIDENCE_THRESHOLD = 0.4
SMALL_FRAME_SCALE = 0.25
FRAME_SKIP = 2
ALERT_COOLDOWN = 5  # seconds delay between sound alerts

# === ALERT STATE ===
last_alert_time = 0
alert_lock = threading.Lock()

# === LOAD ENCODINGS ===
print("[INFO] Loading face encodings...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

tree = KDTree(np.array(data["encodings"]))
print("[INFO] KDTree built for fast face search.")

if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)
    print(f"[INFO] Attendance file '{ATTENDANCE_FILE}' created.")

# === ATTENDANCE FUNCTION ===
def mark_attendance(name):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    df = pd.read_csv(ATTENDANCE_FILE)
    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        df.loc[len(df)] = [name, date, time_str]
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"[INFO] {name} marked present at {time_str} on {date}")

# === PLAY ALERT WITH COOLDOWN ===
def play_alert_with_cooldown():
    global last_alert_time
    with alert_lock:
        now = time.time()
        if now - last_alert_time >= ALERT_COOLDOWN:
            last_alert_time = now
            threading.Thread(target=play_sound, daemon=True).start()

def play_sound():
    try:
        print("[ALERT] Playing impostor sound!")
        playsound(IMPOSTOR_SOUND)
    except Exception as e:
        print(f"[ERROR] Could not play sound: {e}")

# === START VIDEO CAPTURE ===
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("[ERROR] Cannot open webcam.")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        if frame_count % FRAME_SKIP == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=SMALL_FRAME_SCALE, fy=SMALL_FRAME_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            results = []

            for encoding, location in zip(face_encodings, face_locations):
                dist, ind = tree.query([encoding], k=1)
                min_distance = dist[0][0]
                min_index = ind[0][0]

                if min_distance < CONFIDENCE_THRESHOLD:
                    name = data["names"][min_index]
                    box_color = (0, 255, 0)
                    label = f"{name} ({min_distance:.2f})"
                    mark_attendance(name)
                else:
                    name = "IMPOSTOR"
                    box_color = (0, 0, 255)
                    label = "IMPOSTOR!"
                    print("[WARNING] Impostor detected!")
                    play_alert_with_cooldown()

                top, right, bottom, left = [int(coord / SMALL_FRAME_SCALE) for coord in location]
                results.append((left, top, right, bottom, box_color, label))

        if 'results' in locals():
            for (left, top, right, bottom, box_color, label) in results:
                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
                cv2.putText(display_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        cv2.imshow("Mirror Face Recognition (Optimized)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exit requested.")
            break

        frame_count += 1

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Face recognition stopped.")