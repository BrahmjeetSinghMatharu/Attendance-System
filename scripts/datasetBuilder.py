import cv2
import os
import time  # <-- for delay

# Set base directory for dataset
BASE_DIR = "dataset"
os.makedirs(BASE_DIR, exist_ok=True)

# Ask for person’s name
person_name = input("Enter person's name: ")
person_dir = os.path.join(BASE_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
print("[INFO] Capturing images. Press 'q' to quit.")

count = 0
CAPTURE_DELAY = 0.5  # Seconds between captures (adjust as needed)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        break
    
    # Show the frame
    cv2.imshow("Capturing - Press 'q' to quit", frame)

    # Save frame
    img_path = os.path.join(person_dir, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"[INFO] Saved {img_path}")
    
    count += 1
    
    # Wait for the user to press 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Quitting capture...")
        break
    
    time.sleep(CAPTURE_DELAY)  # Delay between captures

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Dataset collected for {person_name}. {count} images saved.")