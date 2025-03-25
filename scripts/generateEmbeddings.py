import os
import cv2
import face_recognition
import pickle

# Set dataset and encoding file paths
BASE_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"

known_encodings = []
known_names = []

# Function to process a single image
def process_image(image_path, person_name):
    print(f"[INFO] Processing {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Unable to read {image_path}, skipping.")
        return
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    
    if len(encodings) == 0:
        print(f"[WARNING] No face found in {image_path}, skipping.")
        return
    
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(person_name)

# Process images sequentially to reduce memory usage
def process_images():
    for person_name in os.listdir(BASE_DIR):
        person_dir = os.path.join(BASE_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        print(f"[INFO] Processing images for {person_name}")
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            process_image(image_path, person_name)

# Process images and store encodings
print("[INFO] Extracting face embeddings...")
process_images()

# Save encodings to a file
print("[INFO] Saving encodings...")
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Face encodings saved successfully.")