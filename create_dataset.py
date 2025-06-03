import os
import cv2
import pickle
import mediapipe as mp

# Set up paths and MediaPipe
DATA_DIR = './data'
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

data = []
labels = []

# Loop through each folder and image
for label in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder_path):
        continue  # Skip non-directories

    print(f"Processing directory: {label}")

    for file_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Skipped (image not readable): {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_, features = [], [], []

                # Collect raw x, y
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                # Normalize landmarks relative to hand bounding box
                for lm in hand_landmarks.landmark:
                    features.append(lm.x - min(x_))
                    features.append(lm.y - min(y_))

                data.append(features)
                labels.append(label)
        else:
            print(f"❌ No hand detected in: {img_path}")

# Save data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"✅ Done. Total samples collected: {len(data)}")
