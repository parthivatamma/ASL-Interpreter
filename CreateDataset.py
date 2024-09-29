import os
import mediapipe as mp
import cv2
import pickle
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Adjust detection confidence if needed
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = "./data"
data = []
labels = []


# Function to pad landmarks to a fixed length
def pad_landmarks(data_aux, target_length=84):
    # If data is shorter, pad with zeros
    if len(data_aux) < target_length:
        data_aux.extend([0] * (target_length - len(data_aux)))
    return data_aux


# Check if the DATA_DIR exists
if not os.path.exists(DATA_DIR):
    print("Data directory does not exist.")
else:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if os.path.isdir(dir_path):
            print(f"Processing directory: {dir_}")
            for img_path in os.listdir(dir_path):
                img_full_path = os.path.join(dir_path, img_path)
                print(f"Processing image: {img_full_path}")
                data_aux = []

                img = cv2.imread(img_full_path)
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x)
                            data_aux.append(y)

                    # Pad landmarks to a fixed length 
                    data_aux = pad_landmarks(data_aux, target_length=84)

                    print(f"Landmarks detected and padded: {len(data_aux)} coordinates")
                    data.append(data_aux)
                    labels.append(dir_)
                else:
                    print(f"No hands detected in image: {img_path}")

# Save data if populated
if data and labels:
    with open("data.pickle", "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    print("Data saved successfully.")
else:
    print("No data collected to save.")
