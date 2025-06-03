import os
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 500

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
for j in range(0,26):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
# import os
# import cv2
# import numpy as np
# import math
# import time
# from cvzone.HandTrackingModule import HandDetector

# # SETTINGS
# DATA_DIR = './data'
# alphabet_labels = [chr(i) for i in range(65, 91)]  # A-Z
# dataset_size = 100
# imgSize = 300
# offset = 20

# # Create data folders
# for label in alphabet_labels:
#     path = os.path.join(DATA_DIR, label)
#     os.makedirs(path, exist_ok=True)

# # Choose mode
# mode = input("Choose mode:\n1 - Raw Image Capture\n2 - Hand Cropped Image Capture\nEnter 1 or 2: ")

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# if mode == '1':
#     # RAW IMAGE COLLECTION
#     for label in alphabet_labels:
#         print(f'Collecting RAW data for class {label}')
#         while True:
#             ret, frame = cap.read()
#             cv2.putText(frame, 'Ready? Press "Q"', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
#             cv2.imshow('frame', frame)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break

#         counter = 0
#         while counter < dataset_size:
#             ret, frame = cap.read()
#             cv2.imshow('frame', frame)
#             cv2.waitKey(25)
#             cv2.imwrite(os.path.join(DATA_DIR, label, f'{counter}.jpg'), frame)
#             counter += 1

# elif mode == '2':
#     # HAND CROPPED IMAGE COLLECTION
#     detector = HandDetector(maxHands=2)

#     for label in alphabet_labels:
#         print(f'Collecting CROPPED HAND data for class {label}')
#         folder = os.path.join(DATA_DIR, label)

#         while True:
#             ret, frame = cap.read()
#             cv2.putText(frame, 'Ready? Press "Q"', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 3)
#             cv2.imshow('Image', frame)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break

#         counter = 0
#         while counter < dataset_size:
#             ret, img = cap.read()
#             hands, img = detector.findHands(img)
#             if hands:
#                 hand = hands[0]
#                 x, y, w, h = hand['bbox']

#                 imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#                 imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

#                 if imgCrop.size == 0:
#                     continue

#                 aspectRatio = h / w

#                 if aspectRatio > 1:
#                     k = imgSize / h
#                     wCal = math.ceil(k * w)
#                     imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                     wGap = math.ceil((imgSize - wCal) / 2)
#                     imgWhite[:, wGap:wCal + wGap] = imgResize
#                 else:
#                     k = imgSize / w
#                     hCal = math.ceil(k * h)
#                     imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                     hGap = math.ceil((imgSize - hCal) / 2)
#                     imgWhite[hGap:hCal + hGap, :] = imgResize

#                 cv2.imshow('ImageWhite', imgWhite)
#                 cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
#                 counter += 1
#                 print(f"Saved image {counter}/{dataset_size}")

#             cv2.imshow('Image', img)
#             cv2.waitKey(1)

# cap.release()
# cv2.destroyAllWindows()
# # 