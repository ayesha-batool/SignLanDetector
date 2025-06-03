import pickle
import cv2
import mediapipe as mp
import numpy as np
import textwrap
import pyttsx3
import nltk
import threading

nltk.download('words', quiet=True)
from nltk.corpus import words
english_vocab = set(w.lower() for w in words.words())
engine = pyttsx3.init()
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create window and set to fullscreen
window_name = '🤖 Enhanced Sign Language Recognition'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

labels_dict = {i: chr(65 + i) for i in range(26)}
output_text = ""
prev_character = ""
stable_counter = 0
required_stable_frames = 15 
WRAP_WIDTH = 55
scroll_offset = 0
scroll_step = 1
max_lines = 6

file_path = 'output.txt'
open(file_path, 'w').close()

def is_valid_word(word):
    return word.lower() in english_vocab

def write_to_file(content):
    with open(file_path, 'w') as f:
        lines = textwrap.wrap(content, width=WRAP_WIDTH)
        for line in lines:
            f.write(line + '\n')

def speak_text(text_list):
    try:
        local_engine = pyttsx3.init()
        text_to_speak = " ".join(text_list)
        local_engine.say(text_to_speak)
        local_engine.runAndWait()
        local_engine.stop()
    except RuntimeError as e:
        print(f"Speech error: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    predicted_character = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_index = int(prediction[0])
            predicted_character = labels_dict[predicted_index]

            if predicted_character == prev_character:
                stable_counter += 1
            else:
                stable_counter = 0
                prev_character = predicted_character

            if stable_counter == required_stable_frames:
                if not (len(output_text) >= 2 and 
                        output_text[-1] == predicted_character and 
                        output_text[-2] == predicted_character):
                    output_text += predicted_character
                    write_to_file(output_text)
                stable_counter = 0

            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H) - 10
            cv2.putText(frame, predicted_character, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        stable_counter = 0
        prev_character = ""

    box_height = 120
    cv2.rectangle(frame, (0, H - box_height), (W, H), (30, 30, 30), -1)
    cv2.rectangle(frame, (5, H - box_height + 5), (W - 5, H - 5), (100, 255, 100), 2)
    cv2.putText(frame, "Output (P: speak, S: space, Q: backspace,ESC: exit):",
                (15, H - box_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    wrapped_lines = textwrap.wrap(output_text, width=WRAP_WIDTH)
    if wrapped_lines:
        start_line = max(0, len(wrapped_lines) - max_lines - scroll_offset)
        for i, line in enumerate(wrapped_lines[start_line:start_line + max_lines]):
            y_offset = H - box_height + 45 + (i * 25)
            cv2.putText(frame, line, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 255, 255), 1)

        bar_height = int(max_lines / len(wrapped_lines) * box_height)
        bar_pos = int((scroll_offset / len(wrapped_lines)) * (box_height - bar_height))
        cv2.rectangle(frame, (W - 20, H - box_height + 5 + bar_pos),
                      (W - 10, H - box_height + 5 + bar_pos + bar_height), (100, 255, 100), -1)
    else:
        cv2.putText(frame, "No text to display.", (10, H - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Keep it full screen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        output_text = output_text[:-1]
        write_to_file(output_text)
    elif key == ord('s'):
        output_text += " "
        write_to_file(output_text)
    elif key == ord('p'):
        cleaned_text = ""
        words_to_check = output_text.strip().split()
        for w in words_to_check:
            if is_valid_word(w):
                cleaned_text += w + " "
            else:
                cleaned_text += f"{w} "
        if cleaned_text.strip():
            threading.Thread(target=speak_text, args=([cleaned_text.strip()],), daemon=True).start()
    elif key == ord('f'):  # Fullscreen toggle
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('w'):  # Windowed toggle
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif key == 27:  # ESC
        break
    elif key == 2490368 and scroll_offset > 0:  # UP
        scroll_offset -= scroll_step
    elif key == 2621440 and scroll_offset < len(wrapped_lines) - max_lines:  # DOWN
        scroll_offset += scroll_step

cap.release()
cv2.destroyAllWindows()
