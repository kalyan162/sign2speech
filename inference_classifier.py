import time
import pickle
import concurrent.futures
import io
from pydub import AudioSegment
import cv2
import mediapipe as mp
import numpy as np
import gtts
import pygame
from googletrans import Translator
import os

# Initialize the Translator
translator = Translator()


# Initialize pygame mixer
pygame.mixer.init()

# Set the path to the ffmpeg binary
AudioSegment.ffmpeg = "C:/Users/HP/Downloads/ffmpeg-7.0.2/bin/ffmpeg.exe"

# Set the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model with absolute path
model_path = os.path.join(script_dir, 'model.p')
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

labels_dict = {0: 'a', 1: 'b', 2: 'c', 3: 't', 4: ' '}

# Create a ThreadPoolExecutor for handling tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Queue for pending speech requests
pending_requests = set()

# Initializing string
current_string = ""

# Time delay settings
prediction_delay = 1  # Time delay between predictions in seconds
last_prediction_time = 0  # Timestamp of the last prediction

def speak_text(text):
    if text in pending_requests:
        return
    pending_requests.add(text)
       # Translate text from English to Telugu
    translated = translator.translate(text, src='en', dest='te')
    telugu_text=translated.text
    tts = gtts.gTTS(text=telugu_text, lang='te')
    with io.BytesIO() as temp_file:
        tts.write_to_fp(temp_file)
        temp_file.seek(0)

        # Load and play audio
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        # Wait until the audio is done playing
        while pygame.mixer.music.get_busy():
            pygame.time.wait(500)

        # Remove from pending requests after playback
        pending_requests.remove(text)

# Function to run speech synthesis asynchronously using ThreadPoolExecutor
def run_speech_async(text):
    executor.submit(speak_text, text)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    # Downscale frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    # Check current time
    current_time = time.time()

    if results.multi_hand_landmarks:
        # Only process if enough time has passed since the last prediction
        if current_time - last_prediction_time > prediction_delay:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Predict the character
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Append predicted character to the string
            current_string += predicted_character

            # Draw the rectangle and predicted character on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Update last prediction time
            last_prediction_time = current_time

            # If predicted character is space, speak the accumulated string
            if predicted_character == ' ':
                print(current_string)
                if current_string.strip():  # If there's anything in the string
                    run_speech_async(current_string.strip())  # Speak the accumulated string
                
    else:
        cv2.waitKey(5000)
        break
    # Display the frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)  # Reduce wait time for smooth real-time performance

cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)
