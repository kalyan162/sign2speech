# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import pyttsx3
# import threading  # Import threading for speech synthesis

# # Load model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)

# text_speech = pyttsx3.init()

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {0: 'one', 1: 'two', 2: 'three', 3: 'four', 4: 'five'}

# # Function to handle speech synthesis in a separate thread
# def speak_text(text):
#     text_speech.say(text)
#     text_speech.runAndWait()

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         prediction = model.predict([np.asarray(data_aux)])

#         predicted_character = labels_dict[int(prediction[0])]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#         # Run speech synthesis in a new thread
#         speech_thread = threading.Thread(target=speak_text, args=(predicted_character,))
#         speech_thread.start()

#     else:
#         print("hello")
#         cv2.waitKey(5000)
#         break

#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)  # Reduce wait time for smooth real-time performance


#MAIN CODE#############################################################################################

# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import gtts
# import io
# import pygame
# import concurrent.futures

# # Initialize pygame mixer
# pygame.mixer.init()

# # Set the path to the ffmpeg binary
# from pydub import AudioSegment
# AudioSegment.ffmpeg = "C:/Users/HP/Downloads/ffmpeg-7.0.2/bin/ffmpeg.exe"

# # Load model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

# labels_dict = {0: 'one', 1: 'two', 2: 'three', 3: 'four', 4: 'five'}

# # Create a ThreadPoolExecutor for handling tasks
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# # Queue for pending speech requests
# pending_requests = set()

# def speak_text(text):
#     if text in pending_requests:
#         return
#     pending_requests.add(text)
#     tts = gtts.gTTS(text=text, lang='en')
#     with io.BytesIO() as temp_file:
#         tts.write_to_fp(temp_file)
#         temp_file.seek(0)
        
#         # Load and play audio
#         pygame.mixer.music.load(temp_file)
#         pygame.mixer.music.play()
        
#         # Wait until the audio is done playing
#         while pygame.mixer.music.get_busy():
#             pygame.time.wait(500)
        
#         # Remove from pending requests after playback
#         pending_requests.remove(text)

# # Function to run speech synthesis asynchronously using ThreadPoolExecutor
# def run_speech_async(text):
#     executor.submit(speak_text, text)

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Downscale frame for faster processing
#     frame = cv2.resize(frame, (640, 480))
#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         prediction = model.predict([np.asarray(data_aux)])

#         predicted_character = labels_dict[int(prediction[0])]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#         # Run speech synthesis asynchronously
#         run_speech_async(predicted_character)

#     else:
#         print("hello")
#         cv2.waitKey(5000)
#         break

#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)  # Reduce wait time for smooth real-time performance

# cap.release()
# cv2.destroyAllWindows()
###################################################################################################################


import pickle
import cv2
import mediapipe as mp
import numpy as np
import gtts
import io
import pygame
import concurrent.futures
from googletrans import Translator

# Initialize pygame mixer
pygame.mixer.init()

# Set the path to the ffmpeg binary
from pydub import AudioSegment

AudioSegment.ffmpeg = "C:/Users/HP/Downloads/ffmpeg-7.0.2/bin/ffmpeg.exe"

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Mapping of predicted numbers to English labels
labels_dict = {0: 'one', 1: 'two', 2: 'three', 3: 'four', 4: 'five'}

# Translator for converting English to Telugu
translator = Translator()

# Create a ThreadPoolExecutor for handling tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Queue for pending speech requests
pending_requests = set()


def speak_text(text):
    if text in pending_requests:
        return
    pending_requests.add(text)
    
    # Translate text to Telugu
    translated = translator.translate(text, src='en', dest='te')
    telugu_text = translated.text
    
    tts = gtts.gTTS(text=telugu_text, lang='te')  # Use Telugu language code 'te'
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

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
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

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Run speech synthesis asynchronously
        run_speech_async(predicted_character)

    else:
        print("hello")
        cv2.waitKey(5000)
        break

    cv2.imshow('frame', frame)
    cv2.waitKey(1)  # Reduce wait time for smooth real-time performance

cap.release()
cv2.destroyAllWindows()
