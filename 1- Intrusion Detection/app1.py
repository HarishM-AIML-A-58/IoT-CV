from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import numpy as np
import os
from PIL import Image
import pyttsx3
from datetime import datetime
import serial
import time
from telegram import Bot

# ====== Voice Setup ======
engine = pyttsx3.init()
engine.setProperty('rate', 140)
engine.say("GuardianEye activated. Monitoring zone.")
engine.runAndWait()

# ====== Telegram Setup ======
TELEGRAM_BOT_TOKEN = "8361461500:AAEWF7UszvsSDeLHbeUEKiLHhkHOB_u6Lvw"
TELEGRAM_CHAT_ID = "6743445373"  # You must replace this with your own Telegram Chat ID
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# ====== Serial Setup (for LED + Buzzer) ======
try:
    ser = serial.Serial('COM6', 9600, timeout=1)  # Update COM5 to your port
    time.sleep(2)
    print("Serial connected to ESP")
except Exception as e:
    print("Serial connection failed:", e)
    ser = None

# ====== Load MTCNN and ResNet ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ====== Load Known Faces ======
known_faces_dir = '1- Intrusion Detection\\known_faces'
known_embeddings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        name = os.path.splitext(filename)[0]
        img = Image.open(os.path.join(known_faces_dir, filename)).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0).to(device))
            known_embeddings.append(embedding)
            known_names.append(name)

# ====== Intruder Image Folder ======
if not os.path.exists("intruder"):
    os.makedirs("intruder")

# ====== Recognition Function ======
def recognize_face(face_img):
    face_img = face_img.convert('RGB')
    face_tensor = mtcnn(face_img)
    if face_tensor is not None:
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0).to(device))
        min_dist = float('inf')
        identity = "Unknown"
        for i, known_embedding in enumerate(known_embeddings):
            dist = (embedding - known_embedding).norm().item()
            if dist < min_dist:
                min_dist = dist
                identity = known_names[i]
        return identity if min_dist < 0.9 else "Intruder"
    return "No Face Detected"

# ====== Start Webcam ======
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    identity = recognize_face(img)

    if identity != "No Face Detected":
        if identity != "Intruder":
            print(f"{identity} detected. Searching for intruders.")
            engine.say(f"{identity} detected. Searching for intruders.")
        else:
            print("Intruder detected!")
            engine.say("Intruder detected!")

            # Save intruder image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intruder/intruder_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Intruder face saved as: {filename}")

            # Serial: Trigger LED and Buzzer
            if ser:
                ser.write(b'I')
                print("Signal sent to ESP for alert")

            # Telegram Alert
            try:
                bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚨 Intruder detected by GuardianEye!")
                with open(filename, 'rb') as photo:
                    bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
                print("Telegram alert sent")
            except Exception as e:
                print("Failed to send Telegram alert:", e)

        engine.runAndWait()

    # Display on screen
    cv2.putText(frame, identity, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if identity == "Intruder" else (0, 255, 0), 2)
    cv2.imshow('GuardianEye Surveillance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
