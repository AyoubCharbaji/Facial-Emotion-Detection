import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# ==============================
# 1️⃣ Load Model & Class Names
# ==============================

model = load_model("emotion_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("Loaded classes:", class_names)

# ==============================
# 2️⃣ Load Face Detector
# ==============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# 3️⃣ Start Webcam
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Resize to model input size
        face = cv2.resize(face, (224, 224))

        # Convert BGR → RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize
        face = face / 255.0

        # Expand dimensions
        face = np.expand_dims(face, axis=0)

        # Predict
        prediction = model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion = class_names[emotion_index]
        confidence = prediction[0][emotion_index]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put text
        text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
