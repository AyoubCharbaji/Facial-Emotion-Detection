# 🎭 Facial Emotion Detection

> Full-stack application for real-time facial emotion recognition using a deep learning model.

---

## 🧰 Tech Stack

| Layer     | Technology              |
|-----------|-------------------------|
| ML        | PyTorch · ONNX          |
| Backend   | FastAPI                 |
| Frontend  | React + TypeScript      |
| DevOps    | Docker / Docker Compose |

---

## 📁 Project Structure

```
emotion-project/
├── ml/                  # Model training scripts
├── backend/             # FastAPI REST & WebSocket API
├── frontend/            # React + TypeScript app
└── docker-compose.yml
```

---

## 🚀 Getting Started

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the ONNX model file at:

```
backend/models/emotion_model.onnx
```

Start the server:

```bash
uvicorn app.main:app --reload
```

API docs available at:

```
http://localhost:8000/docs
```

---

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

App available at:

```
http://localhost:3000
```

---

## 📡 API Endpoints

| Method | Endpoint          | Description                  |
|--------|-------------------|------------------------------|
| GET    | `/health`         | Health check                 |
| POST   | `/predict`        | Predict from image file      |
| POST   | `/predict/base64` | Predict from base64 image    |
| WS     | `/ws/stream`      | Real-time WebSocket stream   |

---

## 🐳 Docker (optional)

```bash
docker-compose up --build
```

---

## 📌 Notes

- ⚠️ Ensure the ONNX model is present before starting the backend
- 💡 Good lighting improves detection accuracy
- 📊 Performance depends on training quality

---

## 📤 Example Response

```json
{
  "dominant_emotion": "happy",
  "confidence": 94.5
}
```

---

## 📄 License

MIT
