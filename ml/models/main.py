"""
EMOTION DETECTION — BACKEND FASTAPI (Phase 4)
==============================================
Lancer : uvicorn app.main:app --reload --port 8000
Docs   : http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import numpy as np
import cv2
import onnxruntime as ort
import base64
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
EMOTION_FR = {
    "angry":     "Colère",
    "disgusted": "Dégoût",
    "fearful":   "Peur",
    "happy":     "Joie",
    "neutral":   "Neutre",
    "sad":       "Tristesse",
    "surprised": "Surprise",
}
EMOTION_EMOJI = {
    "angry": "😠", "disgusted": "🤢", "fearful": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprised": "😲",
}
IMG_SIZE = 48
MODEL_PATH = Path(__file__).parent.parent / "models" / "emotion_model.onnx"
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ─────────────────────────────────────────────
# ML Engine
# ─────────────────────────────────────────────
class EmotionEngine:
    def __init__(self, model_path: str):
        logger.info(f"Chargement du modèle ONNX : {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        logger.info("Modèle + Haar Cascade chargés.")

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def preprocess_face(self, face_gray: np.ndarray) -> np.ndarray:
        """Redimensionner et normaliser un visage 48x48."""
        face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
        face = face.astype(np.float32) / 127.5 - 1.0
        return face.reshape(1, 1, IMG_SIZE, IMG_SIZE)

    def detect_faces(self, img_gray: np.ndarray):
        """Retourner liste de (x, y, w, h) pour chaque visage détecté."""
        return self.face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

    def predict_face(self, face_tensor: np.ndarray) -> dict:
        """Prédire l'émotion pour un visage prétraité."""
        logits = self.session.run(None, {"input": face_tensor})[0][0]
        probs = self.softmax(logits)
        idx = int(probs.argmax())
        return {
            "emotion": EMOTIONS[idx],
            "emotion_fr": EMOTION_FR[EMOTIONS[idx]],
            "emoji": EMOTION_EMOJI[EMOTIONS[idx]],
            "confidence": float(round(probs[idx] * 100, 2)),
            "probabilities": {
                e: float(round(float(p) * 100, 2))
                for e, p in zip(EMOTIONS, probs)
            },
        }

    def process_image(self, img_bgr: np.ndarray) -> dict:
        """Pipeline complet : détection visage(s) + prédiction."""
        h, w = img_bgr.shape[:2]
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(img_gray)

        if len(faces) == 0:
            return {
                "success": False,
                "error": "Aucun visage détecté dans l'image.",
                "faces": [],
            }

        results = []
        for x, y, fw, fh in faces:
            face_crop = img_gray[y:y + fh, x:x + fw]
            tensor = self.preprocess_face(face_crop)
            pred = self.predict_face(tensor)
            pred["bbox"] = {
                "x": int(x), "y": int(y),
                "w": int(fw), "h": int(fh),
                "x_norm": round(x / w, 4),
                "y_norm": round(y / h, 4),
                "w_norm": round(fw / w, 4),
                "h_norm": round(fh / h, 4),
            }
            results.append(pred)

        return {
            "success": True,
            "faces_count": len(results),
            "faces": results,
            "dominant_emotion": results[0]["emotion"],
        }


# ─────────────────────────────────────────────
# App FastAPI
# ─────────────────────────────────────────────
engine: EmotionEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if MODEL_PATH.exists():
        engine = EmotionEngine(str(MODEL_PATH))
        logger.info("Engine ML prêt.")
    else:
        logger.warning(
            f"Modèle introuvable : {MODEL_PATH}. "
            "Placer emotion_model.onnx dans ./models/"
        )
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Emotion Detection API",
    description="Détection d'émotions faciales via CNN (FER dataset)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Emotion Detection API",
        "status": "running",
        "model_loaded": engine is not None,
        "endpoints": ["/predict", "/ws/stream", "/health", "/docs"],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "emotions": EMOTIONS,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyser une image (JPG/PNG) et retourner les émotions détectées.
    Supporte plusieurs visages par image.
    """
    if engine is None:
        raise HTTPException(503, "Modèle non chargé. "
                            "Vérifier models/emotion_model.onnx")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Type de fichier invalide : {file.content_type}")

    t0 = time.time()
    raw = await file.read()

    # Décoder l'image
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Impossible de décoder l'image.")

    result = engine.process_image(img)
    result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
    result["filename"] = file.filename

    return JSONResponse(content=result)


@app.post("/predict/base64")
async def predict_base64(payload: dict):
    """
    Analyser une image encodée en base64 (utile depuis le frontend webcam).
    Body: { "image": "data:image/jpeg;base64,..." }
    """
    if engine is None:
        raise HTTPException(503, "Modèle non chargé.")

    image_data = payload.get("image", "")
    if not image_data:
        raise HTTPException(400, "Champ 'image' manquant.")

    # Retirer le préfixe data URL si présent
    if "," in image_data:
        image_data = image_data.split(",")[1]

    t0 = time.time()
    try:
        raw = base64.b64decode(image_data)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(400, f"Erreur décodage base64 : {e}")

    if img is None:
        raise HTTPException(400, "Image invalide.")

    result = engine.process_image(img)
    result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
    return JSONResponse(content=result)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket pour la détection en temps réel via webcam.
    Le client envoie des frames base64 toutes les ~300ms.
    Le serveur répond avec les prédictions.
    """
    await websocket.accept()
    logger.info("WebSocket connecté.")

    if engine is None:
        await websocket.send_json({"error": "Modèle non chargé."})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            image_b64 = msg.get("frame", "")

            if not image_b64:
                continue

            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]

            try:
                raw = base64.b64decode(image_b64)
                arr = np.frombuffer(raw, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if img is None:
                    await websocket.send_json({"error": "Frame invalide."})
                    continue

                result = engine.process_image(img)
                result["timestamp"] = time.time()
                await websocket.send_json(result)

            except Exception as e:
                logger.warning(f"Erreur traitement frame : {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket déconnecté.")
    except Exception as e:
        logger.error(f"Erreur WebSocket : {e}")
        await websocket.close()
