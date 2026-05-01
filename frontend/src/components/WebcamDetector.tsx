import { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import EmotionBars from "./EmotionBars";
import type { DetectionResult, FaceResult } from "../App";

const API_URL = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";
const EMOTIONS_ORDER = ["happy","neutral","sad","angry","surprised","fearful","disgusted"];

type Props = { onResult: (r: DetectionResult) => void };

export default function WebcamDetector({ onResult }: Props) {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fpsRef = useRef({ count: 0, last: Date.now() });

  const [active, setActive] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const drawOverlay = useCallback((faces: FaceResult[], vw: number, vh: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = vw; canvas.height = vh;
    ctx.clearRect(0, 0, vw, vh);
    faces.forEach((face) => {
      const { x, y, w, h } = face.bbox;
      const color = face.emotion === "happy" ? "#22c55e"
        : face.emotion === "angry"  ? "#ef4444"
        : face.emotion === "sad"    ? "#3b82f6" : "#a855f7";
      ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
      const label = `${face.emoji} ${face.emotion_fr} ${face.confidence.toFixed(0)}%`;
      ctx.font = "bold 13px sans-serif";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(x, y - 24, tw + 12, 22);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 6, y - 7);
    });
  }, []);

  const capture = useCallback(async () => {
    const screenshot = webcamRef.current?.getScreenshot();
    if (!screenshot) return;
    try {
      const res = await fetch(`${API_URL}/predict/base64`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: screenshot }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: DetectionResult = await res.json();
      data.timestamp = Date.now();
      setResult(data); setError(null); onResult(data);
      fpsRef.current.count++;
      const now = Date.now();
      if (now - fpsRef.current.last >= 1000) {
        setFps(fpsRef.current.count);
        fpsRef.current = { count: 0, last: now };
      }
      const video = webcamRef.current?.video;
      if (video && data.success && data.faces.length) {
        drawOverlay(data.faces, video.videoWidth, video.videoHeight);
      } else {
        canvasRef.current?.getContext("2d")
          ?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    } catch (e: unknown) {
      if (e instanceof Error) setError(e.message);
    }
  }, [drawOverlay, onResult]);

  useEffect(() => {
    if (active) {
      intervalRef.current = setInterval(capture, 400);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
      canvasRef.current?.getContext("2d")
        ?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [active, capture]);

  const face = result?.faces?.[0];

  return (
    <div className="detector-card">
      <div className="video-wrapper">
        <Webcam ref={webcamRef} screenshotFormat="image/jpeg" screenshotQuality={0.8}
          videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
          className="webcam-video" mirrored />
        <canvas ref={canvasRef} className="overlay-canvas" />
        <div className="video-badges">
          {active && <span className="badge badge-live"><span className="live-dot" />LIVE</span>}
          {active && <span className="badge badge-fps">{fps} fps</span>}
        </div>
      </div>

      <div className="controls">
        <button className={`btn-main ${active ? "btn-stop" : "btn-start"}`}
          onClick={() => setActive((v) => !v)}>
          {active ? "⏹ Arrêter" : "▶ Démarrer la détection"}
        </button>
      </div>

      {error && <div className="alert alert-error"><strong>Erreur :</strong> {error}</div>}

      {face && (
        <div className="result-section">
          <div className="result-main">
            <span className="result-emoji">{face.emoji}</span>
            <div>
              <div className="result-emotion">{face.emotion_fr}</div>
              <div className="result-confidence">{face.confidence.toFixed(1)}% de confiance</div>
            </div>
            {result?.processing_time_ms && (
              <div className="result-latency">{result.processing_time_ms} ms</div>
            )}
          </div>
          {result && result.faces_count > 1 && (
            <div className="faces-count">{result.faces_count} visages détectés</div>
          )}
          <EmotionBars probabilities={face.probabilities} dominant={face.emotion} order={EMOTIONS_ORDER} />
        </div>
      )}

      {active && result && !result.success && (
        <div className="alert alert-warn">Aucun visage détecté — centrez votre visage dans la caméra.</div>
      )}
    </div>
  );
}
