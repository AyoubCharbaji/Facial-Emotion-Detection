import { useState, useRef, useCallback } from "react";
import EmotionBars from "./EmotionBars";
import type { DetectionResult } from "../App";

const API_URL = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";

type Props = { onResult: (r: DetectionResult) => void };

export default function ImageUpload({ onResult }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Fichier invalide. Uploader une image JPG ou PNG."); return;
    }
    setError(null); setResult(null);
    setPreview(URL.createObjectURL(file));
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: DetectionResult = await res.json();
      data.timestamp = Date.now();
      setResult(data); onResult(data);
    } catch (e: unknown) {
      if (e instanceof Error) setError(e.message);
    } finally { setLoading(false); }
  }, [onResult]);

  const drawBboxes = useCallback(() => {
    const img = imgRef.current; const canvas = canvasRef.current;
    if (!img || !canvas || !result?.success) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    canvas.width = img.offsetWidth; canvas.height = img.offsetHeight;
    const sx = img.offsetWidth / img.naturalWidth;
    const sy = img.offsetHeight / img.naturalHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    result.faces.forEach((face) => {
      const x = face.bbox.x * sx, y = face.bbox.y * sy;
      const w = face.bbox.w * sx, h = face.bbox.h * sy;
      const color = face.emotion === "happy" ? "#22c55e"
        : face.emotion === "angry" ? "#ef4444"
        : face.emotion === "sad"   ? "#3b82f6" : "#a855f7";
      ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
      const label = `${face.emoji} ${face.emotion_fr} ${face.confidence.toFixed(0)}%`;
      ctx.font = "bold 13px sans-serif";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = color; ctx.fillRect(x, y - 24, tw + 12, 22);
      ctx.fillStyle = "#fff"; ctx.fillText(label, x + 6, y - 7);
    });
  }, [result]);

  const face = result?.faces?.[0];

  return (
    <div className="detector-card">
      <div className={`drop-zone ${dragOver ? "drag-over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) processFile(f); }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}>
        {preview ? (
          <div className="preview-wrapper">
            <img ref={imgRef} src={preview} alt="preview" className="preview-img" onLoad={drawBboxes} />
            <canvas ref={canvasRef} className="overlay-canvas" />
          </div>
        ) : (
          <div className="drop-placeholder">
            <div className="drop-icon">⬆</div>
            <div className="drop-text">Glisser-déposer une image ici</div>
            <div className="drop-sub">ou cliquer pour sélectionner</div>
          </div>
        )}
      </div>

      <input ref={inputRef} type="file" accept="image/*" className="hidden-input"
        onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])} />

      {preview && (
        <button className="btn-main btn-start" onClick={() => inputRef.current?.click()} style={{ marginTop: 8 }}>
          Changer d'image
        </button>
      )}

      {loading && <div className="loading-row"><div className="spinner" /><span>Analyse en cours…</span></div>}
      {error && <div className="alert alert-error"><strong>Erreur :</strong> {error}</div>}

      {face && !loading && (
        <div className="result-section">
          <div className="result-main">
            <span className="result-emoji">{face.emoji}</span>
            <div>
              <div className="result-emotion">{face.emotion_fr}</div>
              <div className="result-confidence">{face.confidence.toFixed(1)}% de confiance</div>
            </div>
            {result?.processing_time_ms && <div className="result-latency">{result.processing_time_ms} ms</div>}
          </div>
          {result && result.faces_count > 1 && <div className="faces-count">{result.faces_count} visages détectés</div>}
          <EmotionBars probabilities={face.probabilities} dominant={face.emotion} />
        </div>
      )}

      {result && !result.success && !loading && (
        <div className="alert alert-warn">{result.error ?? "Aucun visage détecté dans cette image."}</div>
      )}
    </div>
  );
}
