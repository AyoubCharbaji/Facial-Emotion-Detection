import { useState } from "react";
import WebcamDetector from "./components/WebcamDetector";
import ImageUpload from "./components/ImageUpload";
import EmotionHistory from "./components/EmotionHistory";

export type FaceResult = {
  emotion: string;
  emotion_fr: string;
  emoji: string;
  confidence: number;
  probabilities: Record<string, number>;
  bbox: { x: number; y: number; w: number; h: number;
          x_norm: number; y_norm: number; w_norm: number; h_norm: number; };
};

export type DetectionResult = {
  success: boolean;
  faces_count: number;
  dominant_emotion: string;
  faces: FaceResult[];
  processing_time_ms: number;
  timestamp?: number;
  error?: string;
};

export type HistoryEntry = {
  emotion: string;
  emotion_fr: string;
  emoji: string;
  confidence: number;
  timestamp: number;
};

type Tab = "webcam" | "upload";

export default function App() {
  const [tab, setTab] = useState<Tab>("webcam");
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const addToHistory = (result: DetectionResult) => {
    if (!result.success || !result.faces.length) return;
    const face = result.faces[0];
    setHistory((prev) => [
      {
        emotion: face.emotion,
        emotion_fr: face.emotion_fr,
        emoji: face.emoji,
        confidence: face.confidence,
        timestamp: result.timestamp ?? Date.now(),
      },
      ...prev.slice(0, 99),
    ]);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">◎</span>
            <span className="logo-text">EmotiScan</span>
          </div>
          <nav className="tab-nav">
            <button className={`tab-btn ${tab === "webcam" ? "active" : ""}`}
              onClick={() => setTab("webcam")}>
              Webcam live
            </button>
            <button className={`tab-btn ${tab === "upload" ? "active" : ""}`}
              onClick={() => setTab("upload")}>
              Uploader une image
            </button>
          </nav>
        </div>
      </header>

      <main className="app-main">
        <div className="left-panel">
          {tab === "webcam"
            ? <WebcamDetector onResult={addToHistory} />
            : <ImageUpload onResult={addToHistory} />}
        </div>
        <div className="right-panel">
          <EmotionHistory history={history} />
        </div>
      </main>
    </div>
  );
}
