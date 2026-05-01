const COLORS: Record<string, string> = {
  happy: "#22c55e", neutral: "#94a3b8", sad: "#3b82f6",
  angry: "#ef4444", surprised: "#f59e0b", fearful: "#8b5cf6", disgusted: "#f97316",
};
const LABELS_FR: Record<string, string> = {
  happy: "Joie", neutral: "Neutre", sad: "Tristesse",
  angry: "Colère", surprised: "Surprise", fearful: "Peur", disgusted: "Dégoût",
};
const EMOJIS: Record<string, string> = {
  happy: "😊", neutral: "😐", sad: "😢",
  angry: "😠", surprised: "😲", fearful: "😨", disgusted: "🤢",
};

type Props = {
  probabilities: Record<string, number>;
  dominant: string;
  order?: string[];
};

export default function EmotionBars({ probabilities, dominant, order }: Props) {
  const keys = order ?? Object.keys(probabilities);
  return (
    <div className="emotion-bars">
      <div className="bars-title">Probabilités par émotion</div>
      {keys.map((emotion) => {
        const val = probabilities[emotion] ?? 0;
        const isTop = emotion === dominant;
        const color = COLORS[emotion] ?? "#64748b";
        return (
          <div key={emotion} className={`bar-row ${isTop ? "bar-row-top" : ""}`}>
            <span className="bar-emoji">{EMOJIS[emotion]}</span>
            <span className="bar-label">{LABELS_FR[emotion] ?? emotion}</span>
            <div className="bar-track">
              <div className="bar-fill"
                style={{ width: `${val}%`, background: color, opacity: isTop ? 1 : 0.55 }} />
            </div>
            <span className="bar-value" style={{ color }}>{val.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
