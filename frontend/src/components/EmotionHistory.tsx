import { BarChart, Bar, Cell, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import type { HistoryEntry } from "../App";

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
const ALL_EMOTIONS = ["happy","neutral","sad","angry","surprised","fearful","disgusted"];

type Props = { history: HistoryEntry[] };

export default function EmotionHistory({ history }: Props) {
  const distribution = ALL_EMOTIONS.map((e) => ({
    emotion: e, label: LABELS_FR[e], emoji: EMOJIS[e],
    count: history.filter((h) => h.emotion === e).length,
    pct: history.length > 0
      ? Math.round(history.filter((h) => h.emotion === e).length / history.length * 100) : 0,
  }));

  const timeline = history.slice(0, 30).reverse().map((h, i) => ({
    idx: i + 1,
    confidence: Math.round(h.confidence),
    emotion: h.emotion, label: LABELS_FR[h.emotion], emoji: EMOJIS[h.emotion],
  }));

  const dominant = distribution.reduce((a, b) => b.count > a.count ? b : a, distribution[0]);
  const distFiltered = distribution.filter((d) => d.count > 0);

  return (
    <div className="history-panel">
      <div className="history-header">
        <span className="history-title">Historique</span>
        <span className="history-count">{history.length} détections</span>
      </div>

      {history.length === 0 ? (
        <div className="history-empty">Lancez la détection pour voir l'historique ici.</div>
      ) : (
        <>
          <div className="dominant-card">
            <div className="dominant-label">Émotion dominante</div>
            <div className="dominant-value">
              <span>{EMOJIS[dominant.emotion]}</span>
              <span>{LABELS_FR[dominant.emotion]}</span>
              <span className="dominant-pct">{dominant.pct}%</span>
            </div>
          </div>

          {distFiltered.length > 0 && (
            <div className="chart-section">
              <div className="chart-label">Distribution</div>
              <ResponsiveContainer width="100%" height={130}>
                <BarChart data={distFiltered} margin={{ top: 4, right: 4, left: -20, bottom: 4 }}>
                  <XAxis dataKey="emoji" tick={{ fontSize: 16 }} interval={0} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(val: number, _n, { payload }: any) => [`${val} (${payload.pct}%)`, payload.label]}
                    contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                  <Bar dataKey="count" radius={[4,4,0,0]}>
                    {distFiltered.map((e) => <Cell key={e.emotion} fill={COLORS[e.emotion]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {timeline.length > 2 && (
            <div className="chart-section">
              <div className="chart-label">Confiance dans le temps</div>
              <ResponsiveContainer width="100%" height={90}>
                <LineChart data={timeline} margin={{ top: 4, right: 4, left: -20, bottom: 4 }}>
                  <XAxis dataKey="idx" hide />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(val: number, _n, { payload }: any) => [`${val}%`, `${payload.emoji} ${payload.label}`]}
                    contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                  <Line type="monotone" dataKey="confidence" stroke="#a855f7" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="chart-label" style={{ marginTop: 12 }}>Récentes</div>
          <div className="history-list">
            {history.slice(0, 12).map((h, i) => (
              <div key={i} className="history-item">
                <span className="hi-emoji">{h.emoji}</span>
                <span className="hi-label">{h.emotion_fr}</span>
                <span className="hi-conf" style={{ color: COLORS[h.emotion] }}>{h.confidence.toFixed(0)}%</span>
                <span className="hi-time">
                  {new Date(h.timestamp).toLocaleTimeString("fr-FR", { hour:"2-digit", minute:"2-digit", second:"2-digit" })}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
