import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { getSeverityInfo } from '../utils/colors';
import { useResponsive } from '../hooks/useResponsive';

const formatTimestamp = (value) => {
  if (!value) return 'Unknown';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);

  const today = new Date();
  const isSameDay =
    date.getFullYear() === today.getFullYear() &&
    date.getMonth() === today.getMonth() &&
    date.getDate() === today.getDate();

  const timeLabel = new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit'
  }).format(date);

  if (isSameDay) return `Today, ${timeLabel}`;

  const dateLabel = new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric'
  }).format(date);

  return `${dateLabel}, ${timeLabel}`;
};

const normalizeHistory = (history) =>
  (Array.isArray(history) ? history : [])
    .map((entry, index) => {
      const timestamp =
        entry.timestamp || entry.created_at || entry.createdAt || entry.date || entry.time || index;

      return {
        disease: entry.disease || entry.disease_name || entry.name || 'Unknown',
        confidence: Number(entry.confidence ?? entry.confidence_score ?? 0) * 100,
        severity: entry.severity || 'Unknown',
        crop: entry.crop || entry.crop_name || 'Unknown',
        timestamp,
        displayTimestamp: formatTimestamp(timestamp),
        sortTimestamp: Number.isNaN(new Date(timestamp).getTime()) ? index : new Date(timestamp).getTime()
      };
    })
    .sort((left, right) => left.sortTimestamp - right.sortTimestamp);

/**
 * @param {{
 *   history: Array<Record<string, unknown>>,
 *   loading: boolean,
 *   error: string | null,
 *   labels: Record<string, string>,
 *   onRefresh: () => void
 * }} props
 */
export default function DiseaseHistory({ history, loading, error, onRefresh, labels }) {
  const { isMobile } = useResponsive();
  const normalizedHistory = normalizeHistory(history);
  const chartData = normalizedHistory.map((entry) => ({
    ...entry,
    confidence: Number.isFinite(entry.confidence) ? entry.confidence : 0
  }));
  const recentPredictions = [...normalizedHistory].slice(-5).reverse();

  return (
    <section className="rounded-[28px] border border-borderSoft bg-white/65 p-5 shadow-soft backdrop-blur sm:p-7">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.28em] text-textSecondary/70">05</p>
          <h2 className="mt-2 text-xl font-bold text-textPrimary sm:text-2xl">{labels.diseaseHistory}</h2>
          <p className="mt-1 text-sm text-textSecondary">{labels.showHistoryGraph}</p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          aria-label="Refresh"
          className="rounded-full border border-borderSoft bg-page p-2 text-textPrimary transition-all duration-200 hover:border-green-300 hover:bg-green-50"
        >
          {loading ? (
            <span className="flex h-5 w-5 items-center justify-center">
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
            </span>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          )}
        </button>
      </div>

      <div className="mt-5 rounded-[22px] bg-page p-4">
        {loading ? (
          <p className="text-sm text-textSecondary">{labels.loadingHistory}</p>
        ) : error ? (
          <p className="text-sm text-red-600">{error === 'Failed to fetch' ? (labels.errorFetch || error) : error}</p>
        ) : chartData.length ? (
          <div className="h-[220px] w-full md:h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="#E8E8E8" />
                <XAxis
                  dataKey="displayTimestamp"
                  tick={{ fontSize: isMobile ? 10 : 12, fill: '#7F8C8D' }}
                  interval={isMobile ? 'preserveStartEnd' : 0}
                  minTickGap={18}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: isMobile ? 10 : 12, fill: '#7F8C8D' }}
                  width={32}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: '16px',
                    border: '1px solid #E8E8E8',
                    background: '#fff'
                  }}
                  formatter={(value, name, entry) => [`${Math.round(Number(value))}%`, 'Confidence']}
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#2C3E50"
                  strokeWidth={3}
                  dot={{ r: 4, fill: '#FFA500' }}
                  activeDot={{ r: 6, fill: '#E74C3C' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="flex items-start gap-3 rounded-2xl border border-dashed border-borderSoft bg-white p-4 text-sm text-textSecondary">
            <span aria-hidden="true" className="text-lg">
              🌿
            </span>
            <p className="whitespace-pre-line">{labels.noHistory}</p>
          </div>
        )}
      </div>

      <div className="mt-5">
        <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-textSecondary">
          {labels.lastFive}
        </h3>
        <div className="mt-3 grid gap-3">
          {recentPredictions.length ? (
            recentPredictions.map((entry) => {
              const severityInfo = getSeverityInfo(entry.severity);
              const severityLabel = `${severityInfo.icon} ${severityInfo.label.toUpperCase()}`;

              return (
                <article
                  key={`${entry.timestamp}-${entry.disease}`}
                  className="rounded-2xl border border-borderSoft bg-white p-4 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-base font-bold text-textPrimary">{entry.disease}</p>
                      <p className="mt-1 text-xs text-textSecondary">{entry.displayTimestamp}</p>
                    </div>
                    <span
                      className="rounded-full px-3 py-1 text-xs font-semibold"
                      style={{ backgroundColor: severityInfo.color, color: severityInfo.textColor }}
                    >
                      {severityLabel}
                    </span>
                  </div>
                  <p className="mt-3 text-sm text-textSecondary">{entry.crop}</p>
                </article>
              );
            })
          ) : (
            <p className="text-sm text-textSecondary">{labels.noHistoryList}</p>
          )}
        </div>
      </div>
    </section>
  );
}