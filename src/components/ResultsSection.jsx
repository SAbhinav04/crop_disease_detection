import { useState } from 'react';
import { getSeverityInfo } from '../utils/colors';

/**
 * @param {{
 *   prediction: { disease?: string, confidence?: number, severity?: string, crop?: string } | null,
 *   labels: Record<string, string>,
 *   onRequestAdvice: () => void,
 *   onRequestAudio: () => void,
 *   adviceLoading: boolean,
 *   audioLoading: boolean
 * }} props
 */
export default function ResultsSection({
  prediction,
  labels,
  onRequestAdvice,
  onRequestAudio,
  adviceLoading,
  audioLoading
}) {
  const [feedback, setFeedback] = useState('');

  if (!prediction) return null;

  const severityInfo = getSeverityInfo(prediction.severity);
  const confidence = Number(prediction.confidence || 0);
  const severityBadgeLabel = `${severityInfo.icon} ${severityInfo.label.toUpperCase()}`;

  const buildSummary = () => {
    const parts = [
      `${labels.disease}: ${prediction.disease || 'Unknown'}`,
      `${labels.confidence}: ${Math.round(confidence)}%`,
      `${labels.severity}: ${severityInfo.label}`,
      `${labels.crop}: ${prediction.crop || 'Unknown'}`
    ];

    return parts.join('\n');
  };

  const flashFeedback = (message) => {
    setFeedback(message);
    window.clearTimeout(flashFeedback.timeoutId);
    flashFeedback.timeoutId = window.setTimeout(() => setFeedback(''), 2200);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(buildSummary());
      flashFeedback(labels.copied);
    } catch {
      flashFeedback(labels.copyFailed);
    }
  };

  const handleShare = async () => {
    const text = `${buildSummary()}\n\n${window.location.href}`;
    const shareData = {
      title: labels.results,
      text,
      url: window.location.href
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
        return;
      }

      const whatsappUrl = `https://wa.me/?text=${encodeURIComponent(text)}`;
      window.open(whatsappUrl, '_blank', 'noopener,noreferrer');
    } catch {
      flashFeedback(labels.shareFailed);
    }
  };

  return (
    <section className="result-reveal rounded-[28px] border border-borderSoft bg-white/95 p-5 shadow-soft backdrop-blur sm:p-7">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.28em] text-textSecondary/70">02</p>
          <h2 className="mt-2 text-xl font-bold text-textPrimary sm:text-2xl">{labels.results}</h2>
        </div>
        <div
          className="rounded-full px-4 py-2 text-sm font-semibold shadow-sm"
          style={{ backgroundColor: severityInfo.color, color: severityInfo.textColor }}
        >
          {severityBadgeLabel}
        </div>
      </div>

      <div className="rounded-[22px] bg-page p-4 sm:p-5">
        <div className="space-y-2">
          <p className="text-sm uppercase tracking-[0.18em] text-textSecondary">{labels.disease}</p>
          <p className="text-2xl font-extrabold text-textPrimary sm:text-3xl">
            {prediction.disease || 'Unknown'}
          </p>
        </div>

        <div className="mt-5 space-y-2">
          <div className="flex items-center justify-between text-sm font-semibold text-textPrimary">
            <span>{labels.confidence}</span>
            <span title={`${labels.confidenceTooltip || 'Exact confidence: '}${Math.round(confidence)}%`}>
              {Math.round(confidence)}%
            </span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-white ring-1 ring-inset ring-borderSoft" title={`${labels.confidenceTooltip || 'Exact confidence: '}${Math.round(confidence)}%`}>
            <div
              className="h-full rounded-full bg-gradient-to-r from-severityMild via-severityModerate to-severitySevere transition-all duration-300"
              style={{ width: `${Math.min(Math.max(confidence, 0), 100)}%` }}
            />
          </div>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          <div className="rounded-2xl border border-borderSoft bg-white p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">{labels.severity}</p>
            <p className="mt-1 text-base font-bold text-textPrimary">{severityInfo.label}</p>
          </div>
          <div className="rounded-2xl border border-borderSoft bg-white p-4">
            <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">{labels.crop}</p>
            <p className="mt-1 text-base font-bold text-textPrimary">{prediction.crop || 'Unknown'}</p>
          </div>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={onRequestAdvice}
            className="inline-flex min-h-11 items-center justify-center rounded-full bg-textPrimary px-4 py-3 text-sm font-semibold text-white shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:scale-[1.01] hover:bg-slate-800 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={adviceLoading}
          >
            {adviceLoading ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                {labels.getAdvice}
              </span>
            ) : (
              labels.getAdvice
            )}
          </button>
          <button
            type="button"
            onClick={onRequestAudio}
            className="inline-flex min-h-11 items-center justify-center rounded-full border border-borderSoft bg-white px-4 py-3 text-sm font-semibold text-textPrimary shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:scale-[1.01] hover:border-orange-300 hover:bg-amber-50 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={audioLoading}
          >
            {audioLoading ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                {labels.generatingAudio}
              </span>
            ) : (
              `🔊 ${labels.hearKannada}`
            )}
          </button>
        </div>

        <div className="mt-3 flex flex-col gap-3 sm:flex-row">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex min-h-11 items-center justify-center rounded-full border border-borderSoft bg-white px-4 py-3 text-sm font-semibold text-textPrimary shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:scale-[1.01] hover:border-orange-300 hover:shadow-md"
          >
            {labels.copyResult}
          </button>
          <button
            type="button"
            onClick={handleShare}
            className="inline-flex min-h-11 items-center justify-center rounded-full border border-borderSoft bg-white px-4 py-3 text-sm font-semibold text-textPrimary shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:scale-[1.01] hover:border-orange-300 hover:shadow-md"
          >
            {labels.shareWhatsApp}
          </button>
        </div>

        {feedback ? (
          <p className="mt-3 text-sm font-medium text-textSecondary">{feedback}</p>
        ) : null}
      </div>
    </section>
  );
}
