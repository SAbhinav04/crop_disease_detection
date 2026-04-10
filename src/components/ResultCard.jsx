import React from 'react';

const severityColors = {
  Early: '#FFD700',
  Moderate: '#FFA500',
  Severe: '#E74C3C'
};

function clampConfidence(value) {
  const num = Number(value);
  if (Number.isNaN(num)) return 0;
  const percent = num <= 1 ? num * 100 : num;
  return Math.min(100, Math.max(0, percent));
}

export default function ResultCard({ result }) {
  const confidence = clampConfidence(result?.confidence);
  const severity = result?.severity || 'Early';
  const badgeColor = severityColors[severity] || severityColors.Early;

  return (
    <section className="w-full rounded-2xl border border-amber-200 bg-white p-4 shadow-sm sm:p-6">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Detected Disease</p>
            <h3 className="text-lg font-bold text-slate-900 sm:text-xl">{result?.disease || 'Unknown_Disease'}</h3>
          </div>

          <span
            className="inline-flex w-fit rounded-full px-3 py-1 text-xs font-semibold text-slate-900"
            style={{ backgroundColor: badgeColor }}
          >
            {severity}
          </span>
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between text-sm text-slate-700">
            <span>Confidence</span>
            <span className="font-semibold">{confidence.toFixed(0)}%</span>
          </div>
          <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full rounded-full bg-gradient-to-r from-amber-400 via-orange-500 to-red-500 transition-all duration-500"
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>

        <div className="rounded-xl bg-slate-50 px-3 py-2 text-sm text-slate-700">
          <span className="font-medium text-slate-900">Crop:</span> {result?.crop || 'Unknown'}
        </div>
      </div>
    </section>
  );
}
