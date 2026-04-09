import { adviceLabels } from '../utils/i18n';

const toListItems = (value) => {
  if (!value) return [];
  if (Array.isArray(value)) return value.filter(Boolean);
  return String(value)
    .split(/\n|•|-/)
    .map((item) => item.trim())
    .filter(Boolean);
};

const renderValue = (value) => {
  const items = toListItems(value);
  if (!items.length) {
    return <p className="text-sm text-textSecondary">-</p>;
  }

  if (items.length === 1) {
    return <p className="text-sm leading-6 text-textPrimary">{items[0]}</p>;
  }

  return (
    <ul className="space-y-2 text-sm leading-6 text-textPrimary">
      {items.map((item) => (
        <li key={item} className="flex gap-2">
          <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-severityModerate" />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
};

/**
 * @param {{
 *   remedy: { english?: Record<string, unknown>, kannada?: Record<string, unknown> } | null,
 *   language: 'en' | 'kn',
 *   isOpen: boolean,
 *   onToggle: () => void,
 *   onSwitchLanguage: (language: 'en' | 'kn') => void
 * }} props
 */
export default function AIAdviceSection({ remedy, language, isOpen, onToggle, onSwitchLanguage }) {
  const activeLanguage = language === 'kn' ? 'kn' : 'en';
  const block = remedy?.[activeLanguage] || remedy?.english || remedy?.kannada || null;
  const labels = adviceLabels[activeLanguage];
  const ui = activeLanguage === 'kn'
    ? {
        title: 'AI ಸಲಹೆ',
        collapse: 'ಕುಗ್ಗಿಸಿ',
        expand: 'ವಿಸ್ತರಿಸಿ',
        english: 'English',
        kannada: 'ಕನ್ನಡ',
        noAdvice: "'Get AI Advice' ಒತ್ತಿ ಪರಿಹಾರಗಳನ್ನು ನೋಡಿ.",
        emptyHint: 'ಸಲಹೆ ಲೋಡ್ ಆಗಿಲ್ಲ.'
      }
    : {
        title: 'AI Advice',
        collapse: 'Collapse',
        expand: 'Expand',
        english: 'English',
        kannada: 'ಕನ್ನಡ',
        noAdvice: "Click 'Get AI Advice' to see remedies.",
        emptyHint: 'No advice loaded yet.'
      };

  return (
    <section className="rounded-[28px] border border-borderSoft bg-white/95 p-5 shadow-soft backdrop-blur sm:p-7">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.28em] text-textSecondary/70">03</p>
          <h2 className="mt-2 text-xl font-bold text-textPrimary sm:text-2xl">{ui.title}</h2>
        </div>
        <button
          type="button"
          onClick={onToggle}
          className="rounded-full border border-borderSoft bg-page px-4 py-2 text-sm font-semibold text-textPrimary transition-all duration-200 hover:border-orange-300 hover:bg-amber-50"
        >
          {isOpen ? ui.collapse : ui.expand}
        </button>
      </div>

      <div className="mt-4 flex gap-2 border-b border-borderSoft pb-1">
        <button
          type="button"
          onClick={() => onSwitchLanguage('en')}
          className={`border-b-2 px-4 py-2 text-sm font-semibold transition-all duration-200 ${
            activeLanguage === 'en'
              ? 'border-orange-400 text-textPrimary'
              : 'border-transparent text-textSecondary hover:text-textPrimary'
          }`}
        >
          {ui.english}
        </button>
        <button
          type="button"
          onClick={() => onSwitchLanguage('kn')}
          className={`border-b-2 px-4 py-2 text-sm font-semibold transition-all duration-200 ${
            activeLanguage === 'kn'
              ? 'border-orange-400 text-textPrimary'
              : 'border-transparent text-textSecondary hover:text-textPrimary'
          }`}
        >
          {ui.kannada}
        </button>
      </div>

      {isOpen ? (
        block ? (
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <div className="space-y-4 rounded-[22px] bg-page p-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-textSecondary">
                  {labels.cause}
                </p>
                <div className="mt-2">{renderValue(block.cause)}</div>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-textSecondary">
                  {labels.symptoms}
                </p>
                <div className="mt-2">{renderValue(block.symptoms)}</div>
              </div>
            </div>

            <div className="space-y-4 rounded-[22px] bg-page p-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-textSecondary">
                  {labels.treatment_steps}
                </p>
                <div className="mt-2">{renderValue(block.treatment_steps)}</div>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-textSecondary">
                  {labels.prevention}
                </p>
                <div className="mt-2">{renderValue(block.prevention)}</div>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-textSecondary">
                  {labels.fertilizer_recommendation}
                </p>
                <div className="mt-2">{renderValue(block.fertilizer_recommendation)}</div>
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-4 rounded-[22px] border border-dashed border-borderSoft bg-page p-4 text-sm text-textSecondary">
            <p className="flex items-start gap-3">
              <span aria-hidden="true" className="text-lg">
                🪴
              </span>
              <span>{ui.noAdvice}</span>
            </p>
            <p className="mt-2 text-xs text-textSecondary/80">{ui.emptyHint}</p>
          </div>
        )
      ) : null}
    </section>
  );
}
