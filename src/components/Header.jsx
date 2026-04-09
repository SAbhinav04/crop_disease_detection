/**
 * @param {{ language: 'en' | 'kn', onToggleLanguage: () => void, labels: { title: string, english: string, kannada: string } }} props
 */
export default function Header({ language, onToggleLanguage, labels }) {
  const isKannada = language === 'kn';

  return (
    <header className="sticky top-0 z-30 border-b border-borderSoft/80 bg-page/90 backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-textPrimary text-lg text-white shadow-soft">
            🌾
          </div>
          <div>
            <h1 className="text-xl font-extrabold tracking-tight text-textPrimary sm:text-2xl">
              {labels.title}
            </h1>
          </div>
        </div>

        <button
          type="button"
          onClick={onToggleLanguage}
          className="inline-flex min-h-11 items-center gap-2 rounded-full border border-borderSoft bg-white px-4 py-2 text-sm font-semibold text-textPrimary shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2"
          aria-label={`Switch language to ${isKannada ? 'English' : 'Kannada'}`}
        >
          <span aria-hidden="true" className="text-base">
            🌐
          </span>
          <span>{isKannada ? labels.kannada : labels.english}</span>
        </button>
      </div>
    </header>
  );
}
