import { useEffect, useRef, useState } from 'react';

const isImageFile = (file) => Boolean(file && file.type.startsWith('image/'));

/**
 * @param {{
 *   language: 'en' | 'kn',
 *   labels: Record<string, string>,
 *   selectedFile: File | null,
 *   previewUrl: string,
 *   loading: boolean,
 *   error: string | null,
 *   onFileSelect: (file: File | null) => void,
 *   onInvalidFile: (message: string) => void,
 *   onAnalyze: () => void
 * }} props
 */
export default function UploadSection({
  language,
  labels,
  selectedFile,
  previewUrl,
  loading,
  error,
  onFileSelect,
  onInvalidFile,
  onAnalyze
}) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    if (!isDragging) return undefined;

    const handleDrop = () => setIsDragging(false);
    window.addEventListener('dragend', handleDrop);
    window.addEventListener('drop', handleDrop);

    return () => {
      window.removeEventListener('dragend', handleDrop);
      window.removeEventListener('drop', handleDrop);
    };
  }, [isDragging]);

  const handleFile = (file) => {
    if (!file) return;
    if (!isImageFile(file)) {
      onInvalidFile(labels.errorFileType);
      onFileSelect(null);
      return;
    }
    onFileSelect(file);
  };

  const handleChange = (event) => {
    handleFile(event.target.files?.[0] || null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0] || null);
  };

  return (
    <section className="rounded-[28px] border border-borderSoft bg-white/95 p-5 shadow-soft backdrop-blur sm:p-7">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.28em] text-textSecondary/70">
            01
          </p>
          <h2 className="mt-2 text-xl font-bold text-textPrimary sm:text-2xl">{labels.uploadTitle}</h2>
          <p className="mt-2 text-sm text-textSecondary">{labels.uploadHint}</p>
        </div>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="hidden min-h-11 shrink-0 rounded-full bg-[#FFA500] px-4 py-2 text-sm font-semibold text-white shadow-[0_12px_24px_rgba(255,165,0,0.28)] transition-all duration-200 hover:-translate-y-0.5 hover:bg-[#ff9900] hover:shadow-[0_16px_30px_rgba(255,165,0,0.38)] focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 sm:inline-flex"
        >
          {labels.analyze}
        </button>
      </div>

      <input ref={inputRef} type="file" accept="image/*" onChange={handleChange} className="sr-only" />

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDrop={handleDrop}
        className={`group flex w-full flex-col items-center justify-center rounded-[24px] border-2 border-dashed px-4 py-8 text-center transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 ${
          isDragging ? 'border-severityModerate bg-amber-50 shadow-[0_16px_40px_rgba(255,165,0,0.16)]' : 'border-borderSoft bg-panel/70 hover:border-severityModerate hover:bg-amber-50/60 hover:shadow-[0_16px_40px_rgba(44,62,80,0.06)]'
        }`}
      >
        {previewUrl ? (
          <div className="w-full max-w-md">
            <img
              src={previewUrl}
              alt={selectedFile ? selectedFile.name : 'Selected crop image preview'}
              className="mx-auto max-h-64 w-full rounded-2xl object-cover shadow-md"
            />
          </div>
        ) : (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white text-2xl shadow-sm">
              📷
            </div>
            <p className="mt-4 text-base font-semibold text-textPrimary">
              {isDragging ? labels.dragActive : labels.readyToUpload}
            </p>
            <p className="mt-1 max-w-sm text-sm text-textSecondary">
              {language === 'kn'
                ? 'JPEG, PNG, ಅಥವಾ ಇತರ ಚಿತ್ರಗಳನ್ನು ಇಲ್ಲಿ ಎಳೆದು ಬಿಡಿ.'
                : 'Drop a JPEG, PNG, or other image file here.'}
            </p>
          </>
        )}
      </button>

      <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-h-11 rounded-2xl border border-borderSoft bg-page px-4 py-3 text-sm text-textPrimary">
          {selectedFile ? (
            <>
              <span className="font-semibold">{labels.selectedFile}:</span> {selectedFile.name}
            </>
          ) : (
            <div className="flex items-start gap-3 whitespace-pre-line text-textSecondary">
              <span aria-hidden="true" className="text-lg">
                🍃
              </span>
              <span>{labels.noPrediction}</span>
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onAnalyze}
          disabled={loading || !selectedFile}
          className="inline-flex min-h-11 items-center justify-center rounded-full bg-[#FFA500] px-5 py-3 text-sm font-semibold text-white shadow-[0_12px_24px_rgba(255,165,0,0.28)] transition-all duration-200 hover:-translate-y-0.5 hover:bg-[#ff9900] hover:shadow-[0_16px_30px_rgba(255,165,0,0.38)] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? (
            <span className="inline-flex items-center gap-2">
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
              {labels.analyzing}
            </span>
          ) : (
            labels.analyze
          )}
        </button>
      </div>

      {error ? (
        <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
          {error}
        </div>
      ) : null}
    </section>
  );
}
