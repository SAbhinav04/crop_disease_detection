import { useEffect, useRef, useState } from 'react';

/**
 * @param {{
 *   src: string | null,
 *   loading: boolean,
 *   language: 'en' | 'kn',
 *   labels: Record<string, string>,
 *   playNonce?: number
 * }} props
 */
export default function AudioPlayer({ src, loading, language, labels, playNonce = 0 }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setPlaying(false);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.load();
    }
  }, [src]);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  useEffect(() => {
    if (!audioRef.current || !src) return;
    const tryAutoPlay = async () => {
      try {
        await audioRef.current.play();
        setPlaying(true);
      } catch {
        setPlaying(false);
      }
    };
    tryAutoPlay();
  }, [src, playNonce]);

  const togglePlayback = async () => {
    if (!audioRef.current || !src) return;

    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
      return;
    }

    try {
      await audioRef.current.play();
      setPlaying(true);
    } catch {
      setPlaying(false);
    }
  };

  return (
    <section className="rounded-[28px] border border-borderSoft bg-white/65 p-5 shadow-soft backdrop-blur sm:p-7">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.28em] text-textSecondary/70">04</p>
          <h2 className="mt-2 text-xl font-bold text-textPrimary sm:text-2xl">{labels.hearKannada}</h2>
        </div>
        {loading ? (
          <span className="inline-flex items-center gap-2 rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-800">
            <span className="h-3 w-3 animate-spin rounded-full border-2 border-amber-500 border-t-transparent" />
            {labels.generatingAudio}
          </span>
        ) : null}
      </div>

      <div className="mt-5 rounded-[22px] bg-page p-4">
        <button
          type="button"
          onClick={togglePlayback}
          disabled={!src || loading}
          className="inline-flex min-h-11 w-full items-center justify-center rounded-full bg-textPrimary px-4 py-3 text-sm font-semibold text-white transition-all duration-200 hover:-translate-y-0.5 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {playing ? `✓ ${labels.playing}` : `🔊 ${labels.playKannada || 'Play in Kannada'}`}
        </button>
        <p className="mt-3 text-center text-sm text-textSecondary">
          {src ? (playing ? labels.playing : labels.readyToPlay) : labels.requestAudio}
        </p>
        {src ? (
          <audio
            ref={audioRef}
            src={src || undefined}
            disableRemotePlayback
            playsInline
            preload="metadata"
            onEnded={() => setPlaying(false)}
            onPause={() => setPlaying(false)}
            onPlay={() => setPlaying(true)}
            className="hidden"
          />
        ) : null}
      </div>
    </section>
  );
}