import { useEffect, useRef, useState } from 'react';

const formatTime = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const total = Math.floor(seconds);
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, '0')}`;
};

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
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const controlLabels = language === 'kn'
    ? {
        play: 'ಆಡಿಸಿ',
        pause: 'ನಿಲ್ಲಿಸಿ',
        slider: 'ಆಡಿಯೋ ಸ್ಥಾನ'
      }
    : {
        play: 'Play',
        pause: 'Pause',
        slider: 'Audio position'
      };

  useEffect(() => {
    setPlaying(false);
    setDuration(0);
    setCurrentTime(0);
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

  useEffect(() => {
    if (!audioRef.current) return;

    const audio = audioRef.current;
    const onLoadedMetadata = () => {
      setDuration(audio.duration || 0);
    };
    const onTimeUpdate = () => {
      setCurrentTime(audio.currentTime || 0);
    };

    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('timeupdate', onTimeUpdate);

    return () => {
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('timeupdate', onTimeUpdate);
    };
  }, [src]);

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

  const onSeek = (event) => {
    if (!audioRef.current) return;
    const value = Number(event.target.value);
    if (!Number.isFinite(value)) return;

    audioRef.current.currentTime = value;
    setCurrentTime(value);
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
        <div className="grid gap-2 sm:grid-cols-2">
          <button
            type="button"
            onClick={togglePlayback}
            disabled={!src || loading || playing}
            className="inline-flex min-h-11 w-full items-center justify-center rounded-full bg-textPrimary px-4 py-3 text-sm font-semibold text-white transition-all duration-200 hover:-translate-y-0.5 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {`▶ ${controlLabels.play}`}
          </button>
          <button
            type="button"
            onClick={togglePlayback}
            disabled={!src || loading || !playing}
            className="inline-flex min-h-11 w-full items-center justify-center rounded-full border border-borderSoft bg-white px-4 py-3 text-sm font-semibold text-textPrimary transition-all duration-200 hover:-translate-y-0.5 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {`⏸ ${controlLabels.pause}`}
          </button>
        </div>

        <div className="mt-4">
          <label htmlFor="audio-seek" className="sr-only">
            {controlLabels.slider}
          </label>
          <input
            id="audio-seek"
            type="range"
            min={0}
            max={Math.max(duration, 0)}
            step="0.1"
            value={Math.min(currentTime, duration || 0)}
            onChange={onSeek}
            disabled={!src || loading || duration <= 0}
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          />
          <div className="mt-1 flex items-center justify-between text-xs text-textSecondary">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

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
            onEnded={() => {
              setPlaying(false);
              setCurrentTime(0);
            }}
            onPause={() => setPlaying(false)}
            onPlay={() => setPlaying(true)}
            className="hidden"
          />
        ) : null}
      </div>
    </section>
  );
}