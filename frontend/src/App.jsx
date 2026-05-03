import { useEffect, useMemo, useState } from 'react';
import './App.css';
import Login from './components/Login';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import UploadSection from './components/UploadSection';
import ResultsSection from './components/ResultsSection';
import AIAdviceSection from './components/AIAdviceSection';
import AudioPlayer from './components/AudioPlayer';
import HowItWorks from './components/HowItWorks';
import Features from './components/Features';
import Footer from './components/Footer';
import { useApi } from './hooks/useApi';
import { uiText } from './utils/i18n';

const MAX_TTS_WORDS_BY_LANGUAGE = {
  en: 78,
  kn: 60
};

const normalizeErrorMessage = (error, fallback) => {
  if (!error) return fallback;
  if (typeof error === 'string') return error;
  if (error.message && error.message.length < 180) return error.message;
  return fallback;
};

const createPreviewUrl = (file) => (file ? URL.createObjectURL(file) : '');

const normalizeSentence = (value) => {
  if (!value) return '';

  const text = Array.isArray(value) ? value.filter(Boolean).join('. ') : String(value);
  return text
    .replace(/\s+/g, ' ')
    .replace(/[•\-*]+\s*/g, ' ')
    .trim();
};

const clampWords = (value, maxWords) => {
  const words = String(value || '').split(/\s+/).filter(Boolean);
  if (words.length <= maxWords) return words.join(' ');
  return `${words.slice(0, maxWords).join(' ')}...`;
};

const compactField = (value, maxWords) => clampWords(normalizeSentence(value), maxWords);

const buildCompactTtsSummary = ({ disease, crop, block, language }) => {
  const isKannada = language === 'kn';
  const maxWords = MAX_TTS_WORDS_BY_LANGUAGE[language] || MAX_TTS_WORDS_BY_LANGUAGE.en;

  const labels = isKannada
    ? {
      disease: 'ರೋಗ',
      crop: 'ಬೆಳೆ',
      cause: 'ಕಾರಣ',
      prevention: 'ತಡೆ',
      cure: 'ಚಿಕಿತ್ಸೆ',
      missing: 'ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ'
    }
    : {
      disease: 'Disease',
      crop: 'Crop',
      cause: 'Cause',
      prevention: 'Prevention',
      cure: 'Treatment',
      missing: 'Not available'
    };

  const diseaseText = compactField(disease, 10) || 'Unknown';
  const causeText = compactField(block?.cause, 20) || labels.missing;
  const remedySource = block?.treatment_steps || block?.prevention;
  const remedyText = compactField(remedySource, 24) || labels.missing;
  const preventionText = compactField(block?.prevention, 14);
  const cropText = compactField(crop, 6);

  // Priority order for 30s TTS: disease -> cause -> remedy. Then add optional context if space allows.
  const mandatoryParts = [
    `${labels.disease}: ${diseaseText}`,
    `${labels.cause}: ${causeText}`,
    `${labels.cure}: ${remedyText}`
  ];

  const optionalParts = [
    preventionText ? `${labels.prevention}: ${preventionText}` : '',
    cropText ? `${labels.crop}: ${cropText}` : ''
  ].filter(Boolean);

  const mandatorySummary = clampWords(mandatoryParts.join('. '), maxWords);
  const withOptional = clampWords(`${mandatorySummary}. ${optionalParts.join('. ')}`, maxWords);

  return optionalParts.length ? withOptional : mandatorySummary;
};

export default function App() {
  const api = useApi();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [language, setLanguage] = useState('en');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [remedy, setRemedy] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loadingAdvice, setLoadingAdvice] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadingAudio, setLoadingAudio] = useState(false);
  const [adviceOpen, setAdviceOpen] = useState(false);
  const [audioSrc, setAudioSrc] = useState(null);
  const [audioPlayNonce, setAudioPlayNonce] = useState(0);
  const [historyError, setHistoryError] = useState(null);

  const labels = uiText[language];

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (audioSrc) URL.revokeObjectURL(audioSrc);
    };
  }, [audioSrc, previewUrl]);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('');
      return undefined;
    }

    const nextUrl = createPreviewUrl(selectedFile);
    setPreviewUrl(nextUrl);

    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [selectedFile]);

  const handleLanguageToggle = () => {
    setLanguage((current) => (current === 'en' ? 'kn' : 'en'));
  };

  const handleFileSelect = (file) => {
    setError(null);
    setHistoryError(null);
    setPrediction(null);
    setRemedy(null);
    setAudioSrc(null);
    setAdviceOpen(false);
    setSelectedFile(file);
  };

  const handleInvalidFile = (message) => {
    setError(message);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError(labels.errorFileType);
      return;
    }

    setError(null);
    setLoadingPrediction(true);
    setPrediction(null);
    setRemedy(null);
    setAudioSrc(null);
    setAdviceOpen(false);

    try {
      const result = await api.predictDisease(selectedFile);
      setPrediction(result);
      await loadHistory();
    } catch (caughtError) {
      setError(normalizeErrorMessage(caughtError, labels.errorGeneric));
    } finally {
      setLoadingPrediction(false);
    }
  };

  const loadHistory = async () => {
    setLoadingHistory(true);
    setHistoryError(null);

    try {
      const result = await api.fetchHistory();
      setHistory(Array.isArray(result) ? result : result?.history || []);
    } catch (caughtError) {
      setHistoryError(normalizeErrorMessage(caughtError, labels.errorGeneric));
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleRequestAdvice = async () => {
    if (!prediction?.disease) return;

    setLoadingAdvice(true);
    setError(null);

    try {
      const result = await api.fetchRemedy(prediction.disease, prediction.crop);
      setRemedy(result);
      setAdviceOpen(true);
    } catch (caughtError) {
      setError(normalizeErrorMessage(caughtError, labels.errorGeneric));
    } finally {
      setLoadingAdvice(false);
    }
  };

  const buildSpeechText = useMemo(() => {
    if (!remedy) return '';

    const remedyKey = language === 'kn' ? 'kannada' : 'english';
    const block = remedy[remedyKey] || remedy.english || remedy.kannada;
    if (!block) return clampWords(prediction?.disease || '');

    return buildCompactTtsSummary({
      disease: prediction?.disease,
      crop: prediction?.crop,
      block,
      language
    });
  }, [language, prediction, remedy]);

  const handleRequestAudio = async () => {
    if (!buildSpeechText) {
      setError(labels.requestAudio);
      return;
    }

    setLoadingAudio(true);
    setError(null);

    try {
      const audioBlob = await api.fetchTts({ text: buildSpeechText, language: 'kn' });
      const nextAudioUrl = URL.createObjectURL(audioBlob);
      setAudioSrc((current) => {
        if (current) URL.revokeObjectURL(current);
        return nextAudioUrl;
      });
      setAudioPlayNonce((value) => value + 1);
    } catch (caughtError) {
      setError(normalizeErrorMessage(caughtError, labels.errorGeneric));
    } finally {
      setLoadingAudio(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setPrediction(null);
    setError(null);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setSelectedFile(null);
    setPreviewUrl('');
    setPrediction(null);
    setRemedy(null);
    setAudioSrc(null);
    setAdviceOpen(false);
    setError(null);
    setHistoryError(null);
  };

  if (!isLoggedIn) {
    return (
      <Login
        initialLanguage={language}
        initialPhone="1234567890"
        onLogin={({ lang }) => {
          setLanguage(lang);
          setIsLoggedIn(true);
        }}
      />
    );
  }

  return (
    <div className="app-root">
      <Navbar language={language} setLanguage={setLanguage} labels={labels} onLogout={handleLogout} />

      <div className="section-container">

        <Hero labels={labels} />

        {!prediction && (
          <>
            <UploadSection
              labels={labels}
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              loading={loadingPrediction}
              error={error}
              onFileSelect={handleFileSelect}
              onAnalyze={handleAnalyze}
            />

            <HowItWorks labels={labels} />
          </>
        )}

        {prediction && (
          <>
            <ResultsSection
              prediction={prediction}
              labels={labels}
              onRequestAdvice={handleRequestAdvice}
              onRequestAudio={handleRequestAudio}
              adviceLoading={loadingAdvice}
              audioLoading={loadingAudio}
            />

            <AIAdviceSection
              remedy={remedy}
              language={language}
              isOpen={adviceOpen}
              onToggle={() => setAdviceOpen((current) => !current)}
              onSwitchLanguage={setLanguage}
            />

            <AudioPlayer
              src={audioSrc}
              loading={loadingAudio}
              language={language}
              labels={labels}
              playNonce={audioPlayNonce}
            />
          </>
        )}

        <Features labels={labels} />

        <Footer />

      </div>
    </div>
  );
}