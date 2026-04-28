import { useEffect, useMemo, useState } from 'react';
import './App.css';
import Login from './components/Login';
import Header from './components/Header';
import ResponsiveLayout from './components/ResponsiveLayout';
import UploadSection from './components/UploadSection';
import ResultsSection from './components/ResultsSection';
import AIAdviceSection from './components/AIAdviceSection';
import DiseaseHistory from './components/DiseaseHistory';
import AudioPlayer from './components/AudioPlayer';
import ResultCard from './components/ResultCard';
import { useApi } from './hooks/useApi';
import { uiText } from './utils/i18n';

const normalizeErrorMessage = (error, fallback) => {
  if (!error) return fallback;
  if (typeof error === 'string') return error;
  if (error.message && error.message.length < 180) return error.message;
  return fallback;
};

const createPreviewUrl = (file) => (file ? URL.createObjectURL(file) : '');

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
      const result = await api.fetchRemedy(prediction.disease);
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
    if (!block) return prediction?.disease || '';

    const parts = [
      prediction?.disease,
      block.cause,
      block.symptoms,
      Array.isArray(block.treatment_steps) ? block.treatment_steps.join('. ') : block.treatment_steps,
      block.prevention,
      block.fertilizer_recommendation
    ].filter(Boolean);

    return parts.join('. ');
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

  if (!isLoggedIn) {
    return (
      <Login
        initialLanguage={language}
        onLogin={({ lang }) => {
          setLanguage(lang);
          setIsLoggedIn(true);
        }}
      />
    );
  }

  return (
    <div className="app-root app-shell">
      <ResponsiveLayout
        header={<Header language={language} onToggleLanguage={handleLanguageToggle} labels={labels} />}
        left={
          <>
            {prediction ? <ResultCard result={prediction} /> : null}
            <UploadSection
              language={language}
              labels={labels}
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              loading={loadingPrediction}
              error={error}
              onFileSelect={handleFileSelect}
              onInvalidFile={handleInvalidFile}
              onAnalyze={handleAnalyze}
            />
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
            
          </>
        }
        right={
          <>
            <DiseaseHistory
              history={history}
              loading={loadingHistory}
              error={historyError}
              labels={labels}
              onRefresh={loadHistory}
            />
            <AudioPlayer
              src={audioSrc}
              loading={loadingAudio}
              language={language}
              labels={labels}
              playNonce={audioPlayNonce}
            />
          </>
          
        }
      />
    </div>
  );
}