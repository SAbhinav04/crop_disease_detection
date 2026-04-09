const getApiBaseUrl = () => {
  const value = import.meta.env.VITE_API_URL;
  return value ? value.replace(/\/$/, '') : '';
};

const buildUrl = (path) => `${getApiBaseUrl()}${path}`;

const parseResponse = async (response) => {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Request failed');
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }

  return response.blob();
};

const base64ToBlob = (base64Value, mimeType = 'audio/wav') => {
  const byteCharacters = atob(base64Value);
  const byteNumbers = new Array(byteCharacters.length);

  for (let index = 0; index < byteCharacters.length; index += 1) {
    byteNumbers[index] = byteCharacters.charCodeAt(index);
  }

  return new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
};

export const predictDisease = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(buildUrl('/predict'), {
    method: 'POST',
    body: formData
  });

  return parseResponse(response);
};

export const fetchRemedy = async (disease) => {
  const response = await fetch(buildUrl(`/remedy-llm?disease=${encodeURIComponent(disease)}`));
  return parseResponse(response);
};

export const fetchHistory = async () => {
  const response = await fetch(buildUrl('/history'));
  return parseResponse(response);
};

export const fetchTts = async ({ text, language }) => {
  const response = await fetch(buildUrl('/tts'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text, language })
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Request failed');
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    const payload = await response.json();
    const base64Value = payload.audio_base64 || payload.audio || payload.data;

    if (typeof base64Value === 'string' && base64Value.length) {
      return base64ToBlob(base64Value, payload.mime_type || payload.mimeType || 'audio/wav');
    }

    throw new Error('Audio response did not include a file payload');
  }

  return response.blob();
};

export const getReadableError = (error) => {
  if (!error) return 'Something went wrong.';
  if (typeof error === 'string') return error;
  if (error.message && error.message.length < 200) return error.message;
  return 'Something went wrong.';
};
