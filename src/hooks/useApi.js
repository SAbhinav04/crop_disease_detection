import { fetchHistory, fetchRemedy, fetchTts, getReadableError, predictDisease } from '../utils/api';

export function useApi() {
  return {
    predictDisease,
    fetchRemedy,
    fetchHistory,
    fetchTts,
    getReadableError
  };
}
