export const severityConfig = {
  mild: {
    label: 'Mild',
    color: '#FFD700',
    textColor: '#6b5600',
    icon: '🟡'
  },
  moderate: {
    label: 'Moderate',
    color: '#FFA500',
    textColor: '#6a3f00',
    icon: '🟠'
  },
  severe: {
    label: 'Severe',
    color: '#E74C3C',
    textColor: '#ffffff',
    icon: '🔴'
  },
  unknown: {
    label: 'Unknown',
    color: '#7F8C8D',
    textColor: '#ffffff',
    icon: '⚪'
  }
};

export const getSeverityKey = (severity) => {
  const value = String(severity || '').toLowerCase();
  if (value.includes('severe')) return 'severe';
  if (value.includes('moderate')) return 'moderate';
  if (value.includes('mild') || value.includes('early')) return 'mild';
  return 'unknown';
};

export const getSeverityInfo = (severity) => {
  const key = getSeverityKey(severity);
  return severityConfig[key];
};
