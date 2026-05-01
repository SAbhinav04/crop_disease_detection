export const severityConfig = {
  high: {
    label: 'High',
    color: '#E74C3C',
    textColor: '#ffffff',
    icon: '🔴'
  },
  medium: {
    label: 'Medium',
    color: '#FFA500',
    textColor: '#6a3f00',
    icon: '🟠'
  },
  low: {
    label: 'Low',
    color: '#FFD700',
    textColor: '#6b5600',
    icon: '🟡'
  },
  unknown: {
    label: 'Unknown',
    color: '#7F8C8D',
    textColor: '#ffffff',
    icon: '⚪'
  }
};

export const getSeverityKey = (severity) => {
  // If a numeric confidence (0-1 or 0-100) is passed, map ranges to low/medium/high
  const num = Number(severity);
  if (!Number.isNaN(num)) {
    let n = num;
    if (n >= 0 && n <= 1) n = n * 100; // convert fraction to percent
    if (n <= 33) return 'low';
    if (n <= 66) return 'medium';
    if (n <= 100) return 'high';
    return 'unknown';
  }

  const value = String(severity || '').toLowerCase();
  if (value.includes('severe') || value.includes('high')) return 'high';
  if (value.includes('moderate') || value.includes('medium')) return 'medium';
  if (value.includes('mild') || value.includes('low') || value.includes('early') || value.includes('normal') || value.includes('healthy')) return 'low';
  return 'unknown';
};

export const getSeverityInfo = (severity) => {
  const key = getSeverityKey(severity);
  return severityConfig[key] || severityConfig.unknown;
};
