/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        page: '#FAFAF8',
        panel: '#F5F5F0',
        textPrimary: '#2C3E50',
        textSecondary: '#7F8C8D',
        borderSoft: '#E8E8E8',
        severityMild: '#FFD700',
        severityModerate: '#FFA500',
        severitySevere: '#E74C3C'
      },
      boxShadow: {
        soft: '0 20px 50px rgba(44, 62, 80, 0.08)'
      },
      backgroundImage: {
        'orchard-radial': 'radial-gradient(circle at top left, rgba(255, 215, 0, 0.16), transparent 32%), radial-gradient(circle at top right, rgba(231, 76, 60, 0.14), transparent 26%), radial-gradient(circle at bottom right, rgba(52, 152, 219, 0.08), transparent 30%)'
      }
    }
  },
  plugins: []
};
