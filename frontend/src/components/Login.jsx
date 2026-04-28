import { useState } from 'react';

const T = {
  en: {
    badge: 'AGRIVISION EDGE',
    headline: 'Crop disease detection built for Indian farmers',
    sub: 'Scan any leaf, get instant diagnosis and treatment advice in Kannada or English.',
    stat1: 'Detection accuracy', stat2: 'Crop diseases', stat3: 'Multilingual support',
    tag1: 'Instant diagnosis', tag2: 'AI-powered', tag3: 'Kannada support',
    formTitle: 'Welcome back',
    formSub: 'Enter your phone number to continue',
    labelPhone: 'PHONE NUMBER',
    labelOtp: 'ENTER OTP',
    sendOtp: 'Send OTP',
    sending: 'Sending...',
    otpHint: 'OTP sent to your number',
    verify: 'Verify & Continue',
    verifying: 'Verifying...',
    resend: 'Resend OTP',
    forFarmers: 'FOR REGISTERED FARMERS ONLY',
    helpCenter: 'Help center',
    support: 'Support',
    footer: '© 2024 AGRIVISION EDGE. EMPOWERING INDIAN AGRICULTURE.',
    privacy: 'PRIVACY POLICY',
    terms: 'TERMS OF SERVICE',
    contact: 'CONTACT SUPPORT',
    errInvalidPhone: 'Please enter a valid 10-digit phone number.',
    errOtpFailed: 'Incorrect OTP. Please try again.',
    errGeneric: 'Something went wrong. Please try again.',
    errTooMany: 'Too many attempts. Please try after some time.',
  },
  kn: {
    badge: 'ಅಗ್ರಿವಿಷನ್ ಎಡ್ಜ್',
    headline: 'ಭಾರತೀಯ ರೈತರಿಗಾಗಿ ಬೆಳೆ ರೋಗ ಪರಿಶೀಲನೆ',
    sub: 'ಯಾವುದೇ ಎಲೆ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ, ಕನ್ನಡ ಅಥವಾ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ತಕ್ಷಣ ರೋಗನಿರ್ಣಯ ಪಡೆಯಿರಿ.',
    stat1: 'ನಿಖರತೆ', stat2: 'ಬೆಳೆ ರೋಗಗಳು', stat3: 'ಬಹುಭಾಷಾ ಬೆಂಬಲ',
    tag1: 'ತ್ವರಿತ ರೋಗನಿರ್ಣಯ', tag2: 'AI ಆಧಾರಿತ', tag3: 'ಕನ್ನಡ ಬೆಂಬಲ',
    formTitle: 'ಸ್ವಾಗತ',
    formSub: 'ಮುಂದುವರಿಯಲು ನಿಮ್ಮ ಫೋನ್ ಸಂಖ್ಯೆ ನಮೂದಿಸಿ',
    labelPhone: 'ಫೋನ್ ಸಂಖ್ಯೆ',
    labelOtp: 'OTP ನಮೂದಿಸಿ',
    sendOtp: 'OTP ಕಳುಹಿಸಿ',
    sending: 'ಕಳುಹಿಸಲಾಗುತ್ತಿದೆ...',
    otpHint: 'OTP ನಿಮ್ಮ ಸಂಖ್ಯೆಗೆ ಕಳುಹಿಸಲಾಗಿದೆ',
    verify: 'ಪರಿಶೀಲಿಸಿ & ಮುಂದುವರಿಯಿರಿ',
    verifying: 'ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ...',
    resend: 'OTP ಮರು-ಕಳುಹಿಸಿ',
    forFarmers: 'ನೋಂದಾಯಿತ ರೈತರಿಗೆ ಮಾತ್ರ',
    helpCenter: 'ಸಹಾಯ ಕೇಂದ್ರ',
    support: 'ಬೆಂಬಲ',
    footer: '© 2024 ಅಗ್ರಿವಿಷನ್ ಎಡ್ಜ್. ಭಾರತೀಯ ಕೃಷಿಗೆ ಶಕ್ತಿ.',
    privacy: 'ಗೌಪ್ಯತಾ ನೀತಿ',
    terms: 'ಸೇವಾ ನಿಯಮಗಳು',
    contact: 'ಬೆಂಬಲ ಸಂಪರ್ಕಿಸಿ',
    errInvalidPhone: 'ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ 10-ಅಂಕಿಯ ಫೋನ್ ಸಂಖ್ಯೆ ನಮೂದಿಸಿ.',
    errOtpFailed: 'ತಪ್ಪಾದ OTP. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
    errGeneric: 'ಏನೋ ತಪ್ಪಾಗಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
    errTooMany: 'ಹಲವು ಬಾರಿ ಪ್ರಯತ್ನಿಸಲಾಗಿದೆ. ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಪ್ರಯತ್ನಿಸಿ.',
  }
};

// Replace this function with your actual OTP send/verify API calls
const mockSendOtp = (phone) =>
  new Promise((resolve) => setTimeout(() => resolve({ success: true, otp: '123456' }), 800));

const mockVerifyOtp = (phone, otp, expectedOtp) =>
  new Promise((resolve, reject) =>
    setTimeout(() => {
      if (otp === expectedOtp) resolve({ success: true });
      else reject(new Error('invalid-otp'));
    }, 800)
  );

export default function Login({ onLogin, initialLanguage = 'en' }) {
  const [lang, setLang] = useState(initialLanguage);
  const [phone, setPhone] = useState('');
  const [otp, setOtp] = useState('');
  const [step, setStep] = useState('phone');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [expectedOtp, setExpectedOtp] = useState('');
  const t = T[lang];

  const handleSendOtp = async () => {
    setError('');
    if (phone.length !== 10) { setError(t.errInvalidPhone); return; }
    setLoading(true);
    try {
      const result = await mockSendOtp(phone);
      // In production: call your backend OTP API here and remove the line below
      setExpectedOtp(result.otp);
      setStep('otp');
    } catch (err) {
      console.error(err);
      setError(t.errGeneric);
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async () => {
    setError('');
    if (otp.length < 6) return;
    setLoading(true);
    try {
      await mockVerifyOtp(phone, otp, expectedOtp);
      if (onLogin) onLogin({ phone, lang });
    } catch (err) {
      console.error(err);
      setError(t.errOtpFailed);
    } finally {
      setLoading(false);
    }
  };

  const handleResend = () => {
    setStep('phone');
    setOtp('');
    setError('');
    setExpectedOtp('');
  };

  const BG = 'url(https://lh3.googleusercontent.com/aida-public/AB6AXuC1JGZgatIpJej__Oy2EjR0cVoILIjReuA1Y0X1srz-mM4K6a43BqcIH8WBQqx0tb-_9JTx4AguaCwKtosFbzTYgwwsvPjELaSPflgimN8oWXdpHJJ_XM2idIBd3IcU8Snsx5LuV9QsAz2_2XmG3f2cV0jjv9o67cgD1mdFyqs7xY7K4ORB1fsgLWWhCSGaLVYMcNCG4BU2fd0nGlI9rKcM0k2kwUbqPTYMlc56FDz_hEADkRlsNw0WHIjrswDNE213GTl8flidnxpM)';

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundImage: BG, backgroundSize: 'cover', backgroundPosition: 'center', backgroundAttachment: 'fixed', fontFamily: "'DM Sans','Noto Sans Kannada',system-ui,sans-serif", position: 'relative' }}>
      <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 0 }} />

      {/* Navbar */}
      <nav style={{ position: 'relative', zIndex: 10, display: 'flex', alignItems: 'center', padding: '1.25rem 2.5rem' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'rgba(255,255,255,0.12)', backdropFilter: 'blur(10px)', border: '0.5px solid rgba(255,255,255,0.2)', borderRadius: 999, padding: '6px 14px 6px 8px' }}>
          <div style={{ width: 26, height: 26, background: '#fff', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13 }}>🌾</div>
          <span style={{ fontSize: 12, fontWeight: 600, color: '#fff', letterSpacing: '0.08em' }}>{t.badge}</span>
        </div>
      </nav>

      {/* Main */}
      <main style={{ flex: 1, position: 'relative', zIndex: 10, display: 'flex', alignItems: 'center', padding: '2rem 2.5rem', gap: '3rem', maxWidth: 1100, margin: '0 auto', width: '100%' }}>

        {/* Left hero */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: '1.5rem', flexWrap: 'wrap' }}>
            {[t.tag1, t.tag2, t.tag3].map(tag => (
              <span key={tag} style={{ fontSize: 11, color: 'rgba(255,255,255,0.75)', border: '0.5px solid rgba(255,255,255,0.3)', borderRadius: 999, padding: '3px 10px', backdropFilter: 'blur(4px)', background: 'rgba(255,255,255,0.08)' }}>{tag}</span>
            ))}
          </div>
          <h1 style={{ fontSize: 'clamp(2rem,4vw,3.25rem)', fontWeight: 700, color: '#fff', lineHeight: 1.15, marginBottom: '1rem', letterSpacing: '-0.02em' }}>{t.headline}</h1>
          <p style={{ fontSize: 15, color: 'rgba(255,255,255,0.6)', lineHeight: 1.7, maxWidth: 420, marginBottom: '2.5rem' }}>{t.sub}</p>
          <div style={{ display: 'flex', gap: '2rem' }}>
            {[{ num: '96%', icon: '✔', label: t.stat1 }, { num: '12+', icon: '🌿', label: t.stat2 }, { num: '2', icon: '⟨A⟩', label: t.stat3 }].map((s, i) => (
              <div key={i}>
                <div style={{ fontSize: 22, fontWeight: 600, color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 14, color: '#7ec850' }}>{s.icon}</span>{s.num}
                </div>
                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.45)', marginTop: 3 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right card */}
        <div style={{ width: 380, flexShrink: 0, background: 'rgba(240,243,237,0.92)', backdropFilter: 'blur(20px)', borderRadius: 20, border: '0.5px solid rgba(255,255,255,0.4)', padding: '2rem', boxShadow: '0 24px 60px rgba(0,0,0,0.25)' }}>

          {/* Lang toggle */}
          <div style={{ display: 'flex', background: 'rgba(0,0,0,0.06)', borderRadius: 10, padding: 3, marginBottom: '1.5rem' }}>
            {['en', 'kn'].map(l => (
              <button key={l} onClick={() => { setLang(l); setError(''); }} style={{ flex: 1, padding: '7px 0', fontSize: 13, fontWeight: 500, border: 'none', cursor: 'pointer', borderRadius: 8, background: lang === l ? '#fff' : 'transparent', color: lang === l ? '#1a2e0f' : 'rgba(0,0,0,0.45)', boxShadow: lang === l ? '0 1px 4px rgba(0,0,0,0.1)' : 'none', transition: 'all 0.15s' }}>
                {l === 'en' ? 'English' : 'ಕನ್ನಡ'}
              </button>
            ))}
          </div>

          <div style={{ fontSize: 18, fontWeight: 600, color: '#1a2e0f', marginBottom: 4 }}>{t.formTitle}</div>
          <div style={{ fontSize: 13, color: 'rgba(0,0,0,0.45)', marginBottom: '1.5rem' }}>{t.formSub}</div>

          {/* Phone */}
          <label style={{ fontSize: 11, fontWeight: 600, color: 'rgba(0,0,0,0.45)', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>{t.labelPhone}</label>
          <div style={{ display: 'flex', alignItems: 'center', background: '#fff', borderRadius: 10, border: '0.5px solid rgba(0,0,0,0.12)', overflow: 'hidden', marginBottom: '1rem', opacity: step === 'otp' ? 0.6 : 1 }}>
            <div style={{ padding: '11px 12px', fontSize: 13, color: 'rgba(0,0,0,0.5)', borderRight: '0.5px solid rgba(0,0,0,0.1)', background: 'rgba(0,0,0,0.03)', whiteSpace: 'nowrap' }}>🇮🇳 +91</div>
            <input type="tel" maxLength={10} value={phone} onChange={e => { setPhone(e.target.value.replace(/\D/g, '')); setError(''); }} placeholder="98XXXXXXXX" disabled={step === 'otp'} style={{ flex: 1, padding: '11px 12px', fontSize: 14, border: 'none', outline: 'none', background: 'transparent', color: '#1a2e0f' }} />
          </div>

          {/* OTP */}
          {step === 'otp' && (
            <>
              <div style={{ fontSize: 11, color: '#3B6D11', marginBottom: 8, fontWeight: 500 }}>✓ {t.otpHint}</div>
              <label style={{ fontSize: 11, fontWeight: 600, color: 'rgba(0,0,0,0.45)', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>{t.labelOtp}</label>
              <div style={{ display: 'flex', alignItems: 'center', background: '#fff', borderRadius: 10, border: '0.5px solid rgba(59,109,17,0.4)', overflow: 'hidden', marginBottom: '1rem', boxShadow: '0 0 0 3px rgba(59,109,17,0.08)' }}>
                <input type="number" value={otp} onChange={e => { setOtp(e.target.value); setError(''); }} placeholder="------" maxLength={6} style={{ flex: 1, padding: '11px 14px', fontSize: 18, border: 'none', outline: 'none', background: 'transparent', color: '#1a2e0f', letterSpacing: 8, fontWeight: 600 }} />
              </div>
            </>
          )}

          {/* Error */}
          {error ? (
            <div style={{ fontSize: 12, color: '#c0392b', background: 'rgba(192,57,43,0.08)', borderRadius: 8, padding: '8px 12px', marginBottom: '0.75rem' }}>{error}</div>
          ) : null}

          {/* Buttons */}
          {step === 'phone' ? (
            <button onClick={handleSendOtp} disabled={phone.length < 10 || loading} style={{ width: '100%', padding: '12px', background: phone.length >= 10 && !loading ? '#1a2e0f' : 'rgba(0,0,0,0.12)', color: phone.length >= 10 && !loading ? '#fff' : 'rgba(0,0,0,0.3)', border: 'none', borderRadius: 10, fontSize: 14, fontWeight: 600, cursor: phone.length >= 10 && !loading ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, transition: 'background 0.15s', marginBottom: '1rem' }}>
              {loading ? t.sending : <>{t.sendOtp} →</>}
            </button>
          ) : (
            <>
              <button onClick={handleVerify} disabled={otp.length < 6 || loading} style={{ width: '100%', padding: '12px', background: otp.length >= 6 && !loading ? '#1a2e0f' : 'rgba(0,0,0,0.12)', color: otp.length >= 6 && !loading ? '#fff' : 'rgba(0,0,0,0.3)', border: 'none', borderRadius: 10, fontSize: 14, fontWeight: 600, cursor: otp.length >= 6 && !loading ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, transition: 'background 0.15s', marginBottom: '0.75rem' }}>
                {loading ? t.verifying : <>{t.verify} →</>}
              </button>
              <div style={{ textAlign: 'center', fontSize: 12, color: 'rgba(0,0,0,0.4)' }}>
                <span onClick={handleResend} style={{ color: '#3B6D11', cursor: 'pointer', fontWeight: 500 }}>{t.resend}</span>
              </div>
            </>
          )}

          <div style={{ borderTop: '0.5px solid rgba(0,0,0,0.08)', margin: '1.25rem 0 1rem' }} />
          <div style={{ textAlign: 'center', fontSize: 10, fontWeight: 700, color: 'rgba(0,0,0,0.35)', letterSpacing: '0.1em', marginBottom: '0.75rem' }}>{t.forFarmers}</div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem' }}>
            {[t.helpCenter, t.support].map(link => (
              <span key={link} style={{ fontSize: 12, color: 'rgba(0,0,0,0.4)', cursor: 'pointer' }}>{link}</span>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{ position: 'relative', zIndex: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 2.5rem', borderTop: '0.5px solid rgba(255,255,255,0.1)', flexWrap: 'wrap', gap: 8 }}>
        <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', letterSpacing: '0.06em' }}>{t.footer}</span>
        <div style={{ display: 'flex', gap: '1.5rem' }}>
          {[t.privacy, t.terms, t.contact].map(link => (
            <span key={link} style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', cursor: 'pointer', letterSpacing: '0.05em' }}>{link}</span>
          ))}
        </div>
      </footer>
    </div>
  );
}