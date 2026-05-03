import { useState } from 'react';

const DEFAULT_BYPASS_PHONE = '1234567890';
const DEFAULT_BYPASS_OTP = '123456';

const getApiBaseUrl = () => {
  const value = import.meta.env.VITE_API_URL;
  if (value) return value.replace(/\/$/, '');
  return 'http://localhost:8000';
};

const T = {
  en: {
    badge: 'AGRIVISION EDGE',
    headline: 'Crop disease detection built for Indian farmers',
    sub: 'Scan any leaf, get instant diagnosis and treatment advice in Kannada or English.',
    stat1: 'Detection accuracy', stat2: 'Crop diseases', stat3: 'Multilingual support',
    tag1: 'Instant diagnosis', tag2: 'AI-powered', tag3: 'Kannada support',
    tabLogin: 'Login', tabSignup: 'Sign Up',
    loginTitle: 'Welcome back', loginSub: 'Enter your phone number to continue',
    signupTitle: 'Create account', signupSub: 'Register as a farmer to get started',
    labelName: 'FULL NAME', namePlaceholder: 'Enter your full name',
    labelPhone: 'PHONE NUMBER', labelOtp: 'ENTER OTP',
    sendOtp: 'Send OTP', sending: 'Sending...',
    otpHint: 'OTP sent to your number',
    verifyLogin: 'Verify & Login',
    verifySignup: 'Verify & Create Account',
    verifying: 'Verifying...',
    resend: 'Resend OTP',
    forFarmers: 'FOR REGISTERED FARMERS ONLY',
    helpCenter: 'Help center', support: 'Support',
    footer: '© 2024 AGRIVISION EDGE. EMPOWERING INDIAN AGRICULTURE.',
    privacy: 'PRIVACY POLICY', terms: 'TERMS OF SERVICE', contact: 'CONTACT SUPPORT',
    errInvalidPhone: 'Please enter a valid 10-digit phone number.',
    errInvalidName: 'Please enter your full name.',
    errOtpFailed: 'Incorrect OTP. Please try again.',
    errGeneric: 'Something went wrong. Please try again.',
  },
  kn: {
    badge: 'ಅಗ್ರಿವಿಷನ್ ಎಡ್ಜ್',
    headline: 'ಭಾರತೀಯ ರೈತರಿಗಾಗಿ ಬೆಳೆ ರೋಗ ಪರಿಶೀಲನೆ',
    sub: 'ಯಾವುದೇ ಎಲೆ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ, ಕನ್ನಡ ಅಥವಾ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ತಕ್ಷಣ ರೋಗನಿರ್ಣಯ ಪಡೆಯಿರಿ.',
    stat1: 'ನಿಖರತೆ', stat2: 'ಬೆಳೆ ರೋಗಗಳು', stat3: 'ಬಹುಭಾಷಾ ಬೆಂಬಲ',
    tag1: 'ತ್ವರಿತ ರೋಗನಿರ್ಣಯ', tag2: 'AI ಆಧಾರಿತ', tag3: 'ಕನ್ನಡ ಬೆಂಬಲ',
    tabLogin: 'ಲಾಗಿನ್', tabSignup: 'ನೋಂದಣಿ',
    loginTitle: 'ಸ್ವಾಗತ', loginSub: 'ಮುಂದುವರಿಯಲು ನಿಮ್ಮ ಫೋನ್ ಸಂಖ್ಯೆ ನಮೂದಿಸಿ',
    signupTitle: 'ಖಾತೆ ತೆರೆಯಿರಿ', signupSub: 'ರೈತರಾಗಿ ನೋಂದಾಯಿಸಿಕೊಳ್ಳಿ',
    labelName: 'ಪೂರ್ಣ ಹೆಸರು', namePlaceholder: 'ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರು ನಮೂದಿಸಿ',
    labelPhone: 'ಫೋನ್ ಸಂಖ್ಯೆ', labelOtp: 'OTP ನಮೂದಿಸಿ',
    sendOtp: 'OTP ಕಳುಹಿಸಿ', sending: 'ಕಳುಹಿಸಲಾಗುತ್ತಿದೆ...',
    otpHint: 'OTP ನಿಮ್ಮ ಸಂಖ್ಯೆಗೆ ಕಳುಹಿಸಲಾಗಿದೆ',
    verifyLogin: 'ಪರಿಶೀಲಿಸಿ & ಲಾಗಿನ್',
    verifySignup: 'ಪರಿಶೀಲಿಸಿ & ಖಾತೆ ತೆರೆಯಿರಿ',
    verifying: 'ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ...',
    resend: 'OTP ಮರು-ಕಳುಹಿಸಿ',
    forFarmers: 'ನೋಂದಾಯಿತ ರೈತರಿಗೆ ಮಾತ್ರ',
    helpCenter: 'ಸಹಾಯ ಕೇಂದ್ರ', support: 'ಬೆಂಬಲ',
    footer: '© 2024 ಅಗ್ರಿವಿಷನ್ ಎಡ್ಜ್. ಭಾರತೀಯ ಕೃಷಿಗೆ ಶಕ್ತಿ.',
    privacy: 'ಗೌಪ್ಯತಾ ನೀತಿ', terms: 'ಸೇವಾ ನಿಯಮಗಳು', contact: 'ಬೆಂಬಲ ಸಂಪರ್ಕಿಸಿ',
    errInvalidPhone: 'ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ 10-ಅಂಕಿಯ ಫೋನ್ ಸಂಖ್ಯೆ ನಮೂದಿಸಿ.',
    errInvalidName: 'ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರು ನಮೂದಿಸಿ.',
    errOtpFailed: 'ತಪ್ಪಾದ OTP. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
    errGeneric: 'ಏನೋ ತಪ್ಪಾಗಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
  }
};

const BG = "url('/bg.jpeg')";
// ─── Reusable input styles ───────────────────────────────────────────────────
const inputWrap = (focused = false, disabled = false) => ({
  display: 'flex', alignItems: 'center', background: '#fff', borderRadius: 10,
  border: focused ? '0.5px solid rgba(59,109,17,0.5)' : '0.5px solid rgba(0,0,0,0.12)',
  overflow: 'hidden', marginBottom: '1rem',
  boxShadow: focused ? '0 0 0 3px rgba(59,109,17,0.08)' : 'none',
  opacity: disabled ? 0.6 : 1,
  transition: 'all 0.15s',
});
const inputStyle = { flex: 1, padding: '11px 12px', fontSize: 14, border: 'none', outline: 'none', background: 'transparent', color: '#1a2e0f' };
const labelStyle = { fontSize: 11, fontWeight: 600, color: 'rgba(0,0,0,0.45)', letterSpacing: '0.06em', display: 'block', marginBottom: 6 };
const primaryBtn = (active) => ({
  width: '100%', padding: '12px', border: 'none', borderRadius: 10,
  fontSize: 14, fontWeight: 600, cursor: active ? 'pointer' : 'not-allowed',
  background: active ? '#1a2e0f' : 'rgba(0,0,0,0.12)',
  color: active ? '#fff' : 'rgba(0,0,0,0.3)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
  transition: 'background 0.15s', marginBottom: '0.75rem',
});

export default function Login({ onLogin, initialLanguage = 'en', initialPhone = DEFAULT_BYPASS_PHONE }) {
  const [lang, setLang] = useState(initialLanguage);
  const [activeTab, setActiveTab] = useState('login'); // 'login' | 'signup'
  const devLoginPhone = String(initialPhone || DEFAULT_BYPASS_PHONE).replace(/\D/g, '').slice(0, 10);

  // Login state
  const [loginPhone, setLoginPhone] = useState(devLoginPhone);
  const [loginOtp, setLoginOtp] = useState('');
  const [loginStep, setLoginStep] = useState('phone');
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState('');

  // Signup state
  const [signupName, setSignupName] = useState('');
  const [signupPhone, setSignupPhone] = useState('');
  const [signupOtp, setSignupOtp] = useState('');
  const [signupStep, setSignupStep] = useState('details');
  const [signupLoading, setSignupLoading] = useState(false);
  const [signupError, setSignupError] = useState('');

  const t = T[lang];
  const canBypass = loginPhone === DEFAULT_BYPASS_PHONE;

  // ─── Tab switch — reset state ──────────────────────────────────────────────
  const switchTab = (tab) => {
    setActiveTab(tab);
    setLoginError('');
    setSignupError('');
  };

  // ─── LOGIN handlers ────────────────────────────────────────────────────────
  const handleLoginSendOtp = async () => {
    setLoginError('');
    if (loginPhone.length !== 10) { setLoginError(t.errInvalidPhone); return; }
    if (canBypass) { setLoginStep('otp'); setLoginOtp(DEFAULT_BYPASS_OTP); return; }
    setLoginLoading(true);
    try {
      const res = await fetch(`${getApiBaseUrl()}/auth/send-otp`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone: loginPhone }),
      });
      if (!res.ok) throw new Error();
      setLoginStep('otp');
    } catch { setLoginError(t.errGeneric); }
    finally { setLoginLoading(false); }
  };

  const handleLoginVerify = async () => {
    setLoginError('');
    if (loginOtp.length < 6) return;
    if (canBypass && loginOtp === DEFAULT_BYPASS_OTP) {
      if (onLogin) onLogin({ phone: loginPhone, lang, name: 'demo' }); return;
    }
    setLoginLoading(true);
    try {
      const res = await fetch(`${getApiBaseUrl()}/auth/verify-otp`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone: loginPhone, otp: loginOtp }),
      });
      if (!res.ok) throw new Error();
      if (onLogin) onLogin({ phone: loginPhone, lang });
    } catch { setLoginError(t.errOtpFailed); }
    finally { setLoginLoading(false); }
  };

  // ─── SIGNUP handlers ───────────────────────────────────────────────────────
  const handleSignupSendOtp = async () => {
    setSignupError('');
    if (!signupName.trim()) { setSignupError(t.errInvalidName); return; }
    if (signupPhone.length !== 10) { setSignupError(t.errInvalidPhone); return; }
    setSignupLoading(true);
    try {
      const res = await fetch(`${getApiBaseUrl()}/auth/send-otp`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone: signupPhone }),
      });
      if (!res.ok) throw new Error();
      setSignupStep('otp');
    } catch { setSignupError(t.errGeneric); }
    finally { setSignupLoading(false); }
  };

  const handleSignupVerify = async () => {
    setSignupError('');
    if (signupOtp.length < 6) return;
    setSignupLoading(true);
    try {
      const res = await fetch(`${getApiBaseUrl()}/auth/verify-otp`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone: signupPhone, otp: signupOtp, name: signupName }),
      });
      if (!res.ok) throw new Error();
      if (onLogin) onLogin({ phone: signupPhone, lang, name: signupName });
    } catch { setSignupError(t.errOtpFailed); }
    finally { setSignupLoading(false); }
  };

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
        <div style={{ width: 390, flexShrink: 0, background: 'rgba(240,243,237,0.92)', backdropFilter: 'blur(20px)', borderRadius: 20, border: '0.5px solid rgba(255,255,255,0.4)', padding: '2rem', boxShadow: '0 24px 60px rgba(0,0,0,0.25)' }}>

          {/* Language toggle */}
          <div style={{ display: 'flex', background: 'rgba(0,0,0,0.06)', borderRadius: 10, padding: 3, marginBottom: '1.25rem' }}>
            {['en', 'kn'].map(l => (
              <button key={l} onClick={() => { setLang(l); setLoginError(''); setSignupError(''); }} style={{ flex: 1, padding: '7px 0', fontSize: 13, fontWeight: 500, border: 'none', cursor: 'pointer', borderRadius: 8, background: lang === l ? '#fff' : 'transparent', color: lang === l ? '#1a2e0f' : 'rgba(0,0,0,0.45)', boxShadow: lang === l ? '0 1px 4px rgba(0,0,0,0.1)' : 'none', transition: 'all 0.15s' }}>
                {l === 'en' ? 'English' : 'ಕನ್ನಡ'}
              </button>
            ))}
          </div>

          {/* Login / Signup tab switcher */}
          <div style={{ display: 'flex', borderBottom: '1px solid rgba(0,0,0,0.08)', marginBottom: '1.25rem' }}>
            {['login', 'signup'].map(tab => (
              <button key={tab} onClick={() => switchTab(tab)} style={{
                flex: 1, padding: '8px 0', fontSize: 13, fontWeight: 600,
                border: 'none', background: 'transparent', cursor: 'pointer',
                color: activeTab === tab ? '#1a2e0f' : 'rgba(0,0,0,0.35)',
                borderBottom: activeTab === tab ? '2px solid #1a2e0f' : '2px solid transparent',
                transition: 'all 0.15s', marginBottom: -1,
              }}>
                {tab === 'login' ? t.tabLogin : t.tabSignup}
              </button>
            ))}
          </div>

          {/* ── LOGIN FORM ── */}
          {activeTab === 'login' && (
            <>
              <div style={{ fontSize: 17, fontWeight: 600, color: '#1a2e0f', marginBottom: 3 }}>{t.loginTitle}</div>
              <div style={{ fontSize: 12, color: 'rgba(0,0,0,0.4)', marginBottom: '1.25rem' }}>{t.loginSub}</div>

              <label style={labelStyle}>{t.labelPhone}</label>
              <div style={inputWrap(false, loginStep === 'otp')}>
                <div style={{ padding: '11px 12px', fontSize: 13, color: 'rgba(0,0,0,0.5)', borderRight: '0.5px solid rgba(0,0,0,0.1)', background: 'rgba(0,0,0,0.03)', whiteSpace: 'nowrap' }}>🇮🇳 +91</div>
                <input type="tel" maxLength={10} value={loginPhone} onChange={e => { setLoginPhone(e.target.value.replace(/\D/g, '')); setLoginError(''); }} placeholder="98XXXXXXXX" disabled={loginStep === 'otp'} style={inputStyle} />
              </div>

              {canBypass && <div style={{ fontSize: 11, color: '#3B6D11', marginBottom: '0.75rem', fontWeight: 500 }}>Demo login enabled for {DEFAULT_BYPASS_PHONE}</div>}

              {loginStep === 'otp' && (
                <>
                  <div style={{ fontSize: 11, color: '#3B6D11', marginBottom: 8, fontWeight: 500 }}>✓ {t.otpHint}</div>
                  <label style={labelStyle}>{t.labelOtp}</label>
                  <div style={inputWrap(true)}>
                    <input type="number" value={loginOtp} onChange={e => { setLoginOtp(e.target.value); setLoginError(''); }} placeholder="------" maxLength={6} style={{ ...inputStyle, fontSize: 18, letterSpacing: 8, fontWeight: 600, padding: '11px 14px' }} />
                  </div>
                </>
              )}

              {loginError && <div style={{ fontSize: 12, color: '#c0392b', background: 'rgba(192,57,43,0.08)', borderRadius: 8, padding: '8px 12px', marginBottom: '0.75rem' }}>{loginError}</div>}

              {loginStep === 'phone' ? (
                <button onClick={handleLoginSendOtp} disabled={loginPhone.length < 10 || loginLoading} style={primaryBtn(loginPhone.length >= 10 && !loginLoading)}>
                  {loginLoading ? t.sending : canBypass ? 'Continue to app →' : `${t.sendOtp} →`}
                </button>
              ) : (
                <>
                  <button onClick={handleLoginVerify} disabled={loginOtp.length < 6 || loginLoading} style={primaryBtn(loginOtp.length >= 6 && !loginLoading)}>
                    {loginLoading ? t.verifying : canBypass ? 'Enter demo code →' : `${t.verifyLogin} →`}
                  </button>
                  <div style={{ textAlign: 'center', fontSize: 12, color: 'rgba(0,0,0,0.4)', marginBottom: '0.5rem' }}>
                    <span onClick={() => { setLoginStep('phone'); setLoginOtp(''); setLoginError(''); }} style={{ color: '#3B6D11', cursor: 'pointer', fontWeight: 500 }}>{t.resend}</span>
                  </div>
                </>
              )}
            </>
          )}

          {/* ── SIGNUP FORM ── */}
          {activeTab === 'signup' && (
            <>
              <div style={{ fontSize: 17, fontWeight: 600, color: '#1a2e0f', marginBottom: 3 }}>{t.signupTitle}</div>
              <div style={{ fontSize: 12, color: 'rgba(0,0,0,0.4)', marginBottom: '1.25rem' }}>{t.signupSub}</div>

              {/* Full Name — only show before OTP step */}
              {signupStep === 'details' && (
                <>
                  <label style={labelStyle}>{t.labelName}</label>
                  <div style={inputWrap()}>
                    <input type="text" value={signupName} onChange={e => { setSignupName(e.target.value); setSignupError(''); }} placeholder={t.namePlaceholder} style={inputStyle} />
                  </div>
                </>
              )}

              {/* Phone */}
              <label style={labelStyle}>{t.labelPhone}</label>
              <div style={inputWrap(false, signupStep === 'otp')}>
                <div style={{ padding: '11px 12px', fontSize: 13, color: 'rgba(0,0,0,0.5)', borderRight: '0.5px solid rgba(0,0,0,0.1)', background: 'rgba(0,0,0,0.03)', whiteSpace: 'nowrap' }}>🇮🇳 +91</div>
                <input type="tel" maxLength={10} value={signupPhone} onChange={e => { setSignupPhone(e.target.value.replace(/\D/g, '')); setSignupError(''); }} placeholder="98XXXXXXXX" disabled={signupStep === 'otp'} style={inputStyle} />
              </div>

              {/* OTP */}
              {signupStep === 'otp' && (
                <>
                  <div style={{ fontSize: 11, color: '#3B6D11', marginBottom: 8, fontWeight: 500 }}>✓ {t.otpHint}</div>
                  <label style={labelStyle}>{t.labelOtp}</label>
                  <div style={inputWrap(true)}>
                    <input type="number" value={signupOtp} onChange={e => { setSignupOtp(e.target.value); setSignupError(''); }} placeholder="------" maxLength={6} style={{ ...inputStyle, fontSize: 18, letterSpacing: 8, fontWeight: 600, padding: '11px 14px' }} />
                  </div>
                </>
              )}

              {signupError && <div style={{ fontSize: 12, color: '#c0392b', background: 'rgba(192,57,43,0.08)', borderRadius: 8, padding: '8px 12px', marginBottom: '0.75rem' }}>{signupError}</div>}

              {signupStep === 'details' ? (
                <button onClick={handleSignupSendOtp} disabled={!signupName.trim() || signupPhone.length < 10 || signupLoading} style={primaryBtn(!!(signupName.trim() && signupPhone.length >= 10 && !signupLoading))}>
                  {signupLoading ? t.sending : `${t.sendOtp} →`}
                </button>
              ) : (
                <>
                  <button onClick={handleSignupVerify} disabled={signupOtp.length < 6 || signupLoading} style={primaryBtn(signupOtp.length >= 6 && !signupLoading)}>
                    {signupLoading ? t.verifying : `${t.verifySignup} →`}
                  </button>
                  <div style={{ textAlign: 'center', fontSize: 12, color: 'rgba(0,0,0,0.4)', marginBottom: '0.5rem' }}>
                    <span onClick={() => { setSignupStep('details'); setSignupOtp(''); setSignupError(''); }} style={{ color: '#3B6D11', cursor: 'pointer', fontWeight: 500 }}>{t.resend}</span>
                  </div>
                </>
              )}

              {/* Already have account */}
              <div style={{ textAlign: 'center', fontSize: 12, color: 'rgba(0,0,0,0.4)' }}>
                Already registered?{' '}
                <span onClick={() => switchTab('login')} style={{ color: '#1a2e0f', cursor: 'pointer', fontWeight: 600 }}>{t.tabLogin}</span>
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