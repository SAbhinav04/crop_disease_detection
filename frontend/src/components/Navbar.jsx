import { useEffect, useRef, useState } from 'react';

export default function Navbar({ language, setLanguage, labels, onLogout, userPhone }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const onDocumentClick = (event) => {
      if (!menuRef.current?.contains(event.target)) {
        setMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', onDocumentClick);
    return () => document.removeEventListener('mousedown', onDocumentClick);
  }, []);

  return (
    <div className="navbar">

      <div className="nav-left">
        <img src="/logo.png" className="logo" />
        <div>
          <h3>{labels.navbarTitle}</h3>
          <p>{labels.navbarSubtitle}</p>
        </div>
      </div>

      <div className="nav-right">
        <button
          onClick={() =>
            setLanguage(language === "en" ? "kn" : "en")
          }
        >
          🌐 {language === "en" ? "English" : "ಕನ್ನಡ"}
        </button>

        <div ref={menuRef} style={{ position: 'relative' }}>
          <button
            type="button"
            className="profile"
            onClick={() => setMenuOpen((current) => !current)}
            aria-label="Profile menu"
            aria-expanded={menuOpen}
          >
            👤
          </button>

          {menuOpen ? (
            <div
              style={{
                position: 'absolute',
                top: '46px',
                right: 0,
                minWidth: '180px',
                background: '#ffffff',
                borderRadius: '12px',
                border: '1px solid #e6ece8',
                boxShadow: '0 12px 24px rgba(0,0,0,0.12)',
                padding: '6px'
              }}
            >
              {userPhone ? (
                <div
                  style={{
                    padding: '8px 12px 10px',
                    borderBottom: '1px solid #eef3ef',
                    marginBottom: '6px'
                  }}
                >
                  <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '3px' }}>
                    Logged in with
                  </div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: '#1f2937' }}>
                    {userPhone}
                  </div>
                </div>
              ) : null}
              <button
                type="button"
                onClick={() => {
                  setMenuOpen(false);
                  onLogout?.();
                }}
                style={{
                  width: '100%',
                  border: 'none',
                  borderRadius: '8px',
                  background: 'transparent',
                  padding: '10px 12px',
                  textAlign: 'left',
                  fontSize: '14px',
                  cursor: 'pointer',
                  color: '#1f2937'
                }}
              >
                Logout
              </button>
            </div>
          ) : null}
        </div>
      </div>

    </div>
  );
}
