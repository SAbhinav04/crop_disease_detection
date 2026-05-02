export default function Hero() {
  return (
    <div className="hero-card">

      <div className="hero-row">

      

        {/* TEXT CONTENT */}
        <div className="hero-text">

          {/* TITLE */}
          <h1 className="hero-title">
            <span>Detect Disease.Get Solution.</span><br/>

            <span>Protect Your Crop.</span>
          </h1>

          {/* SUBTEXT */}
          <p className="hero-sub">
            Upload a leaf image and get instant diagnosis and AI-powered 
            treatment advice in your language.
          </p>
        </div>
        <div className="hero-right">
          <div
            style={{
              width: '250px',
              height: '150px',
              backgroundImage: `url('/farmer.jpeg')`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              borderRadius: '12px'
            }}
          />
        </div>
      </div>

    </div>
  );
}
