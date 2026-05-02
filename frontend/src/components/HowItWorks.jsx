export default function HowItWorks({ labels }) {
  const steps = [
    {
      number: 1,
      icon: '🌿',
      title: labels.step1Title || 'Upload Leaf',
      description: labels.step1Desc || 'Upload a clear leaf image'
    },
    {
      number: 2,
      icon: '🤖',
      title: labels.step2Title || 'AI Detects Disease',
      description: labels.step2Desc || 'Our AI analyzes and detects disease'
    },
    {
      number: 3,
      icon: '💊',
      title: labels.step3Title || 'Get Solution',
      description: labels.step3Desc || 'Get treatment and prevention advice'
    },
    {
      number: 4,
      icon: '🔊',
      title: labels.step4Title || 'Listen in Kannada',
      description: labels.step4Desc || 'Audio guidance in Kannada'
    }
  ];

  const features = [
    {
      title: labels.feature1Title || 'Accurate Detection',
      description: labels.feature1Desc || 'AI-powered disease identification'
    },
    {
      title: labels.feature2Title || 'Smart Solutions',
      description: labels.feature2Desc || 'Personalized treatment and prevention tips'
    },
    {
      title: labels.feature3Title || 'Audio Support',
      description: labels.feature3Desc || 'Listen to advice in Kannada'
    },
    {
      title: labels.feature4Title || 'History Tracking',
      description: labels.feature4Desc || 'View your past diagnosis history'
    }
  ];

  return (
    <section className="how-section">

      {/* TITLE */}
      <div className="how-title">
        <span></span>
        <h2>{labels.howItWorks}</h2>
        <span></span>
      </div>

      {/* STEPS */}
      <div className="how-steps">
        {steps.map((step, index) => (
          <div key={step.number} className="step-flow-item">
            <div className="step">
              <div className="step-number">{step.number}</div>
              <div className="step-icon">{step.icon}</div>
              <h3>{step.title}</h3>
              <p>{step.description}</p>
            </div>
            {index < steps.length - 1 ? <div className="arrow">→</div> : null}
          </div>
        ))}

      </div>

      {/* FEATURES */}
      <div className="features-box">
        {features.map((feature) => (
          <div key={feature.title} className="feature">
            <h4>{feature.title}</h4>
            <p>{feature.description}</p>
          </div>
        ))}
      </div>

    </section>
  );
}
