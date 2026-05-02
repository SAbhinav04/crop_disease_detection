export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="site-footer" aria-label="Site footer">
      <div className="site-footer__top">
        <div className="site-footer__brand">
          <p className="site-footer__eyebrow">Support</p>
          <h3>Need Help?</h3>
          <p>We are here to help you with crop diagnosis and next steps.</p>
        </div>

        <div className="site-footer__contact">
          <p className="site-footer__label">Call Us</p>
          <a href="tel:1234567890" className="site-footer__phone">1234567890</a>
          <p>Mon - Sat: 9 AM to 6 PM</p>
        </div>
      </div>

      <div className="site-footer__bottom">
        <p>© {year} AgriVision Edge</p>
        <p>Built for farmer-first crop care</p>
      </div>
    </footer>
  );
}
