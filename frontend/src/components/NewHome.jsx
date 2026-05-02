import { useState } from "react";
import Navbar from "./Navbar";
import Hero from "./Hero";
import UploadSection from "./UploadSection";
import ResultSection from "./ResultSection";
import HowItWorks from "./HowItWorks";
import Features from "./Features";
import Footer from "./Footer";

export default function NewHome() {
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  return (
    <>
      <Navbar />
      <Hero />

      {!result ? (
        <UploadSection
          setResult={setResult}
          setPreviewUrl={setPreviewUrl}
        />
      ) : (
        <ResultSection
          result={result}
          previewUrl={previewUrl}
        />
      )}

      <HowItWorks />
      <Features />
      <Footer />
    </>
  );
}
