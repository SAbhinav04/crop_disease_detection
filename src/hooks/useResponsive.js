import { useEffect, useState } from 'react';

export function useResponsive(breakpoint = 768) {
  const getMatches = () =>
    typeof window !== 'undefined' ? window.matchMedia(`(min-width: ${breakpoint}px)`).matches : false;

  const [isDesktop, setIsDesktop] = useState(getMatches);

  useEffect(() => {
    const mediaQueryList = window.matchMedia(`(min-width: ${breakpoint}px)`);
    const handleChange = (event) => setIsDesktop(event.matches);

    setIsDesktop(mediaQueryList.matches);
    mediaQueryList.addEventListener('change', handleChange);

    return () => mediaQueryList.removeEventListener('change', handleChange);
  }, [breakpoint]);

  return { isDesktop, isMobile: !isDesktop };
}
