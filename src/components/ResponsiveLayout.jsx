/**
 * @param {{ left: React.ReactNode, right: React.ReactNode, header: React.ReactNode }} props
 */
export default function ResponsiveLayout({ header, left, right }) {
  return (
    <div className="min-h-screen bg-orchard-radial">
      {header}
      <main className="mx-auto w-full max-w-7xl px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
        <div className="grid gap-5 md:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.95fr)]">
          <section className="space-y-5">{left}</section>
          <aside className="space-y-5 md:sticky md:top-[6.75rem] md:self-start">{right}</aside>
        </div>
      </main>
    </div>
  );
}
