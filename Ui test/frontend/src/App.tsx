import React from 'react';
import { InferencePlaygroundPage } from './pages/InferencePlaygroundPage';

export const App: React.FC = () => (
  <div className="min-h-screen bg-dark-bg">
    <header className="border-b border-slate-800 bg-slate-950/80 px-6 py-4 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
        <div>
          <h1 className="text-lg font-bold tracking-wide text-white">V14 ABSA</h1>
          <p className="text-xs text-slate-400">Aspect-Based Sentiment Analysis · Indonesia &amp; English</p>
        </div>
        <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-[11px] font-medium text-amber-300">
          Pre-production candidate
        </span>
      </div>
    </header>
    <main className="mx-auto max-w-7xl p-6">
      <InferencePlaygroundPage />
    </main>
  </div>
);
