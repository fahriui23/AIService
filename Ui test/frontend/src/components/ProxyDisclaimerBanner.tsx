import React from 'react';
import { AlertTriangle } from 'lucide-react';

export const ProxyDisclaimerBanner: React.FC = () => {
  return (
    <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 mb-6 text-amber-200 flex items-start gap-3 shadow-lg shadow-amber-500/5">
      <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
      <div>
        <h4 className="font-semibold text-amber-400 text-sm mb-1 uppercase tracking-wider">
          PROXY EVALUATION — NOT HUMAN-GOLD VALIDATED
        </h4>
        <p className="text-xs text-amber-200/90 leading-relaxed">
          Evaluation menggunakan proxy labels dan belum divalidasi menggunakan Human GOLD.
          Metrik digunakan untuk perbandingan eksperimen internal dan bukan klaim performa produksi.
        </p>
      </div>
    </div>
  );
};
