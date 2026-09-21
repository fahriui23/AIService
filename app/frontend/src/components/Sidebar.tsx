import React from 'react';
import {
  LayoutDashboard,
  Database,
  Sliders,
  PlayCircle,
  BarChart3,
  TrendingUp,
  AlertOctagon,
  GitCompare,
  Sparkles,
  FolderArchive,
  Settings
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'dataset', label: 'Dataset Manager', icon: Database },
    { id: 'experiment', label: 'Eksperimen', icon: Sliders },
    { id: 'training', label: 'Training Monitor', icon: PlayCircle },
    { id: 'evaluation', label: 'Evaluasi', icon: BarChart3 },
    { id: 'learning_curve', label: 'Learning Curve', icon: TrendingUp },
    { id: 'error_analysis', label: 'Error Analysis', icon: AlertOctagon },
    { id: 'ablation', label: 'Ablation Comparison', icon: GitCompare },
    { id: 'inference', label: 'Inference Playground', icon: Sparkles },
    { id: 'models', label: 'Model Registry', icon: FolderArchive },
    { id: 'settings', label: 'Pengaturan', icon: Settings },
  ];

  return (
    <aside className="w-64 border-r border-slate-800 bg-dark-card/30 p-4 flex flex-col justify-between shrink-0 min-h-[calc(100vh-4rem)]">
      <div className="space-y-1">
        <div className="px-3 py-2 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
          Navigasi Utama
        </div>

        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30 shadow-md shadow-blue-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-slate-500'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/80 text-[11px] text-slate-400">
        <p className="font-semibold text-slate-300 mb-0.5">Evaluation Mode</p>
        <p className="text-[10px] text-amber-400 font-mono">proxy_without_human_gold</p>
      </div>
    </aside>
  );
};
