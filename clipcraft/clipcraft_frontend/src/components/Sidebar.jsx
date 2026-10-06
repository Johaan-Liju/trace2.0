import { Search, ScanFace, Wand2, LayoutDashboard } from 'lucide-react';
import './Sidebar.css';

const features = [
  { id: 'search', label: 'Dialogue Search', icon: Search },
  { id: 'face', label: 'Face Detection', icon: ScanFace },
  { id: 'prompt', label: 'Prompt Clipping', icon: Wand2 },
];

export default function Sidebar({ activeFeature, onFeatureChange }) {
  return (
    <aside className="sidebar glass-panel">
      <div className="sidebar-header">
        <div className="logo-box">
          <LayoutDashboard size={24} className="logo-icon" />
        </div>
        <h1 className="brand">Clipcraft</h1>
      </div>

      <nav className="sidebar-nav">
        {features.map((feat) => {
          const Icon = feat.icon;
          const isActive = activeFeature === feat.id;
          return (
            <button 
              key={feat.id} 
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => onFeatureChange(feat.id)}
            >
              <Icon size={20} className="nav-icon" />
              <span>{feat.label}</span>
              {isActive && <div className="active-indicator" />}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="user-badge">
          <div className="avatar">A</div>
          <div className="user-info">
            <span className="user-name">Admin</span>
            <span className="user-role">Pro License</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
