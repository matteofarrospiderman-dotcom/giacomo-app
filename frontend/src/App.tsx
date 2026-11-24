import { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard";
import Workspace from "./components/Workspace";

/** Minimal hash router: #/ (dashboard) and #/project/<id>[/<tab>] */
function useRoute(): string[] {
  const parse = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [r, setR] = useState(parse);
  useEffect(() => {
    const on = () => setR(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return r;
}

export const navigate = (path: string) => {
  window.location.hash = path;
};

export default function App() {
  const route = useRoute();
  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">
          <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="7" fill="#1e2536" />
            <path d="M6 9h20M6 14h20M6 19h20M6 24h20" stroke="#59627a" strokeWidth="1.3" />
            <circle cx="12" cy="14" r="3.6" fill="#f59e0b" />
            <circle cx="21" cy="19" r="3.6" fill="#38bdf8" />
          </svg>
          TabForge <small>AI guitar transcription</small>
        </a>
        <div className="spacer" />
        <a className="btn" href="#/">Projects</a>
      </header>
      {route[0] === "project" && route[1] ? <Workspace key={route[1]} id={route[1]} tab={route[2]} /> : <Dashboard />}
    </div>
  );
}
