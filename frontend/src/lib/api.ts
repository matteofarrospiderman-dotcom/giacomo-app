import type { Analysis, Health, Project, Recommendation, Track, Tuning } from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/health"),
  tunings: () => req<Tuning[]>("/tunings"),
  gear: () => req<{ amps: Record<string, { label: string; kind: string }>; pedals: Record<string, { label: string; role: string }> }>("/gear"),
  learning: () => req<Record<string, any>>("/learning"),
  projects: () => req<Project[]>("/projects"),
  project: (id: string) => req<Project>(`/projects/${id}`),
  deleteProject: (id: string) => req<void>(`/projects/${id}`, { method: "DELETE" }),
  reprocess: (id: string, opts: Record<string, unknown>) => req<Project>(`/projects/${id}/reprocess`, json("POST", opts)),
  analysis: (id: string) => req<Analysis>(`/projects/${id}/analysis`),
  tracks: (id: string) => req<{ id: string; name: string; instrument: string; role: string; stats: Track["stats"] }[]>(`/projects/${id}/tracks`),
  track: (id: string, tid: string) => req<Track>(`/projects/${id}/tracks/${tid}`),
  editNote: (id: string, tid: string, nid: string, body: Record<string, unknown>, refinger = false) =>
    req<Track>(`/projects/${id}/tracks/${tid}/notes/${nid}?refinger=${refinger}`, json("PATCH", body)),
  addNote: (id: string, tid: string, body: Record<string, unknown>) => req<Track>(`/projects/${id}/tracks/${tid}/notes`, json("POST", body)),
  deleteNote: (id: string, tid: string, nid: string) => req<Track>(`/projects/${id}/tracks/${tid}/notes/${nid}`, { method: "DELETE" }),
  restore: (id: string, tid: string, index: number) => req<Track>(`/projects/${id}/tracks/${tid}/restore`, json("POST", { index })),
  trackSettings: (id: string, tid: string, body: Record<string, unknown>) => req<Track>(`/projects/${id}/tracks/${tid}/settings`, json("POST", body)),
  reoptimize: (id: string, tid: string) => req<Track>(`/projects/${id}/tracks/${tid}/reoptimize`, { method: "POST" }),
  transcribe: (id: string, stem: string, instrument: string, tuning = "auto") =>
    req<Project>(`/projects/${id}/transcribe`, json("POST", { stem, instrument, tuning })),
  recommend: (id: string, stem: string, amp: string, pedals: string[], guitar?: string) =>
    req<Recommendation>(`/projects/${id}/tone/${stem}/recommend?amp=${encodeURIComponent(amp)}&pedals=${encodeURIComponent(pedals.join(","))}${guitar ? `&guitar=${guitar}` : ""}`),
  fileUrl: (id: string, path: string) => `${BASE}/projects/${id}/files/${path}`,
  exportUrl: (id: string, fmt: string, tracks?: string[], quantized = false) =>
    `${BASE}/projects/${id}/export/${fmt}?${tracks && tracks.length ? `tracks=${encodeURIComponent(tracks.join(","))}&` : ""}quantized=${quantized}`,
  upload: (file: File, opts: Record<string, string | boolean | number | null>, onProgress?: (f: number) => void) =>
    new Promise<Project>((resolve, reject) => {
      const fd = new FormData();
      fd.append("file", file);
      for (const [k, v] of Object.entries(opts)) if (v !== null && v !== "") fd.append(k, String(v));
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/projects`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
      xhr.onload = () => {
        try {
          const body = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) resolve(body);
          else reject(new ApiError(xhr.status, body.detail ?? xhr.statusText));
        } catch {
          reject(new ApiError(xhr.status, xhr.statusText));
        }
      };
      xhr.onerror = () => reject(new ApiError(0, "Network error"));
      xhr.send(fd);
    }),
};

export async function fetchPeaks(id: string, path: string): Promise<{ sr: number; min: number[]; max: number[] }> {
  const res = await fetch(api.fileUrl(id, path));
  if (!res.ok) throw new ApiError(res.status, "peaks unavailable");
  return res.json();
}
