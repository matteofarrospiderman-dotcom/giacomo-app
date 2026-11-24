export type Level = "high" | "medium" | "low" | "user";

export interface Technique {
  type: string;
  confidence?: number;
  semitones?: number;
  direction?: string;
  [k: string]: unknown;
}

export interface Alternative {
  string: number;
  fret: number;
  cost_delta: number;
  shape?: [number, number][] | null;
}

export interface Note {
  id: string;
  start: number;
  end: number;
  pitch: number;
  name: string;
  velocity: number;
  confidence: number;
  level: Level;
  sources: Record<string, number | string>;
  flags: string[];
  techniques: Technique[];
  string: number | null;
  fret: number | null;
  locked: boolean;
  edited: boolean;
  dead?: boolean;
  alternatives?: Alternative[];
  group?: number;
  tick?: number;
  dur_ticks?: number;
  measure?: number;
  contour?: { t0: number; hop: number; values: number[] };
}

export interface Tuning {
  id: string;
  name: string;
  notes: number[];
  note_names: string[];
  strings: number;
  instrument: string;
  label: string;
}

export interface Measure {
  index: number;
  start: number;
  end: number;
  start_tick: number;
  ticks: number;
  time_signature: [number, number];
  bpm: number;
}

export interface NotatedBeat {
  tick: number;
  ticks: number;
  value: number;
  dotted: boolean;
  tuplet: [number, number] | null;
  rest: boolean;
  notes: { id: string; tie: boolean }[];
}

export interface ReviewItem {
  note_id: string;
  time: number;
  flag: string;
  message: string;
  severity: "high" | "medium" | "low";
}

export interface Track {
  id: string;
  name: string;
  stem: string;
  instrument: "guitar" | "bass";
  role: string;
  tuning: Tuning;
  capo: number;
  max_fret: number;
  profile: string;
  notes: Note[];
  rejected: { start: number; end: number; pitch: number; confidence: number; reason: string }[];
  measures: Measure[];
  notation: { index: number; beats: NotatedBeat[]; time_signature: [number, number] }[];
  grid: { time_signature: [number, number]; beats_per_bar: number; ticks_per_beat: number; bpm: number };
  review: ReviewItem[];
  stats: { notes: number; levels: Record<Level, number>; flagged: number; mean_confidence: number; techniques: Record<string, number> };
  playability: Record<string, number>;
  fingering: { total_cost: number; profile: string; profile_label: string; uncertain_groups: number };
  alternative_fingerings: Record<string, { label: string; changes: Record<string, [number, number]>; n_changed: number; report: Record<string, number> }>;
  tuning_detection?: TuningDetection | null;
  downbeat_offset?: number;
  time_signature?: [number, number];
  engines: Record<string, string>;
  dead_strums?: { time: number; confidence: number }[];
  part_split?: { adopted: boolean; overlay_notes: number; confidence: number } | null;
  version: number;
}

export interface TuningDetection {
  best: Tuning;
  confidence: number;
  ranking: { tuning: Tuning; probability: number }[];
  reference: { cents: number; a4_hz: number; note: string | null };
  evidence: string[];
}

export interface StemInfo {
  name: string;
  label: string;
  wav: string;
  preview: string;
  peaks: string;
  parent?: string;
  pan?: number;
  analysis?: {
    energy_share: number;
    active_ratio: number;
    present: boolean;
    crest_db: number;
    rolloff_hz: number;
    guitar_type?: { type: string; p_acoustic: number; confidence: number; evidence: string[] };
    content?: { guesses: { label: string; confidence: number }[] };
  };
}

export interface Chord {
  start: number;
  end: number;
  name: string;
  root: number | null;
  quality: string | null;
  confidence: number;
  confidence_level: string;
  flag?: string;
}

export interface Effect {
  effect: string;
  key: string;
  detected: boolean;
  confidence: number;
  tier: "high" | "probable" | "possible" | "uncertain";
  summary: string;
  details: Record<string, any>;
}

export interface ToneAnalysis {
  available: boolean;
  effects: Effect[];
  amp_character: { style: string; gain: number; eq: Record<string, number>; cabinet: string };
  eq: { spectrum: { freqs: number[]; db: number[] }; character: string[]; band_levels_db: Record<string, number> };
  disclaimer: string;
}

export interface Analysis {
  duration: number;
  signal: { peak_dbfs: number; rms_dbfs: number; clipping_ratio: number };
  mix: { wav: string; preview: string; peaks: string };
  separation: { engine: string; model: string; quality: string; notes: string[] };
  stems: Record<string, StemInfo>;
  rhythm: {
    bpm: number;
    tempo_confidence: number;
    tempo_stability: number;
    tempo_candidates: { bpm: number; score: number }[];
    beats: number[];
    downbeats: number[];
    time_signature: [number, number];
    meter_confidence: number;
    feel: string;
    source: string;
  };
  key: { name: string; confidence: number; alternatives: { name: string; correlation: number }[] };
  chords: Chord[];
  chord_validation?: { chords_checked: number; agreement: number | null };
  tuning?: TuningDetection;
  tone?: Record<string, ToneAnalysis>;
  tracks?: { id: string; name: string; stem: string; instrument: string; role: string }[];
  guitar_split?: { analysis: { modes?: { pan: number; share: number }[]; pan_histogram?: number[]; reason?: string }; double_tracked: boolean; parts: string[] };
  processing_seconds?: number;
  reference_deviation_cents?: number;
}

export interface ProjectStatus {
  state: "uploaded" | "queued" | "processing" | "done" | "failed";
  stage: string;
  progress: number;
  message: string;
  log: { t: number; stage: string; message: string }[];
  error?: string;
}

export interface Project {
  id: string;
  name: string;
  created_at: number;
  updated_at: number;
  status: ProjectStatus;
  summary: null | {
    duration: number;
    bpm: number;
    time_signature: [number, number];
    key: string;
    tuning: string | null;
    tracks: number;
    separation: string;
    notes: number;
  };
  source: null | { original_name: string; bytes: number };
  options?: Record<string, unknown>;
}

export interface Health {
  status: string;
  ffmpeg: boolean;
  engines: Record<string, Record<string, unknown>>;
  device: string;
  limits: { max_upload_mb: number; max_duration_s: number; formats: string[] };
}

export interface Recommendation {
  chain: { block: string; tier: string; available: boolean; settings: Record<string, unknown>; character?: string }[];
  notes: string[];
}
