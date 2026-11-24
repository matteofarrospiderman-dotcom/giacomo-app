const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

export const midiName = (m: number) => `${NAMES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`;

export function fmtTime(t: number): string {
  if (!isFinite(t)) return "0:00.0";
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export const pct = (x: number) => `${Math.round(x * 100)}%`;

export const levelColor: Record<string, string> = {
  high: "var(--ok)",
  medium: "var(--warn)",
  low: "var(--bad)",
  user: "var(--user)",
};

export const tierColor: Record<string, string> = {
  high: "var(--ok)",
  probable: "var(--accent)",
  possible: "var(--warn)",
  uncertain: "var(--muted)",
};

export const TECHNIQUES: { type: string; label: string; short: string }[] = [
  { type: "hammer_on", label: "Hammer-on", short: "h" },
  { type: "pull_off", label: "Pull-off", short: "p" },
  { type: "slide", label: "Slide", short: "/" },
  { type: "bend", label: "Bend", short: "b" },
  { type: "bend_release", label: "Bend & release", short: "br" },
  { type: "prebend_release", label: "Pre-bend release", short: "pb" },
  { type: "vibrato", label: "Vibrato", short: "~" },
  { type: "palm_mute", label: "Palm mute", short: "PM" },
  { type: "let_ring", label: "Let ring", short: "LR" },
  { type: "harmonic", label: "Natural harmonic", short: "<>" },
  { type: "tremolo_picking", label: "Tremolo picking", short: "tr" },
];

export const FLAG_LABELS: Record<string, string> = {
  ambiguous_pitch: "Ambiguous pitch",
  octave_disagreement: "Octave disagreement",
  possible_harmonic: "Possible overtone",
  uncertain_string: "Uncertain string",
  impossible_fingering: "Impossible fingering",
  unusual_timing: "Unusual timing",
  overlapping_instrument: "Overlapping instrument",
  single_source: "Single engine only",
  weak_evidence: "Weak evidence",
  out_of_range: "Out of range",
  chord_disagreement: "Chord disagreement",
};

export const ENGINE_LABELS: Record<string, string> = {
  basic_pitch: "Basic Pitch",
  dsp: "DSP salience",
  f0: "f0 tracker",
  dsp_primary: "DSP (primary)",
  user: "You",
  restored: "Restored",
};
