"""Tuning catalogue and pitch helpers.

Conventions used throughout the code base (matching Guitar Pro / MusicXML TAB):

* Tunings are stored low string -> high string as MIDI note numbers.
* Strings are numbered 1..N from the **highest** pitched string (1 = high e).
"""

from __future__ import annotations

from dataclasses import dataclass

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def midi_to_name(m: int, flats: bool = False) -> str:
    names = NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP
    return f"{names[m % 12]}{m // 12 - 1}"


def name_to_midi(name: str) -> int:
    name = name.strip()
    letter = name[0].upper()
    rest = name[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    octave = int(rest) if rest else 4
    return (octave + 1) * 12 + base + acc


@dataclass(frozen=True)
class Tuning:
    id: str
    name: str
    notes: tuple[int, ...]  # low -> high, MIDI
    instrument: str = "guitar"  # guitar | bass
    prior: float = 0.01  # how common the tuning is (used as a Bayesian prior)

    @property
    def strings(self) -> int:
        return len(self.notes)

    def open_midi(self, string: int) -> int:
        """Open-string pitch for string number (1 = highest)."""
        return self.notes[self.strings - string]

    def label(self) -> str:
        return " ".join(midi_to_name(n)[:-1] for n in self.notes)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "notes": list(self.notes),
            "note_names": [midi_to_name(n) for n in self.notes],
            "strings": self.strings,
            "instrument": self.instrument,
            "label": self.label(),
        }


def _t(id_: str, name: str, notes: str, prior: float, instrument: str = "guitar") -> Tuning:
    return Tuning(id_, name, tuple(name_to_midi(n) for n in notes.split()), instrument, prior)


TUNINGS: dict[str, Tuning] = {
    t.id: t
    for t in [
        # Priors are rough frequencies across popular guitar music (sum need not be 1).
        _t("standard", "Standard (E A D G B E)", "E2 A2 D3 G3 B3 E4", 0.55),
        _t("eb_standard", "Half step down (Eb)", "Eb2 Ab2 Db3 Gb3 Bb3 Eb4", 0.10),
        _t("drop_d", "Drop D", "D2 A2 D3 G3 B3 E4", 0.10),
        _t("d_standard", "D standard (whole step down)", "D2 G2 C3 F3 A3 D4", 0.04),
        _t("drop_csharp", "Drop C#", "C#2 G#2 C#3 F#3 A#3 D#4", 0.03),
        _t("drop_c", "Drop C", "C2 G2 C3 F3 A3 D4", 0.04),
        _t("c_standard", "C standard", "C2 F2 A#2 D#3 G3 C4", 0.015),
        _t("drop_b", "Drop B", "B1 F#2 B2 E3 G#3 C#4", 0.015),
        _t("b_standard", "B standard", "B1 E2 A2 D3 F#3 B3", 0.01),
        _t("open_g", "Open G", "D2 G2 D3 G3 B3 D4", 0.02),
        _t("open_d", "Open D", "D2 A2 D3 F#3 A3 D4", 0.015),
        _t("open_e", "Open E", "E2 B2 E3 G#3 B3 E4", 0.01),
        _t("open_c", "Open C", "C2 G2 C3 G3 C4 E4", 0.005),
        _t("dadgad", "DADGAD", "D2 A2 D3 G3 A3 D4", 0.015),
        _t("double_drop_d", "Double drop D", "D2 A2 D3 G3 B3 D4", 0.005),
        _t("7_standard", "7-string standard (B)", "B1 E2 A2 D3 G3 B3 E4", 0.03),
        _t("7_drop_a", "7-string drop A", "A1 E2 A2 D3 G3 B3 E4", 0.01),
        _t("8_standard", "8-string standard (F#)", "F#1 B1 E2 A2 D3 G3 B3 E4", 0.01),
        _t("8_drop_e", "8-string drop E", "E1 B1 E2 A2 D3 G3 B3 E4", 0.005),
        _t("bass_standard", "Bass standard (E A D G)", "E1 A1 D2 G2", 0.8, "bass"),
        _t("bass_drop_d", "Bass drop D", "D1 A1 D2 G2", 0.1, "bass"),
        _t("bass_5_standard", "5-string bass (B E A D G)", "B0 E1 A1 D2 G2", 0.1, "bass"),
        _t("bass_eb", "Bass half step down", "Eb1 Ab1 Db2 Gb2", 0.05, "bass"),
    ]
}


def get_tuning(tuning_id: str) -> Tuning:
    return TUNINGS[tuning_id]


def custom_tuning(notes: list[int], name: str = "Custom", instrument: str = "guitar") -> Tuning:
    if not 4 <= len(notes) <= 9:
        raise ValueError("A tuning needs between 4 and 9 strings")
    if any(not 18 <= n <= 80 for n in notes):
        raise ValueError("Tuning note out of range")
    return Tuning("custom", name, tuple(int(n) for n in notes), instrument, 0.001)


def tuning_from_dict(d: dict) -> Tuning:
    tid = d.get("id", "custom")
    if tid in TUNINGS and tuple(d.get("notes", TUNINGS[tid].notes)) == TUNINGS[tid].notes:
        return TUNINGS[tid]
    return custom_tuning(list(d["notes"]), d.get("name", "Custom"), d.get("instrument", "guitar"))


def candidate_tunings(instrument: str, strings: int | None = None) -> list[Tuning]:
    out = [t for t in TUNINGS.values() if t.instrument == instrument]
    if strings:
        out = [t for t in out if t.strings == strings]
    return out
