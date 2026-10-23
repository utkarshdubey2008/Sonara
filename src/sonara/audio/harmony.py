"""Harmonic analysis: chroma extraction, key finding, and chord progression estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (standard perceptual weights for tonal music)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class ChordEvent:
    """Estimated chord occurrence."""
    chord_symbol: str
    root: str
    quality: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class HarmonicAnalysisResult:
    """Results from harmonic and key analysis."""
    primary_key: str
    mode: str
    key_confidence: float
    tonal_stability: float
    chord_vocabulary: List[str]
    chord_sequence: List[ChordEvent]
    chroma_profile: List[float]
    uncertainty_flags: List[str] = field(default_factory=list)


class HarmonyAnalyzer:
    """Performs chromagram analysis, key detection, and chord recognition."""

    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        self.hop_length = hop_length
        self.n_fft = n_fft
        self._init_chord_templates()

    def _init_chord_templates(self) -> None:
        """Initialize binary chroma templates for standard triads and sevenths."""
        self.templates: Dict[str, np.ndarray] = {}
        for root_idx, root in enumerate(PITCH_CLASSES):
            # Major triad: root, +4, +7
            maj = np.zeros(12)
            maj[root_idx] = 1.0
            maj[(root_idx + 4) % 12] = 1.0
            maj[(root_idx + 7) % 12] = 1.0
            maj /= np.linalg.norm(maj)
            self.templates[f"{root}:maj"] = maj

            # Minor triad: root, +3, +7
            min_chord = np.zeros(12)
            min_chord[root_idx] = 1.0
            min_chord[(root_idx + 3) % 12] = 1.0
            min_chord[(root_idx + 7) % 12] = 1.0
            min_chord /= np.linalg.norm(min_chord)
            self.templates[f"{root}:min"] = min_chord

            # Dominant 7th: root, +4, +7, +10
            dom7 = np.zeros(12)
            dom7[root_idx] = 1.0
            dom7[(root_idx + 4) % 12] = 1.0
            dom7[(root_idx + 7) % 12] = 1.0
            dom7[(root_idx + 10) % 12] = 1.0
            dom7 /= np.linalg.norm(dom7)
            self.templates[f"{root}:7"] = dom7

            # Diminished: root, +3, +6
            dim = np.zeros(12)
            dim[root_idx] = 1.0
            dim[(root_idx + 3) % 12] = 1.0
            dim[(root_idx + 6) % 12] = 1.0
            dim /= np.linalg.norm(dim)
            self.templates[f"{root}:dim"] = dim

    def analyze(
        self, y: np.ndarray, sr: int, beat_timestamps: Optional[List[float]] = None
    ) -> HarmonicAnalysisResult:
        """
        Analyze audio for harmonic content, key, and chords.

        Args:
            y: Audio waveform array.
            sr: Sample rate.
            beat_timestamps: Optional beat times for beat-synchronous chord tracking.

        Returns:
            HarmonicAnalysisResult.
        """
        uncertainty_flags: List[str] = []
        duration = float(len(y)) / float(sr) if sr > 0 else 0.0

        if duration < 0.5 or np.all(np.abs(y) < 1e-5):
            return HarmonicAnalysisResult(
                primary_key="unknown",
                mode="unknown",
                key_confidence=0.0,
                tonal_stability=0.0,
                chord_vocabulary=[],
                chord_sequence=[],
                chroma_profile=[0.0] * 12,
                uncertainty_flags=["silent_or_too_short"],
            )

        # Compute harmonic component via HPSS to separate tonal content from drums
        try:
            y_harmonic, _ = librosa.effects.hpss(y)
        except Exception:
            y_harmonic = y

        # Chroma CENS (Chroma Energy Normalized Statistics) for stable key detection
        try:
            chroma_cens = librosa.feature.chroma_cens(
                y=y_harmonic, sr=sr, hop_length=self.hop_length
            )
            # Chroma STFT for responsive chord tracking
            chroma_stft = librosa.feature.chroma_stft(
                y=y_harmonic, sr=sr, hop_length=self.hop_length, n_fft=self.n_fft
            )
        except Exception as e:
            uncertainty_flags.append(f"chroma_extraction_failed: {str(e)}")
            return HarmonicAnalysisResult(
                primary_key="unknown",
                mode="unknown",
                key_confidence=0.0,
                tonal_stability=0.0,
                chord_vocabulary=[],
                chord_sequence=[],
                chroma_profile=[0.0] * 12,
                uncertainty_flags=uncertainty_flags,
            )

        # Mean chroma vector across the entire track
        mean_chroma = np.mean(chroma_cens, axis=1)
        chroma_norm = mean_chroma / (np.linalg.norm(mean_chroma) + 1e-8)

        # Krumhansl-Schmuckler key finding
        key_name, mode, key_conf, tonal_stability = self._estimate_key(chroma_norm)

        # Estimate chord progression
        chords = self._estimate_chords(chroma_stft, sr, duration, beat_timestamps)
        vocab = sorted(list(set(c.chord_symbol for c in chords)))

        if key_conf < 0.25:
            uncertainty_flags.append("ambiguous_tonal_center")

        return HarmonicAnalysisResult(
            primary_key=key_name,
            mode=mode,
            key_confidence=round(key_conf, 3),
            tonal_stability=round(tonal_stability, 3),
            chord_vocabulary=vocab,
            chord_sequence=chords,
            chroma_profile=[round(float(v), 4) for v in chroma_norm],
            uncertainty_flags=uncertainty_flags,
        )

    def _estimate_key(self, chroma: np.ndarray) -> Tuple[str, str, float, float]:
        """Estimate key and mode using Pearson correlation against Krumhansl profiles."""
        best_score = -1.0
        best_key = "C"
        best_mode = "major"
        scores: List[Tuple[float, str, str]] = []

        for shift in range(12):
            root = PITCH_CLASSES[shift]
            # Roll profiles to align with root pitch class
            maj_shifted = np.roll(MAJOR_PROFILE, shift)
            min_shifted = np.roll(MINOR_PROFILE, shift)

            # Pearson correlation
            r_maj = float(np.corrcoef(chroma, maj_shifted)[0, 1])
            r_min = float(np.corrcoef(chroma, min_shifted)[0, 1])

            scores.append((r_maj, root, "major"))
            scores.append((r_min, root, "minor"))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_score, best_key, best_mode = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else 0.0

        # Confidence: margin over runner-up normalized by score magnitude
        margin = max(0.0, top_score - second_score)
        confidence = float(np.clip((top_score * 0.6) + (margin * 1.4), 0.0, 1.0))

        # Tonal stability is correlation strength
        tonal_stability = float(np.clip(top_score, 0.0, 1.0))

        return best_key, best_mode, confidence, tonal_stability

    def _estimate_chords(
        self,
        chroma: np.ndarray,
        sr: int,
        duration: float,
        beat_timestamps: Optional[List[float]],
    ) -> List[ChordEvent]:
        """Match chroma segments to chord templates."""
        n_frames = chroma.shape[1]
        if n_frames == 0:
            return []

        frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=self.hop_length)

        # Segment by beats if available, otherwise by 0.5s windows
        if beat_timestamps and len(beat_timestamps) >= 2:
            time_boundaries = [0.0] + [t for t in beat_timestamps if t < duration] + [duration]
        else:
            time_boundaries = list(np.arange(0.0, duration, 0.5)) + [duration]

        raw_chords: List[ChordEvent] = []

        for i in range(len(time_boundaries) - 1):
            t_start = time_boundaries[i]
            t_end = time_boundaries[i + 1]
            if t_end - t_start < 0.05:
                continue

            frame_mask = (frame_times >= t_start) & (frame_times < t_end)
            if not np.any(frame_mask):
                continue

            seg_chroma = np.mean(chroma[:, frame_mask], axis=1)
            norm = np.linalg.norm(seg_chroma)
            if norm < 1e-4:
                continue
            seg_chroma /= norm

            # Find best matching chord template
            best_chord = "N"
            best_sim = -1.0
            for name, tmpl in self.templates.items():
                sim = float(np.dot(seg_chroma, tmpl))
                if sim > best_sim:
                    best_sim = sim
                    best_chord = name

            if best_sim >= 0.5:
                root_part, quality_part = best_chord.split(":")
                raw_chords.append(
                    ChordEvent(
                        chord_symbol=best_chord,
                        root=root_part,
                        quality=quality_part,
                        start_time=round(t_start, 3),
                        end_time=round(t_end, 3),
                        confidence=round(float(np.clip(best_sim, 0.0, 1.0)), 3),
                    )
                )

        # Merge adjacent identical chords
        merged: List[ChordEvent] = []
        for ch in raw_chords:
            if merged and merged[-1].chord_symbol == ch.chord_symbol and abs(merged[-1].end_time - ch.start_time) < 0.1:
                merged[-1].end_time = ch.end_time
                merged[-1].confidence = round((merged[-1].confidence + ch.confidence) / 2.0, 3)
            else:
                merged.append(ch)

        return merged
