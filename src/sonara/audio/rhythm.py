"""Rhythm analysis: tempo estimation, beat tracking, onset detection, and groove metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import librosa
import numpy as np


@dataclass
class RhythmAnalysisResult:
    """Results from rhythmic and metric analysis."""
    estimated_bpm: float
    tempo_confidence: float
    time_signature_estimate: str
    beat_timestamps: List[float]
    onset_timestamps: List[float]
    onset_count: int
    onset_density_per_second: float
    syncopation_index: float
    pulse_clarity: float
    is_rhythmic: bool
    uncertainty_flags: List[str] = field(default_factory=list)


class RhythmAnalyzer:
    """Analyzes rhythmic properties of audio waveforms."""

    def __init__(self, hop_length: int = 512):
        self.hop_length = hop_length

    def analyze(self, y: np.ndarray, sr: int) -> RhythmAnalysisResult:
        """
        Extract rhythm, tempo, beats, and onsets.

        Args:
            y: Audio waveform array (1D float).
            sr: Sample rate.

        Returns:
            RhythmAnalysisResult instance.
        """
        uncertainty_flags: List[str] = []
        duration = float(len(y)) / float(sr) if sr > 0 else 0.0

        if duration < 0.2 or np.all(np.abs(y) < 1e-5):
            return RhythmAnalysisResult(
                estimated_bpm=0.0,
                tempo_confidence=0.0,
                time_signature_estimate="unknown",
                beat_timestamps=[],
                onset_timestamps=[],
                onset_count=0,
                onset_density_per_second=0.0,
                syncopation_index=0.0,
                pulse_clarity=0.0,
                is_rhythmic=False,
                uncertainty_flags=["insufficient_audio_or_silent"],
            )

        # Onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Onset peak picking
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, backtrack=True
        )
        onset_times = [float(t) for t in librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)]
        onset_count = len(onset_times)
        density = float(onset_count / duration) if duration > 0 else 0.0

        # Pulse clarity (autocorrelation of onset envelope)
        if len(onset_env) > 10:
            ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
            ac_normalized = ac / (np.max(ac) + 1e-8)
            # Pulse clarity = ratio of second highest peak to total energy
            if len(ac_normalized) > 50:
                pulse_clarity = float(np.max(ac_normalized[10:50]))
            else:
                pulse_clarity = 0.0
        else:
            pulse_clarity = 0.0

        # Beat tracking & tempo
        try:
            tempo_arr, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
            )
            bpm = float(tempo_arr[0]) if isinstance(tempo_arr, np.ndarray) and tempo_arr.ndim > 0 else float(tempo_arr)
            beat_times = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)]
        except Exception as e:
            bpm = 0.0
            beat_times = []
            uncertainty_flags.append(f"beat_tracking_failed: {str(e)}")

        # Assess tempo confidence and rhythmicity
        if bpm <= 0 or len(beat_times) < 2 or pulse_clarity < 0.05:
            tempo_confidence = 0.1
            is_rhythmic = False
            uncertainty_flags.append("weak_or_irregular_beat")
        else:
            tempo_confidence = min(1.0, max(0.1, pulse_clarity * 1.5))
            is_rhythmic = True

        # Syncopation estimation: check what proportion of onsets fall far from beat times
        syncopation_idx = 0.0
        if is_rhythmic and len(beat_times) > 1 and len(onset_times) > 2:
            beat_interval = 60.0 / bpm
            tolerance = beat_interval * 0.15  # within 15% of a beat is on-beat
            off_beat_count = 0
            for ot in onset_times:
                closest_dist = min(abs(ot - bt) for bt in beat_times)
                if closest_dist > tolerance:
                    off_beat_count += 1
            syncopation_idx = float(off_beat_count / len(onset_times))

        # Time signature heuristic (simple 4/4 vs 3/4 check based on beat accentuation)
        time_sig = "4/4"  # Default assumption in western popular music
        if is_rhythmic and len(beat_frames) >= 8 and len(onset_env) > 0:
            beat_strengths = []
            for bf in beat_frames:
                if bf < len(onset_env):
                    beat_strengths.append(float(onset_env[bf]))
            if len(beat_strengths) >= 12:
                # Check 3-beat periodicity vs 4-beat periodicity
                b_arr = np.array(beat_strengths)
                corr3 = np.corrcoef(b_arr[:-3], b_arr[3:])[0, 1] if len(b_arr) > 3 else 0
                corr4 = np.corrcoef(b_arr[:-4], b_arr[4:])[0, 1] if len(b_arr) > 4 else 0
                if not np.isnan(corr3) and not np.isnan(corr4):
                    if corr3 > corr4 + 0.15 and corr3 > 0.3:
                        time_sig = "3/4"
                    else:
                        time_sig = "4/4"

        return RhythmAnalysisResult(
            estimated_bpm=round(bpm, 2),
            tempo_confidence=round(tempo_confidence, 3),
            time_signature_estimate=time_sig,
            beat_timestamps=beat_times,
            onset_timestamps=onset_times,
            onset_count=onset_count,
            onset_density_per_second=round(density, 2),
            syncopation_index=round(syncopation_idx, 3),
            pulse_clarity=round(pulse_clarity, 3),
            is_rhythmic=is_rhythmic,
            uncertainty_flags=uncertainty_flags,
        )
