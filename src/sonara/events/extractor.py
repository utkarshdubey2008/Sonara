"""Discrete musical event extraction from audio waveforms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import librosa
import numpy as np
from scipy.signal import find_peaks

from sonara.events.classifier import LayerClassifier


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_num: int) -> str:
    """Convert MIDI note number (0-127) to standard note name with octave (e.g. 60 -> C4)."""
    pitch_class = NOTE_NAMES[midi_num % 12]
    octave = (midi_num // 12) - 1
    return f"{pitch_class}{octave}"


def hz_to_midi(freq_hz: float) -> Optional[int]:
    """Convert frequency in Hz to closest MIDI note number."""
    if freq_hz <= 0 or math.isnan(freq_hz):
        return None
    midi = 69 + 12 * math.log2(freq_hz / 440.0)
    return int(round(midi))


@dataclass
class ExtractedEvent:
    """A discrete musical event detected in audio."""
    event_id: str
    start_time: float
    duration: float
    end_time: float
    layer: str  # percussion, bass, harmonic, melodic, unknown
    event_type: str  # note, hit, chord, rest
    pitch_midi: Optional[int]
    pitch_name: Optional[str]
    frequency_hz: Optional[float]
    cents_deviation: Optional[float]
    velocity: int  # 0 to 127
    confidence: float
    uncertainty_flags: List[str] = field(default_factory=list)


class EventExtractor:
    """Extracts discrete musical events and note parameters from audio signals."""

    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_events(
        self, y: np.ndarray, sr: int, onset_timestamps: List[float]
    ) -> List[ExtractedEvent]:
        """
        Extract discrete musical events at detected onset locations.

        Args:
            y: Audio waveform array.
            sr: Sample rate.
            onset_timestamps: List of onset times in seconds.

        Returns:
            List of ExtractedEvent instances.
        """
        events: List[ExtractedEvent] = []
        total_duration = float(len(y)) / float(sr) if sr > 0 else 0.0

        if not onset_timestamps or total_duration <= 0.0:
            return events

        # Sort onsets
        sorted_onsets = sorted([t for t in onset_timestamps if 0.0 <= t < total_duration])
        if not sorted_onsets:
            return events

        # Precompute spectral features for quick frame querying
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        max_rms = float(np.max(rms)) if len(rms) > 0 and np.max(rms) > 0 else 1.0

        n_frames = len(rms)
        frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=self.hop_length)

        for i, t_start in enumerate(sorted_onsets):
            event_id = f"evt_{i:04d}"
            # Duration extends until the next onset or max 2.0s
            t_next = sorted_onsets[i + 1] if i + 1 < len(sorted_onsets) else total_duration
            raw_dur = min(2.0, max(0.05, t_next - t_start))
            t_end = min(total_duration, t_start + raw_dur)
            dur = t_end - t_start

            # Find matching frame index
            frame_idx = int(np.clip(t_start * sr / self.hop_length, 0, n_frames - 1))
            cur_rms = float(rms[frame_idx]) if frame_idx < n_frames else 0.0
            cur_cent = float(spec_cent[frame_idx]) if frame_idx < len(spec_cent) else 1000.0
            cur_zcr = float(zcr[frame_idx]) if frame_idx < len(zcr) else 0.0

            # Velocity normalized from RMS amplitude
            velocity = int(np.clip(127.0 * (cur_rms / max_rms), 10, 127))

            # Extract windowed slice for pitch estimation
            s_start = int(t_start * sr)
            s_end = min(len(y), s_start + int(min(dur, 0.4) * sr))
            slice_y = y[s_start:s_end]

            pitch_midi: Optional[int] = None
            pitch_name: Optional[str] = None
            f0_hz: Optional[float] = None
            cents: Optional[float] = None
            is_harmonic = False
            uncertainty_flags: List[str] = []

            if len(slice_y) >= self.n_fft // 2:
                f0, harmonicity = self._estimate_pitch(slice_y, sr)
                if f0 is not None and 30.0 <= f0 <= 4000.0 and harmonicity > 0.35:
                    f0_hz = round(f0, 2)
                    raw_midi = 69 + 12 * math.log2(f0 / 440.0)
                    m_int = int(round(raw_midi))
                    if 12 <= m_int <= 120:
                        pitch_midi = m_int
                        pitch_name = midi_to_note_name(m_int)
                        cents = round((raw_midi - m_int) * 100.0, 1)
                        is_harmonic = True
                        if harmonicity < 0.55:
                            uncertainty_flags.append("pitch_uncertain")
                else:
                    uncertainty_flags.append("unpitched_or_percussive")
            else:
                uncertainty_flags.append("short_transient")

            # Layer assignment
            layer = LayerClassifier.classify_event(
                pitch_midi=pitch_midi,
                frequency_hz=f0_hz,
                spectral_centroid=cur_cent,
                zero_crossing_rate=cur_zcr,
                is_harmonic=is_harmonic,
            )

            # Event type
            if layer == LayerClassifier.LAYER_PERCUSSION:
                event_type = "hit"
            elif pitch_midi is not None:
                event_type = "note"
            else:
                event_type = "onset"

            # Confidence score
            base_conf = 0.85 if is_harmonic else 0.65
            if "pitch_uncertain" in uncertainty_flags:
                base_conf -= 0.20
            if cur_rms / max_rms < 0.1:
                base_conf -= 0.15
                uncertainty_flags.append("low_energy")

            confidence = float(np.clip(base_conf, 0.1, 1.0))

            events.append(
                ExtractedEvent(
                    event_id=event_id,
                    start_time=round(t_start, 3),
                    duration=round(dur, 3),
                    end_time=round(t_end, 3),
                    layer=layer,
                    event_type=event_type,
                    pitch_midi=pitch_midi,
                    pitch_name=pitch_name,
                    frequency_hz=f0_hz,
                    cents_deviation=cents,
                    velocity=velocity,
                    confidence=round(confidence, 3),
                    uncertainty_flags=uncertainty_flags,
                )
            )

        return events

    def _estimate_pitch(self, audio_slice: np.ndarray, sr: int) -> tuple[Optional[float], float]:
        """Estimate fundamental frequency via normalized autocorrelation on windowed slice."""
        n = len(audio_slice)
        if n < 256:
            return None, 0.0

        # Apply Hanning window
        windowed = audio_slice * np.hanning(n)

        # Autocorrelation via FFT
        fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fft = np.fft.rfft(windowed, n=fft_size)
        ac = np.fft.irfft(fft * np.conj(fft), n=fft_size)[:n]

        if ac[0] <= 1e-8:
            return None, 0.0

        ac_norm = ac / ac[0]

        # Search in reasonable musical pitch range (50 Hz to 2000 Hz)
        min_lag = max(2, int(sr / 2000.0))
        max_lag = min(n - 1, int(sr / 50.0))

        if min_lag >= max_lag:
            return None, 0.0

        search_region = ac_norm[min_lag:max_lag]
        if len(search_region) == 0:
            return None, 0.0

        # Find peaks
        peaks, props = find_peaks(search_region, height=0.3, distance=int(sr / 4000.0) + 1)
        if len(peaks) == 0:
            return None, 0.0

        best_peak_idx = peaks[np.argmax(props["peak_heights"])]
        actual_lag = min_lag + best_peak_idx
        harmonicity = float(ac_norm[actual_lag])

        # Parabolic interpolation for fine sub-sample lag resolution
        if 0 < actual_lag < n - 1:
            alpha = ac_norm[actual_lag - 1]
            beta = ac_norm[actual_lag]
            gamma = ac_norm[actual_lag + 1]
            denom = 2 * (alpha - 2 * beta + gamma)
            if abs(denom) > 1e-6:
                delta = (alpha - gamma) / denom
                refined_lag = actual_lag + delta
            else:
                refined_lag = float(actual_lag)
        else:
            refined_lag = float(actual_lag)

        if refined_lag > 0:
            f0 = sr / refined_lag
            return f0, harmonicity

        return None, 0.0
