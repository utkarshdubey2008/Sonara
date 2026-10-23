"""Discrete musical event extraction from audio waveforms with multi-layer decomposition."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    source_id: Optional[str] = None
    instrument_role: Optional[str] = None


class EventExtractor:
    """Extracts discrete musical events and note parameters with multi-layer decomposition."""

    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_events(
        self, y: np.ndarray, sr: int, onset_timestamps: List[float]
    ) -> List[ExtractedEvent]:
        """
        Extract multi-layered discrete musical events (percussion, bass, melodic).

        Args:
            y: Audio waveform array.
            sr: Sample rate.
            onset_timestamps: List of onset times in seconds.

        Returns:
            List of ExtractedEvent instances sorted by start time.
        """
        events: List[ExtractedEvent] = []
        total_duration = float(len(y)) / float(sr) if sr > 0 else 0.0

        if not onset_timestamps or total_duration <= 0.0:
            return events

        # Filter and sort onsets
        sorted_onsets = sorted([t for t in onset_timestamps if 0.0 <= t < total_duration])
        if not sorted_onsets:
            return events

        # Precompute short-time Fourier transform for spectral querying
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        max_rms = float(np.max(rms)) if len(rms) > 0 and np.max(rms) > 0 else 1.0

        # Separate transient percussive energy and harmonic resonance via HPSS
        y_harm, y_perc = y, y
        if len(y) >= self.n_fft * 2:
            try:
                y_harm, y_perc = librosa.effects.hpss(y)
            except Exception:
                y_harm, y_perc = y, y

        hop = self.hop_length
        # Percussive activity envelope to gate drum hits during quiet intros and breakdowns
        rms_perc = librosa.feature.rms(y=y_perc, hop_length=hop)[0]
        max_rms_perc = float(np.max(rms_perc)) if len(rms_perc) > 0 and np.max(rms_perc) > 0 else 1.0

        # Sub-band signals on percussive and harmonic components
        from scipy.signal import butter, sosfilt
        sos_k = butter(4, [35.0, 120.0], btype="bandpass", fs=sr, output="sos")
        y_kick = sosfilt(sos_k, y_perc)
        rms_k = librosa.feature.rms(y=y_kick, hop_length=self.hop_length)[0]
        max_k = float(np.max(rms_k)) if len(rms_k) > 0 and np.max(rms_k) > 0 else 1.0

        sos_s = butter(4, [250.0, 2500.0], btype="bandpass", fs=sr, output="sos")
        y_snare = sosfilt(sos_s, y_perc)
        rms_s = librosa.feature.rms(y=y_snare, hop_length=self.hop_length)[0]
        max_s = float(np.max(rms_s)) if len(rms_s) > 0 and np.max(rms_s) > 0 else 1.0

        sos_b = butter(4, 140.0, btype="lowpass", fs=sr, output="sos")
        y_bass = sosfilt(sos_b, y_harm)
        rms_b = librosa.feature.rms(y=y_bass, hop_length=self.hop_length)[0]
        max_b = float(np.max(rms_b)) if len(rms_b) > 0 and np.max(rms_b) > 0 else 1.0

        low_mask = (freqs >= 30.0) & (freqs <= 160.0)
        mid_mask = (freqs >= 200.0) & (freqs <= 2500.0)
        high_mask = (freqs >= 4000.0)

        n_frames = stft.shape[1]
        event_counter = 0
        prev_kick_time: Optional[float] = None

        for i, t_start in enumerate(sorted_onsets):
            t_next = sorted_onsets[i + 1] if i + 1 < len(sorted_onsets) else total_duration
            dur = min(2.0, max(0.06, t_next - t_start))
            t_end = min(total_duration, t_start + dur)

            frame_idx = int(np.clip(t_start * sr / self.hop_length, 0, n_frames - 1))
            cur_rms = float(rms[frame_idx]) if frame_idx < len(rms) else 0.0
            norm_vel = int(np.clip(127.0 * (cur_rms / max_rms), 20, 127))

            k_val = float(rms_k[frame_idx]) if frame_idx < len(rms_k) else 0.0
            s_val = float(rms_s[frame_idx]) if frame_idx < len(rms_s) else 0.0
            b_val = float(rms_b[frame_idx]) if frame_idx < len(rms_b) else 0.0

            frame_spec = stft[:, frame_idx]
            total_frame_energy = np.sum(frame_spec) + 1e-8
            low_energy_ratio = float(np.sum(frame_spec[low_mask]) / total_frame_energy)
            mid_energy_ratio = float(np.sum(frame_spec[mid_mask]) / total_frame_energy)
            high_energy_ratio = float(np.sum(frame_spec[high_mask]) / total_frame_energy)

            # Windowed slice for fundamental frequency analysis
            s_start = int(t_start * sr)
            s_end = min(len(y), s_start + int(min(dur, 0.35) * sr))
            slice_y = y[s_start:s_end]

            f0, harmonicity = self._estimate_pitch(slice_y, sr) if len(slice_y) >= self.n_fft // 2 else (None, 0.0)

            # Percussion Engine: only trigger when genuine percussive energy exists
            # In quiet intros, breakdowns, or outro fades, k_val < 0.04 and s_val < 0.03
            is_perc_active = (k_val >= 0.05 or s_val >= 0.045) or (low_energy_ratio > 0.32 and cur_rms / max_rms > 0.20)

            if is_perc_active:
                # 1. Kick Drum: strong sub-120Hz percussive punch
                if k_val >= 0.06 or (low_energy_ratio > 0.30 and cur_rms / max_rms > 0.18):
                    kick_vel = int(np.clip(127.0 * (k_val / max_k * 1.15), 40, 127))
                    # Drop entrance accentuation: if kick follows prolonged silence (>2.0s), boost impact
                    if prev_kick_time is None or (t_start - prev_kick_time > 2.0):
                        kick_vel = int(np.clip(kick_vel * 1.5, 95, 127))
                    prev_kick_time = t_start

                    events.append(
                        ExtractedEvent(
                            event_id=f"evt_{event_counter:04d}",
                            start_time=round(t_start, 3),
                            duration=round(min(0.25, dur), 3),
                            end_time=round(t_start + min(0.25, dur), 3),
                            layer=LayerClassifier.LAYER_PERCUSSION,
                            event_type="hit",
                            pitch_midi=36,  # Standard General MIDI Kick
                            pitch_name="C1",
                            frequency_hz=65.4,
                            cents_deviation=0.0,
                            velocity=kick_vel,
                            confidence=0.92,
                            uncertainty_flags=[],
                            source_id="src_kick",
                            instrument_role="kick_drum",
                        )
                    )
                    event_counter += 1

                # 2. Snare Drum: crisp mid-range percussive crack
                if s_val >= 0.055 or (mid_energy_ratio > 0.35 and (f0 is None or harmonicity < 0.45) and cur_rms / max_rms > 0.18):
                    snare_vel = int(np.clip(127.0 * (s_val / max_s * 1.15), 35, 127))
                    events.append(
                        ExtractedEvent(
                            event_id=f"evt_{event_counter:04d}",
                            start_time=round(t_start, 3),
                            duration=round(min(0.18, dur), 3),
                            end_time=round(t_start + min(0.18, dur), 3),
                            layer=LayerClassifier.LAYER_PERCUSSION,
                            event_type="hit",
                            pitch_midi=38,  # Standard General MIDI Snare
                            pitch_name="D1",
                            frequency_hz=73.4,
                            cents_deviation=0.0,
                            velocity=snare_vel,
                            confidence=0.88,
                            uncertainty_flags=[],
                            source_id="src_snare",
                            instrument_role="snare_backbeat",
                        )
                    )
                    event_counter += 1

                # 3. Closed Hi-Hat: high-frequency metallic crispness
                if high_energy_ratio > 0.16 and cur_rms / max_rms > 0.08:
                    hat_vel = int(np.clip(127.0 * (cur_rms / max_rms * 0.75), 20, 115))
                    events.append(
                        ExtractedEvent(
                            event_id=f"evt_{event_counter:04d}",
                            start_time=round(t_start, 3),
                            duration=round(min(0.08, dur), 3),
                            end_time=round(t_start + min(0.08, dur), 3),
                            layer=LayerClassifier.LAYER_PERCUSSION,
                            event_type="hit",
                            pitch_midi=42,  # Standard General MIDI Closed Hat
                            pitch_name="F#1",
                            frequency_hz=92.5,
                            cents_deviation=0.0,
                            velocity=hat_vel,
                            confidence=0.85,
                            uncertainty_flags=[],
                            source_id="src_hihat",
                            instrument_role="hihat_timekeeper",
                        )
                    )
                    event_counter += 1

            # Pitched Note Extraction (Bass or Strings/Harmonic Bed)
            if f0 is not None and 35.0 <= f0 <= 220.0 and harmonicity > 0.35:
                raw_midi = 69 + 12 * math.log2(f0 / 440.0)
                m_int = int(round(raw_midi))
                if 20 <= m_int <= 55:
                    # Bassline: requires genuine low-frequency bass energy (b_val >= 0.08)
                    # and active rhythm section, preventing spurious bass in strings-only intro/outro
                    if (b_val >= 0.08 and is_perc_active) or (low_energy_ratio > 0.35 and cur_rms / max_rms > 0.20):
                        cents = round((raw_midi - m_int) * 100.0, 1)
                        p_name = midi_to_note_name(m_int)
                        bass_vel = int(np.clip(127.0 * (b_val / max_b * 1.15), 35, 127))
                        events.append(
                            ExtractedEvent(
                                event_id=f"evt_{event_counter:04d}",
                                start_time=round(t_start, 3),
                                duration=round(dur, 3),
                                end_time=round(t_end, 3),
                                layer=LayerClassifier.LAYER_BASS,
                                event_type="note",
                                pitch_midi=m_int,
                                pitch_name=p_name,
                                frequency_hz=round(f0, 2),
                                cents_deviation=cents,
                                velocity=bass_vel,
                                confidence=0.88,
                                uncertainty_flags=[],
                                source_id="src_bass",
                                instrument_role="synth_bass",
                            )
                        )
                        event_counter += 1
            elif f0 is not None and f0 > 220.0 and harmonicity > 0.70 and dur >= 0.18:
                raw_midi = 69 + 12 * math.log2(f0 / 440.0)
                m_int = int(round(raw_midi))
                if 55 < m_int <= 108:
                    cents = round((raw_midi - m_int) * 100.0, 1)
                    p_name = midi_to_note_name(m_int)
                    # Route to LAYER_HARMONIC (strings/pads) to eliminate electric piano plucks!
                    events.append(
                        ExtractedEvent(
                            event_id=f"evt_{event_counter:04d}",
                            start_time=round(t_start, 3),
                            duration=round(dur, 3),
                            end_time=round(t_end, 3),
                            layer=LayerClassifier.LAYER_HARMONIC,
                            event_type="note",
                            pitch_midi=m_int,
                            pitch_name=p_name,
                            frequency_hz=round(f0, 2),
                            cents_deviation=cents,
                            velocity=int(norm_vel * 0.70),
                            confidence=round(harmonicity, 3),
                            uncertainty_flags=[],
                            source_id="src_strings",
                            instrument_role="ostinato_strings",
                        )
                    )
                    event_counter += 1

        # Sort all extracted events chronologically
        events.sort(key=lambda e: (e.start_time, e.layer))
        return events

    def _estimate_pitch(self, audio_slice: np.ndarray, sr: int) -> Tuple[Optional[float], float]:
        """Estimate fundamental frequency via normalized autocorrelation on windowed slice."""
        n = len(audio_slice)
        if n < 256:
            return None, 0.0

        windowed = audio_slice * np.hanning(n)
        fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fft = np.fft.rfft(windowed, n=fft_size)
        ac = np.fft.irfft(fft * np.conj(fft), n=fft_size)[:n]

        if ac[0] <= 1e-8:
            return None, 0.0

        ac_norm = ac / ac[0]
        min_lag = max(2, int(sr / 2000.0))
        max_lag = min(n - 1, int(sr / 45.0))

        if min_lag >= max_lag:
            return None, 0.0

        search_region = ac_norm[min_lag:max_lag]
        if len(search_region) == 0:
            return None, 0.0

        peaks, props = find_peaks(search_region, height=0.3, distance=int(sr / 4000.0) + 1)
        if len(peaks) == 0:
            return None, 0.0

        best_peak_idx = peaks[np.argmax(props["peak_heights"])]
        actual_lag = min_lag + best_peak_idx
        harmonicity = float(ac_norm[actual_lag])

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
