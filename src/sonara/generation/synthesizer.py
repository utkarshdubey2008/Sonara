"""Procedural multi-timbral audio synthesizer for SMR document rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

from sonara.events.classifier import LayerClassifier
from sonara.representation.schema import SMRDocument


class SMRSynthesizer:
    """Renders SMRDocument musical events directly into high-fidelity audio (WAV)."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def render(self, document: SMRDocument, output_path: str | Path) -> Path:
        """
        Synthesize audio waveform from SMRDocument events and save to WAV.

        Args:
            document: SMRDocument instance.
            output_path: Path for output WAV file.

        Returns:
            Resolved Path of the written audio file.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        total_duration = max(1.0, document.metadata.duration_seconds)
        total_samples = int(np.ceil((total_duration + 1.0) * self.sample_rate))

        # Stereo mix buffers: [channels, samples]
        mix = np.zeros((2, total_samples), dtype=np.float32)

        # Render each event
        for ev in document.events:
            start_sample = int(ev.start_time * self.sample_rate)
            dur_samples = int(max(0.04, ev.duration) * self.sample_rate)
            if start_sample >= total_samples:
                continue

            end_sample = min(total_samples, start_sample + dur_samples)
            actual_dur = (end_sample - start_sample) / self.sample_rate
            if actual_dur <= 0.005:
                continue

            amp = (ev.velocity / 127.0) * 0.35

            if ev.layer == LayerClassifier.LAYER_PERCUSSION:
                left_sig, right_sig = self._synth_drum(ev.pitch_midi, actual_dur, amp)
            elif ev.layer == LayerClassifier.LAYER_BASS:
                freq = ev.frequency_hz or 55.0
                left_sig, right_sig = self._synth_bass(freq, actual_dur, amp)
            elif ev.layer == LayerClassifier.LAYER_HARMONIC:
                freq = ev.frequency_hz or 261.63
                left_sig, right_sig = self._synth_pad(freq, actual_dur, amp * 0.7)
            else:  # Melodic lead or default
                freq = ev.frequency_hz or 440.0
                left_sig, right_sig = self._synth_lead(freq, actual_dur, amp)

            # Accumulate into mix buffer
            n_write = min(len(left_sig), total_samples - start_sample)
            mix[0, start_sample : start_sample + n_write] += left_sig[:n_write]
            mix[1, start_sample : start_sample + n_write] += right_sig[:n_write]

        # Normalization with 0.5 dB headroom
        peak = float(np.max(np.abs(mix)))
        if peak > 1e-4:
            target_peak = 0.94  # approx -0.5 dB
            mix = (mix / peak) * target_peak

        # Transpose to (samples, channels) for soundfile
        audio_out = mix.T

        sf.write(str(path), audio_out, self.sample_rate, subtype="PCM_16")
        return path

    def _synth_drum(
        self, pitch_midi: Optional[int], dur: float, amp: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize punchy production drum hits (kick, snare, hi-hat, open hat)."""
        sr = self.sample_rate
        n = max(32, int(dur * sr))
        t = np.linspace(0.0, dur, n, endpoint=False)

        # Kick Drum (MIDI 36)
        if pitch_midi is None or pitch_midi <= 36:
            f_start = 220.0
            f_sub = 48.0
            pitch_env = f_sub + (f_start - f_sub) * np.exp(-t * 85.0)
            phase = 2 * np.pi * np.cumsum(pitch_env) / sr
            body = np.sin(phase) * np.exp(-t * 9.5)
            click = np.sin(2 * np.pi * 1800.0 * t) * np.exp(-t * 380.0) * 0.45
            sig = np.tanh((body + click) * 1.35) * amp * 1.3
            return sig, sig  # Centered punch

        # Snare Drum / Rimshot (MIDI 38)
        elif pitch_midi <= 40:
            pitch_env = 165.0 + 45.0 * np.exp(-t * 60.0)
            phase = 2 * np.pi * np.cumsum(pitch_env) / sr
            body = np.sin(phase) * np.exp(-t * 24.0)
            raw_noise = np.random.rand(n) * 2 - 1
            from scipy.signal import butter, sosfilt
            sos = butter(4, 1100.0, btype="highpass", fs=sr, output="sos")
            noise_filt = sosfilt(sos, raw_noise) * np.exp(-t * 18.0)
            sig = np.tanh(body * 0.85 + noise_filt * 1.1) * amp * 1.15
            return sig * 0.95, sig * 1.05  # Stereo spread

        # Open Hat / Crash (MIDI 46)
        elif pitch_midi == 46:
            raw_noise = np.random.rand(n) * 2 - 1
            from scipy.signal import butter, sosfilt
            hi_cut = min(sr * 0.48, 12000.0)
            low_cut = min(hi_cut - 500.0, 4500.0)
            sos = butter(4, [low_cut, hi_cut], btype="bandpass", fs=sr, output="sos")
            noise_filt = sosfilt(sos, raw_noise) * np.exp(-t * 12.0)
            sig = np.tanh(noise_filt * 1.8) * amp * 0.85
            return sig * 1.1, sig * 0.9

        # Closed Hi-Hat (MIDI 42 or other percussion)
        else:
            raw_noise = np.random.rand(n) * 2 - 1
            from scipy.signal import butter, sosfilt
            hi_cut = min(sr * 0.48, 10000.0)
            low_cut = min(hi_cut - 500.0, 5000.0)
            sos = butter(4, [low_cut, hi_cut], btype="bandpass", fs=sr, output="sos")
            noise_filt = sosfilt(sos, raw_noise) * np.exp(-t * 55.0)
            sig = np.tanh(noise_filt * 2.2) * amp * 0.75
            return sig * 1.05, sig * 0.95

    def _synth_bass(self, freq: float, dur: float, amp: float) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize deep, warm analog Moog-style sub-bass."""
        sr = self.sample_rate
        n = max(32, int(dur * sr))
        t = np.linspace(0.0, dur, n, endpoint=False)

        freq = max(35.0, min(240.0, freq))
        sub = np.sin(2 * np.pi * freq * t)
        saw = 2 * (t * freq - np.floor(0.5 + t * freq))
        from scipy.signal import butter, sosfilt
        sos = butter(4, 260.0, btype="lowpass", fs=sr, output="sos")
        saw_filt = sosfilt(sos, saw)

        env = np.ones(n)
        rel_len = min(n // 2, int(0.08 * sr))
        if rel_len > 0:
            env[-rel_len:] = np.linspace(1, 0, rel_len)

        sig = np.tanh(sub * 0.75 + saw_filt * 0.50) * env * amp * 1.2
        return sig, sig  # Centered sub

    def _synth_pad(self, freq: float, dur: float, amp: float) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize warm stereo strings/pad with lush detuned chorus."""
        sr = self.sample_rate
        n = max(32, int(dur * sr))
        t = np.linspace(0.0, dur, n, endpoint=False)

        # Smooth attack & release envelope (prevents clicking)
        attack_time = min(0.12, dur * 0.25)
        release_time = min(0.25, dur * 0.35)
        env = np.ones(n)
        att_samples = int(attack_time * sr)
        rel_samples = int(release_time * sr)
        if att_samples > 0:
            env[:att_samples] = np.linspace(0, 1, att_samples)
        if rel_samples > 0:
            env[-rel_samples:] = np.linspace(1, 0, rel_samples)

        # Detuned stereo oscillators for width
        osc_l = np.sin(2 * np.pi * (freq * 0.998) * t) + 0.35 * np.sin(2 * np.pi * (freq * 1.996) * t)
        osc_r = np.sin(2 * np.pi * (freq * 1.002) * t) + 0.35 * np.sin(2 * np.pi * (freq * 2.004) * t)

        return osc_l * env * amp * 0.9, osc_r * env * amp * 0.9

    def _synth_lead(self, freq: float, dur: float, amp: float) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize smooth analog synth lead with vibrato (non-percussive)."""
        sr = self.sample_rate
        n = max(32, int(dur * sr))
        t = np.linspace(0.0, dur, n, endpoint=False)

        # Vibrato LFO: 4.5 Hz
        vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * 4.5 * t)
        phase = 2 * np.pi * np.cumsum(freq * vibrato) / sr

        # Saturated sawtooth through lowpass filter
        saw = 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
        from scipy.signal import butter, sosfilt
        sos = butter(4, min(sr * 0.45, max(800.0, freq * 3.5)), btype="lowpass", fs=sr, output="sos")
        saw_filt = sosfilt(sos, saw)

        # Smooth sustained envelope with soft attack and decay (not a piano pluck)
        att_len = min(n // 4, int(0.05 * sr))
        rel_len = min(n // 4, int(0.08 * sr))
        env = np.ones(n)
        if att_len > 0:
            env[:att_len] = np.linspace(0, 1, att_len)
        if rel_len > 0:
            env[-rel_len:] = np.linspace(1, 0, rel_len)

        sig = np.tanh(saw_filt * 1.1) * env * amp
        return sig * 0.95, sig * 1.05
