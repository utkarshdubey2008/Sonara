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
        """Synthesize drum hits (kick, snare, hi-hat)."""
        sr = self.sample_rate
        n = int(dur * sr)
        t = np.linspace(0.0, dur, n, endpoint=False)

        # Kick
        if pitch_midi is None or pitch_midi <= 36:
            f_start = 140.0
            f_end = 45.0
            freq_curve = f_start * (f_end / f_start) ** (t / max(0.1, dur))
            phase = 2 * np.pi * np.cumsum(freq_curve) / sr
            envelope = np.exp(-t * 14.0)
            sig = np.sin(phase) * envelope * amp * 1.2
            return sig, sig  # Center panned

        # Snare
        elif pitch_midi <= 40:
            env_body = np.exp(-t * 22.0)
            body = np.sin(2 * np.pi * 185.0 * t) * env_body
            env_noise = np.exp(-t * 16.0)
            noise = (np.random.rand(n) * 2 - 1) * env_noise
            sig = (body * 0.6 + noise * 0.7) * amp
            return sig * 0.9, sig * 1.1  # Slight right pan

        # Hi-hat / cymbal
        else:
            env = np.exp(-t * 45.0)
            noise = (np.random.rand(n) * 2 - 1) * env
            sig = noise * amp * 0.5
            return sig * 1.1, sig * 0.9  # Slight left pan

    def _synth_bass(self, freq: float, dur: float, amp: float) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize warm sub-bass with harmonic richness."""
        sr = self.sample_rate
        n = int(dur * sr)
        t = np.linspace(0.0, dur, n, endpoint=False)

        env = np.exp(-t * 2.8)
        # Fundamental sine + 2nd harmonic
        f1 = np.sin(2 * np.pi * freq * t)
        f2 = 0.35 * np.sin(2 * np.pi * (freq * 2) * t)
        f3 = 0.15 * np.sin(2 * np.pi * (freq * 3) * t)

        sig = (f1 + f2 + f3) * env * amp * 1.1
        return sig, sig  # Center panned

    def _synth_pad(self, freq: float, dur: float, amp: float) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize warm stereo pad with detuned chorus effect."""
        sr = self.sample_rate
        n = int(dur * sr)
        t = np.linspace(0.0, dur, n, endpoint=False)

        # Smooth attack & release envelope
        attack_time = min(0.1, dur * 0.2)
        release_time = min(0.2, dur * 0.3)
        env = np.ones(n)
        att_samples = int(attack_time * sr)
        rel_samples = int(release_time * sr)
        if att_samples > 0:
            env[:att_samples] = np.linspace(0, 1, att_samples)
        if rel_samples > 0:
            env[-rel_samples:] = np.linspace(1, 0, rel_samples)

        # Detuned stereo oscillators for width
        osc_l = np.sin(2 * np.pi * (freq * 0.998) * t) + 0.3 * np.sin(2 * np.pi * (freq * 1.996) * t)
        osc_r = np.sin(2 * np.pi * (freq * 1.002) * t) + 0.3 * np.sin(2 * np.pi * (freq * 2.004) * t)

        return osc_l * env * amp, osc_r * env * amp

    def _synth_lead(self, freq: float, dur: float, amp: float) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize melodic lead synth with vibrato."""
        sr = self.sample_rate
        n = int(dur * sr)
        t = np.linspace(0.0, dur, n, endpoint=False)

        # Vibrato LFO: 5.0 Hz
        vibrato = 1.0 + 0.015 * np.sin(2 * np.pi * 5.0 * t)
        phase = 2 * np.pi * np.cumsum(freq * vibrato) / sr

        # Band-limited rounded square/triangle wave
        osc = np.sin(phase) + 0.25 * np.sin(3 * phase)

        # Attack-decay envelope
        env = np.exp(-t * 2.0)
        sig = osc * env * amp

        return sig * 0.95, sig * 1.05
