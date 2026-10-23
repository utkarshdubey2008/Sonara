"""Audio stem separation and multi-track isolation for beat deconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


class StemSeparator:
    """Decomposes an audio mix into isolated multi-track stems for sampling and DAW import."""

    STEM_DRUMS = "drums"
    STEM_BASS = "bass"
    STEM_MELODY_SAMPLES = "melody_samples"
    STEM_VOCALS = "vocals"

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def separate(self, audio_path: str | Path, output_dir: str | Path) -> Dict[str, Path]:
        """
        Decompose audio mix into separate audio stems (drums, bass, melody_samples, vocals).

        Args:
            audio_path: Path to input audio mix.
            output_dir: Directory where isolated stem WAVs will be written.

        Returns:
            Dictionary mapping stem names to their resolved Path objects.
        """
        in_p = Path(audio_path)
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        y, sr = librosa.load(str(in_p), sr=self.sample_rate, mono=True)
        if len(y) == 0:
            raise ValueError(f"Empty audio waveform in {audio_path}")

        # 1. Harmonic-Percussive Source Separation (HPSS)
        # Separate transient percussive energy from pitched harmonic resonance
        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.2, 1.2))

        # 2. Drums Stem
        # y_percussive contains the primary drum kit (kick transient, snare crack, hi-hat ticks)
        drums = y_percussive

        # 3. Bass Stem
        # Low-pass filter the harmonic component below 240 Hz (Linkwitz-Riley 4th order filter)
        sos_bass = butter(4, 240.0, btype="lowpass", fs=sr, output="sos")
        bass = sosfilt(sos_bass, y_harmonic)

        # 4. Melody & Chopped Samples Stem (Mid/High harmonics without bass)
        # High-pass filter above 240 Hz, band-limited up to 3500 Hz
        sos_melody = butter(4, [240.0, 4500.0], btype="bandpass", fs=sr, output="sos")
        melody_samples = sosfilt(sos_melody, y_harmonic)

        # 5. Vocals / High Air Stem (> 3000 Hz harmonics and vocal presence)
        sos_vocals = butter(4, 1000.0, btype="highpass", fs=sr, output="sos")
        vocals = sosfilt(sos_vocals, y_harmonic)

        # Normalize stems with headroom
        stem_arrays = {
            self.STEM_DRUMS: drums,
            self.STEM_BASS: bass,
            self.STEM_MELODY_SAMPLES: melody_samples,
            self.STEM_VOCALS: vocals,
        }

        saved_stems: Dict[str, Path] = {}
        for name, sig in stem_arrays.items():
            peak = float(np.max(np.abs(sig)))
            if peak > 1e-4:
                sig_norm = (sig / peak) * 0.92
            else:
                sig_norm = sig

            out_stem_path = out_d / f"{name}.wav"
            sf.write(str(out_stem_path), sig_norm, sr, subtype="PCM_16")
            saved_stems[name] = out_stem_path

        return saved_stems
