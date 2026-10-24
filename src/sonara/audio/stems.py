"""Audio stem separation and multi-track isolation for beat deconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


class StemSeparator:
    """Decomposes an audio mix into isolated multi-track stems with stereo Mid-Side vocal decoupling."""

    STEM_DRUMS = "drums"
    STEM_BASS = "bass"
    STEM_MELODY_SAMPLES = "melody_samples"
    STEM_VOCALS = "vocals"

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def separate(self, audio_path: str | Path, output_dir: str | Path) -> Dict[str, Path]:
        """
        Decompose audio mix into separate audio stems (drums, bass, melody_samples, vocals).
        Applies Mid-Side stereo decomposition to isolate and decouple center lead vocals.

        Args:
            audio_path: Path to input audio mix.
            output_dir: Directory where isolated stem WAVs will be written.

        Returns:
            Dictionary mapping stem names to their resolved Path objects.
        """
        in_p = Path(audio_path)
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        y_raw, sr = librosa.load(str(in_p), sr=self.sample_rate, mono=False)
        if y_raw.size == 0:
            raise ValueError(f"Empty audio waveform in {audio_path}")

        # Check if stereo
        if y_raw.ndim > 1 and y_raw.shape[0] >= 2:
            L = y_raw[0]
            R = y_raw[1]
            M = 0.5 * (L + R)
            S = 0.5 * (L - R)

            # 1. Bass: low-pass < 220 Hz on center and side
            sos_bass = butter(4, 220.0, btype="lowpass", fs=sr, output="sos")
            bass = sosfilt(sos_bass, M)

            # 2. Mid/High Center (> 150 Hz) contains lead vocal + center drum crack
            sos_midhigh = butter(4, 150.0, btype="highpass", fs=sr, output="sos")
            M_midhigh = sosfilt(sos_midhigh, M)

            # 3. Harmonic-Percussive Source Separation
            M_harm, M_perc = librosa.effects.hpss(M_midhigh, margin=(1.5, 1.2))
            S_harm, S_perc = librosa.effects.hpss(S, margin=(1.2, 1.2))

            # 4. Drums: All percussive transients
            drums = M_perc + S_perc

            # 5. Lead Vocals: Center harmonic line
            vocals = M_harm

            # 6. Melody & Samples: Stereo side harmonics + filtered non-lead backing
            melody_samples = S_harm + 0.12 * M_harm

        else:
            # Mono fallback
            y_mono = y_raw if y_raw.ndim == 1 else np.mean(y_raw, axis=0)
            y_harmonic, y_percussive = librosa.effects.hpss(y_mono, margin=(1.2, 1.2))

            drums = y_percussive
            sos_bass = butter(4, 220.0, btype="lowpass", fs=sr, output="sos")
            bass = sosfilt(sos_bass, y_harmonic)

            sos_melody = butter(4, [220.0, 4500.0], btype="bandpass", fs=sr, output="sos")
            melody_samples = sosfilt(sos_melody, y_harmonic)

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
            sf.write(str(out_stem_path), sig_norm.astype(np.float32), sr, subtype="PCM_16")
            saved_stems[name] = out_stem_path

        return saved_stems
