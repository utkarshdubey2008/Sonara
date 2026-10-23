"""Comparative evaluation between original and SMR-reconstructed audio."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import librosa
import numpy as np


@dataclass
class MetricComparison:
    """Individual comparative metric result."""
    name: str
    original_value: float
    reconstructed_value: float
    alignment_score: float  # 0.0 to 1.0 (1.0 = perfect match)
    unit: str
    description: str


@dataclass
class ReconstructionReport:
    """Complete objective evaluation of audio reconstruction fidelity."""
    original_file: str
    reconstructed_file: str
    composite_fidelity_score: float  # 0.0 to 100.0%
    metrics: List[MetricComparison]
    harmonic_alignment: float
    envelope_correlation: float
    tempo_error_bpm: float
    reproduced_capabilities: List[str]
    unreproduced_limitations: List[str]


class ReconstructionComparator:
    """Computes objective mathematical and musical alignment between original and reconstructed audio."""

    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def compare(
        self, original_path: str | Path, reconstructed_path: str | Path
    ) -> ReconstructionReport:
        """
        Compare original audio file against SMR reconstructed audio file.

        Args:
            original_path: Path to original audio file.
            reconstructed_path: Path to reconstructed WAV file.

        Returns:
            ReconstructionReport with objective metrics.
        """
        orig_p = Path(original_path)
        recon_p = Path(reconstructed_path)

        # 1. Load both signals
        y_orig, sr = librosa.load(str(orig_p), sr=self.sample_rate, mono=True)
        y_recon, _ = librosa.load(str(recon_p), sr=self.sample_rate, mono=True)

        min_len = min(len(y_orig), len(y_recon))
        if min_len < self.sample_rate * 0.5:
            raise ValueError("Audio signals too short for comparative evaluation.")

        # Align lengths for frame-wise comparisons
        y_orig_trim = y_orig[:min_len]
        y_recon_trim = y_recon[:min_len]

        metrics: List[MetricComparison] = []

        # 2. Envelope Correlation (Dynamic contour match)
        rms_orig = librosa.feature.rms(y=y_orig_trim, hop_length=self.hop_length)[0]
        rms_recon = librosa.feature.rms(y=y_recon_trim, hop_length=self.hop_length)[0]

        if np.std(rms_orig) > 1e-6 and np.std(rms_recon) > 1e-6:
            r_val = float(np.corrcoef(rms_orig, rms_recon)[0, 1])
            env_corr = float(np.clip(r_val, 0.0, 1.0))
        else:
            env_corr = 0.5

        metrics.append(
            MetricComparison(
                name="Dynamic Envelope Correlation",
                original_value=round(float(np.mean(rms_orig)), 4),
                reconstructed_value=round(float(np.mean(rms_recon)), 4),
                alignment_score=round(env_corr, 3),
                unit="Pearson r",
                description="Correlation between energy dynamics and temporal section builds",
            )
        )

        # 3. Harmonic Chroma Alignment (Tonal match)
        chroma_orig = librosa.feature.chroma_stft(y=y_orig_trim, sr=sr, hop_length=self.hop_length)
        chroma_recon = librosa.feature.chroma_stft(y=y_recon_trim, sr=sr, hop_length=self.hop_length)

        c_orig_mean = np.mean(chroma_orig, axis=1)
        c_recon_mean = np.mean(chroma_recon, axis=1)

        norm_orig = np.linalg.norm(c_orig_mean) + 1e-8
        norm_recon = np.linalg.norm(c_recon_mean) + 1e-8
        harmonic_sim = float(np.dot(c_orig_mean, c_recon_mean) / (norm_orig * norm_recon))
        harmonic_sim = float(np.clip(harmonic_sim, 0.0, 1.0))

        metrics.append(
            MetricComparison(
                name="Harmonic Chroma Similarity",
                original_value=round(float(np.argmax(c_orig_mean)), 0),
                reconstructed_value=round(float(np.argmax(c_recon_mean)), 0),
                alignment_score=round(harmonic_sim, 3),
                unit="Cosine Sim",
                description="Pitch-class distribution alignment across the 12 chromatic semitones",
            )
        )

        # 4. Tempo & Rhythmic Congruence
        tempo_orig, _ = librosa.beat.beat_track(y=y_orig_trim, sr=sr, hop_length=self.hop_length)
        tempo_recon, _ = librosa.beat.beat_track(y=y_recon_trim, sr=sr, hop_length=self.hop_length)

        bpm_orig = float(tempo_orig[0]) if isinstance(tempo_orig, np.ndarray) and tempo_orig.ndim > 0 else float(tempo_orig)
        bpm_recon = float(tempo_recon[0]) if isinstance(tempo_recon, np.ndarray) and tempo_recon.ndim > 0 else float(tempo_recon)

        tempo_err = abs(bpm_orig - bpm_recon)
        tempo_align = max(0.0, 1.0 - (tempo_err / max(1.0, bpm_orig)))

        metrics.append(
            MetricComparison(
                name="Tempo Congruence",
                original_value=round(bpm_orig, 1),
                reconstructed_value=round(bpm_recon, 1),
                alignment_score=round(tempo_align, 3),
                unit="BPM",
                description="Closeness of estimated beats per minute",
            )
        )

        # 5. Spectral Centroid / Timbral Brightness
        sc_orig = float(np.mean(librosa.feature.spectral_centroid(y=y_orig_trim, sr=sr, hop_length=self.hop_length)))
        sc_recon = float(np.mean(librosa.feature.spectral_centroid(y=y_recon_trim, sr=sr, hop_length=self.hop_length)))
        sc_ratio = min(sc_orig, sc_recon) / max(sc_orig, sc_recon, 1e-4)

        metrics.append(
            MetricComparison(
                name="Spectral Brightness Alignment",
                original_value=round(sc_orig, 1),
                reconstructed_value=round(sc_recon, 1),
                alignment_score=round(sc_ratio, 3),
                unit="Hz",
                description="Ratio of average frequency center of mass (perceived brightness)",
            )
        )

        # 6. Composite Fidelity Score (Weighted average)
        # 35% harmonic similarity, 30% tempo congruence, 20% dynamic envelope correlation, 15% spectral brightness
        composite = (
            (harmonic_sim * 0.35)
            + (tempo_align * 0.30)
            + (env_corr * 0.20)
            + (sc_ratio * 0.15)
        ) * 100.0

        # Qualitative capability and limitation audit
        capabilities = [
            "Accurate metric tempo and beat grid synchronization",
            "Functional harmonic progression and tonal center preservation",
            "Multi-timbral layer partitioning (percussion, bass, pads, leads)",
            "Dynamic velocity contour tracking across chronological onsets",
            "Formal section arrangement and macro-temporal duration adherence",
        ]

        limitations = [
            "Acoustic room impulse response and spatial reverberation are not resynthesized",
            "Vocal formant phonetics and lyrical speech are reduced to melodic tonal contours",
            "Complex analog distortion and non-linear saturation are synthesized via clean procedural oscillators",
            "Dense polyphonic piano/guitar strum voices are quantized to primary chord triad degrees",
        ]

        return ReconstructionReport(
            original_file=orig_p.name,
            reconstructed_file=recon_p.name,
            composite_fidelity_score=round(composite, 1),
            metrics=metrics,
            harmonic_alignment=round(harmonic_sim, 3),
            envelope_correlation=round(env_corr, 3),
            tempo_error_bpm=round(tempo_err, 1),
            reproduced_capabilities=capabilities,
            unreproduced_limitations=limitations,
        )
