"""Acoustic, spectral, and timbral feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

import librosa
import numpy as np


@dataclass
class AcousticFeatureSummary:
    """Summary of acoustic and timbral features."""
    rms_energy_mean: float
    rms_energy_std: float
    peak_amplitude: float
    dynamic_range_db: float
    crest_factor: float
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_bandwidth_mean: float
    spectral_rolloff_mean: float
    spectral_flatness_mean: float
    zero_crossing_rate_mean: float
    warmth_index: float
    brightness_index: float
    spectral_contrast_mean: List[float]
    mfcc_mean: List[float]
    mfcc_std: List[float]
    uncertainty_flags: List[str] = field(default_factory=list)


class FeatureExtractor:
    """Extracts DSP and acoustic descriptors from audio signals."""

    def __init__(self, hop_length: int = 512, n_fft: int = 2048, n_mfcc: int = 13):
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc

    def extract(self, y: np.ndarray, sr: int) -> AcousticFeatureSummary:
        """
        Extract acoustic, spectral, and timbral features from waveform.

        Args:
            y: Audio waveform array (1D float).
            sr: Sample rate.

        Returns:
            AcousticFeatureSummary instance.
        """
        uncertainty_flags: List[str] = []

        if y.size == 0 or np.all(np.abs(y) < 1e-6):
            return AcousticFeatureSummary(
                rms_energy_mean=0.0,
                rms_energy_std=0.0,
                peak_amplitude=0.0,
                dynamic_range_db=0.0,
                crest_factor=1.0,
                spectral_centroid_mean=0.0,
                spectral_centroid_std=0.0,
                spectral_bandwidth_mean=0.0,
                spectral_rolloff_mean=0.0,
                spectral_flatness_mean=0.0,
                zero_crossing_rate_mean=0.0,
                warmth_index=0.0,
                brightness_index=0.0,
                spectral_contrast_mean=[0.0] * 7,
                mfcc_mean=[0.0] * self.n_mfcc,
                mfcc_std=[0.0] * self.n_mfcc,
                uncertainty_flags=["completely_silent_signal"],
            )

        # Dynamic and amplitude features
        peak = float(np.max(np.abs(y)))
        rms_frames = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_mean = float(np.mean(rms_frames))
        rms_std = float(np.std(rms_frames))

        crest_factor = float(peak / (rms_mean + 1e-8))
        if rms_mean > 1e-6:
            min_rms = max(float(np.min(rms_frames[rms_frames > 1e-6])), 1e-6)
            max_rms = max(float(np.max(rms_frames)), 1e-6)
            dyn_range_db = float(20.0 * np.log10(max_rms / min_rms))
        else:
            dyn_range_db = 0.0

        if peak >= 0.999:
            uncertainty_flags.append("signal_clipping_detected")
        if rms_mean < 0.01:
            uncertainty_flags.append("low_signal_to_noise_ratio")

        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        spec_bw = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        spec_roll = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85
        )[0]
        spec_flat = librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            y=y, hop_length=self.hop_length
        )[0]

        # Spectral contrast
        try:
            spec_cont = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            contrast_mean = [float(val) for val in np.mean(spec_cont, axis=1)]
        except Exception:
            contrast_mean = [0.0] * 7
            uncertainty_flags.append("spectral_contrast_failed")

        # MFCC extraction
        try:
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
            )
            mfcc_m = [float(val) for val in np.mean(mfccs, axis=1)]
            mfcc_s = [float(val) for val in np.std(mfccs, axis=1)]
        except Exception:
            mfcc_m = [0.0] * self.n_mfcc
            mfcc_s = [0.0] * self.n_mfcc
            uncertainty_flags.append("mfcc_computation_failed")

        # Frequency band distribution for warmth (<500Hz) and brightness (>3000Hz)
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        total_energy = np.sum(stft) + 1e-8

        low_mask = freqs <= 500.0
        high_mask = freqs >= 3000.0
        warmth = float(np.sum(stft[low_mask, :]) / total_energy)
        brightness = float(np.sum(stft[high_mask, :]) / total_energy)

        return AcousticFeatureSummary(
            rms_energy_mean=rms_mean,
            rms_energy_std=rms_std,
            peak_amplitude=peak,
            dynamic_range_db=dyn_range_db,
            crest_factor=crest_factor,
            spectral_centroid_mean=float(np.mean(spec_cent)),
            spectral_centroid_std=float(np.std(spec_cent)),
            spectral_bandwidth_mean=float(np.mean(spec_bw)),
            spectral_rolloff_mean=float(np.mean(spec_roll)),
            spectral_flatness_mean=float(np.mean(spec_flat)),
            zero_crossing_rate_mean=float(np.mean(zcr)),
            warmth_index=warmth,
            brightness_index=brightness,
            spectral_contrast_mean=contrast_mean,
            mfcc_mean=mfcc_m,
            mfcc_std=mfcc_s,
            uncertainty_flags=uncertainty_flags,
        )
