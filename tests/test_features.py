"""Tests for FeatureExtractor module."""

import numpy as np
import pytest

from sonara.audio.features import FeatureExtractor


def test_feature_extraction_sine() -> None:
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    extractor = FeatureExtractor()
    feat = extractor.extract(y, sr)

    # 440 Hz tone should have centroid close to 440 Hz
    assert 400.0 <= feat.spectral_centroid_mean <= 480.0
    assert feat.peak_amplitude > 0.45
    assert feat.rms_energy_mean > 0.3
    assert feat.warmth_index > 0.5  # 440Hz is below 500Hz
    assert feat.brightness_index < 0.1
    assert len(feat.mfcc_mean) == 13
    assert len(feat.spectral_contrast_mean) == 7
    assert "completely_silent_signal" not in feat.uncertainty_flags


def test_feature_extraction_silence() -> None:
    sr = 22050
    y = np.zeros(sr, dtype=np.float32)

    extractor = FeatureExtractor()
    feat = extractor.extract(y, sr)

    assert feat.rms_energy_mean == 0.0
    assert feat.peak_amplitude == 0.0
    assert "completely_silent_signal" in feat.uncertainty_flags


def test_feature_extraction_clipping_flag() -> None:
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5))
    y = np.clip(2.0 * np.sin(2 * np.pi * 300.0 * t), -1.0, 1.0)

    extractor = FeatureExtractor()
    feat = extractor.extract(y, sr)

    assert feat.peak_amplitude >= 0.999
    assert "signal_clipping_detected" in feat.uncertainty_flags
