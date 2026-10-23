"""Tests for RhythmAnalyzer and HarmonyAnalyzer modules."""

import numpy as np
import pytest

from sonara.audio.harmony import HarmonyAnalyzer
from sonara.audio.rhythm import RhythmAnalyzer


def test_rhythm_analyzer_beats() -> None:
    sr = 22050
    duration = 4.0
    total_samples = int(sr * duration)
    y = np.zeros(total_samples, dtype=np.float32)

    # Impulses at 120 BPM (every 0.5s)
    beat_interval = int(0.5 * sr)
    for i in range(0, total_samples, beat_interval):
        hit_len = min(500, total_samples - i)
        y[i : i + hit_len] = 0.9

    analyzer = RhythmAnalyzer()
    res = analyzer.analyze(y, sr)

    assert res.is_rhythmic
    assert 110.0 <= res.estimated_bpm <= 130.0
    assert res.tempo_confidence > 0.5
    assert res.onset_count >= 7
    assert res.time_signature_estimate in ("4/4", "3/4")


def test_rhythm_analyzer_silence() -> None:
    sr = 22050
    y = np.zeros(sr, dtype=np.float32)
    analyzer = RhythmAnalyzer()
    res = analyzer.analyze(y, sr)

    assert not res.is_rhythmic
    assert res.estimated_bpm == 0.0
    assert "insufficient_audio_or_silent" in res.uncertainty_flags


def test_harmony_analyzer_c_major() -> None:
    sr = 22050
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # C major triad: C4 (261.63), E4 (329.63), G4 (392.00)
    y = (
        0.33 * np.sin(2 * np.pi * 261.63 * t)
        + 0.33 * np.sin(2 * np.pi * 329.63 * t)
        + 0.33 * np.sin(2 * np.pi * 392.00 * t)
    )

    analyzer = HarmonyAnalyzer()
    res = analyzer.analyze(y, sr)

    assert res.primary_key == "C"
    assert res.mode == "major"
    assert res.key_confidence > 0.4
    assert "C:maj" in res.chord_vocabulary


def test_harmony_analyzer_silence() -> None:
    sr = 22050
    y = np.zeros(sr, dtype=np.float32)
    analyzer = HarmonyAnalyzer()
    res = analyzer.analyze(y, sr)

    assert res.primary_key == "unknown"
    assert res.key_confidence == 0.0
    assert "silent_or_too_short" in res.uncertainty_flags
