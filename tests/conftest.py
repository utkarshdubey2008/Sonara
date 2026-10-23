"""Test fixtures and synthetic audio generation helpers for Sonara test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator
import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sine_audio_file(temp_dir: Path) -> Path:
    """Generate a clean 440 Hz (A4) 2-second mono WAV audio file."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine tone with fade-in and fade-out
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    window = np.hanning(len(y))
    y = y * window

    file_path = temp_dir / "sine_440hz.wav"
    sf.write(str(file_path), y, sr, subtype="PCM_16")
    return file_path


@pytest.fixture
def silent_audio_file(temp_dir: Path) -> Path:
    """Generate a 1-second completely silent audio file."""
    sr = 22050
    duration = 1.0
    y = np.zeros(int(sr * duration), dtype=np.float32)

    file_path = temp_dir / "silence.wav"
    sf.write(str(file_path), y, sr, subtype="PCM_16")
    return file_path


@pytest.fixture
def rhythmic_audio_file(temp_dir: Path) -> Path:
    """Generate a 4-second audio track with synthetic beats at 120 BPM."""
    sr = 22050
    duration = 4.0
    total_samples = int(sr * duration)
    y = np.zeros(total_samples, dtype=np.float32)

    # 120 BPM = 2 beats per second (every 0.5s)
    beat_interval = int(0.5 * sr)
    for i in range(0, total_samples, beat_interval):
        # Add transient click / burst
        hit_len = min(1000, total_samples - i)
        t_hit = np.linspace(0, 0.05, hit_len, endpoint=False)
        decay = np.exp(-t_hit * 80.0)
        y[i : i + hit_len] += (0.8 * np.sin(2 * np.pi * 120.0 * t_hit) * decay).astype(np.float32)

    file_path = temp_dir / "rhythm_120bpm.wav"
    sf.write(str(file_path), y, sr, subtype="PCM_16")
    return file_path


@pytest.fixture
def chord_audio_file(temp_dir: Path) -> Path:
    """Generate a 3-second C major triad (C4, E4, G4) audio file."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    f_c4 = 261.63
    f_e4 = 329.63
    f_g4 = 392.00

    y = (
        0.3 * np.sin(2 * np.pi * f_c4 * t)
        + 0.3 * np.sin(2 * np.pi * f_e4 * t)
        + 0.3 * np.sin(2 * np.pi * f_g4 * t)
    )
    window = np.hanning(len(y))
    y = y * window

    file_path = temp_dir / "chord_c_major.wav"
    sf.write(str(file_path), y, sr, subtype="PCM_16")
    return file_path
