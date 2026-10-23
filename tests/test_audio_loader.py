"""Tests for AudioLoader module."""

from pathlib import Path
import numpy as np
import pytest

from sonara.audio.loader import AudioLoader, AudioLoadError


def test_inspect_valid_sine(sine_audio_file: Path) -> None:
    loader = AudioLoader(default_sr=22050)
    meta = loader.inspect(sine_audio_file)

    assert meta.file_name == "sine_440hz.wav"
    assert meta.duration_seconds > 1.9
    assert meta.sample_rate == 22050
    assert meta.channels == 1
    assert not meta.is_silent
    assert len(meta.sha256_hash) == 64


def test_load_valid_sine(sine_audio_file: Path) -> None:
    loader = AudioLoader(default_sr=22050)
    waveform, meta = loader.load(sine_audio_file)

    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 1
    assert len(waveform) == int(meta.duration_seconds * 22050)
    assert meta.peak_amplitude > 0.4
    assert meta.rms_amplitude > 0.1


def test_load_silent_file(silent_audio_file: Path) -> None:
    loader = AudioLoader(default_sr=22050)
    waveform, meta = loader.load(silent_audio_file)

    assert meta.is_silent
    assert meta.peak_amplitude < 1e-4
    assert meta.rms_amplitude < 1e-4


def test_load_nonexistent_file(temp_dir: Path) -> None:
    loader = AudioLoader()
    with pytest.raises(AudioLoadError):
        loader.load(temp_dir / "does_not_exist.wav")


def test_load_empty_file(temp_dir: Path) -> None:
    empty_path = temp_dir / "empty.wav"
    empty_path.touch()
    loader = AudioLoader()
    with pytest.raises(AudioLoadError):
        loader.load(empty_path)
