"""Edge cases, robustness, and failure mode testing for Sonara."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from sonara.pipeline import SonaraPipeline
from sonara.representation.serializer import SMRSerializer
from sonara.representation.validator import SMRValidator


def test_pipeline_on_silence(silent_audio_file: Path) -> None:
    pipeline = SonaraPipeline()
    doc = pipeline.analyze(silent_audio_file)

    assert doc.metadata.is_silent
    assert doc.rhythm.estimated_bpm == 0.0
    assert not doc.rhythm.is_rhythmic
    assert doc.harmony.primary_key == "unknown"
    assert "completely_silent_signal" in doc.acoustic_features.uncertainty_flags
    assert SMRValidator.validate(doc).is_valid


def test_pipeline_on_high_sample_rate(temp_dir: Path) -> None:
    # 96,000 Hz sample rate audio
    sr = 96000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 500.0 * t)

    file_path = temp_dir / "high_sr_96k.wav"
    sf.write(str(file_path), y, sr, subtype="PCM_24")

    pipeline = SonaraPipeline(target_sr=22050)
    doc = pipeline.analyze(file_path)

    assert doc.metadata.sample_rate == 96000
    assert doc.metadata.duration_seconds > 0.9
    assert SMRValidator.validate(doc).is_valid


def test_pipeline_on_stereo_audio(temp_dir: Path) -> None:
    # 2-channel stereo audio
    sr = 22050
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 880.0 * t)
    stereo = np.column_stack([left, right])

    file_path = temp_dir / "stereo_440_880.wav"
    sf.write(str(file_path), stereo, sr, subtype="PCM_16")

    pipeline = SonaraPipeline(target_sr=22050)
    doc = pipeline.analyze(file_path)

    assert doc.metadata.channels == 2
    assert doc.metadata.duration_seconds > 1.4
    assert SMRValidator.validate(doc).is_valid


def test_pipeline_on_clipped_audio(temp_dir: Path) -> None:
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Severe digital clipping
    y = np.clip(5.0 * np.sin(2 * np.pi * 200.0 * t), -1.0, 1.0)

    file_path = temp_dir / "clipped.wav"
    sf.write(str(file_path), y, sr, subtype="PCM_16")

    pipeline = SonaraPipeline(target_sr=22050)
    doc = pipeline.analyze(file_path)

    assert doc.metadata.is_clipped
    assert "signal_clipping_detected" in doc.acoustic_features.uncertainty_flags
    assert SMRValidator.validate(doc).is_valid


def test_pipeline_on_dc_offset(temp_dir: Path) -> None:
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 0.3 DC bias offset + 300 Hz tone
    y = 0.3 + 0.3 * np.sin(2 * np.pi * 300.0 * t)

    file_path = temp_dir / "dc_offset.wav"
    sf.write(str(file_path), y, sr, subtype="PCM_16")

    pipeline = SonaraPipeline(target_sr=22050)
    doc = pipeline.analyze(file_path)

    assert doc.metadata.duration_seconds > 0.9
    assert SMRValidator.validate(doc).is_valid


def test_invalid_json_deserialization() -> None:
    with pytest.raises(Exception):
        SMRSerializer.from_json('{"invalid": "json", "smr_version": "1.0.0"}')
