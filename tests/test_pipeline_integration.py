"""Integration tests for SonaraPipeline."""

from pathlib import Path
import pytest

from sonara.pipeline import SonaraPipeline
from sonara.representation.validator import SMRValidator


def test_pipeline_on_sine(sine_audio_file: Path, temp_dir: Path) -> None:
    pipeline = SonaraPipeline()
    doc = pipeline.analyze(sine_audio_file)

    assert doc.metadata.duration_seconds > 1.9
    assert doc.metadata.sample_rate == 22050
    assert doc.acoustic_features.spectral_centroid_mean > 300.0

    val = SMRValidator.validate(doc)
    assert val.is_valid

    # Export report and verify
    report_file = temp_dir / "sine_report.md"
    pipeline.export_report(doc, report_file)
    assert report_file.exists()
    assert report_file.stat().st_size > 200

    # Export MIDI and verify
    midi_file = temp_dir / "sine.mid"
    pipeline.export_midi(doc, midi_file)
    assert midi_file.exists()


def test_pipeline_on_chord(chord_audio_file: Path, temp_dir: Path) -> None:
    pipeline = SonaraPipeline()
    doc = pipeline.analyze(chord_audio_file)

    assert doc.harmony.primary_key == "C"
    assert doc.harmony.mode == "major"
    assert len(doc.events) > 0

    val = SMRValidator.validate(doc)
    assert val.is_valid
