"""Tests for bidirectional MIDI conversion."""

from pathlib import Path
import pytest

from sonara.generation.generator import StyleConditionedGenerator
from sonara.representation.midi_converter import MIDIConverter


def test_smr_to_midi_and_back(temp_dir: Path) -> None:
    generator = StyleConditionedGenerator(seed=777)
    doc = generator.generate(duration_seconds=8.0, target_bpm=120.0, target_key="G")

    midi_path = temp_dir / "test_export.mid"
    exported_path = MIDIConverter.smr_to_midi(doc, midi_path)

    assert exported_path.exists()
    assert exported_path.stat().st_size > 100

    # Test reverse conversion
    imported_smr = MIDIConverter.midi_to_smr(exported_path)
    assert imported_smr.rhythm.estimated_bpm == 120.0
    assert len(imported_smr.events) > 0
    assert imported_smr.metadata.duration_seconds > 5.0
