"""Tests for SMR schema, serialization, and validation."""

from pathlib import Path
import pytest

from sonara.generation.generator import StyleConditionedGenerator
from sonara.representation.schema import SMRDocument, SMR_SCHEMA_VERSION
from sonara.representation.serializer import SMRSerializer
from sonara.representation.validator import SMRValidator


def test_smr_schema_export() -> None:
    schema = SMRSerializer.export_json_schema()
    assert isinstance(schema, dict)
    assert schema.get("title") == "SMRDocument"
    assert "properties" in schema


def test_smr_serialization_roundtrip(temp_dir: Path) -> None:
    generator = StyleConditionedGenerator(seed=42)
    doc = generator.generate(duration_seconds=16.0, target_bpm=120.0, target_key="C")

    assert doc.smr_version == SMR_SCHEMA_VERSION
    assert len(doc.events) > 10

    # Test JSON string roundtrip
    json_str = SMRSerializer.to_json(doc)
    parsed = SMRSerializer.from_json(json_str)

    assert parsed.metadata.duration_seconds == doc.metadata.duration_seconds
    assert parsed.rhythm.estimated_bpm == doc.rhythm.estimated_bpm
    assert len(parsed.events) == len(doc.events)

    # Test file roundtrip
    out_file = temp_dir / "test_roundtrip.json"
    SMRSerializer.save_json(doc, out_file)
    loaded = SMRSerializer.load_json(out_file)

    assert loaded.metadata.file_name == doc.metadata.file_name


def test_smr_validator_valid_document() -> None:
    generator = StyleConditionedGenerator(seed=123)
    doc = generator.generate(duration_seconds=8.0)

    report = SMRValidator.validate(doc)
    assert report.is_valid
    assert len(report.errors) == 0


def test_smr_validator_detects_invalid_event() -> None:
    generator = StyleConditionedGenerator(seed=123)
    doc = generator.generate(duration_seconds=8.0)

    # Corrupt an event: end_time before start_time
    doc.events[0].end_time = doc.events[0].start_time - 1.0

    report = SMRValidator.validate(doc)
    assert not report.is_valid
    assert any("start_time" in e for e in report.errors)
