"""Semantic and schema validation for Sonara Musical Representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from sonara.representation.schema import SMRDocument


@dataclass
class ValidationReport:
    """Report detailing validation status, errors, and warnings."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SMRValidator:
    """Validates structural and semantic invariants of SMR documents."""

    @classmethod
    def validate(cls, document: SMRDocument) -> ValidationReport:
        """
        Validate an SMRDocument for structural and semantic consistency.

        Args:
            document: SMRDocument instance.

        Returns:
            ValidationReport with validation findings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        total_dur = document.metadata.duration_seconds
        tolerance = 1.0  # Allowed timestamp margin in seconds

        # 1. Check sections
        prev_end = 0.0
        for s in document.structure:
            if s.start_time > s.end_time:
                errors.append(f"Section {s.section_id}: start_time ({s.start_time}) > end_time ({s.end_time})")
            if s.duration <= 0.0:
                errors.append(f"Section {s.section_id}: non-positive duration ({s.duration})")
            if s.end_time > total_dur + tolerance:
                warnings.append(f"Section {s.section_id} ends at {s.end_time}s, exceeding audio duration {total_dur}s")
            if s.start_time < prev_end - 0.5:
                warnings.append(f"Section {s.section_id} overlaps previous section ending at {prev_end}s")
            prev_end = s.end_time

        # 2. Check events
        for e in document.events:
            if e.start_time > e.end_time:
                errors.append(f"Event {e.event_id}: start_time ({e.start_time}) > end_time ({e.end_time})")
            if e.duration <= 0.0:
                errors.append(f"Event {e.event_id}: non-positive duration ({e.duration})")
            if not (0 <= e.velocity <= 127):
                errors.append(f"Event {e.event_id}: velocity ({e.velocity}) outside MIDI range [0, 127]")
            if not (0.0 <= e.confidence <= 1.0):
                errors.append(f"Event {e.event_id}: confidence ({e.confidence}) outside range [0.0, 1.0]")
            if e.pitch_midi is not None and not (0 <= e.pitch_midi <= 127):
                errors.append(f"Event {e.event_id}: pitch_midi ({e.pitch_midi}) outside MIDI range [0, 127]")

        # 3. Check chords
        for ch in document.harmony.chord_sequence:
            if ch.start_time > ch.end_time:
                errors.append(f"Chord {ch.chord_symbol}: start_time ({ch.start_time}) > end_time ({ch.end_time})")

        # 4. Check rhythm consistency
        if document.rhythm.is_rhythmic and document.rhythm.estimated_bpm <= 0.0:
            warnings.append("Track marked is_rhythmic=True but estimated_bpm is 0.0")

        return ValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
