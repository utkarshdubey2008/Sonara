"""Tests for StructureAnalyzer module."""

import numpy as np
import pytest

from sonara.audio.structure import StructureAnalyzer


def test_structure_short_audio() -> None:
    sr = 22050
    y = np.sin(2 * np.pi * 440.0 * np.linspace(0, 2.0, sr * 2))
    analyzer = StructureAnalyzer(min_segment_duration=3.0)
    res = analyzer.analyze(y, sr)

    assert res.num_sections == 1
    assert res.sections[0].section_type == "body"
    assert "track_too_short_for_segmentation" in res.uncertainty_flags


def test_structure_multi_section_audio() -> None:
    sr = 22050
    # Create 12 seconds audio with two distinct harmonic segments:
    # Segment 1 (0-6s): 220 Hz tone
    # Segment 2 (6-12s): 880 Hz tone
    t1 = np.linspace(0, 6, sr * 6, endpoint=False)
    y1 = 0.5 * np.sin(2 * np.pi * 220.0 * t1)
    t2 = np.linspace(0, 6, sr * 6, endpoint=False)
    y2 = 0.5 * np.sin(2 * np.pi * 880.0 * t2)
    y = np.concatenate([y1, y2])

    analyzer = StructureAnalyzer(min_segment_duration=2.5)
    res = analyzer.analyze(y, sr)

    assert res.num_sections >= 2
    assert res.sections[0].start_time == 0.0
    assert res.sections[-1].end_time > 11.0
