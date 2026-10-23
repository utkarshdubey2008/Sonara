"""Tests for StyleConditionedGenerator and neural interface."""

import pytest

from sonara.analysis.collection import CollectionAnalyzer
from sonara.generation.generator import StyleConditionedGenerator
from sonara.generation.neural_interface import NeuralPromptConditioner


def test_generation_with_profile() -> None:
    gen = StyleConditionedGenerator(seed=42)
    doc_base = gen.generate(duration_seconds=8.0, target_bpm=130.0, target_key="D")

    profile = CollectionAnalyzer.analyze_collection([doc_base], profile_name="Profile_D")

    # Generate conditioned on profile
    gen_conditioned = StyleConditionedGenerator(seed=101)
    doc_new = gen_conditioned.generate(style_profile=profile, duration_seconds=12.0)

    assert doc_new.metadata.duration_seconds >= 10.0
    assert len(doc_new.events) > 20
    assert len(doc_new.structure) >= 1
    assert doc_new.harmony.primary_key == "D"


def test_neural_prompt_conditioner() -> None:
    gen = StyleConditionedGenerator(seed=42)
    doc = gen.generate(duration_seconds=8.0, target_bpm=120.0, target_key="C")

    prompt = NeuralPromptConditioner.from_smr_document(doc)

    assert "C" in prompt.key
    assert prompt.bpm == 120.0
    assert "120 BPM" in prompt.text_prompt
    assert len(prompt.conditioning_tags) >= 2
