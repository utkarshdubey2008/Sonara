"""Tests for multi-song collection analysis, style profiling, and similarity."""

from pathlib import Path
import pytest

from sonara.analysis.collection import CollectionAnalyzer
from sonara.analysis.similarity import SimilarityEngine
from sonara.generation.generator import StyleConditionedGenerator


def test_collection_analysis_and_style_profile() -> None:
    gen = StyleConditionedGenerator()
    doc1 = gen.generate(duration_seconds=8.0, target_bpm=120.0, target_key="C")
    doc2 = gen.generate(duration_seconds=8.0, target_bpm=128.0, target_key="G")
    doc3 = gen.generate(duration_seconds=8.0, target_bpm=116.0, target_key="C")

    docs = [doc1, doc2, doc3]
    profile = CollectionAnalyzer.analyze_collection(docs, profile_name="Tri_Track_Profile")

    assert profile.tracks_analyzed == 3
    assert 115.0 <= profile.rhythm.mean_bpm <= 125.0
    assert profile.rhythm.dominant_meter == "4/4"
    assert "C" in profile.harmony.key_distribution
    assert profile.harmony.key_distribution["C"] > 0.5  # 2 out of 3 are in C
    assert len(profile.evidence) > 0
    assert profile.timbre.mean_spectral_centroid > 0


def test_cross_track_similarity() -> None:
    gen = StyleConditionedGenerator()
    doc_c_120 = gen.generate(duration_seconds=8.0, target_bpm=120.0, target_key="C")
    doc_c_122 = gen.generate(duration_seconds=8.0, target_bpm=122.0, target_key="C")
    doc_f_160 = gen.generate(duration_seconds=8.0, target_bpm=160.0, target_key="F#")

    match_close = SimilarityEngine.compute_pairwise(doc_c_120, doc_c_122)
    match_far = SimilarityEngine.compute_pairwise(doc_c_120, doc_f_160)

    # Tracks with similar key and tempo should have higher cosine similarity than far tracks
    assert match_close.cosine_similarity >= match_far.cosine_similarity
