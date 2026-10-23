"""Style profile data model and generator for multi-track collections."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from sonara.representation.schema import SMRDocument


class EvidenceObservation(BaseModel):
    """Explicit evidence pointing back to measured track data."""
    metric_name: str = Field(..., description="Name of the measured property")
    track_name: str = Field(..., description="Source track basename")
    measured_value: float = Field(..., description="Observed numeric value")


class RhythmicTendencies(BaseModel):
    """Aggregated rhythmic profile across tracks."""
    mean_bpm: float = Field(..., description="Average tempo in BPM")
    bpm_std: float = Field(..., description="Tempo variation across collection")
    bpm_range: List[float] = Field(default_factory=list, description="[min_bpm, max_bpm]")
    mean_syncopation: float = Field(..., description="Average syncopation index")
    mean_onset_density: float = Field(..., description="Average onset density (events/sec)")
    mean_pulse_clarity: float = Field(..., description="Average pulse clarity")
    dominant_meter: str = Field("4/4", description="Most frequent time signature")


class HarmonicGrammar(BaseModel):
    """Aggregated harmonic patterns and chord progression tendencies."""
    key_distribution: Dict[str, float] = Field(default_factory=dict, description="Relative frequency of keys")
    mode_distribution: Dict[str, float] = Field(default_factory=dict, description="Major vs minor frequency")
    mean_tonal_stability: float = Field(..., description="Average tonal stability index")
    collective_chord_vocabulary: List[str] = Field(default_factory=list, description="All unique chords observed")
    chord_transition_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="First-order Markov transition probabilities P(chord_next | chord_current)"
    )


class StructuralGrammar(BaseModel):
    """Typical arrangement and sectional architecture."""
    average_section_count: float = Field(..., description="Average sections per track")
    average_section_duration: float = Field(..., description="Average duration of a section in seconds")
    repetition_density_mean: float = Field(..., description="Mean proportion of repeated sections")
    section_type_frequencies: Dict[str, float] = Field(
        default_factory=dict, description="Frequency of section types (verse, chorus, etc.)"
    )


class TimbralAesthetic(BaseModel):
    """Acoustic and production characteristics."""
    mean_spectral_centroid: float = Field(..., description="Average brightness in Hz")
    mean_warmth_index: float = Field(..., description="Average low-frequency warmth ratio")
    mean_brightness_index: float = Field(..., description="Average high-frequency brightness ratio")
    mean_dynamic_range_db: float = Field(..., description="Average dynamic range in dB")
    mean_crest_factor: float = Field(..., description="Average crest factor")


class StyleProfile(BaseModel):
    """Comprehensive, machine-readable style profile learned from a track collection."""
    profile_version: str = Field("1.0.0", description="Style profile specification version")
    profile_name: str = Field("Artist_Collection_Profile", description="Identifier for this profile")
    tracks_analyzed: int = Field(..., ge=1, description="Number of tracks analyzed")
    track_names: List[str] = Field(default_factory=list, description="List of included track filenames")
    total_audio_duration_seconds: float = Field(..., description="Total cumulative analyzed audio in seconds")
    rhythm: RhythmicTendencies
    harmony: HarmonicGrammar
    structure: StructuralGrammar
    timbre: TimbralAesthetic
    evidence: List[EvidenceObservation] = Field(default_factory=list, description="Data points supporting the profile")
