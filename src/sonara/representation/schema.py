"""Sonara Musical Representation (SMR) v1.0.0 Schema Specification."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


SMR_SCHEMA_VERSION = "1.0.0"


class AudioMetadataModel(BaseModel):
    """Audio physical and container attributes."""
    file_path: str = Field(..., description="Path to source audio file")
    file_name: str = Field(..., description="Audio file basename")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    sha256_hash: str = Field(..., description="SHA-256 integrity hash")
    duration_seconds: float = Field(..., ge=0.0, description="Total duration in seconds")
    sample_rate: int = Field(..., gt=0, description="Audio sample rate in Hz")
    channels: int = Field(..., gt=0, description="Number of audio channels")
    format: str = Field(..., description="Audio container format (e.g. WAV, MP3, FLAC)")
    is_silent: bool = Field(False, description="Flag indicating completely silent audio")
    is_clipped: bool = Field(False, description="Flag indicating potential amplitude clipping")
    peak_amplitude: float = Field(0.0, ge=0.0, le=1.0, description="Peak absolute sample amplitude")
    rms_amplitude: float = Field(0.0, ge=0.0, description="Root-mean-square amplitude")


class AcousticFeaturesModel(BaseModel):
    """Acoustic, timbral, and spectral descriptors."""
    rms_energy_mean: float = Field(..., description="Mean frame-wise RMS energy")
    dynamic_range_db: float = Field(..., description="Dynamic range in decibels")
    crest_factor: float = Field(..., description="Ratio of peak amplitude to RMS energy")
    spectral_centroid_mean: float = Field(..., description="Perceived brightness (centroid) in Hz")
    spectral_centroid_std: float = Field(..., description="Standard deviation of spectral centroid")
    spectral_bandwidth_mean: float = Field(..., description="Spectral frequency spread in Hz")
    spectral_rolloff_mean: float = Field(..., description="85% energy spectral rolloff in Hz")
    spectral_flatness_mean: float = Field(..., description="Wiener entropy / noisiness (0.0 to 1.0)")
    zero_crossing_rate_mean: float = Field(..., description="Average zero crossing rate")
    warmth_index: float = Field(..., description="Proportion of spectral energy below 500 Hz")
    brightness_index: float = Field(..., description="Proportion of spectral energy above 3000 Hz")
    spectral_contrast_mean: List[float] = Field(default_factory=list, description="Octave-band spectral contrast")
    mfcc_mean: List[float] = Field(default_factory=list, description="Mean MFCC vector (13 coefficients)")
    mfcc_std: List[float] = Field(default_factory=list, description="Standard deviation MFCC vector")
    uncertainty_flags: List[str] = Field(default_factory=list, description="DSP uncertainty warnings")


class RhythmProfileModel(BaseModel):
    """Rhythmic, tempo, and groove metrics."""
    estimated_bpm: float = Field(..., ge=0.0, description="Estimated beats per minute")
    tempo_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in tempo estimation")
    time_signature_estimate: str = Field("4/4", description="Estimated metric meter (e.g. 4/4, 3/4)")
    beat_count: int = Field(..., ge=0, description="Total detected beats")
    onset_count: int = Field(..., ge=0, description="Total detected transient onsets")
    onset_density_per_second: float = Field(..., ge=0.0, description="Onset frequency per second")
    syncopation_index: float = Field(..., ge=0.0, le=1.0, description="Proportion of syncopated onsets")
    pulse_clarity: float = Field(..., ge=0.0, le=1.0, description="Periodicity / clarity of rhythmic pulse")
    is_rhythmic: bool = Field(True, description="Whether track exhibits discernible rhythm")
    beat_timestamps: List[float] = Field(default_factory=list, description="Timestamps of detected beats")
    uncertainty_flags: List[str] = Field(default_factory=list, description="Rhythm estimation flags")


class ChordItemModel(BaseModel):
    """Harmonic chord segment."""
    chord_symbol: str = Field(..., description="Chord symbol (e.g. C:maj, A:min)")
    root: str = Field(..., description="Root note (e.g. C, G#)")
    quality: str = Field(..., description="Triad quality (maj, min, 7, dim)")
    start_time: float = Field(..., ge=0.0, description="Onset time in seconds")
    end_time: float = Field(..., ge=0.0, description="Offset time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class HarmonicProfileModel(BaseModel):
    """Tonal and harmonic characteristics."""
    primary_key: str = Field(..., description="Estimated tonal center (e.g. C, F#, Bb)")
    mode: str = Field(..., description="Mode (major, minor, or unknown)")
    key_confidence: float = Field(..., ge=0.0, le=1.0, description="Key estimation confidence")
    tonal_stability: float = Field(..., ge=0.0, le=1.0, description="Correlation to ideal diatonic profile")
    chord_vocabulary: List[str] = Field(default_factory=list, description="Unique chords used across the track")
    chord_sequence: List[ChordItemModel] = Field(default_factory=list, description="Chronological chord progression")
    chroma_profile: List[float] = Field(default_factory=list, description="12-bin pitch class distribution")
    uncertainty_flags: List[str] = Field(default_factory=list, description="Harmonic ambiguity warnings")


class StructuralSectionModel(BaseModel):
    """Hierarchical formal section of a musical composition."""
    section_id: str = Field(..., description="Unique section identifier (e.g. sec_0)")
    section_type: str = Field(..., description="Functional role: intro, verse, chorus, bridge, outro, transition, body")
    start_time: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time: float = Field(..., ge=0.0, description="End time in seconds")
    duration: float = Field(..., ge=0.0, description="Duration of section in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Segmentation confidence")
    is_repetition: bool = Field(False, description="Whether this section repeats an earlier section")
    repeat_of_id: Optional[str] = Field(None, description="Section ID of original section if repeated")


class MusicalEventModel(BaseModel):
    """Atomic musical event (note, transient hit, or chord)."""
    event_id: str = Field(..., description="Unique event identifier (e.g. evt_0012)")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    duration: float = Field(..., ge=0.0, description="Estimated duration in seconds")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds")
    layer: str = Field(..., description="Functional layer: percussion, bass, harmonic, melodic, or unknown")
    event_type: str = Field(..., description="Event type: note, hit, chord, onset, rest")
    pitch_midi: Optional[int] = Field(None, ge=0, le=127, description="MIDI note number (0-127)")
    pitch_name: Optional[str] = Field(None, description="Standard note name (e.g. C4, A#2)")
    frequency_hz: Optional[float] = Field(None, ge=0.0, description="Fundamental frequency in Hz")
    cents_deviation: Optional[float] = Field(None, description="Tuning deviation in cents (-50 to +50)")
    velocity: int = Field(..., ge=0, le=127, description="Estimated velocity/dynamics (0-127)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score (0.0 to 1.0)")
    uncertainty_flags: List[str] = Field(default_factory=list, description="Uncertainty markers")


class LayerInteractionModel(BaseModel):
    """Statistical interaction between different instrument layers."""
    layer_pair: str = Field(..., description="E.g. bass_percussion, melodic_harmonic")
    synchrony_score: float = Field(..., ge=0.0, le=1.0, description="Rhythmic alignment between layers")
    density_ratio: float = Field(..., ge=0.0, description="Ratio of event density between the two layers")
    harmonic_support_overlap: float = Field(0.0, ge=0.0, le=1.0, description="Harmonic alignment ratio")


class AnalysisProvenanceModel(BaseModel):
    """Metadata regarding analysis execution, versions, and limitations."""
    system_name: str = Field("Sonara", description="Analysis engine name")
    system_version: str = Field(..., description="Sonara software version")
    smr_schema_version: str = Field(SMR_SCHEMA_VERSION, description="SMR format schema version")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC execution timestamp")
    execution_time_seconds: float = Field(..., ge=0.0, description="Processing duration in seconds")
    limitations: List[str] = Field(default_factory=list, description="Explicit analysis limitations")
    warnings: List[str] = Field(default_factory=list, description="Runtime execution warnings")


class SMRDocument(BaseModel):
    """Complete Sonara Musical Representation (SMR) Document."""
    smr_version: str = Field(SMR_SCHEMA_VERSION, description="SMR specification version")
    metadata: AudioMetadataModel
    acoustic_features: AcousticFeaturesModel
    rhythm: RhythmProfileModel
    harmony: HarmonicProfileModel
    structure: List[StructuralSectionModel] = Field(default_factory=list)
    events: List[MusicalEventModel] = Field(default_factory=list)
    layer_interactions: List[LayerInteractionModel] = Field(default_factory=list)
    provenance: AnalysisProvenanceModel
