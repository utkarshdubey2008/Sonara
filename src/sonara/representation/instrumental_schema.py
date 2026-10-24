"""Sonara Instrumental Representation (SMR v2.0 / SIR) Specification.

A comprehensive, machine-readable format for representing every meaningful detail
of a song's instrumental music, sound design, and production architecture,
strictly excluding the lead singer's vocal performance while preserving
backing instrumentation, vocoders, and vocal-like sound design instruments.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


SMR_V2_SPEC_VERSION = "2.0.0"


class InstrumentCategory(str, Enum):
    PERCUSSION = "percussion"
    BASS = "bass"
    CHORDAL = "chordal"
    MELODIC = "melodic"
    SAMPLE_INSTRUMENT = "sample_instrument"
    TEXTURAL_FX = "textural_fx"


class GeneratorType(str, Enum):
    SAMPLE_ONE_SHOT = "sample_one_shot"
    SAMPLE_SLICER = "sample_slicer"
    ANALOG_SUBTRACTIVE = "analog_subtractive"
    WAVETABLE_SYNTH = "wavetable_synth"
    FM_SYNTH = "fm_synth"
    PHYSICAL_MODEL = "physical_model"
    HYBRID_DSP = "hybrid_dsp"


class PreservationStrategy(str, Enum):
    PURE_SYMBOLIC = "pure_symbolic"
    HYBRID_SAMPLE_MODELED = "hybrid_sample_modeled"
    RAW_AUDIO_SLICE = "raw_audio_slice"


class ADSRSettings(BaseModel):
    """Attack, Decay, Sustain, Release envelope parameters in milliseconds."""
    attack_ms: float = Field(5.0, ge=0.0, description="Attack time in milliseconds")
    decay_ms: float = Field(150.0, ge=0.0, description="Decay time in milliseconds")
    sustain_level: float = Field(0.7, ge=0.0, le=1.0, description="Sustain amplitude level (0.0 to 1.0)")
    release_ms: float = Field(200.0, ge=0.0, description="Release time in milliseconds")


class FilterSettings(BaseModel):
    """DSP filter configuration."""
    filter_type: str = Field("lowpass_24db", description="Filter topology (lowpass_24db, bandpass_12db, highpass_12db)")
    cutoff_hz: float = Field(2000.0, ge=20.0, le=22050.0, description="Filter cutoff frequency in Hz")
    resonance_q: float = Field(0.707, ge=0.1, le=18.0, description="Filter resonance Q factor")
    envelope_amount: float = Field(0.0, ge=-1.0, le=1.0, description="Filter envelope modulation amount")


class EffectsChainSettings(BaseModel):
    """Audio production effects and spatialization."""
    drive_saturation: float = Field(0.0, ge=0.0, le=1.0, description="Harmonic saturation / overdrive depth")
    reverb_send: float = Field(0.0, ge=0.0, le=1.0, description="Wet reverb send amount")
    reverb_room_size_ms: float = Field(1200.0, ge=50.0, description="Simulated acoustic space decay time")
    delay_send: float = Field(0.0, ge=0.0, le=1.0, description="Echo/delay send amount")
    delay_time_beats: float = Field(0.75, ge=0.0, description="Delay offset in musical beats")
    sidechain_source: Optional[str] = Field(None, description="Instrument ID driving sidechain compression ducking")
    sidechain_duck_db: float = Field(0.0, le=0.0, ge=-24.0, description="Gain reduction depth during sidechain trigger")
    sidechain_release_ms: float = Field(65.0, ge=5.0, description="Sidechain compressor release recovery time")


class SoundDesignModel(BaseModel):
    """Complete sound synthesis and timbral profile of an instrumental entity."""
    generator_type: GeneratorType = Field(GeneratorType.ANALOG_SUBTRACTIVE, description="Synthesis / sampling engine")
    sample_asset_ref: Optional[str] = Field(None, description="Relative path to audio payload in SMR container (e.g. assets/kick_sample.wav)")
    oscillator_topology: str = Field("saw_sub_sine", description="Oscillator waveforms or algorithm")
    envelope: ADSRSettings = Field(default_factory=ADSRSettings, description="Amplitude dynamic envelope")
    filter: FilterSettings = Field(default_factory=FilterSettings, description="Timbral filtering")
    effects: EffectsChainSettings = Field(default_factory=EffectsChainSettings, description="Effects and sidechain routing")
    stereo_pan: float = Field(0.0, ge=-1.0, le=1.0, description="Default stereo field position (-1.0 Left to +1.0 Right)")
    gain_trim_db: float = Field(0.0, ge=-48.0, le=12.0, description="Output gain trim in decibels")


class TimbralFingerprintModel(BaseModel):
    """Objective spectral measurements capturing instrument timbre."""
    spectral_centroid_hz: float = Field(..., ge=0.0, description="Center of spectral mass")
    spectral_rolloff_hz: float = Field(..., ge=0.0, description="Frequency below which 85% of energy resides")
    spectral_flatness: float = Field(..., ge=0.0, le=1.0, description="Noise vs tonal harmonicity (0=pure tone, 1=white noise)")
    warmth_index: float = Field(..., description="Ratio of sub-300Hz energy to high-frequency energy")
    harmonic_to_noise_ratio_db: float = Field(..., description="Harmonic clarity in dB")


class InstrumentEntityModel(BaseModel):
    """An individual sound source, synthesizer, or instrument within the arrangement."""
    instrument_id: str = Field(..., description="Unique entity ID (e.g. inst_kick_808, inst_strings_ostinato)")
    name: str = Field(..., description="Descriptive instrument name")
    category: InstrumentCategory = Field(..., description="Broad instrumental classification")
    musical_role: str = Field(..., description="Role: downbeat_anchor, snare_backbeat, ostinato_lead, sub_foundation, etc.")
    preservation_strategy: PreservationStrategy = Field(PreservationStrategy.HYBRID_SAMPLE_MODELED)
    sound_design: SoundDesignModel = Field(default_factory=SoundDesignModel)
    timbral_fingerprint: Optional[TimbralFingerprintModel] = None
    frequency_band_primary_hz: List[float] = Field(default_factory=list, description="Primary frequency occupancy [min_hz, max_hz]")


class PatternEventModel(BaseModel):
    """Atomic musical micro-event inside a reusable pattern."""
    event_id: str = Field(..., description="Unique pattern event ID")
    start_beat: float = Field(..., ge=0.0, description="Start position in musical beats relative to pattern start (e.g. 0.0, 1.5)")
    duration_beats: float = Field(..., ge=0.0, description="Duration in musical beats")
    pitch_midi: Optional[int] = Field(None, ge=0, le=127, description="General MIDI pitch number or null for unpitched percussion")
    pitch_cents: float = Field(0.0, ge=-50.0, le=50.0, description="Microtonal detuning deviation")
    velocity: int = Field(100, ge=1, le=127, description="MIDI dynamic velocity")
    micro_timing_offset_ms: float = Field(0.0, description="Humanized micro-groove push/pull in milliseconds")
    transient_sharpness: float = Field(0.8, ge=0.0, le=1.0, description="Attack transient intensity")


class PatternModel(BaseModel):
    """A reusable musical motif, drum groove, bassline, or chord progression."""
    pattern_id: str = Field(..., description="Unique pattern identifier (e.g. pat_kick_main_groove, pat_strings_riff)")
    name: str = Field(..., description="Pattern descriptive label")
    length_bars: int = Field(1, ge=1, description="Length of pattern in whole musical bars")
    time_signature: str = Field("4/4", description="Meter signature")
    events: List[PatternEventModel] = Field(default_factory=list, description="Sequence of micro-events within pattern")


class TrackPlacementModel(BaseModel):
    """Invocation of an instrument playing a pattern at a specific bar in the arrangement."""
    placement_id: str = Field(..., description="Unique placement identifier")
    instrument_id: str = Field(..., description="Target instrument entity")
    pattern_id: str = Field(..., description="Pattern played")
    start_bar: int = Field(..., ge=1, description="Bar index where playback begins (1-based)")
    repeat_count: int = Field(1, ge=1, description="Number of sequential pattern repetitions")
    mute_mask_beats: List[float] = Field(default_factory=list, description="Beats muted for fill/breakdown drops")
    gain_offset_db: float = Field(0.0, description="Dynamic volume offset for this section")


class DropMarkerModel(BaseModel):
    """Formal musical drop, breakdown, build-up, or climactic transition."""
    drop_id: str = Field(..., description="Unique transition identifier")
    bar_index: int = Field(..., ge=1, description="Exact bar where drop or breakdown occurs")
    time_seconds: float = Field(..., ge=0.0, description="Exact timestamp in seconds")
    transition_type: str = Field(..., description="drop_impact, breakdown_dropout, build_riser, section_transition")
    impact_intensity: float = Field(1.0, ge=0.0, le=1.0, description="Perceptual impact / dynamic weight (0.0 to 1.0)")
    muted_instruments: List[str] = Field(default_factory=list, description="Instruments explicitly cut during breakdown")
    entering_instruments: List[str] = Field(default_factory=list, description="Instruments slamming in at drop impact")


class AutomationCurveModel(BaseModel):
    """Continuous production automation curve over time."""
    automation_id: str = Field(..., description="Unique automation curve identifier")
    target_instrument_id: str = Field(..., description="Target instrument receiving automation")
    parameter: str = Field(..., description="Parameter: filter_cutoff, volume_gain, sidechain_duck, reverb_send")
    timestamps: List[float] = Field(default_factory=list, description="Sequence of timestamps in seconds")
    values: List[float] = Field(default_factory=list, description="Sequence of parameter values")


class StructuralSectionV2Model(BaseModel):
    """Formal musical macro-section."""
    section_id: str = Field(..., description="Section identifier (e.g. sec_intro, sec_verse_1)")
    name: str = Field(..., description="Intro, Verse, Hook, Breakdown, Climax, Outro")
    start_bar: int = Field(..., ge=1, description="Start bar index")
    end_bar: int = Field(..., ge=1, description="End bar index")
    start_time_seconds: float = Field(..., ge=0.0)
    end_time_seconds: float = Field(..., ge=0.0)
    energy_level: float = Field(0.5, ge=0.0, le=1.0, description="Normalized section energy")


class SidechainRelationalModel(BaseModel):
    """Explicit sidechain interaction graph between instruments."""
    source_instrument_id: str = Field(..., description="Trigger instrument (typically Kick)")
    ducked_instrument_id: str = Field(..., description="Target instrument ducking (e.g. Bass, Strings, Pad)")
    ducking_depth_db: float = Field(-3.0, le=0.0, ge=-24.0, description="Ducking depth in decibels")
    release_time_ms: float = Field(65.0, ge=5.0, description="Release recovery curve duration")


class FrequencyAllocationModel(BaseModel):
    """Acoustic frequency spectrum domain management preventing collisions."""
    instrument_id: str = Field(..., description="Occupant instrument")
    fundamental_hz: float = Field(..., ge=0.0, description="Nominal fundamental frequency")
    allocated_band_hz: List[float] = Field(..., description="Band [low_hz, high_hz] prioritized in the mix")


class RelationalGraphModel(BaseModel):
    """Structural interaction graph describing how instruments cooperate."""
    sidechain_network: List[SidechainRelationalModel] = Field(default_factory=list)
    frequency_allocations: List[FrequencyAllocationModel] = Field(default_factory=list)
    kick_bass_synchrony_ratio: float = Field(0.85, ge=0.0, le=1.0, description="Correlation between kick and bass downbeats")


class VocalExclusionModel(BaseModel):
    """Formal verification of lead vocal exclusion boundary."""
    is_lead_vocal_excluded: bool = Field(True, description="Strictly True: Lead vocal performance is stripped out")
    isolation_method: str = Field("source_separation_decoupling", description="Technique used to exclude lead vocal line")
    suppression_depth_db: float = Field(-42.0, le=-12.0, description="Lead vocal suppression depth in dB")
    retained_instrumental_vocal_elements: List[str] = Field(
        default_factory=list,
        description="Non-lead vocal elements retained as instruments (e.g. vocoders, background chops, vocal synth pads)"
    )


class SMRv2InstrumentalDocument(BaseModel):
    """Complete Sonara Instrumental Representation (SMR v2.0) Master Document.
    
    Encodes the complete instrumental music, sound design, arrangements, patterns,
    micro-timing, and production graphs of a song, excluding lead vocals.
    """
    format_version: str = Field(SMR_V2_SPEC_VERSION, description="SMR format specification version")
    track_title: str = Field(..., description="Song / instrumental composition title")
    total_duration_seconds: float = Field(..., ge=0.0, description="Total playback length in seconds")
    total_bars: int = Field(..., ge=1, description="Total bar count of arrangement")
    master_bpm: float = Field(..., ge=20.0, le=300.0, description="Master project tempo in BPM")
    master_key: str = Field(..., description="Musical key (e.g. F#, C, A#)")
    master_scale: str = Field(..., description="Musical mode / scale (e.g. minor, major, dorian)")
    master_tuning_hz: float = Field(440.0, description="Concert pitch reference (default 440 Hz)")
    
    # Core Architecture Tiers
    instruments: List[InstrumentEntityModel] = Field(default_factory=list, description="Discovered sound design entities")
    patterns: List[PatternModel] = Field(default_factory=list, description="Reusable rhythmic and melodic motifs")
    sections: List[StructuralSectionV2Model] = Field(default_factory=list, description="Macro-formal arrangement sections")
    arrangement_tracks: List[TrackPlacementModel] = Field(default_factory=list, description="Pattern placement timeline")
    drops: List[DropMarkerModel] = Field(default_factory=list, description="Explicit drops, breakdowns, and climaxes")
    automation: List[AutomationCurveModel] = Field(default_factory=list, description="Continuous DSP parameter automation curves")
    relational_graph: RelationalGraphModel = Field(default_factory=RelationalGraphModel, description="Sidechain and frequency relations")
    vocal_exclusion: VocalExclusionModel = Field(default_factory=VocalExclusionModel, description="Lead vocal exclusion audit")
    
    # Metadata & Provenance
    embedded_sample_assets: List[str] = Field(default_factory=list, description="List of audio asset files inside the package")
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict, description="Creation timestamp, source audio hash, engine version")
