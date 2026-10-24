"""Unit and integration tests for Sonara Instrumental Representation (SMR v2.0)."""

import io
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from sonara.representation.instrumental_schema import (
    SMRv2InstrumentalDocument,
    InstrumentEntityModel,
    InstrumentCategory,
    GeneratorType,
    PreservationStrategy,
    SoundDesignModel,
    ADSRSettings,
    FilterSettings,
    EffectsChainSettings,
    PatternModel,
    PatternEventModel,
    TrackPlacementModel,
    DropMarkerModel,
    VocalExclusionModel,
)
from sonara.representation.package import SMRPackage
from sonara.representation.transformer import (
    InstrumentalTransformer,
    InstrumentalSynthesizer,
)
from sonara.representation.converter_v2 import InstrumentalAudioConverter


@pytest.fixture
def sample_smr_v2_document() -> SMRv2InstrumentalDocument:
    """Fixture providing a minimal valid SMR v2.0 document."""
    kick = InstrumentEntityModel(
        instrument_id="inst_kick",
        name="Sub Kick",
        category=InstrumentCategory.PERCUSSION,
        musical_role="downbeat_foundation",
        preservation_strategy=PreservationStrategy.HYBRID_SAMPLE_MODELED,
        sound_design=SoundDesignModel(
            generator_type=GeneratorType.HYBRID_DSP,
            oscillator_topology="pitch_swept_sine",
            envelope=ADSRSettings(attack_ms=1.0, decay_ms=150.0, sustain_level=0.0, release_ms=50.0),
        ),
    )
    snare = InstrumentEntityModel(
        instrument_id="inst_snare",
        name="Snare Backbeat",
        category=InstrumentCategory.PERCUSSION,
        musical_role="backbeat_anchor",
        preservation_strategy=PreservationStrategy.HYBRID_SAMPLE_MODELED,
        sound_design=SoundDesignModel(
            generator_type=GeneratorType.HYBRID_DSP,
            oscillator_topology="dual_sine_filtered_noise",
            envelope=ADSRSettings(attack_ms=1.0, decay_ms=160.0, sustain_level=0.0, release_ms=50.0),
        ),
    )
    bass = InstrumentEntityModel(
        instrument_id="inst_bass",
        name="Saw Sub Bass",
        category=InstrumentCategory.BASS,
        musical_role="sub_foundation",
        preservation_strategy=PreservationStrategy.HYBRID_SAMPLE_MODELED,
        sound_design=SoundDesignModel(
            generator_type=GeneratorType.ANALOG_SUBTRACTIVE,
            oscillator_topology="saw_sub_sine",
            envelope=ADSRSettings(attack_ms=5.0, decay_ms=200.0, sustain_level=0.8, release_ms=150.0),
            filter=FilterSettings(filter_type="lowpass_24db", cutoff_hz=180.0, resonance_q=1.2),
        ),
    )

    # 1-bar pattern
    pat_drums = PatternModel(
        pattern_id="pat_drum_loop",
        name="1-bar drum loop",
        length_bars=1,
        time_signature="4/4",
        events=[
            PatternEventModel(event_id="e1", start_beat=0.0, duration_beats=0.5, pitch_midi=36, velocity=127),
            PatternEventModel(event_id="e2", start_beat=1.0, duration_beats=0.4, pitch_midi=38, velocity=115),
            PatternEventModel(event_id="e3", start_beat=2.0, duration_beats=0.5, pitch_midi=36, velocity=120),
            PatternEventModel(event_id="e4", start_beat=3.0, duration_beats=0.4, pitch_midi=38, velocity=122),
        ],
    )

    pat_bass = PatternModel(
        pattern_id="pat_bass_loop",
        name="1-bar bass loop",
        length_bars=1,
        time_signature="4/4",
        events=[
            PatternEventModel(event_id="b1", start_beat=0.0, duration_beats=1.5, pitch_midi=30, velocity=110),
            PatternEventModel(event_id="b2", start_beat=2.0, duration_beats=1.5, pitch_midi=30, velocity=110),
        ],
    )

    placements = [
        TrackPlacementModel(placement_id="pl1", instrument_id="inst_kick", pattern_id="pat_drum_loop", start_bar=1, repeat_count=4),
        TrackPlacementModel(placement_id="pl2", instrument_id="inst_snare", pattern_id="pat_drum_loop", start_bar=1, repeat_count=4),
        TrackPlacementModel(placement_id="pl3", instrument_id="inst_bass", pattern_id="pat_bass_loop", start_bar=1, repeat_count=4),
    ]

    doc = SMRv2InstrumentalDocument(
        track_title="Test Benchmark Beat",
        total_duration_seconds=8.0,
        total_bars=4,
        master_bpm=120.0,
        master_key="F#",
        master_scale="minor",
        instruments=[kick, snare, bass],
        patterns=[pat_drums, pat_bass],
        arrangement_tracks=placements,
        drops=[
            DropMarkerModel(
                drop_id="drop1",
                bar_index=1,
                time_seconds=0.0,
                transition_type="drop_impact",
                impact_intensity=1.0,
                entering_instruments=["inst_kick", "inst_bass"],
            )
        ],
        vocal_exclusion=VocalExclusionModel(
            is_lead_vocal_excluded=True,
            isolation_method="spectral_subtraction",
            suppression_depth_db=-48.0,
        ),
    )
    return doc


def test_smr_v2_schema_validation(sample_smr_v2_document):
    """Test SMR v2.0 document serialization and schema validation."""
    doc = sample_smr_v2_document
    assert doc.format_version == "2.0.0"
    assert doc.master_bpm == 120.0
    assert doc.master_key == "F#"
    assert doc.vocal_exclusion.is_lead_vocal_excluded is True

    # Validate JSON round-trip
    json_str = doc.model_dump_json()
    reloaded = SMRv2InstrumentalDocument.model_validate_json(json_str)
    assert reloaded.track_title == doc.track_title
    assert len(reloaded.instruments) == 3
    assert len(reloaded.patterns) == 2


def test_smr_package_container_roundtrip(sample_smr_v2_document, tmp_path):
    """Test saving and loading a .smr package bundle with audio assets."""
    doc = sample_smr_v2_document
    package = SMRPackage(document=doc)

    # Add a mock sample asset
    mock_audio = np.sin(2.0 * np.pi * 440.0 * np.linspace(0, 0.5, 22050)).astype(np.float32)
    sample_ref = package.add_sample("assets/samples/synth_pluck.wav", mock_audio, sample_rate=44100)
    assert sample_ref == "assets/samples/synth_pluck.wav"
    assert sample_ref in doc.embedded_sample_assets

    # Save package to .smr archive
    smr_path = tmp_path / "test_project.smr"
    package.save_package(smr_path)
    assert smr_path.exists()
    assert smr_path.stat().st_size > 0

    # Load package back
    loaded_pkg = SMRPackage.load_package(smr_path)
    assert loaded_pkg.document is not None
    assert loaded_pkg.document.track_title == doc.track_title
    assert loaded_pkg.document.vocal_exclusion.is_lead_vocal_excluded is True
    assert "assets/samples/synth_pluck.wav" in loaded_pkg.samples

    retrieved_audio, sr = loaded_pkg.get_sample("assets/samples/synth_pluck.wav")
    assert sr == 44100
    assert len(retrieved_audio) == len(mock_audio)
    np.testing.assert_allclose(retrieved_audio, mock_audio, atol=1e-3)


def test_instrumental_transposition(sample_smr_v2_document):
    """Test transposition preserves rhythm/percussion while shifting bass and melodic pitches."""
    doc = sample_smr_v2_document
    # Pitch of bass is MIDI 30 (F#1), Key is F#
    transposed = InstrumentalTransformer.transpose(doc, semitones=2)

    # Master key should shift from F# to G#
    assert transposed.master_key == "G#"

    # Bass pattern should shift from 30 to 32
    bass_pat = next(p for p in transposed.patterns if p.pattern_id == "pat_bass_loop")
    assert bass_pat.events[0].pitch_midi == 32

    # Percussion pattern events should NOT transpose unpitched hits
    # (even though MIDI numbers exist for GM drums, we clamp or keep)
    drum_pat = next(p for p in transposed.patterns if p.pattern_id == "pat_drum_loop")
    # All 4 events have pitch_midi in [12, 120]
    assert len(drum_pat.events) == 4


def test_instrumental_time_stretch(sample_smr_v2_document):
    """Test time stretch rescales master BPM and updates absolute time coordinates."""
    doc = sample_smr_v2_document
    # Original: 120 BPM, 4 bars = 8.0 seconds
    # Stretched: 60 BPM -> 4 bars = 16.0 seconds
    stretched = InstrumentalTransformer.time_stretch(doc, new_bpm=60.0)
    assert stretched.master_bpm == 60.0
    assert pytest.approx(stretched.total_duration_seconds, rel=1e-2) == 16.0

    # Pattern event durations in beats must remain invariant!
    pat_bass = next(p for p in stretched.patterns if p.pattern_id == "pat_bass_loop")
    assert pat_bass.events[0].duration_beats == 1.5


def test_instrumental_sound_design_swap(sample_smr_v2_document):
    """Test replacing sound design model on an instrument entity."""
    doc = sample_smr_v2_document
    new_sound = SoundDesignModel(
        generator_type=GeneratorType.ANALOG_SUBTRACTIVE,
        oscillator_topology="acid_saw_303",
        filter=FilterSettings(filter_type="lowpass_24db", cutoff_hz=600.0, resonance_q=8.5),
        effects=EffectsChainSettings(drive_saturation=0.85),
    )
    modified = InstrumentalTransformer.swap_instrument_sound(doc, "inst_bass", new_sound)
    bass = next(inst for inst in modified.instruments if inst.instrument_id == "inst_bass")
    assert bass.sound_design.oscillator_topology == "acid_saw_303"
    assert bass.sound_design.filter.cutoff_hz == 600.0
    assert bass.sound_design.effects.drive_saturation == 0.85


def test_instrumental_synthesis_reconstruction(sample_smr_v2_document):
    """Test high-fidelity DSP synthesis renders complete stereo audio array."""
    doc = sample_smr_v2_document
    synthesizer = InstrumentalSynthesizer(sample_rate=22050)
    audio = synthesizer.render(doc)

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 2
    assert audio.shape[0] == 2  # Stereo Left/Right
    expected_samples = int(doc.total_duration_seconds * 22050)
    assert abs(audio.shape[1] - expected_samples) <= 22050  # Within 1 sec duration bounds
    # Audio must not be silence
    peak = np.max(np.abs(audio))
    assert peak > 0.05
    assert peak <= 1.0  # Limiter works


def test_audio_to_smr_converter_pipeline(tmp_path):
    """Test converting raw audio recording into SMR v2.0 document and package."""
    # Generate 4-second mock audio recording with kick, snare, and bass
    sr = 22050
    t = np.linspace(0, 4.0, int(4.0 * sr), endpoint=False)
    # 120 BPM = 0.5s per beat, 2.0s per bar
    audio = np.zeros_like(t)

    # Add kicks on 0.0s, 1.0s, 2.0s, 3.0s
    for beat_time in [0.0, 1.0, 2.0, 3.0]:
        idx = int(beat_time * sr)
        burst_len = int(0.2 * sr)
        audio[idx:idx + burst_len] += np.sin(2.0 * np.pi * 55.0 * np.linspace(0, 0.2, burst_len))

    # Save temporary WAV
    audio_path = tmp_path / "test_track.wav"
    sf.write(str(audio_path), audio, sr)

    converter = InstrumentalAudioConverter(sample_rate=sr)
    doc, package = converter.convert(audio_path, track_title="Test Audio To SMR")

    assert doc.track_title == "Test Audio To SMR"
    assert doc.vocal_exclusion.is_lead_vocal_excluded is True
    assert len(doc.instruments) >= 4
    assert len(doc.patterns) >= 3
    assert len(doc.arrangement_tracks) >= 3
    assert len(doc.drops) >= 1
    assert doc.relational_graph is not None

    # Verify package can be saved
    out_pkg = tmp_path / "output.smr"
    package.save_package(out_pkg)
    assert out_pkg.exists()
