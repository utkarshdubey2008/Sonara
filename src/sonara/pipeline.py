"""End-to-end processing pipeline for Sonara Musical Representation."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from sonara.audio.features import FeatureExtractor
from sonara.audio.harmony import HarmonyAnalyzer
from sonara.audio.loader import AudioLoader
from sonara.audio.rhythm import RhythmAnalyzer
from sonara.audio.structure import StructureAnalyzer
from sonara.events.classifier import LayerClassifier
from sonara.events.extractor import EventExtractor
from sonara.generation.synthesizer import SMRSynthesizer
from sonara.representation.midi_converter import MIDIConverter
from sonara.representation.report import SMRReportGenerator
from sonara.representation.schema import (
    AcousticFeaturesModel,
    AnalysisProvenanceModel,
    AudioMetadataModel,
    ChordItemModel,
    HarmonicProfileModel,
    LayerInteractionModel,
    MusicalEventModel,
    RhythmProfileModel,
    SMRDocument,
    StructuralSectionModel,
)
from sonara.representation.serializer import SMRSerializer
from sonara.representation.validator import SMRValidator


class SonaraPipeline:
    """Master orchestration pipeline for analyzing audio into SMR representations."""

    SYSTEM_VERSION = "0.1.0"

    def __init__(self, target_sr: int = 22050, hop_length: int = 512):
        self.target_sr = target_sr
        self.hop_length = hop_length

        self.loader = AudioLoader(default_sr=target_sr, mono=True)
        self.feature_extractor = FeatureExtractor(hop_length=hop_length)
        self.rhythm_analyzer = RhythmAnalyzer(hop_length=hop_length)
        self.harmony_analyzer = HarmonyAnalyzer(hop_length=hop_length)
        self.structure_analyzer = StructureAnalyzer(hop_length=hop_length * 2)
        self.event_extractor = EventExtractor(hop_length=hop_length)
        self.synthesizer = SMRSynthesizer(sample_rate=44100)

    def analyze(self, audio_path: str | Path) -> SMRDocument:
        """
        Execute end-to-end analysis on an audio file.

        Args:
            audio_path: Path to the target audio file.

        Returns:
            Validated SMRDocument instance.
        """
        t0 = time.time()
        start_utc = datetime.now(timezone.utc).isoformat()
        warnings: List[str] = []
        limitations: List[str] = [
            "Source separation is estimated via spectral energy bands; full stem isolation requires deep neural models.",
            "Polyphonic chord estimation is based on template matching against 12-bin chroma STFT.",
            "Structural section labels (verse, chorus) are heuristic inferences from self-similarity recurrence.",
        ]

        # 1. Load audio and metadata
        waveform, raw_meta = self.loader.load(audio_path, sr=self.target_sr)
        duration = raw_meta.duration_seconds

        # 2. Extract Acoustic & Spectral Features
        acoustic_res = self.feature_extractor.extract(waveform, self.target_sr)
        warnings.extend(acoustic_res.uncertainty_flags)

        # 3. Analyze Rhythm & Metric Grid
        rhythm_res = self.rhythm_analyzer.analyze(waveform, self.target_sr)
        warnings.extend(rhythm_res.uncertainty_flags)

        # 4. Analyze Harmony, Key & Chords
        harmony_res = self.harmony_analyzer.analyze(
            waveform, self.target_sr, beat_timestamps=rhythm_res.beat_timestamps
        )
        warnings.extend(harmony_res.uncertainty_flags)

        # 5. Segment Structure & Recurrence
        struct_res = self.structure_analyzer.analyze(waveform, self.target_sr)
        warnings.extend(struct_res.uncertainty_flags)

        # 6. Extract Discrete Musical Events (Rhythm, Bass, Melodic)
        raw_events = self.event_extractor.extract_events(
            waveform, self.target_sr, onset_timestamps=rhythm_res.onset_timestamps
        )

        # 7. Convert Events to Pydantic Models & Inject Decoded Harmonic Chords
        event_models: List[MusicalEventModel] = []
        for e in raw_events:
            event_models.append(
                MusicalEventModel(
                    event_id=e.event_id,
                    start_time=e.start_time,
                    duration=e.duration,
                    end_time=e.end_time,
                    layer=e.layer,
                    event_type=e.event_type,
                    pitch_midi=e.pitch_midi,
                    pitch_name=e.pitch_name,
                    frequency_hz=e.frequency_hz,
                    cents_deviation=e.cents_deviation,
                    velocity=e.velocity,
                    confidence=e.confidence,
                    uncertainty_flags=e.uncertainty_flags,
                )
            )

        # Inject decoded harmonic chords as sustained polyphonic pad events
        pitch_map = {"C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64, "F": 65, "F#": 66, "G": 67, "G#": 68, "A": 69, "A#": 70, "B": 71}
        chord_counter = len(event_models)
        for ch in harmony_res.chord_sequence:
            root_midi = pitch_map.get(ch.root, 60)
            if ch.quality == "min":
                intervals = [0, 3, 7]
            elif ch.quality == "7":
                intervals = [0, 4, 7, 10]
            elif ch.quality == "dim":
                intervals = [0, 3, 6]
            else:  # maj default
                intervals = [0, 4, 7]

            for interval in intervals:
                c_note = root_midi + interval
                c_freq = round(440.0 * (2 ** ((c_note - 69) / 12)), 2)
                c_dur = round(max(0.1, ch.end_time - ch.start_time), 3)
                event_models.append(
                    MusicalEventModel(
                        event_id=f"evt_harm_{chord_counter:04d}",
                        start_time=round(ch.start_time, 3),
                        duration=c_dur,
                        end_time=round(ch.start_time + c_dur, 3),
                        layer=LayerClassifier.LAYER_HARMONIC,
                        event_type="chord",
                        pitch_midi=c_note,
                        pitch_name=list(pitch_map.keys())[c_note % 12] + str(c_note // 12 - 1),
                        frequency_hz=c_freq,
                        cents_deviation=0.0,
                        velocity=68,
                        confidence=round(ch.confidence, 3),
                        uncertainty_flags=["decoded_from_chroma_progression"],
                    )
                )
                chord_counter += 1

        # Re-sort all events chronologically
        event_models.sort(key=lambda ev: (ev.start_time, ev.layer))

        # 8. Compute Layer Interactions (Bass & Percussion Synchrony)
        layer_interactions: List[LayerInteractionModel] = []
        perc_events = [e for e in event_models if e.layer == LayerClassifier.LAYER_PERCUSSION]
        bass_events = [e for e in event_models if e.layer == LayerClassifier.LAYER_BASS]

        if perc_events and bass_events:
            # Measure synchrony: fraction of bass notes landing within 50ms of a drum hit
            aligned_bass = 0
            for b in bass_events:
                if any(abs(b.start_time - p.start_time) <= 0.05 for p in perc_events):
                    aligned_bass += 1
            sync_score = float(aligned_bass / len(bass_events))
            density_ratio = float(len(perc_events) / max(1, len(bass_events)))

            layer_interactions.append(
                LayerInteractionModel(
                    layer_pair="bass_percussion",
                    synchrony_score=round(sync_score, 3),
                    density_ratio=round(density_ratio, 2),
                    harmonic_support_overlap=0.85,
                )
            )

        # 9. Assemble Pydantic Document Sub-models
        meta_model = AudioMetadataModel(
            file_path=raw_meta.file_path,
            file_name=raw_meta.file_name,
            file_size_bytes=raw_meta.file_size_bytes,
            sha256_hash=raw_meta.sha256_hash,
            duration_seconds=round(duration, 3),
            sample_rate=raw_meta.sample_rate,
            channels=raw_meta.channels,
            format=raw_meta.format,
            is_silent=raw_meta.is_silent,
            is_clipped=raw_meta.is_clipped,
            peak_amplitude=round(raw_meta.peak_amplitude, 4),
            rms_amplitude=round(raw_meta.rms_amplitude, 4),
        )

        acoustics_model = AcousticFeaturesModel(
            rms_energy_mean=round(acoustic_res.rms_energy_mean, 4),
            dynamic_range_db=round(acoustic_res.dynamic_range_db, 2),
            crest_factor=round(acoustic_res.crest_factor, 2),
            spectral_centroid_mean=round(acoustic_res.spectral_centroid_mean, 1),
            spectral_centroid_std=round(acoustic_res.spectral_centroid_std, 1),
            spectral_bandwidth_mean=round(acoustic_res.spectral_bandwidth_mean, 1),
            spectral_rolloff_mean=round(acoustic_res.spectral_rolloff_mean, 1),
            spectral_flatness_mean=round(acoustic_res.spectral_flatness_mean, 4),
            zero_crossing_rate_mean=round(acoustic_res.zero_crossing_rate_mean, 4),
            warmth_index=round(acoustic_res.warmth_index, 4),
            brightness_index=round(acoustic_res.brightness_index, 4),
            spectral_contrast_mean=[round(c, 2) for c in acoustic_res.spectral_contrast_mean],
            mfcc_mean=[round(m, 2) for m in acoustic_res.mfcc_mean],
            mfcc_std=[round(s, 2) for s in acoustic_res.mfcc_std],
            uncertainty_flags=acoustic_res.uncertainty_flags,
        )

        rhythm_model = RhythmProfileModel(
            estimated_bpm=rhythm_res.estimated_bpm,
            tempo_confidence=rhythm_res.tempo_confidence,
            time_signature_estimate=rhythm_res.time_signature_estimate,
            beat_count=len(rhythm_res.beat_timestamps),
            onset_count=rhythm_res.onset_count,
            onset_density_per_second=rhythm_res.onset_density_per_second,
            syncopation_index=rhythm_res.syncopation_index,
            pulse_clarity=rhythm_res.pulse_clarity,
            is_rhythmic=rhythm_res.is_rhythmic,
            beat_timestamps=rhythm_res.beat_timestamps,
            uncertainty_flags=rhythm_res.uncertainty_flags,
        )

        chord_models = [
            ChordItemModel(
                chord_symbol=ch.chord_symbol,
                root=ch.root,
                quality=ch.quality,
                start_time=ch.start_time,
                end_time=ch.end_time,
                confidence=ch.confidence,
            )
            for ch in harmony_res.chord_sequence
        ]

        harmony_model = HarmonicProfileModel(
            primary_key=harmony_res.primary_key,
            mode=harmony_res.mode,
            key_confidence=harmony_res.key_confidence,
            tonal_stability=harmony_res.tonal_stability,
            chord_vocabulary=harmony_res.chord_vocabulary,
            chord_sequence=chord_models,
            chroma_profile=harmony_res.chroma_profile,
            uncertainty_flags=harmony_res.uncertainty_flags,
        )

        section_models = [
            StructuralSectionModel(
                section_id=sec.section_id,
                section_type=sec.section_type,
                start_time=sec.start_time,
                end_time=sec.end_time,
                duration=sec.duration,
                confidence=sec.confidence,
                is_repetition=sec.is_repetition,
                repeat_of_id=sec.repeat_of_id,
            )
            for sec in struct_res.sections
        ]

        elapsed = time.time() - t0

        provenance_model = AnalysisProvenanceModel(
            system_name="Sonara",
            system_version=self.SYSTEM_VERSION,
            timestamp_utc=start_utc,
            execution_time_seconds=round(elapsed, 3),
            limitations=limitations,
            warnings=list(set(warnings)),
        )

        document = SMRDocument(
            metadata=meta_model,
            acoustic_features=acoustics_model,
            rhythm=rhythm_model,
            harmony=harmony_model,
            structure=section_models,
            events=event_models,
            layer_interactions=layer_interactions,
            provenance=provenance_model,
        )

        # Validate invariants
        val_report = SMRValidator.validate(document)
        if not val_report.is_valid:
            for err in val_report.errors:
                provenance_model.warnings.append(f"Validation error: {err}")

        return document

    def export_report(self, document: SMRDocument, output_path: str | Path) -> Path:
        """Generate and save Markdown analysis report."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        report_text = SMRReportGenerator.generate_markdown(document)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return path

    def export_midi(self, document: SMRDocument, output_path: str | Path) -> Path:
        """Export SMRDocument to Standard MIDI."""
        return MIDIConverter.smr_to_midi(document, output_path)

    def render_audio(self, document: SMRDocument, output_path: str | Path) -> Path:
        """Synthesize SMRDocument directly into WAV audio."""
        return self.synthesizer.render(document, output_path)
