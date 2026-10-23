"""Bidirectional conversion between Sonara Musical Representation (SMR) and Standard MIDI."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from sonara.events.classifier import LayerClassifier
from sonara.representation.schema import (
    AcousticFeaturesModel,
    AnalysisProvenanceModel,
    AudioMetadataModel,
    HarmonicProfileModel,
    MusicalEventModel,
    RhythmProfileModel,
    SMRDocument,
)


class MIDIConverter:
    """Converts SMR documents to multi-track MIDI files and imports MIDI files into SMR."""

    TICKS_PER_BEAT = 480

    LAYER_CHANNELS = {
        LayerClassifier.LAYER_PERCUSSION: 9,  # Channel 10 in 1-based indexing
        LayerClassifier.LAYER_BASS: 1,
        LayerClassifier.LAYER_HARMONIC: 2,
        LayerClassifier.LAYER_MELODIC: 0,
        LayerClassifier.LAYER_UNKNOWN: 3,
    }

    LAYER_PROGRAMS = {
        LayerClassifier.LAYER_BASS: 33,      # Electric Bass (finger)
        LayerClassifier.LAYER_HARMONIC: 89,  # Pad 2 (warm)
        LayerClassifier.LAYER_MELODIC: 80,   # Lead 1 (square)
        LayerClassifier.LAYER_UNKNOWN: 0,    # Acoustic Grand Piano
    }

    @classmethod
    def smr_to_midi(cls, document: SMRDocument, output_path: str | Path) -> Path:
        """
        Export an SMR document into a Standard Multi-track MIDI Type 1 file.

        Args:
            document: SMRDocument instance.
            output_path: Path for output .mid file.

        Returns:
            Resolved Path of the saved MIDI file.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        mid = MidiFile(type=1, ticks_per_beat=cls.TICKS_PER_BEAT)
        bpm = max(20.0, min(300.0, document.rhythm.estimated_bpm or 120.0))
        tempo_us = mido.bpm2tempo(bpm)

        # 1. Master Tempo and Meta Track
        meta_track = MidiTrack()
        meta_track.name = f"Sonara Master: {document.metadata.file_name}"
        meta_track.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))

        # Time signature
        ts = document.rhythm.time_signature_estimate or "4/4"
        try:
            num, den = (int(p) for p in ts.split("/"))
        except Exception:
            num, den = 4, 4
        meta_track.append(MetaMessage("time_signature", numerator=num, denominator=den, time=0))
        meta_track.append(MetaMessage("end_of_track", time=0))
        mid.tracks.append(meta_track)

        # 2. Partition events by functional layer
        layer_events: Dict[str, List[MusicalEventModel]] = {
            LayerClassifier.LAYER_PERCUSSION: [],
            LayerClassifier.LAYER_BASS: [],
            LayerClassifier.LAYER_HARMONIC: [],
            LayerClassifier.LAYER_MELODIC: [],
            LayerClassifier.LAYER_UNKNOWN: [],
        }

        for ev in document.events:
            layer_name = ev.layer if ev.layer in layer_events else LayerClassifier.LAYER_UNKNOWN
            layer_events[layer_name].append(ev)

        # Helper to convert seconds to absolute MIDI ticks
        seconds_per_tick = (60.0 / bpm) / cls.TICKS_PER_BEAT

        def sec_to_tick(s: float) -> int:
            return int(round(s / seconds_per_tick))

        # 3. Create MIDI tracks for each active layer
        for layer_name, events in layer_events.items():
            if not events:
                continue

            track = MidiTrack()
            track.name = f"Sonara Layer: {layer_name.capitalize()}"
            channel = cls.LAYER_CHANNELS[layer_name]

            # Set program change if not drum channel
            if channel != 9 and layer_name in cls.LAYER_PROGRAMS:
                track.append(
                    Message("program_change", channel=channel, program=cls.LAYER_PROGRAMS[layer_name], time=0)
                )

            # Build list of raw MIDI actions: (tick, priority, action_type, note, velocity)
            # action_type: 0 for note_off, 1 for note_on (note_off prioritized at same tick)
            raw_actions: List[Tuple[int, int, str, int, int]] = []

            for ev in events:
                t_on = sec_to_tick(ev.start_time)
                t_off = max(t_on + 1, sec_to_tick(ev.end_time))

                if channel == 9:  # Percussion channel
                    # Map to General MIDI drum notes
                    drum_note = 36  # Bass Drum default
                    if ev.pitch_midi is not None:
                        if ev.pitch_midi <= 45:
                            drum_note = 36  # Bass Drum
                        elif ev.pitch_midi <= 65:
                            drum_note = 38  # Acoustic Snare
                        else:
                            drum_note = 42  # Closed Hi-Hat
                    raw_actions.append((t_on, 1, "note_on", drum_note, ev.velocity))
                    raw_actions.append((t_off, 0, "note_off", drum_note, 0))
                else:
                    note_num = ev.pitch_midi if ev.pitch_midi is not None else 60
                    note_num = max(0, min(127, note_num))
                    raw_actions.append((t_on, 1, "note_on", note_num, ev.velocity))
                    raw_actions.append((t_off, 0, "note_off", note_num, 0))

            # Sort actions by tick ascending, note_off before note_on
            raw_actions.sort(key=lambda x: (x[0], x[1]))

            # Write actions with delta ticks
            current_tick = 0
            for tick, _, action, note, vel in raw_actions:
                delta = max(0, tick - current_tick)
                current_tick = tick
                if action == "note_on":
                    track.append(Message("note_on", note=note, velocity=vel, channel=channel, time=delta))
                else:
                    track.append(Message("note_off", note=note, velocity=vel, channel=channel, time=delta))

            track.append(MetaMessage("end_of_track", time=0))
            mid.tracks.append(track)

        mid.save(str(path))
        return path

    @classmethod
    def midi_to_smr(cls, midi_path: str | Path) -> SMRDocument:
        """
        Parse a Standard MIDI file into an SMR document.

        Args:
            midi_path: Path to MIDI file.

        Returns:
            SMRDocument instance.
        """
        path = Path(midi_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"MIDI file not found: {path}")

        mid = MidiFile(str(path))
        ticks_per_beat = mid.ticks_per_beat

        # Determine initial tempo
        current_tempo = 500000  # Default 120 bpm (500,000 microseconds)
        bpm = 120.0
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    current_tempo = msg.tempo
                    bpm = round(mido.tempo2bpm(current_tempo), 2)
                    break

        seconds_per_tick = (current_tempo / 1e6) / ticks_per_beat

        events: List[MusicalEventModel] = []
        event_counter = 0

        # Note tracking per channel: (note) -> (start_time, velocity)
        for track_idx, track in enumerate(mid.tracks):
            current_tick = 0
            active_notes: Dict[Tuple[int, int], Tuple[float, int]] = {}

            for msg in track:
                current_tick += msg.time
                current_time = current_tick * seconds_per_tick

                if msg.type == "set_tempo":
                    current_tempo = msg.tempo
                    seconds_per_tick = (current_tempo / 1e6) / ticks_per_beat
                    bpm = round(mido.tempo2bpm(current_tempo), 2)

                elif msg.type == "note_on" and msg.velocity > 0:
                    key = (msg.channel, msg.note)
                    active_notes[key] = (current_time, msg.velocity)

                elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
                    key = (msg.channel, msg.note)
                    if key in active_notes:
                        start_time, vel = active_notes.pop(key)
                        dur = max(0.05, current_time - start_time)

                        # Determine layer from MIDI channel
                        if msg.channel == 9:
                            layer = LayerClassifier.LAYER_PERCUSSION
                            pitch = None
                            freq = None
                        elif msg.note <= 55:
                            layer = LayerClassifier.LAYER_BASS
                            pitch = msg.note
                            freq = round(440.0 * (2 ** ((msg.note - 69) / 12)), 2)
                        elif msg.channel == 0:
                            layer = LayerClassifier.LAYER_MELODIC
                            pitch = msg.note
                            freq = round(440.0 * (2 ** ((msg.note - 69) / 12)), 2)
                        else:
                            layer = LayerClassifier.LAYER_HARMONIC
                            pitch = msg.note
                            freq = round(440.0 * (2 ** ((msg.note - 69) / 12)), 2)

                        event_id = f"evt_midi_{event_counter:04d}"
                        event_counter += 1

                        events.append(
                            MusicalEventModel(
                                event_id=event_id,
                                start_time=round(start_time, 3),
                                duration=round(dur, 3),
                                end_time=round(start_time + dur, 3),
                                layer=layer,
                                event_type="hit" if layer == LayerClassifier.LAYER_PERCUSSION else "note",
                                pitch_midi=pitch,
                                pitch_name=f"MIDI_{pitch}" if pitch is not None else None,
                                frequency_hz=freq,
                                velocity=vel,
                                confidence=1.0,
                                uncertainty_flags=["parsed_from_midi"],
                            )
                        )

        # Sort events by start time
        events.sort(key=lambda e: e.start_time)
        total_duration = max([e.end_time for e in events], default=1.0)

        meta = AudioMetadataModel(
            file_path=str(path),
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            sha256_hash="midi_import",
            duration_seconds=round(total_duration, 3),
            sample_rate=44100,
            channels=2,
            format="MIDI",
        )

        rhythm = RhythmProfileModel(
            estimated_bpm=bpm,
            tempo_confidence=1.0,
            time_signature_estimate="4/4",
            beat_count=int(total_duration * (bpm / 60.0)),
            onset_count=len(events),
            onset_density_per_second=round(len(events) / total_duration, 2) if total_duration > 0 else 0.0,
            syncopation_index=0.1,
            pulse_clarity=0.95,
            is_rhythmic=True,
        )

        acoustics = AcousticFeaturesModel(
            rms_energy_mean=0.5,
            dynamic_range_db=40.0,
            crest_factor=2.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_bandwidth_mean=1200.0,
            spectral_rolloff_mean=3000.0,
            spectral_flatness_mean=0.1,
            zero_crossing_rate_mean=0.05,
            warmth_index=0.4,
            brightness_index=0.3,
            uncertainty_flags=["synthetic_midi_import"],
        )

        harmony = HarmonicProfileModel(
            primary_key="C",
            mode="major",
            key_confidence=0.8,
            tonal_stability=0.85,
        )

        provenance = AnalysisProvenanceModel(
            system_name="Sonara",
            system_version="0.1.0",
            timestamp_utc="2026-10-23T00:00:00Z",
            execution_time_seconds=0.01,
            limitations=["Imported directly from symbolic MIDI container"],
        )

        return SMRDocument(
            metadata=meta,
            acoustic_features=acoustics,
            rhythm=rhythm,
            harmony=harmony,
            events=events,
            provenance=provenance,
        )
