"""Producer DAW export package generator for FL Studio, Ableton, and Logic Pro."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, Optional

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from sonara.events.classifier import LayerClassifier
from sonara.representation.schema import SMRDocument


class FLStudioExporter:
    """Packages deconstructed musical assets into an FL Studio ready zip bundle."""

    TICKS_PER_BEAT = 480

    @classmethod
    def export_individual_midi_tracks(
        cls, document: SMRDocument, output_dir: str | Path
    ) -> Dict[str, Path]:
        """Export separate MIDI files for drums, bass, chords, and melody."""
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        bpm = max(20.0, min(300.0, document.rhythm.estimated_bpm or 120.0))
        tempo_us = mido.bpm2tempo(bpm)
        seconds_per_tick = (60.0 / bpm) / cls.TICKS_PER_BEAT

        def sec_to_tick(s: float) -> int:
            return int(round(s / seconds_per_tick))

        layer_files = {
            LayerClassifier.LAYER_PERCUSSION: "drums.mid",
            LayerClassifier.LAYER_BASS: "bass.mid",
            LayerClassifier.LAYER_HARMONIC: "chords.mid",
            LayerClassifier.LAYER_MELODIC: "melody.mid",
        }

        exported_paths: Dict[str, Path] = {}

        for layer_name, filename in layer_files.items():
            events = [e for e in document.events if e.layer == layer_name]
            if not events:
                continue

            mid = MidiFile(type=0, ticks_per_beat=cls.TICKS_PER_BEAT)
            track = MidiTrack()
            track.name = f"{document.metadata.file_name} - {layer_name.capitalize()}"
            track.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))

            channel = 9 if layer_name == LayerClassifier.LAYER_PERCUSSION else 0
            raw_actions = []

            for ev in events:
                t_on = sec_to_tick(ev.start_time)
                t_off = max(t_on + 1, sec_to_tick(ev.end_time))
                note = ev.pitch_midi if ev.pitch_midi is not None else (36 if channel == 9 else 60)
                raw_actions.append((t_on, 1, "note_on", note, ev.velocity))
                raw_actions.append((t_off, 0, "note_off", note, 0))

            raw_actions.sort(key=lambda x: (x[0], x[1]))

            cur_tick = 0
            for tick, _, action, note, vel in raw_actions:
                delta = max(0, tick - cur_tick)
                cur_tick = tick
                if action == "note_on":
                    track.append(Message("note_on", note=note, velocity=vel, channel=channel, time=delta))
                else:
                    track.append(Message("note_off", note=note, velocity=vel, channel=channel, time=delta))

            track.append(MetaMessage("end_of_track", time=0))
            mid.tracks.append(track)

            out_path = out_d / filename
            mid.save(str(out_path))
            exported_paths[layer_name] = out_path

        return exported_paths

    @classmethod
    def create_fl_studio_package(
        cls,
        document: SMRDocument,
        stems_dict: Dict[str, Path],
        output_zip_path: str | Path,
    ) -> Path:
        """
        Assemble stems, individual MIDI scores, project metadata, and FL Studio import guide into a .zip.

        Args:
            document: Decoded SMRDocument.
            stems_dict: Dictionary of generated stem WAV file paths.
            output_zip_path: Path for output zip file.

        Returns:
            Resolved Path of the written zip archive.
        """
        zip_p = Path(output_zip_path).resolve()
        zip_p.parent.mkdir(parents=True, exist_ok=True)

        temp_midi_dir = zip_p.parent / "temp_midi_export"
        temp_midi_dir.mkdir(parents=True, exist_ok=True)
        midi_dict = cls.export_individual_midi_tracks(document, temp_midi_dir)

        guide_content = f"""# FL Studio Producer Import Guide for: {document.metadata.file_name}

## 1. Project Setup
- **Master Tempo**: Set your project BPM to **{document.rhythm.estimated_bpm} BPM**.
- **Time Signature**: Set project meter to **{document.rhythm.time_signature_estimate}**.
- **Musical Key**: **{document.harmony.primary_key} {document.harmony.mode}**.

## 2. Importing Audio Stems into the Playlist
1. Open the `stems/` folder inside this package.
2. Drag `drums.wav`, `bass.wav`, `melody_samples.wav`, and `vocals.wav` directly into track 1-4 on the FL Studio Playlist.
3. Align all stems to Bar 1, Beat 1.

## 3. Importing MIDI to the Channel Rack
1. Open the `midi/` folder inside this package.
2. Drag `drums.mid` into an **FPC** drum kit or your favorite drum VST.
   - Note 36 (C1) = Kick
   - Note 38 (D1) = Snare / Clap
   - Note 42 (F#1) = Closed Hi-Hat
3. Drag `bass.mid` into your 808 / Sub-bass instrument (e.g. 3xOSC, BooBass, Serum, SubLab).
4. Drag `chords.mid` into your piano or synth pad plugin (e.g. FL Keys, FLEX, Vital).
5. Drag `melody.mid` into your lead synth or vocal chop sampler.

## 4. Chopping Samples (Kanye / MPC Style)
- For the `melody_samples.wav` stem, load it into **Slicex** or **Fruity Slicer**.
- Set slicing mode to "Beat" or "Transient" to re-trigger and rearrange the chopped sample slices on your MIDI keyboard or pad controller.
"""

        project_info = {
            "track_title": document.metadata.file_name,
            "tempo_bpm": document.rhythm.estimated_bpm,
            "key": f"{document.harmony.primary_key} {document.harmony.mode}",
            "time_signature": document.rhythm.time_signature_estimate,
            "duration_seconds": document.metadata.duration_seconds,
            "events_count": len(document.events),
            "sections": [
                {"type": s.section_type, "start": s.start_time, "end": s.end_time}
                for s in document.structure
            ],
            "chord_progression": [c.chord_symbol for c in document.harmony.chord_sequence[:16]],
        }

        with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add stems
            for stem_name, stem_path in stems_dict.items():
                if stem_path.exists():
                    zf.write(stem_path, arcname=f"stems/{stem_name}.wav")

            # Add MIDI files
            for layer_name, m_path in midi_dict.items():
                if m_path.exists():
                    zf.write(m_path, arcname=f"midi/{m_path.name}")

            # Add Guide and JSON
            zf.writestr("FL_STUDIO_IMPORT_GUIDE.md", guide_content)
            zf.writestr("project_info.json", json.dumps(project_info, indent=2))

        # Cleanup temp midi files
        for p in temp_midi_dir.glob("*.mid"):
            p.unlink()
        temp_midi_dir.rmdir()

        return zip_p
