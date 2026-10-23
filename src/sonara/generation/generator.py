"""Style-conditioned symbolic musical composition generator."""

from __future__ import annotations

import random
from typing import Dict, List, Optional
import numpy as np

from sonara.analysis.style_profile import StyleProfile
from sonara.events.classifier import LayerClassifier
from sonara.representation.schema import (
    AcousticFeaturesModel,
    AnalysisProvenanceModel,
    AudioMetadataModel,
    ChordItemModel,
    HarmonicProfileModel,
    MusicalEventModel,
    RhythmProfileModel,
    SMRDocument,
    StructuralSectionModel,
)


KEY_PITCHES: Dict[str, int] = {
    "C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64, "F": 65,
    "F#": 66, "G": 67, "G#": 68, "A": 69, "A#": 70, "B": 71,
}

# Major scale intervals: root, +2, +4, +5, +7, +9, +11
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
# Natural minor scale intervals: root, +2, +3, +5, +7, +8, +10
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


class StyleConditionedGenerator:
    """Generates novel SMRDocument representations guided by learned StyleProfiles."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(
        self,
        style_profile: Optional[StyleProfile] = None,
        duration_seconds: float = 32.0,
        target_bpm: Optional[float] = None,
        target_key: Optional[str] = None,
    ) -> SMRDocument:
        """
        Synthesize a complete SMR composition adhering to stylistic constraints.

        Args:
            style_profile: Optional StyleProfile to draw artistic grammar from.
            duration_seconds: Target length in seconds.
            target_bpm: Override tempo if desired.
            target_key: Override key if desired.

        Returns:
            Fully formed SMRDocument.
        """
        # 1. Determine Global Tempo & Meter
        if target_bpm:
            bpm = float(target_bpm)
        elif style_profile and style_profile.rhythm.mean_bpm > 0:
            std = max(1.0, min(10.0, style_profile.rhythm.bpm_std))
            bpm = round(float(np.random.normal(style_profile.rhythm.mean_bpm, std * 0.5)), 1)
        else:
            bpm = 120.0
        bpm = max(60.0, min(180.0, bpm))

        # Meter and beat timing
        beat_dur = 60.0 / bpm
        bar_dur = beat_dur * 4  # 4/4 meter
        total_bars = max(4, int(duration_seconds / bar_dur))
        actual_duration = total_bars * bar_dur

        # 2. Determine Key and Mode
        if target_key:
            key_root = target_key.replace("m", "").replace(":maj", "").replace(":min", "")
            mode = "minor" if "m" in target_key.lower() else "major"
        elif style_profile and style_profile.harmony.key_distribution:
            keys = list(style_profile.harmony.key_distribution.keys())
            probs = list(style_profile.harmony.key_distribution.values())
            total_p = sum(probs)
            probs = [p / total_p for p in probs]
            key_root = np.random.choice(keys, p=probs)
            mode = "major" if random.random() > 0.3 else "minor"
        else:
            key_root = "C"
            mode = "major"

        scale_intervals = MAJOR_SCALE if mode == "major" else MINOR_SCALE
        root_midi = KEY_PITCHES.get(key_root, 60)

        # 3. Structural Arrangement
        sections: List[StructuralSectionModel] = []
        sec_bar_counts = [2, 4, 4, 4, 2]  # Intro, Verse 1, Chorus 1, Verse 2, Outro
        sec_names = ["intro", "verse", "chorus", "verse", "outro"]

        current_bar = 0
        for i, (name, b_count) in enumerate(zip(sec_names, sec_bar_counts)):
            if current_bar >= total_bars:
                break
            b_actual = min(b_count, total_bars - current_bar)
            t_start = current_bar * bar_dur
            t_end = (current_bar + b_actual) * bar_dur
            sec_id = f"sec_{i}"
            is_rep = bool(name == "verse" and i > 1)
            rep_id = "sec_1" if is_rep else None

            sections.append(
                StructuralSectionModel(
                    section_id=sec_id,
                    section_type=name,
                    start_time=round(t_start, 3),
                    end_time=round(t_end, 3),
                    duration=round(t_end - t_start, 3),
                    confidence=0.95,
                    is_repetition=is_rep,
                    repeat_of_id=rep_id,
                )
            )
            current_bar += b_actual

        # 4. Harmonic Progression (Chord sequence)
        chord_items: List[ChordItemModel] = []
        # Standard diatonic Roman numeral progressions
        # I - V - vi - IV for major, i - VI - III - VII for minor
        if mode == "major":
            prog_roots = [0, 7, 9, 5]  # I, V, vi, IV
            prog_quals = ["maj", "maj", "min", "maj"]
        else:
            prog_roots = [0, 8, 3, 10]  # i, VI, III, VII
            prog_quals = ["min", "maj", "maj", "maj"]

        for bar_idx in range(total_bars):
            chord_idx = bar_idx % len(prog_roots)
            c_root_int = (root_midi + prog_roots[chord_idx]) % 12
            c_root_name = list(KEY_PITCHES.keys())[list(KEY_PITCHES.values()).index(60 + c_root_int)]
            c_qual = prog_quals[chord_idx]
            symbol = f"{c_root_name}:{c_qual}"

            t_start = bar_idx * bar_dur
            t_end = (bar_idx + 1) * bar_dur

            chord_items.append(
                ChordItemModel(
                    chord_symbol=symbol,
                    root=c_root_name,
                    quality=c_qual,
                    start_time=round(t_start, 3),
                    end_time=round(t_end, 3),
                    confidence=0.90,
                )
            )

        # 5. Event Generation: Drums, Bass, Chords, Lead
        events: List[MusicalEventModel] = []
        event_counter = 0

        sync_tendency = style_profile.rhythm.mean_syncopation if style_profile else 0.25

        for bar_idx in range(total_bars):
            bar_start = bar_idx * bar_dur
            active_chord = chord_items[bar_idx]
            c_root_midi = KEY_PITCHES.get(active_chord.root, 60)
            c_offset = 4 if active_chord.quality == "maj" else 3

            # --- A. Drums / Percussion ---
            for beat in range(4):
                beat_time = bar_start + beat * beat_dur

                # Kick on beat 0 and 2
                if beat in (0, 2):
                    events.append(
                        MusicalEventModel(
                            event_id=f"evt_gen_{event_counter:04d}",
                            start_time=round(beat_time, 3),
                            duration=round(beat_dur * 0.4, 3),
                            end_time=round(beat_time + beat_dur * 0.4, 3),
                            layer=LayerClassifier.LAYER_PERCUSSION,
                            event_type="hit",
                            pitch_midi=36,  # Kick
                            pitch_name="C1",
                            frequency_hz=65.4,
                            velocity=105,
                            confidence=0.95,
                        )
                    )
                    event_counter += 1

                # Snare on beat 1 and 3
                if beat in (1, 3):
                    events.append(
                        MusicalEventModel(
                            event_id=f"evt_gen_{event_counter:04d}",
                            start_time=round(beat_time, 3),
                            duration=round(beat_dur * 0.35, 3),
                            end_time=round(beat_time + beat_dur * 0.35, 3),
                            layer=LayerClassifier.LAYER_PERCUSSION,
                            event_type="hit",
                            pitch_midi=38,  # Snare
                            pitch_name="D1",
                            frequency_hz=73.4,
                            velocity=95,
                            confidence=0.95,
                        )
                    )
                    event_counter += 1

                # Hi-hat on 8th notes
                for sub in [0.0, 0.5]:
                    hh_time = beat_time + sub * beat_dur
                    events.append(
                        MusicalEventModel(
                            event_id=f"evt_gen_{event_counter:04d}",
                            start_time=round(hh_time, 3),
                            duration=round(beat_dur * 0.2, 3),
                            end_time=round(hh_time + beat_dur * 0.2, 3),
                            layer=LayerClassifier.LAYER_PERCUSSION,
                            event_type="hit",
                            pitch_midi=42,  # Closed Hat
                            pitch_name="F#1",
                            frequency_hz=92.5,
                            velocity=75 if sub == 0.0 else 55,
                            confidence=0.95,
                        )
                    )
                    event_counter += 1

            # --- B. Bassline ---
            # Bass plays root and 5th on syncopated rhythmic pattern
            bass_root_midi = c_root_midi - 24  # 2 octaves down
            bass_steps = [0.0, 1.5, 2.0, 3.0] if sync_tendency > 0.2 else [0.0, 1.0, 2.0, 3.0]
            for step in bass_steps:
                b_time = bar_start + step * beat_dur
                b_note = bass_root_midi if step != 3.0 else (bass_root_midi + 7)  # 5th
                events.append(
                    MusicalEventModel(
                        event_id=f"evt_gen_{event_counter:04d}",
                        start_time=round(b_time, 3),
                        duration=round(beat_dur * 0.45, 3),
                        end_time=round(b_time + beat_dur * 0.45, 3),
                        layer=LayerClassifier.LAYER_BASS,
                        event_type="note",
                        pitch_midi=b_note,
                        pitch_name=list(KEY_PITCHES.keys())[b_note % 12] + str(b_note // 12 - 1),
                        frequency_hz=round(440.0 * (2 ** ((b_note - 69) / 12)), 2),
                        velocity=100,
                        confidence=0.92,
                    )
                )
                event_counter += 1

            # --- C. Harmonic Chords / Pad ---
            # Sustained pad playing root, 3rd, 5th
            for interval in [0, c_offset, 7]:
                chord_note = c_root_midi + interval
                events.append(
                    MusicalEventModel(
                        event_id=f"evt_gen_{event_counter:04d}",
                        start_time=round(bar_start, 3),
                        duration=round(bar_dur * 0.95, 3),
                        end_time=round(bar_start + bar_dur * 0.95, 3),
                        layer=LayerClassifier.LAYER_HARMONIC,
                        event_type="chord",
                        pitch_midi=chord_note,
                        pitch_name=list(KEY_PITCHES.keys())[chord_note % 12] + str(chord_note // 12 - 1),
                        frequency_hz=round(440.0 * (2 ** ((chord_note - 69) / 12)), 2),
                        velocity=80,
                        confidence=0.90,
                    )
                )
                event_counter += 1

            # --- D. Melodic Lead ---
            # Lead phrase playing melodic motifs in high register (C5 - C6)
            lead_octave = root_midi + 12
            for beat in range(4):
                if random.random() < 0.75:
                    scale_deg = random.choice([0, 2, 4, 7, 9]) if mode == "major" else random.choice([0, 3, 5, 7, 10])
                    m_note = lead_octave + scale_deg
                    l_time = bar_start + beat * beat_dur
                    events.append(
                        MusicalEventModel(
                            event_id=f"evt_gen_{event_counter:04d}",
                            start_time=round(l_time, 3),
                            duration=round(beat_dur * 0.7, 3),
                            end_time=round(l_time + beat_dur * 0.7, 3),
                            layer=LayerClassifier.LAYER_MELODIC,
                            event_type="note",
                            pitch_midi=m_note,
                            pitch_name=list(KEY_PITCHES.keys())[m_note % 12] + str(m_note // 12 - 1),
                            frequency_hz=round(440.0 * (2 ** ((m_note - 69) / 12)), 2),
                            velocity=90,
                            confidence=0.88,
                        )
                    )
                    event_counter += 1

        # Sort all generated events by time
        events.sort(key=lambda e: e.start_time)

        # Assemble Document
        meta = AudioMetadataModel(
            file_path="synthetic://sonara/generated_composition",
            file_name=f"Sonara_Generated_{key_root}_{mode}_{int(bpm)}bpm",
            file_size_bytes=0,
            sha256_hash="procedural_generation",
            duration_seconds=round(actual_duration, 3),
            sample_rate=44100,
            channels=2,
            format="SMR_SYNTHETIC",
        )

        rhythm = RhythmProfileModel(
            estimated_bpm=bpm,
            tempo_confidence=1.0,
            time_signature_estimate="4/4",
            beat_count=int(total_bars * 4),
            onset_count=len(events),
            onset_density_per_second=round(len(events) / actual_duration, 2),
            syncopation_index=round(sync_tendency, 3),
            pulse_clarity=0.92,
            is_rhythmic=True,
            beat_timestamps=[round(i * beat_dur, 3) for i in range(total_bars * 4)],
        )

        harmony = HarmonicProfileModel(
            primary_key=key_root,
            mode=mode,
            key_confidence=0.95,
            tonal_stability=0.90,
            chord_vocabulary=sorted(list(set(c.chord_symbol for c in chord_items))),
            chord_sequence=chord_items,
        )

        acoustics = AcousticFeaturesModel(
            rms_energy_mean=0.25,
            dynamic_range_db=36.0,
            crest_factor=2.8,
            spectral_centroid_mean=1850.0,
            spectral_centroid_std=240.0,
            spectral_bandwidth_mean=1500.0,
            spectral_rolloff_mean=3600.0,
            spectral_flatness_mean=0.08,
            zero_crossing_rate_mean=0.06,
            warmth_index=0.35,
            brightness_index=0.28,
        )

        provenance = AnalysisProvenanceModel(
            system_name="Sonara",
            system_version="0.1.0",
            timestamp_utc="2026-10-23T00:00:00Z",
            execution_time_seconds=0.05,
            limitations=["Symbolically generated composition via StyleConditionedGenerator"],
        )

        return SMRDocument(
            metadata=meta,
            acoustic_features=acoustics,
            rhythm=rhythm,
            harmony=harmony,
            structure=sections,
            events=events,
            provenance=provenance,
        )
