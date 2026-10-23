"""Multi-track collection analysis and style profile learning."""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from sonara.analysis.style_profile import (
    EvidenceObservation,
    HarmonicGrammar,
    RhythmicTendencies,
    StructuralGrammar,
    StyleProfile,
    TimbralAesthetic,
)
from sonara.representation.schema import SMRDocument


class CollectionAnalyzer:
    """Aggregates multiple SMR documents into a coherent StyleProfile."""

    @classmethod
    def analyze_collection(
        cls, documents: List[SMRDocument], profile_name: str = "Learned_Collection_Profile"
    ) -> StyleProfile:
        """
        Analyze a collection of SMR documents and produce a learned StyleProfile.

        Args:
            documents: List of SMRDocument instances.
            profile_name: Name or tag for the generated profile.

        Returns:
            StyleProfile instance.
        """
        if not documents:
            raise ValueError("Cannot analyze empty collection of documents.")

        n_tracks = len(documents)
        track_names = [d.metadata.file_name for d in documents]
        total_duration = sum(d.metadata.duration_seconds for d in documents)

        evidence: List[EvidenceObservation] = []

        # 1. Rhythmic Tendencies
        bpms = [d.rhythm.estimated_bpm for d in documents if d.rhythm.estimated_bpm > 0]
        syncs = [d.rhythm.syncopation_index for d in documents]
        densities = [d.rhythm.onset_density_per_second for d in documents]
        clarities = [d.rhythm.pulse_clarity for d in documents]
        meters = [d.rhythm.time_signature_estimate for d in documents]

        for d in documents:
            evidence.append(
                EvidenceObservation(
                    metric_name="estimated_bpm",
                    track_name=d.metadata.file_name,
                    measured_value=d.rhythm.estimated_bpm,
                )
            )
            evidence.append(
                EvidenceObservation(
                    metric_name="syncopation_index",
                    track_name=d.metadata.file_name,
                    measured_value=d.rhythm.syncopation_index,
                )
            )

        mean_bpm = float(np.mean(bpms)) if bpms else 120.0
        std_bpm = float(np.std(bpms)) if len(bpms) > 1 else 0.0
        bpm_range = [float(min(bpms)), float(max(bpms))] if bpms else [120.0, 120.0]
        dom_meter = max(set(meters), key=meters.count) if meters else "4/4"

        rhythm_profile = RhythmicTendencies(
            mean_bpm=round(mean_bpm, 1),
            bpm_std=round(std_bpm, 2),
            bpm_range=[round(b, 1) for b in bpm_range],
            mean_syncopation=round(float(np.mean(syncs)), 3),
            mean_onset_density=round(float(np.mean(densities)), 2),
            mean_pulse_clarity=round(float(np.mean(clarities)), 3),
            dominant_meter=dom_meter,
        )

        # 2. Harmonic Grammar
        keys = [d.harmony.primary_key for d in documents]
        modes = [d.harmony.mode for d in documents]
        tonal_stabilities = [d.harmony.tonal_stability for d in documents]

        key_dist: Dict[str, float] = {}
        for k in keys:
            key_dist[k] = round(keys.count(k) / n_tracks, 3)

        mode_dist: Dict[str, float] = {}
        for m in modes:
            mode_dist[m] = round(modes.count(m) / n_tracks, 3)

        all_chords: set[str] = set()
        transition_counts: Dict[str, Dict[str, int]] = {}

        for d in documents:
            all_chords.update(d.harmony.chord_vocabulary)
            seq = [c.chord_symbol for c in d.harmony.chord_sequence]
            for i in range(len(seq) - 1):
                c_curr = seq[i]
                c_next = seq[i + 1]
                if c_curr not in transition_counts:
                    transition_counts[c_curr] = {}
                transition_counts[c_curr][c_next] = transition_counts[c_curr].get(c_next, 0) + 1

        # Normalize transition probabilities
        transition_matrix: Dict[str, Dict[str, float]] = {}
        for c_curr, next_dict in transition_counts.items():
            total_transitions = sum(next_dict.values())
            transition_matrix[c_curr] = {}
            for c_next, count in next_dict.items():
                transition_matrix[c_curr][c_next] = round(count / total_transitions, 3)

        harmony_profile = HarmonicGrammar(
            key_distribution=key_dist,
            mode_distribution=mode_dist,
            mean_tonal_stability=round(float(np.mean(tonal_stabilities)), 3),
            collective_chord_vocabulary=sorted(list(all_chords)),
            chord_transition_matrix=transition_matrix,
        )

        # 3. Structural Grammar
        section_counts = [len(d.structure) for d in documents]
        section_durs: List[float] = []
        rep_densities: List[float] = []
        sec_types: List[str] = []

        for d in documents:
            reps = sum(1 for s in d.structure if s.is_repetition)
            rep_densities.append(reps / len(d.structure) if d.structure else 0.0)
            for s in d.structure:
                section_durs.append(s.duration)
                sec_types.append(s.section_type)

        type_freqs: Dict[str, float] = {}
        total_secs = max(1, len(sec_types))
        for st in set(sec_types):
            type_freqs[st] = round(sec_types.count(st) / total_secs, 3)

        structure_profile = StructuralGrammar(
            average_section_count=round(float(np.mean(section_counts)), 1),
            average_section_duration=round(float(np.mean(section_durs)), 2) if section_durs else 16.0,
            repetition_density_mean=round(float(np.mean(rep_densities)), 3) if rep_densities else 0.0,
            section_type_frequencies=type_freqs,
        )

        # 4. Timbral Aesthetic
        centroids = [d.acoustic_features.spectral_centroid_mean for d in documents]
        warmths = [d.acoustic_features.warmth_index for d in documents]
        brightnesses = [d.acoustic_features.brightness_index for d in documents]
        dyn_ranges = [d.acoustic_features.dynamic_range_db for d in documents]
        crests = [d.acoustic_features.crest_factor for d in documents]

        for d in documents:
            evidence.append(
                EvidenceObservation(
                    metric_name="spectral_centroid_mean",
                    track_name=d.metadata.file_name,
                    measured_value=d.acoustic_features.spectral_centroid_mean,
                )
            )

        timbre_profile = TimbralAesthetic(
            mean_spectral_centroid=round(float(np.mean(centroids)), 1),
            mean_warmth_index=round(float(np.mean(warmths)), 3),
            mean_brightness_index=round(float(np.mean(brightnesses)), 3),
            mean_dynamic_range_db=round(float(np.mean(dyn_ranges)), 2),
            mean_crest_factor=round(float(np.mean(crests)), 2),
        )

        return StyleProfile(
            profile_name=profile_name,
            tracks_analyzed=n_tracks,
            track_names=track_names,
            total_audio_duration_seconds=round(total_duration, 2),
            rhythm=rhythm_profile,
            harmony=harmony_profile,
            structure=structure_profile,
            timbre=timbre_profile,
            evidence=evidence,
        )
