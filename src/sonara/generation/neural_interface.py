"""Conditioning and adapter interface for neural generative foundation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sonara.analysis.style_profile import StyleProfile
from sonara.representation.schema import SMRDocument


@dataclass
class NeuralConditioningPrompt:
    """Conditioning parameters formatted for downstream neural audio models."""
    text_prompt: str
    bpm: float
    key: str
    mode: str
    target_duration_seconds: float
    conditioning_tags: List[str]
    structural_markers: List[Dict[str, Any]]


class NeuralPromptConditioner:
    """Translates SMR representations and StyleProfiles into neural generative model prompts."""

    @classmethod
    def from_style_profile(
        cls, profile: StyleProfile, duration_seconds: float = 30.0
    ) -> NeuralConditioningPrompt:
        """
        Derive neural conditioning parameters from a learned StyleProfile.

        Args:
            profile: Learned StyleProfile instance.
            duration_seconds: Target generation duration.

        Returns:
            NeuralConditioningPrompt tailored for models like MusicGen or Stable Audio.
        """
        # Determine dominant key
        top_key = "C"
        if profile.harmony.key_distribution:
            top_key = max(profile.harmony.key_distribution, key=profile.harmony.key_distribution.get)

        top_mode = "major"
        if profile.harmony.mode_distribution:
            top_mode = max(profile.harmony.mode_distribution, key=profile.harmony.mode_distribution.get)

        bpm = profile.rhythm.mean_bpm

        # Timbral adjectives based on measured metrics
        tags: List[str] = []
        if profile.timbre.mean_warmth_index > 0.35:
            tags.append("warm low-end")
        if profile.timbre.mean_spectral_centroid > 2200.0:
            tags.append("bright highs")
        elif profile.timbre.mean_spectral_centroid < 1400.0:
            tags.append("mellow vintage tone")

        if profile.rhythm.mean_syncopation > 0.30:
            tags.append("syncopated groove")
        else:
            tags.append("driving steady beat")

        if profile.timbre.mean_dynamic_range_db > 30.0:
            tags.append("dynamic uncompressed acoustic production")
        else:
            tags.append("tight modern punchy master")

        tags.append(f"{top_key} {top_mode}")
        tags.append(f"{bpm:.0f} BPM")

        # Synthesize natural language prompt for text-to-audio models (MusicGen / AudioCraft)
        prompt_str = (
            f"A high quality musical production in {top_key} {top_mode} at {bpm:.0f} BPM. "
            f"Featuring a {' '.join(tags[:-2])}, with clear melodic layers, rich harmony, "
            f"and an expressive rhythmic section in {profile.rhythm.dominant_meter} meter."
        )

        return NeuralConditioningPrompt(
            text_prompt=prompt_str,
            bpm=bpm,
            key=top_key,
            mode=top_mode,
            target_duration_seconds=duration_seconds,
            conditioning_tags=tags,
            structural_markers=[],
        )

    @classmethod
    def from_smr_document(cls, document: SMRDocument) -> NeuralConditioningPrompt:
        """
        Derive neural conditioning parameters from an existing SMR document.

        Args:
            document: SMRDocument instance.

        Returns:
            NeuralConditioningPrompt.
        """
        r = document.rhythm
        h = document.harmony
        a = document.acoustic_features

        tags = [
            f"{h.primary_key} {h.mode}",
            f"{r.estimated_bpm:.0f} BPM",
            f"Meter {r.time_signature_estimate}",
        ]

        if a.warmth_index > 0.35:
            tags.append("warm analog tone")
        if a.brightness_index > 0.25:
            tags.append("crisp modern presence")

        markers = [
            {"id": s.section_id, "type": s.section_type, "start": s.start_time, "end": s.end_time}
            for s in document.structure
        ]

        prompt_str = (
            f"Original musical track in {h.primary_key} {h.mode} at {r.estimated_bpm:.0f} BPM. "
            f"Arrangement includes {len(document.structure)} distinct sections with "
            f"{', '.join(h.chord_vocabulary[:4])} chord progression."
        )

        return NeuralConditioningPrompt(
            text_prompt=prompt_str,
            bpm=r.estimated_bpm,
            key=h.primary_key,
            mode=h.mode,
            target_duration_seconds=document.metadata.duration_seconds,
            conditioning_tags=tags,
            structural_markers=markers,
        )
