"""Musical event layer and sound-source classification."""

from __future__ import annotations

from typing import List, Optional


class LayerClassifier:
    """Classifies musical events into functional musical layers."""

    LAYER_PERCUSSION = "percussion"
    LAYER_BASS = "bass"
    LAYER_HARMONIC = "harmonic"
    LAYER_MELODIC = "melodic"
    LAYER_UNKNOWN = "unknown"

    @classmethod
    def classify_event(
        cls,
        pitch_midi: Optional[int],
        frequency_hz: Optional[float],
        spectral_centroid: float,
        zero_crossing_rate: float,
        is_harmonic: bool,
    ) -> str:
        """
        Determine musical layer for a detected event.

        Args:
            pitch_midi: Estimated MIDI note number (0-127) or None.
            frequency_hz: Fundamental frequency in Hz or None.
            spectral_centroid: Centroid frequency in Hz.
            zero_crossing_rate: ZCR value.
            is_harmonic: Whether the event has clear harmonic peaks.

        Returns:
            Layer string: 'percussion', 'bass', 'harmonic', 'melodic', or 'unknown'.
        """
        # High zero crossing rate or very high centroid with no clear pitch -> Percussion / noise
        if not is_harmonic or pitch_midi is None or frequency_hz is None:
            if zero_crossing_rate > 0.15 or spectral_centroid > 3500.0 or spectral_centroid < 150.0:
                return cls.LAYER_PERCUSSION
            return cls.LAYER_UNKNOWN

        # Low frequency register (< 260 Hz, MIDI < 60 / C4) -> Bass
        if pitch_midi <= 55 or frequency_hz < 220.0:
            return cls.LAYER_BASS

        # Mid-to-high register with harmonic content
        if pitch_midi > 55:
            # If centroid is tightly clustered around f0 -> Melodic lead
            # If wide bandwidth or multiple chord components -> Harmonic
            if spectral_centroid < 2500.0:
                return cls.LAYER_HARMONIC
            else:
                return cls.LAYER_MELODIC

        return cls.LAYER_UNKNOWN
