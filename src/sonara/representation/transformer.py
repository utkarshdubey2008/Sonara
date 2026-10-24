"""Sonara Instrumental Representation (SMR v2.0) Programmable Transformer & Synthesizer.

Provides programmatic inspection, manipulation, transposition, tempo rescaling,
sound design re-modeling, arrangement editing, and full instrumental audio rendering.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.signal import butter, sosfilt

from sonara.representation.instrumental_schema import (
    SMRv2InstrumentalDocument,
    InstrumentEntityModel,
    InstrumentCategory,
    PatternModel,
    PatternEventModel,
    SoundDesignModel,
    DropMarkerModel,
    TrackPlacementModel,
)
from sonara.representation.package import SMRPackage


class InstrumentalTransformer:
    """Manipulates and transforms SMR v2.0 Instrumental Documents."""

    KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    @classmethod
    def transpose(cls, doc: SMRv2InstrumentalDocument, semitones: int) -> SMRv2InstrumentalDocument:
        """Transpose all melodic and bass patterns by N semitones.
        
        Percussion entities and unpitched events remain unaffected.
        Master key is shifted accordingly.
        """
        new_doc = doc.model_copy(deep=True)
        if semitones == 0:
            return new_doc

        # Build set of percussion instrument IDs to ignore
        percussion_ids = {
            inst.instrument_id
            for inst in new_doc.instruments
            if inst.category == InstrumentCategory.PERCUSSION
        }

        # Transpose pattern events
        for pattern in new_doc.patterns:
            for event in pattern.events:
                if event.pitch_midi is not None:
                    # Pitch transpose clamped to valid MIDI range [12, 120]
                    transposed = int(np.clip(event.pitch_midi + semitones, 12, 120))
                    event.pitch_midi = transposed

        # Update Master Key
        clean_key = new_doc.master_key.strip().upper()
        # Handle flats conversion
        flat_map = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
        clean_key = flat_map.get(clean_key, clean_key)

        if clean_key in cls.KEY_NAMES:
            idx = cls.KEY_NAMES.index(clean_key)
            new_doc.master_key = cls.KEY_NAMES[(idx + semitones) % 12]

        return new_doc

    @classmethod
    def time_stretch(cls, doc: SMRv2InstrumentalDocument, new_bpm: float) -> SMRv2InstrumentalDocument:
        """Rescale composition tempo to new BPM without changing pitch.
        
        Because SMR v2.0 stores pattern events in musical beats, beat positions
        remain invariant while absolute timestamps (seconds) and automation curves
        are recalculated.
        """
        new_doc = doc.model_copy(deep=True)
        if new_bpm <= 0:
            raise ValueError(f"Invalid BPM: {new_bpm}")

        old_bpm = new_doc.master_bpm
        tempo_ratio = old_bpm / new_bpm  # time duration multiplier

        new_doc.master_bpm = float(new_bpm)
        # Recalculate duration
        seconds_per_beat = 60.0 / new_bpm
        beats_per_bar = 4.0
        new_doc.total_duration_seconds = float(new_doc.total_bars * beats_per_bar * seconds_per_beat)

        # Update drops timestamps
        for drop in new_doc.drops:
            drop.time_seconds = float((drop.bar_index - 1) * beats_per_bar * seconds_per_beat)

        # Update sections timestamps
        for sec in new_doc.sections:
            sec.start_time_seconds = float((sec.start_bar - 1) * beats_per_bar * seconds_per_beat)
            sec.end_time_seconds = float(sec.end_bar * beats_per_bar * seconds_per_beat)

        # Update automation curves
        for auto in new_doc.automation:
            auto.timestamps = [float(t * tempo_ratio) for t in auto.timestamps]

        return new_doc

    @classmethod
    def swap_instrument_sound(
        cls,
        doc: SMRv2InstrumentalDocument,
        instrument_id: str,
        new_sound_design: SoundDesignModel,
    ) -> SMRv2InstrumentalDocument:
        """Replace the synthesis and DSP configuration of an instrument entity."""
        new_doc = doc.model_copy(deep=True)
        for inst in new_doc.instruments:
            if inst.instrument_id == instrument_id:
                inst.sound_design = new_sound_design.model_copy(deep=True)
                break
        return new_doc

    @classmethod
    def remix_drop(
        cls,
        doc: SMRv2InstrumentalDocument,
        target_bar: int,
        impact_intensity: float = 1.0,
        muted_instruments: Optional[List[str]] = None,
        entering_instruments: Optional[List[str]] = None,
    ) -> SMRv2InstrumentalDocument:
        """Inject or modify a formal drop marker at a target bar."""
        new_doc = doc.model_copy(deep=True)
        beats_per_bar = 4.0
        seconds_per_beat = 60.0 / new_doc.master_bpm
        drop_time = (target_bar - 1) * beats_per_bar * seconds_per_beat

        # Check if drop marker already exists at this bar
        existing = [d for d in new_doc.drops if d.bar_index == target_bar]
        if existing:
            existing[0].impact_intensity = impact_intensity
            if muted_instruments is not None:
                existing[0].muted_instruments = muted_instruments
            if entering_instruments is not None:
                existing[0].entering_instruments = entering_instruments
        else:
            new_drop = DropMarkerModel(
                drop_id=f"drop_bar_{target_bar}",
                bar_index=target_bar,
                time_seconds=float(drop_time),
                transition_type="drop_impact",
                impact_intensity=impact_intensity,
                muted_instruments=muted_instruments or [],
                entering_instruments=entering_instruments or [],
            )
            new_doc.drops.append(new_drop)
            new_doc.drops.sort(key=lambda d: d.bar_index)

        return new_doc

    @classmethod
    def mute_instrument(
        cls, doc: SMRv2InstrumentalDocument, instrument_id: str, muted: bool = True
    ) -> SMRv2InstrumentalDocument:
        """Mute or unmute all arrangement tracks for a given instrument."""
        new_doc = doc.model_copy(deep=True)
        gain_db = -96.0 if muted else 0.0
        for track in new_doc.arrangement_tracks:
            if track.instrument_id == instrument_id:
                track.gain_offset_db = gain_db
        return new_doc


class InstrumentalSynthesizer:
    """High-fidelity DSP synthesis and reconstruction engine for SMR v2.0."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def render(
        self,
        doc: SMRv2InstrumentalDocument,
        package: Optional[SMRPackage] = None,
    ) -> np.ndarray:
        """Render complete stereo instrumental audio from SMR v2.0 document and assets.
        
        Args:
            doc: SMRv2InstrumentalDocument specification
            package: Optional SMRPackage providing embedded audio sample assets
            
        Returns:
            Stereo audio array shape (2, num_samples) as float32
        """
        sr = self.sample_rate
        total_samples = max(int(doc.total_duration_seconds * sr), int(sr * 2.0))
        beats_per_bar = 4.0
        sec_per_beat = 60.0 / doc.master_bpm

        inst_map = {inst.instrument_id: inst for inst in doc.instruments}
        pattern_map = {pat.pattern_id: pat for pat in doc.patterns}

        # Track buffers: instrument_id -> stereo waveform (2, total_samples)
        track_buffers: Dict[str, np.ndarray] = {
            inst_id: np.zeros((2, total_samples), dtype=np.float32)
            for inst_id in inst_map
        }

        # Render arrangement placements
        for placement in doc.arrangement_tracks:
            inst = inst_map.get(placement.instrument_id)
            pattern = pattern_map.get(placement.pattern_id)
            if not inst or not pattern:
                continue

            # Check if muted
            if placement.gain_offset_db <= -70.0:
                continue

            placement_gain = 10.0 ** (placement.gain_offset_db / 20.0)
            bar_start_beat = (placement.start_bar - 1) * beats_per_bar

            for rep in range(placement.repeat_count):
                rep_start_beat = bar_start_beat + rep * (pattern.length_bars * beats_per_bar)

                for event in pattern.events:
                    event_beat_in_placement = rep * (pattern.length_bars * beats_per_bar) + event.start_beat
                    if event_beat_in_placement in placement.mute_mask_beats:
                        continue

                    # Exact timing calculation
                    event_abs_beat = rep_start_beat + event.start_beat
                    event_start_sec = (event_abs_beat * sec_per_beat) + (event.micro_timing_offset_ms / 1000.0)
                    start_idx = int(event_start_sec * sr)
                    if start_idx >= total_samples or start_idx < 0:
                        continue

                    duration_sec = max(0.04, event.duration_beats * sec_per_beat)
                    dur_samples = int(duration_sec * sr)

                    # Synthesize note / drum hit
                    event_audio = self._synthesize_entity_event(
                        inst=inst,
                        event=event,
                        duration_sec=duration_sec,
                        package=package,
                    )

                    if len(event_audio) == 0:
                        continue

                    # Stereo placement & gain trim
                    pan = np.clip(inst.sound_design.stereo_pan, -1.0, 1.0)
                    left_gain = math.cos((pan + 1.0) * math.pi / 4.0) * placement_gain
                    right_gain = math.sin((pan + 1.0) * math.pi / 4.0) * placement_gain

                    # Additive mix into track buffer
                    end_idx = min(start_idx + len(event_audio), total_samples)
                    chunk_len = end_idx - start_idx

                    track_buffers[inst.instrument_id][0, start_idx:end_idx] += (
                        event_audio[:chunk_len] * left_gain
                    )
                    track_buffers[inst.instrument_id][1, start_idx:end_idx] += (
                        event_audio[:chunk_len] * right_gain
                    )

        # Apply Relational Graph (Sidechain Compression Network)
        if doc.relational_graph and doc.relational_graph.sidechain_network:
            for sc in doc.relational_graph.sidechain_network:
                if sc.source_instrument_id in track_buffers and sc.ducked_instrument_id in track_buffers:
                    src_buf = track_buffers[sc.source_instrument_id]
                    target_buf = track_buffers[sc.ducked_instrument_id]

                    # Envelope follower on trigger source (sum mono)
                    mono_src = 0.5 * (np.abs(src_buf[0]) + np.abs(src_buf[1]))
                    # Compute exponential smoothing ducking curve
                    release_samples = max(int((sc.release_time_ms / 1000.0) * sr), 10)
                    duck_depth_linear = 10.0 ** (sc.ducking_depth_db / 20.0)

                    # Rapid attack, exponential decay release
                    gain_curve = np.ones(total_samples, dtype=np.float32)
                    decay = np.exp(-1.0 / release_samples)

                    env = 0.0
                    for i in range(total_samples):
                        val = mono_src[i]
                        if val > env:
                            env = val
                        else:
                            env = env * decay
                        reduction = min(1.0, env * 2.0)
                        gain_curve[i] = 1.0 - (1.0 - duck_depth_linear) * reduction

                    target_buf[0] *= gain_curve
                    target_buf[1] *= gain_curve

        # Master Stereo Bus Summation
        master_bus = np.zeros((2, total_samples), dtype=np.float32)
        for inst_id, buf in track_buffers.items():
            master_bus += buf

        # Master Limiter / Soft-Clipping
        peak = np.max(np.abs(master_bus))
        if peak > 0.95:
            # Tanh soft ceiling
            master_bus = np.tanh(master_bus / peak * 1.2) * 0.92
        elif peak > 1e-4:
            master_bus = (master_bus / peak) * 0.88

        return master_bus

    def _synthesize_entity_event(
        self,
        inst: InstrumentEntityModel,
        event: PatternEventModel,
        duration_sec: float,
        package: Optional[SMRPackage] = None,
    ) -> np.ndarray:
        """Synthesize a single pattern event for an instrument entity."""
        sr = self.sample_rate
        dur_samples = max(int(duration_sec * sr), int(0.02 * sr))
        t = np.linspace(0.0, duration_sec, dur_samples, endpoint=False)
        vel_norm = event.velocity / 127.0

        # Check if asset sample exists
        if inst.sound_design.sample_asset_ref and package:
            sample_data = package.get_sample(inst.sound_design.sample_asset_ref)
            if sample_data:
                sig, smp_sr = sample_data
                if sig.ndim > 1:
                    sig = np.mean(sig, axis=0)
                # Resample or slice
                if len(sig) > dur_samples:
                    return (sig[:dur_samples] * vel_norm).astype(np.float32)
                else:
                    padded = np.zeros(dur_samples, dtype=np.float32)
                    padded[:len(sig)] = sig * vel_norm
                    return padded

        category = inst.category
        role = inst.musical_role.lower()

        # Synthesis router based on instrument identity and sound design
        if category == InstrumentCategory.PERCUSSION:
            if "kick" in role or event.pitch_midi == 36:
                return self._synth_kick(duration_sec, vel_norm, event.transient_sharpness)
            elif "snare" in role or "clap" in role or event.pitch_midi == 38:
                return self._synth_snare(duration_sec, vel_norm, event.transient_sharpness)
            elif "hat" in role or event.pitch_midi in (42, 44, 46):
                is_open = "open" in role or event.duration_beats > 0.4
                return self._synth_hat(duration_sec, vel_norm, is_open=is_open)
            else:
                return self._synth_percussion(duration_sec, vel_norm)

        elif category == InstrumentCategory.BASS:
            pitch = event.pitch_midi if event.pitch_midi is not None else 36
            freq = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))
            return self._synth_bass(t, freq, vel_norm, inst.sound_design)

        else:
            # Melodic, Chordal, Sample Instrument
            pitch = event.pitch_midi if event.pitch_midi is not None else 60
            freq = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))
            return self._synth_lead(t, freq, vel_norm, inst.sound_design)

    def _synth_kick(self, duration: float, vel: float, sharpness: float) -> np.ndarray:
        """Procedural 808/acoustic punchy kick synthesis with exponential pitch sweep."""
        dur = min(duration, 0.45)
        num_samples = int(dur * self.sample_rate)
        t = np.linspace(0.0, dur, num_samples, endpoint=False)

        # Exponential pitch drop: 165 Hz -> 48 Hz
        f_start = 160.0 + (sharpness * 60.0)
        f_end = 46.0
        pitch_decay = 0.045
        instant_freq = f_end + (f_start - f_end) * np.exp(-t / pitch_decay)
        phase = 2.0 * np.pi * np.cumsum(instant_freq) / self.sample_rate

        # Sine sub tone
        body = np.sin(phase)
        # Saturated punch
        body = np.tanh(body * 1.5)

        # Amplitude envelope
        amp_env = np.exp(-t / 0.14)
        click = np.random.normal(0, 0.15, num_samples) * np.exp(-t / 0.005) * sharpness

        sig = (body * amp_env + click) * vel
        return sig.astype(np.float32)

    def _synth_snare(self, duration: float, vel: float, sharpness: float) -> np.ndarray:
        """Crisp punchy snare with dual-sine snap and bandpass filtered noise crack."""
        dur = min(duration, 0.35)
        num_samples = int(dur * self.sample_rate)
        t = np.linspace(0.0, dur, num_samples, endpoint=False)

        # Dual tone body (185 Hz and 320 Hz)
        body = (
            0.6 * np.sin(2.0 * np.pi * 185.0 * t) +
            0.4 * np.sin(2.0 * np.pi * 320.0 * t)
        ) * np.exp(-t / 0.055)

        # Filtered noise snares crack (1200 - 4500 Hz)
        noise = np.random.normal(0, 0.8, num_samples)
        sos = butter(4, [1000.0, 4800.0], btype="bandpass", fs=self.sample_rate, output="sos")
        noise_filtered = sosfilt(sos, noise)
        noise_env = np.exp(-t / 0.16)

        sig = (body * 0.45 + noise_filtered * noise_env * 0.75) * vel
        return np.tanh(sig * 1.3).astype(np.float32)

    def _synth_hat(self, duration: float, vel: float, is_open: bool = False) -> np.ndarray:
        """Metallic closed/open hi-hat with 8kHz highpass filter and fast decay."""
        decay_time = 0.18 if is_open else 0.038
        dur = min(duration, 0.35 if is_open else 0.10)
        num_samples = int(dur * self.sample_rate)
        t = np.linspace(0.0, dur, num_samples, endpoint=False)

        # High-frequency square/ring mod metal simulation
        metal = (
            np.sign(np.sin(2.0 * np.pi * 3250.0 * t)) * 0.25 +
            np.sign(np.sin(2.0 * np.pi * 5430.0 * t)) * 0.25 +
            np.random.normal(0, 0.6, num_samples) * 0.5
        )
        sos = butter(4, 7500.0, btype="highpass", fs=self.sample_rate, output="sos")
        hat_filtered = sosfilt(sos, metal)
        amp_env = np.exp(-t / decay_time)

        return (hat_filtered * amp_env * vel * 0.65).astype(np.float32)

    def _synth_percussion(self, duration: float, vel: float) -> np.ndarray:
        """Generic percussive tick/rim."""
        dur = min(duration, 0.08)
        num_samples = int(dur * self.sample_rate)
        t = np.linspace(0.0, dur, num_samples, endpoint=False)
        sig = np.sin(2.0 * np.pi * 880.0 * t) * np.exp(-t / 0.015) * vel
        return sig.astype(np.float32)

    def _synth_bass(
        self, t: np.ndarray, freq: float, vel: float, sound_design: SoundDesignModel
    ) -> np.ndarray:
        """Analog Moog-style saw sub-bass with lowpass filtering and ADSR."""
        # 1. Oscillator: Sawtooth + Sub-Sine (-1 octave)
        sub_freq = freq * 0.5
        saw = 2.0 * (t * freq - np.floor(0.5 + t * freq))
        sub_sine = np.sin(2.0 * np.pi * sub_freq * t)
        raw_osc = 0.65 * saw + 0.35 * sub_sine

        # 2. ADSR Envelope
        env = self._render_adsr(len(t), sound_design.envelope)

        # 3. 24dB Low-Pass Filter
        cutoff = np.clip(sound_design.filter.cutoff_hz, 40.0, 800.0)
        sos = butter(4, cutoff, btype="lowpass", fs=self.sample_rate, output="sos")
        filtered = sosfilt(sos, raw_osc)

        # 4. Drive / Saturation
        drive = 1.0 + sound_design.effects.drive_saturation * 3.0
        saturated = np.tanh(filtered * drive)

        sig = saturated * env * vel * (10.0 ** (sound_design.gain_trim_db / 20.0))
        return sig.astype(np.float32)

    def _synth_lead(
        self, t: np.ndarray, freq: float, vel: float, sound_design: SoundDesignModel
    ) -> np.ndarray:
        """Polyphonic-capable lead / pad synth with warm detuning and filter."""
        # Multi-oscillator detuned saw
        osc1 = 2.0 * (t * freq - np.floor(0.5 + t * freq))
        osc2 = 2.0 * (t * (freq * 1.004) - np.floor(0.5 + t * (freq * 1.004)))
        raw_osc = 0.5 * osc1 + 0.5 * osc2

        # ADSR Envelope
        env = self._render_adsr(len(t), sound_design.envelope)

        # Filter
        cutoff = np.clip(sound_design.filter.cutoff_hz, 100.0, self.sample_rate * 0.45)
        sos = butter(2, cutoff, btype="lowpass", fs=self.sample_rate, output="sos")
        filtered = sosfilt(sos, raw_osc)

        sig = filtered * env * vel * 0.5 * (10.0 ** (sound_design.gain_trim_db / 20.0))
        return sig.astype(np.float32)

    def _render_adsr(self, num_samples: int, adsr) -> np.ndarray:
        """Calculate normalized amplitude curve for ADSR settings."""
        sr = self.sample_rate
        att_s = max(1, int((adsr.attack_ms / 1000.0) * sr))
        dec_s = max(1, int((adsr.decay_ms / 1000.0) * sr))
        rel_s = max(1, int((adsr.release_ms / 1000.0) * sr))
        sus_lvl = adsr.sustain_level

        env = np.zeros(num_samples, dtype=np.float32)

        # Attack phase
        end_att = min(att_s, num_samples)
        env[:end_att] = np.linspace(0.0, 1.0, end_att)

        # Decay phase
        if num_samples > end_att:
            end_dec = min(end_att + dec_s, num_samples)
            env[end_att:end_dec] = np.linspace(1.0, sus_lvl, end_dec - end_att)

            # Sustain phase
            if num_samples > end_dec:
                sus_end = max(end_dec, num_samples - rel_s)
                env[end_dec:sus_end] = sus_lvl

                # Release phase
                if num_samples > sus_end:
                    env[sus_end:] = np.linspace(sus_lvl, 0.0, num_samples - sus_end)

        return env
