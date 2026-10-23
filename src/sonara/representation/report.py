"""Human-readable report generation for Sonara Musical Representation."""

from __future__ import annotations

from typing import Dict
from sonara.representation.schema import SMRDocument


class SMRReportGenerator:
    """Produces structured markdown reports from SMR documents."""

    @classmethod
    def generate_markdown(cls, document: SMRDocument) -> str:
        """
        Create a detailed Markdown inspection report for an SMR document.

        Args:
            document: SMRDocument instance.

        Returns:
            Markdown formatted report string.
        """
        meta = document.metadata
        rhythm = document.rhythm
        harmony = document.harmony
        acoustics = document.acoustic_features
        prov = document.provenance

        # Layer event counts
        layer_counts: Dict[str, int] = {}
        for ev in document.events:
            layer_counts[ev.layer] = layer_counts.get(ev.layer, 0) + 1

        md: list[str] = [
            f"# Sonara Musical Analysis Report: {meta.file_name}",
            "",
            "## 1. Overview & Provenance",
            f"- **System**: {prov.system_name} v{prov.system_version} (SMR Specification v{document.smr_version})",
            f"- **Analysis Timestamp**: {prov.timestamp_utc}",
            f"- **Execution Time**: {prov.execution_time_seconds:.3f} seconds",
            f"- **Source File**: `{meta.file_path}`",
            f"- **File Size**: {meta.file_size_bytes:,} bytes",
            f"- **SHA-256**: `{meta.sha256_hash}`",
            f"- **Audio Format**: {meta.format} | {meta.channels} ch | {meta.sample_rate:,} Hz",
            f"- **Duration**: {meta.duration_seconds:.2f} seconds",
            "",
            "## 2. Musical Foundation",
            f"- **Estimated Tempo**: **{rhythm.estimated_bpm:.1f} BPM** (Confidence: {rhythm.tempo_confidence:.0%})",
            f"- **Meter / Time Signature**: {rhythm.time_signature_estimate}",
            f"- **Rhythmic Pulse Clarity**: {rhythm.pulse_clarity:.2f} / 1.00",
            f"- **Syncopation Index**: {rhythm.syncopation_index:.1%}",
            f"- **Primary Key / Mode**: **{harmony.primary_key} {harmony.mode}** (Confidence: {harmony.key_confidence:.0%})",
            f"- **Tonal Stability**: {harmony.tonal_stability:.2f} / 1.00",
            f"- **Chord Vocabulary**: {', '.join(harmony.chord_vocabulary) if harmony.chord_vocabulary else 'None detected'}",
            "",
            "## 3. Acoustic & Timbral Fingerprint",
            f"- **Dynamic Range**: {acoustics.dynamic_range_db:.1f} dB",
            f"- **Peak Amplitude**: {meta.peak_amplitude:.3f} | **RMS Energy**: {acoustics.rms_energy_mean:.4f}",
            f"- **Crest Factor**: {acoustics.crest_factor:.2f}",
            f"- **Spectral Centroid (Brightness)**: {acoustics.spectral_centroid_mean:.1f} Hz (± {acoustics.spectral_centroid_std:.1f} Hz)",
            f"- **Spectral Bandwidth**: {acoustics.spectral_bandwidth_mean:.1f} Hz",
            f"- **Spectral Rolloff (85%)**: {acoustics.spectral_rolloff_mean:.1f} Hz",
            f"- **Spectral Flatness (Noisiness)**: {acoustics.spectral_flatness_mean:.4f}",
            f"- **Warmth Index (<500Hz)**: {acoustics.warmth_index:.1%}",
            f"- **Brightness Index (>3000Hz)**: {acoustics.brightness_index:.1%}",
            "",
            "## 4. Structural Arrangement",
            f"- **Total Detected Sections**: {len(document.structure)}",
            "",
            "| Section ID | Type | Start (s) | End (s) | Duration (s) | Repetition | Confidence |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for s in document.structure:
            rep_str = f"Yes (repeats {s.repeat_of_id})" if s.is_repetition else "No"
            md.append(
                f"| `{s.section_id}` | **{s.section_type.upper()}** | {s.start_time:.2f} | {s.end_time:.2f} | {s.duration:.2f} | {rep_str} | {s.confidence:.0%} |"
            )

        md.extend([
            "",
            "## 5. Event & Layer Breakdown",
            f"- **Total Extracted Events**: {len(document.events)}",
            f"- **Onset Density**: {rhythm.onset_density_per_second:.1f} events/sec",
            "",
            "| Functional Layer | Event Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])

        total_evts = max(1, len(document.events))
        for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
            md.append(f"| **{layer.capitalize()}** | {count} | {count / total_evts:.1%} |")

        if document.events:
            md.extend([
                "",
                "### Event Timeline (Sample)",
                "| Event ID | Time (s) | Duration (s) | Layer | Type | Note | Freq (Hz) | Vel | Conf |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for ev in document.events[:15]:
                note_str = ev.pitch_name or "-"
                freq_str = f"{ev.frequency_hz:.1f}" if ev.frequency_hz is not None else "-"
                md.append(
                    f"| `{ev.event_id}` | {ev.start_time:.2f} | {ev.duration:.2f} | {ev.layer} | {ev.event_type} | {note_str} | {freq_str} | {ev.velocity} | {ev.confidence:.0%} |"
                )
            if len(document.events) > 15:
                md.append(f"*(... {len(document.events) - 15} more events omitted for brevity)*")

        md.extend([
            "",
            "## 6. Uncertainty Flags & Explicit Limitations",
        ])

        all_flags = (
            meta_flags := ["silent_audio"] if meta.is_silent else []
        ) + (
            ["clipping_detected"] if meta.is_clipped else []
        ) + acoustics.uncertainty_flags + rhythm.uncertainty_flags + harmony.uncertainty_flags + prov.warnings

        if all_flags:
            md.append("The following estimation uncertainties were recorded during DSP analysis:")
            for flag in sorted(set(all_flags)):
                md.append(f"- `[{flag}]`")
        else:
            md.append("No critical uncertainty flags were triggered. Analysis completed within nominal bounds.")

        if prov.limitations:
            md.append("")
            md.append("### Stated Limitations:")
            for lim in prov.limitations:
                md.append(f"- {lim}")

        md.append("")
        return "\n".join(md)
