"""Comprehensive beat deconstruction, stem separation, reconstruction, and benchmarking runner for Flashing Lights."""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import requests
import soundfile as sf
import librosa

from sonara.pipeline import SonaraPipeline
from sonara.audio.stems import StemSeparator
from sonara.generation.synthesizer import SMRSynthesizer
from sonara.analysis.comparator import ReconstructionComparator
from sonara.representation.fl_studio import FLStudioExporter
from sonara.representation.midi_converter import MIDIConverter
from sonara.representation.serializer import SMRSerializer
from sonara.representation.report import SMRReportGenerator


def upload_to_tmpfiles(file_path: Path) -> str | None:
    """Upload a file to tmpfiles.org and return direct download URL."""
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": f},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                page_url = data.get("data", {}).get("url")
                if page_url:
                    # Convert page URL (tmpfiles.org/XXXX/file.ext) to direct download URL (tmpfiles.org/dl/XXXX/file.ext)
                    dl_url = page_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    return dl_url
    except Exception as e:
        print(f"Failed to upload {file_path.name} to tmpfiles.org: {e}")
    return None


def generate_interactive_timeline_html(document, output_path: Path) -> Path:
    """Generate an interactive HTML visual timeline with audio playback capabilities."""
    events = sorted(document.events, key=lambda x: x.start_time)
    duration = document.metadata.duration_seconds
    bpm = document.rhythm.estimated_bpm
    key = f"{document.harmony.primary_key} {document.harmony.mode}"

    # Extract percussion events for focused timeline
    drum_events = [e for e in events if e.layer == "percussion"]
    bass_events = [e for e in events if e.layer == "bass"]
    harmonic_events = [e for e in events if e.layer == "harmonic"]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sonara — Visual Beat Deconstruction Timeline | {document.metadata.file_name}</title>
    <style>
        body {{
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            margin: 0;
            padding: 24px;
        }}
        h1 {{ margin: 0 0 8px 0; font-size: 24px; color: #38bdf8; }}
        .meta-bar {{
            display: flex; gap: 20px; font-size: 14px; color: #94a3b8;
            background: #1e293b; padding: 12px 18px; border-radius: 8px; margin-bottom: 24px;
        }}
        .meta-item strong {{ color: #f8fafc; }}
        .track-container {{
            background: #111827; border: 1px solid #334155; border-radius: 8px;
            padding: 16px; margin-bottom: 16px; position: relative; overflow-x: auto;
        }}
        .track-label {{
            font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;
            margin-bottom: 8px; display: flex; justify-content: space-between;
        }}
        .track-lane {{
            height: 48px; background: #0f172a; position: relative; border-radius: 6px;
            border-left: 2px solid #38bdf8; min-width: 3200px;
        }}
        .event-block {{
            position: absolute; top: 6px; bottom: 6px; border-radius: 4px;
            font-size: 10px; display: flex; align-items: center; justify-content: center;
            font-weight: bold; color: #000; cursor: pointer; transition: transform 0.1s;
        }}
        .event-block:hover {{ transform: scaleY(1.15); z-index: 10; }}
        .kick {{ background: #ef4444; }}
        .snare {{ background: #3b82f6; }}
        .hat {{ background: #10b981; }}
        .bass {{ background: #f59e0b; color: #000; }}
        .strings {{ background: #8b5cf6; color: #fff; }}
        .grid-line {{
            position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(255,255,255,0.06);
        }}
        .bar-line {{
            position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(56,189,248,0.25);
        }}
        .legend {{ display: flex; gap: 16px; margin-top: 16px; font-size: 12px; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>Sonara — Beat Deconstruction Timeline</h1>
    <div class="meta-bar">
        <div class="meta-item">Source: <strong>{document.metadata.file_name}</strong></div>
        <div class="meta-item">Duration: <strong>{duration:.2f}s</strong></div>
        <div class="meta-item">Tempo: <strong>{bpm:.2f} BPM</strong></div>
        <div class="meta-item">Key: <strong>{key}</strong></div>
        <div class="meta-item">Total Events: <strong>{len(events)}</strong></div>
        <div class="meta-item">Kicks: <strong>{len([e for e in drum_events if e.pitch_midi == 36])}</strong></div>
        <div class="meta-item">Snares: <strong>{len([e for e in drum_events if e.pitch_midi == 38])}</strong></div>
        <div class="meta-item">Hi-Hats: <strong>{len([e for e in drum_events if e.pitch_midi == 42])}</strong></div>
    </div>

    <div class="track-container">
        <div class="track-label">
            <span style="color: #ef4444;">Kick Drum (General MIDI 36)</span>
            <span>{len([e for e in drum_events if e.pitch_midi == 36])} Hits Detected</span>
        </div>
        <div class="track-lane">
            <!-- Kick Events -->
            {''.join([f'<div class="event-block kick" style="left: {round((e.start_time / duration) * 100, 3)}%; width: max(4px, {round((e.duration / duration) * 100, 3)}%);" title="t={e.start_time}s vel={e.velocity}">K</div>' for e in drum_events if e.pitch_midi == 36])}
        </div>
    </div>

    <div class="track-container">
        <div class="track-label">
            <span style="color: #3b82f6;">Snare / Rimshot (General MIDI 38)</span>
            <span>{len([e for e in drum_events if e.pitch_midi == 38])} Hits Detected</span>
        </div>
        <div class="track-lane">
            <!-- Snare Events -->
            {''.join([f'<div class="event-block snare" style="left: {round((e.start_time / duration) * 100, 3)}%; width: max(4px, {round((e.duration / duration) * 100, 3)}%);" title="t={e.start_time}s vel={e.velocity}">S</div>' for e in drum_events if e.pitch_midi == 38])}
        </div>
    </div>

    <div class="track-container">
        <div class="track-label">
            <span style="color: #10b981;">Closed Hi-Hat (General MIDI 42)</span>
            <span>{len([e for e in drum_events if e.pitch_midi == 42])} Hits Detected</span>
        </div>
        <div class="track-lane">
            <!-- Hi-Hat Events -->
            {''.join([f'<div class="event-block hat" style="left: {round((e.start_time / duration) * 100, 3)}%; width: max(3px, {round((e.duration / duration) * 100, 3)}%);" title="t={e.start_time}s vel={e.velocity}">H</div>' for e in drum_events if e.pitch_midi == 42])}
        </div>
    </div>

    <div class="track-container">
        <div class="track-label">
            <span style="color: #f59e0b;">Analog Synth Bassline (General MIDI 33)</span>
            <span>{len(bass_events)} Notes Detected</span>
        </div>
        <div class="track-lane">
            <!-- Bass Events -->
            {''.join([f'<div class="event-block bass" style="left: {round((e.start_time / duration) * 100, 3)}%; width: max(4px, {round((e.duration / duration) * 100, 3)}%);" title="t={e.start_time}s pitch={e.pitch_name} vel={e.velocity}">{e.pitch_name or "B"}</div>' for e in bass_events[:300]])}
        </div>
    </div>

    <div class="track-container">
        <div class="track-label">
            <span style="color: #8b5cf6;">Synth Strings Ostinato & Harmonic Chords</span>
            <span>{len(harmonic_events)} Harmonic Nodes</span>
        </div>
        <div class="track-lane">
            <!-- Harmonic Events -->
            {''.join([f'<div class="event-block strings" style="left: {round((e.start_time / duration) * 100, 3)}%; width: max(4px, {round((e.duration / duration) * 100, 3)}%);" title="t={e.start_time}s pitch={e.pitch_name}">{e.pitch_name or "C"}</div>' for e in harmonic_events[:300]])}
        </div>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-color kick"></div> Kick Drum (GM 36)</div>
        <div class="legend-item"><div class="legend-color snare"></div> Snare Drum (GM 38)</div>
        <div class="legend-item"><div class="legend-color hat"></div> Closed Hi-Hat (GM 42)</div>
        <div class="legend-item"><div class="legend-color bass"></div> Synth Bassline</div>
        <div class="legend-item"><div class="legend-color strings"></div> Ostinato Strings</div>
    </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path


def main():
    print("=== SONARA BEAT DECONSTRUCTION TEST ===")
    target_audio = Path("/tmp/flashing_lights_target.wav")
    if not target_audio.exists():
        print(f"Error: Target audio file not found: {target_audio}")
        return

    out_dir = Path("/root/Sonara/output/flashing_lights")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Pipeline Analysis
    print("\n[Step 1/6] Running SonaraPipeline Analysis...")
    t0 = time.time()
    pipeline = SonaraPipeline()
    document = pipeline.analyze(target_audio)
    t1 = time.time()
    print(f"Pipeline finished in {t1 - t0:.2f}s")
    print(f"Tempo: {document.rhythm.estimated_bpm:.2f} BPM, Key: {document.harmony.primary_key} {document.harmony.mode}")
    print(f"Total Decoded Events: {len(document.events)}")

    # Save SMR JSON
    smr_json_path = out_dir / "flashing_lights_smr.json"
    SMRSerializer.save_json(document, smr_json_path)
    print(f"SMR JSON exported to: {smr_json_path}")

    # Save SMR Markdown Report
    smr_report_path = out_dir / "flashing_lights_smr_report.md"
    pipeline.export_report(document, smr_report_path)
    print(f"SMR Report exported to: {smr_report_path}")

    # 2. Run Stem Separation
    print("\n[Step 2/6] Running Stem Separation (HPSS + Phase-Aligned Filter Banks)...")
    stems_dir = out_dir / "stems"
    separator = StemSeparator()
    stems_dict = separator.separate(target_audio, stems_dir)
    print(f"Stems saved to: {stems_dir}")
    for name, p in stems_dict.items():
        print(f"  {name}: {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Export Multi-track MIDIs & FL Studio Package
    print("\n[Step 3/6] Packaging FL Studio Stems & Multi-track MIDIs...")
    zip_path = out_dir / "flashing_lights_fl_studio_package.zip"
    FLStudioExporter.create_fl_studio_package(document, stems_dict, zip_path)
    print(f"FL Studio Package created: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

    # Export Full Project Multi-track MIDI
    full_midi_path = out_dir / "midi" / "full_project.mid"
    full_midi_path.parent.mkdir(parents=True, exist_ok=True)
    MIDIConverter.smr_to_midi(document, full_midi_path)
    print(f"Full Multi-track MIDI saved to: {full_midi_path}")

    # 4. Synthesize Reconstructed Audio
    print("\n[Step 4/6] Synthesizing Audio Reconstructions...")
    # A. Symbolic synthesis from SMR events
    symbolic_wav = out_dir / "reconstruction_symbolic.wav"
    synth = SMRSynthesizer(sample_rate=44100)
    synth.render(document, symbolic_wav)
    print(f"Symbolic Reconstruction saved: {symbolic_wav} ({symbolic_wav.stat().st_size / (1024*1024):.2f} MB)")

    # B. Pure rhythm engine (Synthesized drums + analog sub-bass only)
    rhythm_doc = document.model_copy(deep=True)
    rhythm_doc.events = [e for e in document.events if e.layer in ["percussion", "bass"]]
    rhythm_wav = out_dir / "rhythm_engine.wav"
    synth.render(rhythm_doc, rhythm_wav)
    y_rhythm, sr_synth = sf.read(str(rhythm_wav))

    # C. Hybrid reconstruction (Clean rhythm engine + authentic sample stems with kick sidechain)
    hybrid_wav = out_dir / "reconstruction_hybrid.wav"
    y_melody, sr_mel = sf.read(str(stems_dict["melody_samples"]))
    y_vocals, sr_voc = sf.read(str(stems_dict["vocals"]))

    # Resample melody and vocals to 44.1kHz if needed
    if sr_mel != sr_synth:
        y_melody = librosa.resample(y_melody, orig_sr=sr_mel, target_sr=sr_synth)
    if sr_voc != sr_synth:
        y_vocals = librosa.resample(y_vocals, orig_sr=sr_voc, target_sr=sr_synth)

    # Compute kick sidechain ducking envelope on strings stem
    kick_times = [e.start_time for e in document.events if e.layer == "percussion" and e.pitch_midi == 36]
    duck_len = int(0.065 * sr_synth)  # 65ms duck
    duck_env = np.ones(len(y_melody), dtype=np.float32)
    duck_shape = 1.0 - 0.30 * np.exp(-np.linspace(0, 3.5, duck_len))

    for kt in kick_times:
        idx = int(kt * sr_synth)
        if idx + duck_len < len(duck_env):
            duck_env[idx : idx + duck_len] = np.minimum(duck_env[idx : idx + duck_len], duck_shape)

    min_len = min(y_rhythm.shape[0], len(y_melody), len(y_vocals))
    ducked_melody = y_melody[:min_len] * duck_env[:min_len]

    # Combine punchy rhythm engine + ducked authentic strings + vocal hook
    y_hybrid = np.zeros((min_len, 2), dtype=np.float32)
    y_hybrid[:, 0] = y_rhythm[:min_len, 0] * 0.85 + ducked_melody * 0.70 + y_vocals[:min_len] * 0.45
    y_hybrid[:, 1] = y_rhythm[:min_len, 1] * 0.85 + ducked_melody * 0.70 + y_vocals[:min_len] * 0.45
    peak = np.max(np.abs(y_hybrid))
    if peak > 0:
        y_hybrid = (y_hybrid / peak) * 0.94
    sf.write(str(hybrid_wav), y_hybrid, sr_synth, subtype="PCM_16")
    print(f"Hybrid Reconstruction saved: {hybrid_wav} ({hybrid_wav.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Objective Comparison Benchmarking
    print("\n[Step 5/6] Evaluating Reconstruction Fidelity with ReconstructionComparator...")
    comparator = ReconstructionComparator()
    comp_symbolic = comparator.compare(target_audio, symbolic_wav)
    comp_hybrid = comparator.compare(target_audio, hybrid_wav)
    print(f"\n--- SYMBOLIC RECONSTRUCTION BENCHMARK ---")
    print(f"Composite Fidelity Score: {comp_symbolic.composite_fidelity_score:.1f}%")
    print(f"Harmonic Cosine Similarity: {comp_symbolic.harmonic_alignment:.3f}")
    print(f"Envelope Correlation: {comp_symbolic.envelope_correlation:.3f}")
    print(f"Tempo Congruence Error: {comp_symbolic.tempo_error_bpm:.2f} BPM")

    print(f"\n--- HYBRID RECONSTRUCTION BENCHMARK ---")
    print(f"Composite Fidelity Score: {comp_hybrid.composite_fidelity_score:.1f}%")
    print(f"Harmonic Cosine Similarity: {comp_hybrid.harmonic_alignment:.3f}")
    print(f"Envelope Correlation: {comp_hybrid.envelope_correlation:.3f}")
    print(f"Tempo Congruence Error: {comp_hybrid.tempo_error_bpm:.2f} BPM")

    # 6. Interactive Visual Timeline
    print("\n[Step 6/6] Generating Visual Event Timeline HTML...")
    timeline_html = out_dir / "timeline.html"
    generate_interactive_timeline_html(document, timeline_html)
    print(f"Timeline HTML generated: {timeline_html}")

    import zipfile
    timeline_zip = out_dir / "timeline_bundle.zip"
    with zipfile.ZipFile(timeline_zip, "w") as zf:
        zf.write(timeline_html, "timeline.html")

    # 7. Upload Key Deliverables to tmpfiles.org
    print("\n[Upload] Uploading deliverables to tmpfiles.org...")
    files_to_upload = [
        ("SMR JSON Document", smr_json_path),
        ("SMR Analysis Report", smr_report_path),
        ("FL Studio Project Package ZIP", zip_path),
        ("Full Multi-track MIDI", full_midi_path),
        ("Symbolic Reconstructed Audio", symbolic_wav),
        ("Hybrid Reconstructed Audio", hybrid_wav),
        ("Visual Timeline HTML Bundle", timeline_zip),
    ]

    upload_results = {}
    for label, fpath in files_to_upload:
        if fpath.exists():
            print(f"Uploading {label} ({fpath.name})...")
            url = upload_to_tmpfiles(fpath)
            upload_results[label] = url or "Failed to upload"
            print(f"  -> {upload_results[label]}")

    results_summary = {
        "track_name": document.metadata.file_name,
        "duration_seconds": document.metadata.duration_seconds,
        "tempo_bpm": document.rhythm.estimated_bpm,
        "key": f"{document.harmony.primary_key} {document.harmony.mode}",
        "total_events": len(document.events),
        "kick_count": len([e for e in document.events if e.layer == "percussion" and e.pitch_midi == 36]),
        "snare_count": len([e for e in document.events if e.layer == "percussion" and e.pitch_midi == 38]),
        "hihat_count": len([e for e in document.events if e.layer == "percussion" and e.pitch_midi == 42]),
        "bass_count": len([e for e in document.events if e.layer == "bass"]),
        "harmonic_count": len([e for e in document.events if e.layer == "harmonic"]),
        "fidelity_symbolic_percent": comp_symbolic.composite_fidelity_score,
        "fidelity_hybrid_percent": comp_hybrid.composite_fidelity_score,
        "uploaded_links": upload_results,
    }

    summary_json_path = out_dir / "flashing_lights_test_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nBenchmark completed. Summary written to: {summary_json_path}")


if __name__ == "__main__":
    main()
