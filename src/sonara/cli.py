"""Command-line interface for the Sonara Musical Intelligence System."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from sonara.analysis.collection import CollectionAnalyzer
from sonara.analysis.similarity import SimilarityEngine
from sonara.analysis.style_profile import StyleProfile
from sonara.generation.generator import StyleConditionedGenerator
from sonara.generation.synthesizer import SMRSynthesizer
from sonara.pipeline import SonaraPipeline
from sonara.representation.midi_converter import MIDIConverter
from sonara.representation.report import SMRReportGenerator
from sonara.representation.schema import SMRDocument
from sonara.representation.serializer import SMRSerializer


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze an audio file into SMR representation."""
    input_path = Path(args.audio_path)
    if not input_path.exists():
        print(f"Error: Audio file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"[*] Ingesting and analyzing: {input_path}")
    pipeline = SonaraPipeline()
    try:
        smr_doc = pipeline.analyze(input_path)
    except Exception as e:
        print(f"[-] Pipeline analysis failed: {str(e)}", file=sys.stderr)
        return 1

    # Output JSON SMR
    out_json = args.output or input_path.with_suffix(".smr.json")
    saved_json = SMRSerializer.save_json(smr_doc, out_json)
    print(f"[+] SMR Representation written to: {saved_json}")

    # Output Markdown report if requested or default
    if args.report:
        report_path = pipeline.export_report(smr_doc, args.report)
        print(f"[+] Markdown Analysis Report written to: {report_path}")

    # Output MIDI if requested
    if args.midi:
        midi_path = pipeline.export_midi(smr_doc, args.midi)
        print(f"[+] Standard MIDI exported to: {midi_path}")

    # Render audio if requested
    if args.render:
        render_path = pipeline.render_audio(smr_doc, args.render)
        print(f"[+] Rendered WAV audio written to: {render_path}")

    print(
        f"[✓] Analysis complete: {smr_doc.rhythm.estimated_bpm:.1f} BPM | "
        f"{smr_doc.harmony.primary_key} {smr_doc.harmony.mode} | "
        f"{len(smr_doc.events)} events extracted."
    )
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Analyze a collection of tracks and generate a learned StyleProfile."""
    pipeline = SonaraPipeline()
    documents: List[SMRDocument] = []

    for path_str in args.files:
        p = Path(path_str)
        if not p.exists():
            print(f"[!] Warning: File not found: {p}", file=sys.stderr)
            continue

        if p.suffix.lower() == ".json":
            try:
                doc = SMRSerializer.load_json(p)
                documents.append(doc)
                print(f"[*] Loaded existing SMR: {p.name}")
            except Exception as e:
                print(f"[!] Warning: Failed to parse {p}: {str(e)}", file=sys.stderr)
        else:
            print(f"[*] Analyzing audio file: {p.name}")
            try:
                doc = pipeline.analyze(p)
                documents.append(doc)
            except Exception as e:
                print(f"[!] Warning: Failed to analyze {p}: {str(e)}", file=sys.stderr)

    if not documents:
        print("Error: No valid tracks could be analyzed.", file=sys.stderr)
        return 1

    profile = CollectionAnalyzer.analyze_collection(documents, profile_name=args.name or "Collection_Profile")
    out_path = Path(args.output or "style_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(profile.model_dump_json(indent=2))

    print(f"[+] Style Profile written to: {out_path.resolve()}")
    print(
        f"[✓] Profile Summary: {profile.tracks_analyzed} tracks | "
        f"Mean BPM: {profile.rhythm.mean_bpm} | Dominant Keys: {list(profile.harmony.key_distribution.keys())}"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare tracks and output cross-track similarity matrix."""
    pipeline = SonaraPipeline()
    documents: List[SMRDocument] = []

    for path_str in args.files:
        p = Path(path_str)
        if p.suffix.lower() == ".json":
            documents.append(SMRSerializer.load_json(p))
        else:
            documents.append(pipeline.analyze(p))

    if len(documents) < 2:
        print("Error: Provide at least 2 tracks to compare.", file=sys.stderr)
        return 1

    names, sim_matrix = SimilarityEngine.compute_distance_matrix(documents)
    print("\n--- Cross-Track Cosine Similarity Matrix ---")
    header = f"{'Track':<25} | " + " | ".join(f"{name[:10]:<10}" for name in names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        row = f"{name[:25]:<25} | " + " | ".join(f"{sim_matrix[i, j]:<10.3f}" for j in range(len(names)))
        print(row)
    print("-------------------------------------------\n")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate a novel musical composition conditioned on a StyleProfile."""
    profile: StyleProfile | None = None
    if args.profile:
        p_path = Path(args.profile)
        if not p_path.exists():
            print(f"Error: Style profile file not found: {p_path}", file=sys.stderr)
            return 1
        with open(p_path, "r", encoding="utf-8") as f:
            profile = StyleProfile.model_validate_json(f.read())
        print(f"[*] Loaded Style Profile: {profile.profile_name} ({profile.tracks_analyzed} source tracks)")

    generator = StyleConditionedGenerator(seed=args.seed)
    doc = generator.generate(
        style_profile=profile,
        duration_seconds=args.duration,
        target_bpm=args.bpm,
        target_key=args.key,
    )

    out_smr = Path(args.output_smr or "generated_composition.smr.json")
    SMRSerializer.save_json(doc, out_smr)
    print(f"[+] Generated SMR Document written to: {out_smr.resolve()}")

    if args.midi:
        midi_path = MIDIConverter.smr_to_midi(doc, args.midi)
        print(f"[+] Exported MIDI written to: {midi_path.resolve()}")

    if args.render:
        synth = SMRSynthesizer(sample_rate=44100)
        wav_path = synth.render(doc, args.render)
        print(f"[+] Rendered WAV audio written to: {wav_path.resolve()}")

    print(
        f"[✓] Generation complete: {doc.rhythm.estimated_bpm} BPM | "
        f"{doc.harmony.primary_key} {doc.harmony.mode} | "
        f"{len(doc.events)} events over {doc.metadata.duration_seconds:.1f}s."
    )
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    """Export the SMR formal JSON Schema."""
    schema = SMRSerializer.export_json_schema()
    out_path = Path(args.output or "smr_schema_v1.0.0.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"[+] SMR JSON Schema exported to: {out_path.resolve()}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Display Sonara engine information."""
    print("Sonara: Musical Representation and Intelligence System")
    print("Version: 0.1.0")
    print("Specification: SMR v1.0.0 (Sonara Musical Representation)")
    print("Supported audio inputs: WAV, MP3, FLAC, OGG, AIFF")
    print("Features: Beat/tempo tracking, onset detection, chroma key/chord extraction,")
    print("          recurrence segmentation, layer separation, multi-song style profiling,")
    print("          symbolic composition, multi-track MIDI export, procedural audio synthesis.")
    return 0


def main() -> int:
    """CLI entrypoint dispatcher."""
    parser = argparse.ArgumentParser(
        prog="sonara",
        description="Sonara: Musical Representation and Intelligence System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze an audio track into SMR")
    p_analyze.add_argument("audio_path", help="Path to input audio file")
    p_analyze.add_argument("--output", "-o", help="Path to save SMR JSON representation")
    p_analyze.add_argument("--report", "-r", help="Path to save Markdown analysis report")
    p_analyze.add_argument("--midi", "-m", help="Path to export Standard MIDI file")
    p_analyze.add_argument("--render", "-w", help="Path to render synthesized WAV audio")

    # profile
    p_profile = subparsers.add_parser("profile", help="Generate a style profile from a track collection")
    p_profile.add_argument("files", nargs="+", help="Audio or SMR JSON files to analyze")
    p_profile.add_argument("--name", "-n", default="Collection_Style_Profile", help="Profile name")
    p_profile.add_argument("--output", "-o", default="style_profile.json", help="Path to save StyleProfile JSON")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare tracks and compute similarity matrix")
    p_compare.add_argument("files", nargs="+", help="Audio or SMR files to compare")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate original music guided by style profile")
    p_gen.add_argument("--profile", "-p", help="Path to input StyleProfile JSON")
    p_gen.add_argument("--duration", "-d", type=float, default=32.0, help="Duration in seconds (default: 32)")
    p_gen.add_argument("--bpm", type=float, help="Target BPM override")
    p_gen.add_argument("--key", type=str, help="Target key override (e.g. C, G, Am)")
    p_gen.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    p_gen.add_argument("--output-smr", help="Path to save generated SMR JSON")
    p_gen.add_argument("--midi", "-m", help="Path to export generated MIDI file")
    p_gen.add_argument("--render", "-w", help="Path to render generated WAV audio")

    # schema
    p_schema = subparsers.add_parser("schema", help="Export SMR JSON Schema")
    p_schema.add_argument("--output", "-o", default="smr_schema_v1.0.0.json", help="Output JSON path")

    # info
    subparsers.add_parser("info", help="Display system and specification information")

    args = parser.parse_args()

    dispatch = {
        "analyze": cmd_analyze,
        "profile": cmd_profile,
        "compare": cmd_compare,
        "generate": cmd_generate,
        "schema": cmd_schema,
        "info": cmd_info,
    }

    handler = dispatch.get(args.command)
    if handler:
        return handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
