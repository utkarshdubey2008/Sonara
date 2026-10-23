"""Demonstrate multi-track analysis, style profiling, and guided music generation."""

from pathlib import Path
from sonara.analysis.collection import CollectionAnalyzer
from sonara.analysis.similarity import SimilarityEngine
from sonara.generation.generator import StyleConditionedGenerator
from sonara.generation.synthesizer import SMRSynthesizer
from sonara.pipeline import SonaraPipeline
from sonara.representation.midi_converter import MIDIConverter
from sonara.representation.serializer import SMRSerializer


def main() -> None:
    demo_dir = Path("examples/demo_tracks")
    wav_files = sorted(list(demo_dir.glob("*.wav")))

    if not wav_files:
        print("[!] No WAV files found in examples/demo_tracks. Run generate_synthetic_demos.py first.")
        return

    print(f"[*] Ingesting and analyzing {len(wav_files)} tracks with SonaraPipeline...")
    pipeline = SonaraPipeline()
    documents = []

    for wav_path in wav_files:
        print(f"    - Analyzing {wav_path.name}...")
        doc = pipeline.analyze(wav_path)
        documents.append(doc)

    print("\n[*] Learning Style Profile from track collection...")
    profile = CollectionAnalyzer.analyze_collection(documents, profile_name="Demo_Artist_Profile")
    profile_path = demo_dir / "artist_style_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        f.write(profile.model_dump_json(indent=2))
    print(f"[+] Saved Style Profile to {profile_path.name}")
    print(f"    - Mean BPM: {profile.rhythm.mean_bpm} ± {profile.rhythm.bpm_std}")
    print(f"    - Dominant Key Distribution: {profile.harmony.key_distribution}")
    print(f"    - Mean Spectral Brightness: {profile.timbre.mean_spectral_centroid:.1f} Hz")

    print("\n[*] Computing Cross-Track Similarity Matrix...")
    names, sim_matrix = SimilarityEngine.compute_distance_matrix(documents)
    print(f"{'Track':<24} | " + " | ".join(f"{n[:8]:<8}" for n in names))
    print("-" * 55)
    for i, name in enumerate(names):
        row = f"{name[:24]:<24} | " + " | ".join(f"{sim_matrix[i, j]:<8.3f}" for j in range(len(names)))
        print(row)

    print("\n[*] Generating brand new composition conditioned on Demo_Artist_Profile...")
    generator = StyleConditionedGenerator(seed=888)
    new_doc = generator.generate(style_profile=profile, duration_seconds=16.0)

    new_smr_path = demo_dir / "guided_original_track.smr.json"
    SMRSerializer.save_json(new_doc, new_smr_path)

    new_midi_path = demo_dir / "guided_original_track.mid"
    MIDIConverter.smr_to_midi(new_doc, new_midi_path)

    new_wav_path = demo_dir / "guided_original_track.wav"
    synth = SMRSynthesizer(sample_rate=44100)
    synth.render(new_doc, new_wav_path)

    print(f"[+] Guided Original Track generated:")
    print(f"    - SMR:  {new_smr_path.name}")
    print(f"    - MIDI: {new_midi_path.name}")
    print(f"    - WAV:  {new_wav_path.name}")
    print(f"    - Key:  {new_doc.harmony.primary_key} {new_doc.harmony.mode}")
    print(f"    - BPM:  {new_doc.rhythm.estimated_bpm}")
    print(f"    - Events: {len(new_doc.events)}")

    print("\n[✓] Multi-song analysis and guided generation workflow verified successfully.")


if __name__ == "__main__":
    main()
