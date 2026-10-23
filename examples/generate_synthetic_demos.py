"""Generate synthetic multi-track audio demos for testing and profiling."""

from pathlib import Path
from sonara.generation.generator import StyleConditionedGenerator
from sonara.generation.synthesizer import SMRSynthesizer
from sonara.representation.midi_converter import MIDIConverter
from sonara.representation.serializer import SMRSerializer


def main() -> None:
    output_dir = Path("examples/demo_tracks")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = [
        {"name": "track_1_funk_groove", "bpm": 115.0, "key": "D", "duration": 16.0, "seed": 101},
        {"name": "track_2_pop_energy", "bpm": 124.0, "key": "C", "duration": 16.0, "seed": 202},
        {"name": "track_3_lofi_chill", "bpm": 86.0, "key": "F", "duration": 16.0, "seed": 303},
    ]

    synth = SMRSynthesizer(sample_rate=44100)

    for t in tracks:
        print(f"[*] Generating {t['name']}: {t['bpm']} BPM in {t['key']}...")
        gen = StyleConditionedGenerator(seed=t["seed"])
        doc = gen.generate(duration_seconds=t["duration"], target_bpm=t["bpm"], target_key=t["key"])

        # Save SMR JSON
        json_path = output_dir / f"{t['name']}.smr.json"
        SMRSerializer.save_json(doc, json_path)

        # Save MIDI
        midi_path = output_dir / f"{t['name']}.mid"
        MIDIConverter.smr_to_midi(doc, midi_path)

        # Render audio
        wav_path = output_dir / f"{t['name']}.wav"
        synth.render(doc, wav_path)

        print(f"    [+] Saved {json_path.name}, {midi_path.name}, {wav_path.name}")

    print("\n[✓] Synthetic demo track generation complete.")


if __name__ == "__main__":
    main()
