# Getting Started with Sonara

This guide provides practical instructions for installing Sonara, analyzing audio recordings, generating style profiles across track collections, and synthesizing original music.

---

## 1. Installation

### Requirements
- Python 3.10 or higher
- `ffmpeg` or `libsndfile` installed on the host system

### Setup via pip
```bash
# Clone the repository
git clone https://github.com/utkarshdubey2008/Sonara.git
cd Sonara

# Install dependencies and editable package
pip install -e .

# (Optional) Install development dependencies for running tests
pip install -e ".[dev]"
```

Verify your installation:
```bash
sonara info
```

---

## 2. Quickstart: Analyzing an Audio File

Analyze any local audio file (WAV, MP3, FLAC, OGG):

```bash
sonara analyze path/to/song.wav \
  --output song.smr.json \
  --report song_report.md \
  --midi song.mid \
  --render song_synthesized.wav
```

### Outputs Generated:
1. `song.smr.json`: Full machine-readable Sonara Musical Representation document.
2. `song_report.md`: Human-readable Markdown summary with tempo, key, acoustic traits, and event timeline.
3. `song.mid`: Multi-track Standard MIDI file (drums, bass, pads, lead).
4. `song_synthesized.wav`: Direct procedural audio mixdown rendered from SMR events.

---

## 3. Multi-Track Analysis & Style Profiling

To analyze a collection of songs and learn an artist's style profile:

```bash
# Analyze a collection of audio files or SMR JSON files
sonara profile track1.wav track2.wav track3.wav \
  --name "Artist_Collection" \
  --output artist_profile.json
```

### Comparing Tracks
Compare acoustic and harmonic similarity across tracks:
```bash
sonara compare track1.wav track2.wav track3.wav
```
This produces an $N \times N$ normalized cosine similarity matrix identifying tracks with matching tempo, key, and timbral aesthetics.

---

## 4. Generating Original Music from a Style Profile

Synthesize a brand new composition guided by the learned style profile:

```bash
sonara generate \
  --profile artist_profile.json \
  --duration 32.0 \
  --output-smr new_track.smr.json \
  --midi new_track.mid \
  --render new_track.wav
```

---

## 5. Python API Usage

You can embed Sonara directly into your Python research pipelines:

```python
from sonara.pipeline import SonaraPipeline
from sonara.analysis.collection import CollectionAnalyzer
from sonara.generation.generator import StyleConditionedGenerator
from sonara.generation.synthesizer import SMRSynthesizer

# 1. Analyze Audio
pipeline = SonaraPipeline()
doc = pipeline.analyze("song.wav")
print(f"BPM: {doc.rhythm.estimated_bpm}, Key: {doc.harmony.primary_key} {doc.harmony.mode}")

# 2. Export Multi-track MIDI
pipeline.export_midi(doc, "song.mid")

# 3. Learn Style Profile
profile = CollectionAnalyzer.analyze_collection([doc], profile_name="Single_Track_Profile")

# 4. Generate New Composition
generator = StyleConditionedGenerator(seed=42)
new_doc = generator.generate(style_profile=profile, duration_seconds=16.0)

# 5. Render WAV
synth = SMRSynthesizer(sample_rate=44100)
synth.render(new_doc, "new_song.wav")
```

---

## 6. Running Tests

Execute the automated test suite:
```bash
pytest
```
Run with duration benchmarks:
```bash
pytest --durations=10
```
