# Sonara: Musical Representation and Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![SMR Specification](https://img.shields.io/badge/SMR-v1.0.0-green.svg)](docs/smr_specification.md)

**Sonara** is a research-oriented musical intelligence system designed to decode acoustic audio into a structured, machine-readable representation—the **Sonara Musical Representation (SMR)**—that computers can analyze, understand, learn from across collections, and use to guide original music generation.

---

## 1. Vision & Core Philosophy

Traditional music computation treats music either as **raw acoustic waveforms** (opaque, continuous, non-symbolic) or as **MIDI note streams** (symbolic, but blind to production aesthetics, timbre, mixing, and spatial presence). Modern neural audio tokenizers (EnCodec, DAC) enable generative audio but produce uninterpretable token streams that cannot be easily inspected, edited, or reasoned about.

Sonara establishes a **hybrid representation paradigm**:
- **Explicit Symbolic Structure**: Transparent, verifiable musical elements—tempo (BPM), metric grids, key center, mode, chord sequences, and structural sections (`intro`, `verse`, `chorus`, `bridge`, `outro`).
- **Continuous Acoustic Descriptors**: Production and timbral fingerprints—spectral centroid (brightness), spectral rolloff, bandwidth, Wiener entropy flatness, dynamic range (dB), crest factor, and low-end warmth index.
- **Hierarchical Arrangement**: Inter-layer dynamics, capturing how the rhythm section (kick/snare) synchronizes with basslines and supports melodic leads.
- **Anti-Hallucination & Uncertainty Tracking**: Every extracted event, key, and tempo estimate includes confidence scores and explicit uncertainty flags when signal quality degrades.

---

## 2. Architecture & Modules

```
Sonara Master Pipeline
├── sonara.audio
│   ├── loader.py        # Format detection (WAV, MP3, FLAC), hash, validation, resampling
│   ├── features.py      # Acoustic, spectral, and timbral descriptor extraction
│   ├── rhythm.py        # Onset detection, beat tracking, tempo, pulse clarity, syncopation
│   ├── harmony.py       # HPSS, Chroma CENS/STFT, Krumhansl-Schmuckler key, chord progression
│   └── structure.py     # Self-Similarity Matrix (SSM), Foote novelty curve, section labeling
├── sonara.events
│   ├── classifier.py    # Functional layer classification (percussion, bass, harmonic, lead)
│   └── extractor.py     # Parabolic autocorrelation pitch extraction, note velocity & duration
├── sonara.representation
│   ├── schema.py        # SMR v1.0.0 Pydantic specification models
│   ├── serializer.py    # JSON serialization, deserialization, JSON Schema exporter
│   ├── validator.py     # Invariant and structural validation
│   ├── midi_converter.py# Bidirectional conversion between SMR and Multi-track Standard MIDI
│   └── report.py        # Human-readable Markdown report generator
├── sonara.analysis
│   ├── collection.py    # Multi-track collection aggregation
│   ├── style_profile.py # StyleProfile data model with explicit evidence attribution
│   └── similarity.py    # 23-dimensional feature embedding, cosine similarity, distance matrices
├── sonara.generation
│   ├── generator.py     # Style-conditioned symbolic composition engine
│   ├── synthesizer.py   # Procedural multi-timbral 44.1 kHz stereo audio renderer (WAV)
│   └── neural_interface.py # Conditioning adapter for deep generative models (e.g. MusicGen)
└── sonara.cli           # Unified CLI interface (analyze, profile, compare, generate, schema)
```

---

## 3. Quickstart

### Installation
```bash
git clone https://github.com/utkarshdubey2008/Sonara.git
cd Sonara
pip install -e .
```

### 1. Analyze an Audio Recording
```bash
sonara analyze song.wav \
  --output song.smr.json \
  --report song_report.md \
  --midi song.mid \
  --render song_resynth.wav
```

### 2. Learn an Artist Style Profile across a Track Collection
```bash
sonara profile track1.wav track2.wav track3.wav \
  --name "Artist_Collection" \
  --output artist_profile.json
```

### 3. Compare Multi-Track Similarity
```bash
sonara compare track1.wav track2.wav track3.wav
```

### 4. Generate an Original Track from a Learned Profile
```bash
sonara generate \
  --profile artist_profile.json \
  --duration 32.0 \
  --output-smr new_track.smr.json \
  --midi new_track.mid \
  --render new_track.wav
```

---

## 4. Python API Example

```python
from sonara.pipeline import SonaraPipeline
from sonara.analysis.collection import CollectionAnalyzer
from sonara.generation.generator import StyleConditionedGenerator
from sonara.generation.synthesizer import SMRSynthesizer

# Analyze track
pipeline = SonaraPipeline()
doc = pipeline.analyze("audio.wav")
print(f"Tempo: {doc.rhythm.estimated_bpm} BPM | Key: {doc.harmony.primary_key} {doc.harmony.mode}")

# Export Markdown Inspection Report & MIDI
pipeline.export_report(doc, "report.md")
pipeline.export_midi(doc, "track.mid")

# Multi-track style aggregation
profile = CollectionAnalyzer.analyze_collection([doc], profile_name="Session_Profile")

# Conditioned composition generation
generator = StyleConditionedGenerator(seed=42)
new_doc = generator.generate(style_profile=profile, duration_seconds=16.0)

# Procedural audio rendering
synth = SMRSynthesizer(sample_rate=44100)
synth.render(new_doc, "rendered_output.wav")
```

---

## 5. Documentation & Specifications

- [Architecture & Technical Foundation](docs/architecture.md)
- [Sonara Musical Representation (SMR v1.0.0) Specification](docs/smr_specification.md)
- [Research Foundation & Prior Art Comparative Analysis](docs/research_foundation.md)
- [Getting Started & User Guide](docs/getting_started.md)

---

## 6. Milestone Roadmap & Implementation Status

- [x] **Milestone 1: End-to-End Prototype**
  - Audio ingestion, format detection, validation, checksums.
  - DSP feature extraction (RMS, crest factor, spectral centroid, rolloff, flatness, warmth/brightness).
  - Beat tracking, tempo estimation, onset detection, syncopation index.
  - Harmonic analysis (HPSS, Chroma CENS/STFT, Krumhansl key detection, chord tracking).
  - Recurrence analysis (SSM, Foote novelty filtering, structural sectioning).
  - Discrete musical event extraction with layer classification.
  - SMR v1.0.0 JSON schema serialization, validation, and Markdown report generation.
  - Multi-track MIDI export and import.
- [x] **Milestone 2: Multi-Song Analysis & Style Profiling**
  - Collection aggregation across multiple SMR documents.
  - Learned `StyleProfile` model with explicit evidence attribution.
  - First-order Markov chord transition matrices.
  - Multidimensional feature vectors and cross-track similarity matrix.
- [x] **Milestone 3: Guided Generation & Audio Synthesis**
  - Style-conditioned symbolic composition generator.
  - Multi-timbral procedural audio synthesizer rendering 44.1 kHz stereo audio.
  - Neural conditioning adapter for text-to-audio foundation models.
- [ ] **Milestone 4: Advanced Neural Perception (Future Work)**
  - Optional deep stem separation integration (Demucs v4).
  - Polyphonic transformer transcription (MT3 / Basic Pitch fine-tuned).
  - Direct EnCodec/DAC discrete audio token roundtrip.

---

## 7. License

Distributed under the Apache-2.0 License. See `LICENSE` for details.
