# Sonara Architecture & Technical Foundation

## 1. System Vision & Purpose

Sonara is a research-oriented musical intelligence system designed to decode acoustic musical audio into an expressive, structured, machine-readable representation—the **Sonara Musical Representation (SMR)**. 

While conventional audio formats (WAV, MP3) capture raw acoustic pressure waves, and symbolic formats (MIDI, MusicXML) capture performative note triggers, neither format adequately bridges the continuum between symbolic composition, acoustic timbral production, and macro-structural organization. Sonara decodes audio signals into an explicit multi-tiered representation that computers can inspect, analyze across collections, and use to guide original music generation.

```
                      +-----------------------------+
                      | Raw Audio (WAV, MP3, FLAC)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Audio Ingestion &      |
                      |    Signal Standardization   |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+-----------------+         +-----------------+         +-----------------+
| Acoustic & DSP  |         | Rhythm & Metric |         | Harmonic & Key  |
|  Feature Layer  |         | Tracking Layer  |         | Analysis Layer  |
+--------+--------+         +--------+--------+         +--------+--------+
         |                           |                           |
         +---------------------------+---------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Structural Recurrence &   |
                      |    Boundary Segmentation    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Discrete Event Extraction & |
                      | Functional Layer Separation |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Sonara Musical Format     |
                      |      (SMR v1.0.0 JSON)      |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+-----------------+         +-----------------+         +-----------------+
| Multi-Track     |         | Standard MIDI   |         | Procedural &    |
| Collection &    |         | Interop Export  |         | Neural Audio    |
| Style Profiling |         |  (.mid files)   |         | Synthesis (.wav)|
+-----------------+         +-----------------+         +-----------------+
```

---

## 2. Core Architectural Design Decisions

### 2.1 The Hybrid Representation Paradigm: Symbolic vs. Continuous

A fundamental architectural question in music representation is whether to represent music purely as **symbolic discrete tokens** (such as MIDI note numbers and quantized tick grids) or as **continuous latent embeddings** (such as EnCodec, DAC, or CLAP embeddings).

Sonara adopts an explicit **hybrid architecture**:
1. **Discrete Symbolic Layer**: Used for interpretable, compositional attributes:
   - Pitch names (e.g. `C4`, `F#2`), MIDI pitch integers, frequency in Hz, and cent deviations.
   - Discrete event onsets, durations, and dynamic velocities.
   - Metric beat grids, tempo (BPM), and time signature indicators.
   - Hierarchical structural sections (`intro`, `verse`, `chorus`, `bridge`, `outro`).
2. **Continuous Acoustic Descriptors Layer**: Captures production, mixing, and timbral nuance that symbolic notation cannot convey:
   - Spectral centroid (perceived spectral brightness).
   - Spectral rolloff and bandwidth (harmonic distribution and high-frequency extent).
   - Spectral flatness (Wiener entropy, measuring tonality vs. noisiness).
   - Dynamic range and crest factor (quantifying compression, transient punch, and master dynamics).
   - Warmth and brightness energy ratios.
   - MFCC timbral vectors (13 coefficients capturing acoustic fingerprint).

#### Trade-off Analysis:
| Approach | Interpretability | Editability | Timbral Expressiveness | Model Portability |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Symbolic (MIDI / MusicXML)** | High | High | Low (lacks production context) | High |
| **Pure Continuous (Neural Latents)** | Low (opaque vectors) | Low | High | Low (locked to specific model weights) |
| **Sonara Hybrid (SMR v1.0.0)** | **High** | **High** | **High** | **High (JSON / schema-standardized)** |

### 2.2 Determinism, Uncertainty, and Anti-Hallucination

Unlike generative models that generate hypothetical notes or instruments without qualification, Sonara treats music analysis as an empirical measurement task:
- **Confidence Scoring**: Every extracted event, chord, tempo estimation, and section boundary carries a confidence score between `0.0` and `1.0`.
- **Uncertainty Flags**: When signal conditions degrade (clipping, silence, low SNR, ambiguous tonal centers, high polyphonic density), the engine records explicit uncertainty flags (`low_signal_to_noise_ratio`, `ambiguous_tonal_center`, `pitch_uncertain`, `signal_clipping_detected`).
- **Graceful Fallbacks**: Missing or unresolvable attributes are explicitly represented as `null` / `None` rather than fabricating arbitrary ground truth.

### 2.3 Modular Decoupling and Lightweight Footprint

Heavyweight neural foundation models require dedicated GPU hardware, large memory footprints, and multi-gigabyte checkpoint downloads. Sonara is engineered with a strict **layered dependency design**:
- The core pipeline runs efficiently on standard CPU architectures using optimized digital signal processing algorithms (`numpy`, `scipy`, `librosa`, `soundfile`).
- Heavyweight model integrations (e.g., Demucs stem separation, MT3 transcription, Audiocraft/MusicGen neural generation) are accessed through decoupled abstract interfaces (`NeuralPromptConditioner`), ensuring baseline accessibility across any Linux, macOS, or Windows environment without GPU mandates.

---

## 3. Dataflow & Pipeline Modules

### 3.1 Audio Ingestion (`sonara.audio.loader`)
- Validates file integrity, calculates SHA-256 cryptographic checksums, detects container formats (WAV, MP3, FLAC, OGG, AIFF).
- Downmixes multi-channel audio to mono for signal analysis while preserving multi-channel metadata.
- Resamples waveforms to a standardized target sample rate (default 22,050 Hz).
- Performs peak and RMS amplitude analysis, identifying silent or digital-clipped waveforms.

### 3.2 Acoustic & Timbral Extraction (`sonara.audio.features`)
- Computes frame-wise Root-Mean-Square (RMS) amplitude and dynamic range in decibels.
- Calculates the Crest Factor ($\text{Peak} / \text{RMS}$), measuring transient sharpness and dynamic compression.
- Extracts spectral centroid, spectral bandwidth, spectral rolloff (85% energy threshold), spectral flatness, and zero-crossing rate.
- Analyzes octave-band spectral contrast and 13-coefficient Mel-Frequency Cepstral Coefficients (MFCCs).
- Derives the **Warmth Index** (energy below 500 Hz / total energy) and **Brightness Index** (energy above 3000 Hz / total energy).

### 3.3 Rhythm & Metric Analysis (`sonara.audio.rhythm`)
- Computes complex spectral flux onset strength envelopes.
- Performs peak picking with backtracking to pinpoint onset timestamps.
- Evaluates pulse clarity through autocorrelation of the onset envelope.
- Implements beat tracking to estimate Tempo (BPM) and absolute beat grid timestamps.
- Estimates the metric time signature (evaluating 4/4 vs. 3/4 accentuation periodicities).
- Calculates the **Syncopation Index**, measuring the proportion of transient onsets landing off the primary beat grid.

### 3.4 Harmonic & Tonal Analysis (`sonara.audio.harmony`)
- Performs Harmonic-Percussive Source Separation (HPSS) to isolate harmonic components from percussive transients.
- Computes 12-bin Chroma CENS (Chroma Energy Normalized Statistics) and Chroma STFT.
- Executes the Krumhansl-Schmuckler key-finding algorithm, correlating observed pitch-class distributions against perceptual major and minor profiles across all 24 keys.
- Matches chroma segments against chord template matrices (major, minor, diminished, dominant 7th) to generate time-aligned chord sequences.

### 3.5 Structural Segmentation (`sonara.audio.structure`)
- Constructs a composite feature matrix combining normalized Chroma and MFCC vectors.
- Calculates an $N \times N$ Self-Similarity Matrix (SSM) using cosine distance.
- Applies a 2D Foote checkerboard kernel with Gaussian tapering along the diagonal to generate a structural novelty curve.
- Detects local peaks exceeding adaptive variance thresholds to establish section boundaries.
- Assigns formal musical labels (`intro`, `verse`, `chorus`, `bridge`, `outro`, `transition`) based on positional heuristics and recurrence clustering.

### 3.6 Event Extraction & Layer Classification (`sonara.events`)
- Extracts discrete musical events at detected onset boundaries.
- Estimates event duration via energy decay curves and inter-onset intervals.
- Computes fundamental frequency ($f_0$) using parabolic interpolation over normalized autocorrelation peaks.
- Converts frequency in Hz to standard MIDI note numbers, note names, and cents deviations.
- Classifies events into functional layers (`percussion`, `bass`, `harmonic`, `melodic`, `unknown`) based on pitch register, spectral centroid, zero-crossing rate, and harmonicity.

---

## 4. Multi-Track Analysis & Style Profiling (`sonara.analysis`)

To achieve the long-term vision of learning musical styles across artist collections:
1. **Collection Analyzer**: Aggregates multiple SMR documents into empirical probability distributions:
   - Rhythmic tendencies: mean tempo, tempo dispersion ($\sigma_{\text{BPM}}$), syncopation tendencies, onset density.
   - Harmonic grammar: relative key distributions, modal preference (major vs. minor), collective chord vocabulary, and first-order Markov chord transition matrices ($P(c_{t+1} \mid c_t)$).
   - Structural grammar: average section counts, section duration averages, repetition density.
   - Timbral aesthetic: average spectral brightness, low-end warmth, dynamic range compression.
2. **Evidence Attribution**: Every stylistic observation is tied to concrete track measurements (`EvidenceObservation`), preventing unverifiable generalities.
3. **Cross-Track Similarity Engine**: Extracts 23-dimensional normalized feature vectors from SMR documents and calculates cosine similarity and Euclidean distance matrices across the collection.

---

## 5. Composition Generation & Rendering (`sonara.generation`)

Sonara provides a three-stage generation architecture:
1. **Style-Conditioned Symbolic Generator (`StyleConditionedGenerator`)**:
   - Synthesizes a new `SMRDocument` constrained by a learned `StyleProfile`.
   - Generates chord progressions from the style's Markov transition probabilities.
   - Composes multi-voice arrangements: syncopated basslines, drum tracks (kick, snare, hi-hat), sustained harmonic pads, and melodic leads.
2. **Procedural Multi-Timbral Synthesizer (`SMRSynthesizer`)**:
   - Directly renders SMR events into high-fidelity 44.1 kHz stereo audio (WAV).
   - Uses procedural synthesis algorithms tailored per layer (frequency-swept kicks, tone/noise snares, metallic hats, sub-bass with harmonic overtones, detuned stereo pads, vibrato leads).
   - Mixes, pans, and applies soft-limiting with headroom normalization.
3. **Neural Conditioning Interface (`NeuralPromptConditioner`)**:
   - Formulates structured textual and token conditioning representations for deep neural audio foundation models (e.g. Meta MusicGen / AudioCraft, Stable Audio).
