# Research Foundation & Comparative Analysis

## 1. Prior Art in Musical Representation

The representation of music in computational systems has spanned multiple decades and paradigm shifts. Understanding existing systems clarifies precisely where Sonara innovates and what gaps remain unaddressed.

### 1.1 Symbolic Representations

| Format | Strengths | Critical Limitations |
| :--- | :--- | :--- |
| **Standard MIDI (SMF)** | Universal hardware & DAW adoption; lightweight; event-based. | Lacks pitch spelling ($C\sharp$ vs $D\flat$); no timbral or acoustic descriptors; no formal section hierarchy; channel capacity limited to 16. |
| **MusicXML** | Comprehensive for engraving and notation; explicit bars, clefs, dynamics. | Overly verbose; difficult for machine learning ingestion; ignores production acoustics and timbral nuance. |
| **MEI (Music Encoding Initiative)** | Rigorous scholarly musicology representation; XML-based. | Heavy syntactic overhead; research-focused rather than pipeline-oriented. |
| **REMI / ABC / Humdrum** | Tokenized representations popular in symbolic NLP/transformer models. | Tied strictly to discrete note triggers; discard all acoustic, mixing, and spatial realities. |

**Sonara's Contribution**: SMR preserves the interpretability of symbolic note events while integrating continuous timbral descriptors, structural hierarchy, and uncertainty metrics into an easily parseable, validated JSON format.

---

## 2. Automatic Music Transcription (AMT) & Source Separation

### 2.1 State-of-the-Art in Audio Source Separation
- **Demucs (Défossez et al., Meta AI)**: Hybrid convolutional/transformer network operating on waveform and spectrogram domains. Industry standard for 4-stem separation (drums, bass, vocals, other). Requires significant GPU memory for batch inference.
- **Spleeter (Hennequin et al., Deezer)**: U-Net based separator. Fast on CPU, but suffers from phase artifacts and bleed across stems.

### 2.2 State-of-the-Art in Automatic Music Transcription
- **Basic Pitch (Bittner et al., Spotify)**: Lightweight polyphonic transcription using convolutional neural networks with multipitch and note-level tracking. Efficient and CPU-friendly, but constrained to pitched notes without full arrangement context.
- **MT3 (Gardner et al., Google Brain)**: Multi-task, multi-instrument transcription using a sequence-to-sequence T5 architecture. Can transcribe dozens of instrument tokens, but requires massive training corpora (MAESTRO, Slakh) and heavy compute.

**Sonara's Engineering Position**: Rather than requiring a 10 GB model checkpoint download before users can run basic analysis, Sonara establishes an efficient DSP baseline (HPSS, spectral autocorrelation, and chroma template matching) with clear uncertainty tracking, providing decoupled interfaces to plug in neural separators when available.

---

## 3. Discrete Audio Tokenization & Neural Audio Embeddings

### 3.1 Audio Tokenizers
- **SoundStream (Zeghidour et al., Google)**: Residual Vector Quantization (RVQ) autoencoder operating at 24 kHz.
- **EnCodec (Défossez et al., Meta AI)**: Neural audio codec supporting multi-rate RVQ down to 1.5–6 kbps.
- **DAC (Descript Audio Codec, Kumar et al.)**: High-fidelity discrete audio tokenization with improved adversarial and multi-scale STFT loss functions.

### 3.2 Representation Limitations of Neural Tokenizers
While discrete neural audio tokens allow language models to generate waveform audio, **the tokens themselves are acoustically uninterpretable**:
- A sequence of EnCodec codebook indices provides no direct information about whether a section is in 4/4 or 3/4 time, whether the key is C minor, or how the bass relates to the kick drum.
- Editing an EnCodec sequence symbolically (e.g. "change this chord to a dominant 7th" or "shorten the verse by 2 bars") is virtually impossible without re-synthesizing and re-encoding.

**Sonara's Resolution**: Sonara provides an interpretable symbolic and structural scaffolding (SMR) that can condition downstream neural models (via `NeuralPromptConditioner`), giving creators and algorithms structural control over neural generation.

---

## 4. Benchmark Datasets in Music Information Retrieval (MIR)

Sonara's analysis algorithms are grounded in established MIR evaluation standards:

1. **Rhythm & Beat Tracking**:
   - Evaluated against datasets such as **SMC**, **Beatles**, and **GTZAN**.
   - Standard metrics: Cemgil accuracy, AMLt (continuity within double/half tempo tolerances), and F-measure.
2. **Key & Chord Recognition**:
   - Evaluated on **Isophonics** (Beatles, Queen) and **Billboard** chord corpora.
   - Standard metrics: Mirex Weighted Chord Symbol Recall (WCSR).
3. **Structural Segmentation**:
   - Evaluated on the **SALAMI** (Structural Analysis of Large Audio Music Information) dataset.
   - Standard metrics: Pairwise F-measure at 0.5s and 3.0s boundary tolerance.
4. **Symbolic Music Collections**:
   - **Lakh MIDI Dataset (LMD)** and **MAESTRO** for multi-track symbolic modeling.

---

## 5. Scope & Reality of Multi-Song Learning

A crucial engineering principle of Sonara is **scientific honesty regarding training data and model scale**:
- A collection of 10 or 20 songs by a single artist **cannot** train a foundation generative music model from scratch.
- What a 10-track collection **can** achieve is an empirical statistical extraction of stylistic grammar:
  - Rhythmic tempo distributions and syncopation ratios.
  - Tonal preferences and chord transition probability matrices ($P(c_{t+1} \mid c_t)$).
  - Arrangement patterns (typical section counts and durations).
  - Production timbral fingerprints (spectral brightness, dynamic range).

Sonara explicitly separates measured empirical features from speculative inferences, documenting the specific tracks and metric values that substantiate every learned style profile.
