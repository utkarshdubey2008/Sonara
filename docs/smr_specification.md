# Sonara Musical Representation (SMR) Specification

- **Specification Version**: `1.0.0`
- **Schema Format**: JSON Schema Draft 2020-12
- **MIME Type**: `application/vnd.sonara.smr+json`
- **Default Extension**: `.smr.json`

---

## 1. Overview

The **Sonara Musical Representation (SMR)** is a structured, versioned, machine-readable data format for encapsulating musical compositions, acoustic recordings, and analytical intelligence. SMR documents capture both explicit symbolic notation (notes, chords, bars, sections) and continuous acoustic properties (timbral features, dynamic compression, energy distributions, layer interactions).

---

## 2. Document Hierarchy

An SMR document consists of a root object with the following top-level keys:

```json
{
  "smr_version": "1.0.0",
  "metadata": { ... },
  "acoustic_features": { ... },
  "rhythm": { ... },
  "harmony": { ... },
  "structure": [ ... ],
  "events": [ ... ],
  "layer_interactions": [ ... ],
  "provenance": { ... }
}
```

---

## 3. Schema Definitions

### 3.1 `metadata` (AudioMetadataModel)
Physical and container attributes of the source audio file.

| Field | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `string` | Absolute or relative path to the source audio file. |
| `file_name` | `string` | Base filename. |
| `file_size_bytes` | `integer` (>= 0) | Size of file in bytes. |
| `sha256_hash` | `string` | 64-character hexadecimal SHA-256 integrity checksum. |
| `duration_seconds` | `number` (>= 0.0) | Total audio duration in seconds. |
| `sample_rate` | `integer` (> 0) | Native audio sample rate in Hz (e.g., 44100). |
| `channels` | `integer` (> 0) | Number of audio channels (1 = mono, 2 = stereo). |
| `format` | `string` | Container format identifier (e.g., `WAV`, `MP3`, `FLAC`). |
| `is_silent` | `boolean` | True if the entire waveform has negligible amplitude (< 1e-4). |
| `is_clipped` | `boolean` | True if any sample reaches or exceeds digital full scale (1.0). |
| `peak_amplitude` | `number` [0.0, 1.0] | Maximum absolute sample value. |
| `rms_amplitude` | `number` (>= 0.0) | Global Root-Mean-Square signal amplitude. |

### 3.2 `acoustic_features` (AcousticFeaturesModel)
Timbral, spectral, and dynamic production descriptors.

| Field | Type | Description |
| :--- | :--- | :--- |
| `rms_energy_mean` | `number` | Mean frame-wise RMS energy across the track. |
| `dynamic_range_db` | `number` | Dynamic range in decibels ($20 \log_{10}(\text{max} / \text{min})$). |
| `crest_factor` | `number` | Ratio of peak amplitude to mean RMS energy. |
| `spectral_centroid_mean`| `number` | Perceived spectral brightness in Hz. |
| `spectral_centroid_std` | `number` | Standard deviation of spectral brightness across frames. |
| `spectral_bandwidth_mean`| `number`| Frequency spread around the centroid in Hz. |
| `spectral_rolloff_mean` | `number` | Frequency below which 85% of spectral energy resides. |
| `spectral_flatness_mean`| `number` [0.0, 1.0] | Wiener entropy (tonality vs. noisiness). |
| `zero_crossing_rate_mean`| `number` | Average rate of sign-changes in the signal. |
| `warmth_index` | `number` [0.0, 1.0] | Fraction of total energy in the sub-500 Hz frequency band. |
| `brightness_index` | `number` [0.0, 1.0] | Fraction of total energy in the above-3000 Hz frequency band. |
| `spectral_contrast_mean`| `array[number]` | Subband spectral contrast across 7 frequency octave bands. |
| `mfcc_mean` | `array[number]` | 13-dimensional mean MFCC vector. |
| `mfcc_std` | `array[number]` | 13-dimensional standard deviation MFCC vector. |
| `uncertainty_flags` | `array[string]` | DSP execution flags or warnings. |

### 3.3 `rhythm` (RhythmProfileModel)
Tempo, meter, beat tracking, and groove metrics.

| Field | Type | Description |
| :--- | :--- | :--- |
| `estimated_bpm` | `number` (>= 0.0) | Estimated tempo in beats per minute. |
| `tempo_confidence` | `number` [0.0, 1.0] | Confidence score for tempo tracking. |
| `time_signature_estimate`| `string` | Metric meter (e.g. `4/4`, `3/4`). |
| `beat_count` | `integer` (>= 0) | Number of detected beat occurrences. |
| `onset_count` | `integer` (>= 0) | Total transient onsets detected. |
| `onset_density_per_second`| `number` (>= 0.0) | Average onsets per second. |
| `syncopation_index` | `number` [0.0, 1.0] | Fraction of onsets falling outside the primary beat grid. |
| `pulse_clarity` | `number` [0.0, 1.0] | Strength of rhythmic periodicity. |
| `is_rhythmic` | `boolean` | True if track possesses discernible meter. |
| `beat_timestamps` | `array[number]` | List of timestamps (seconds) for each beat. |

### 3.4 `harmony` (HarmonicProfileModel)
Tonal center, mode, and chord progressions.

| Field | Type | Description |
| :--- | :--- | :--- |
| `primary_key` | `string` | Pitch-class root (e.g., `C`, `F#`, `Bb`). |
| `mode` | `string` | Musical mode (`major`, `minor`, `unknown`). |
| `key_confidence` | `number` [0.0, 1.0] | Key determination confidence score. |
| `tonal_stability` | `number` [0.0, 1.0] | Correlation against ideal diatonic profile. |
| `chord_vocabulary` | `array[string]` | Set of unique chord symbols (e.g., `["C:maj", "A:min"]`). |
| `chord_sequence` | `array[ChordItem]` | Chronological list of chord events. |
| `chroma_profile` | `array[number]` | 12-element pitch-class energy distribution (C to B). |

### 3.5 `structure` (`array[StructuralSectionModel]`)
Hierarchical formal arrangement sections.

| Field | Type | Description |
| :--- | :--- | :--- |
| `section_id` | `string` | Unique identifier (e.g., `sec_0`). |
| `section_type` | `string` | Role (`intro`, `verse`, `chorus`, `bridge`, `outro`, `transition`, `body`). |
| `start_time` | `number` (>= 0.0) | Section onset timestamp in seconds. |
| `end_time` | `number` (>= 0.0) | Section offset timestamp in seconds. |
| `duration` | `number` (>= 0.0) | Section length in seconds. |
| `confidence` | `number` [0.0, 1.0] | Segmentation confidence. |
| `is_repetition` | `boolean` | Indicates if section repeats an earlier section. |
| `repeat_of_id` | `string` / `null` | Reference section ID if repeated. |

### 3.6 `events` (`array[MusicalEventModel]`)
Discrete atomic musical events (notes, drum hits, chords).

| Field | Type | Description |
| :--- | :--- | :--- |
| `event_id` | `string` | Unique identifier (e.g., `evt_0042`). |
| `start_time` | `number` (>= 0.0) | Event onset timestamp in seconds. |
| `duration` | `number` (>= 0.0) | Event duration in seconds. |
| `end_time` | `number` (>= 0.0) | Event offset timestamp in seconds. |
| `layer` | `string` | Functional layer: `percussion`, `bass`, `harmonic`, `melodic`, `unknown`. |
| `event_type` | `string` | Event classification: `note`, `hit`, `chord`, `onset`, `rest`. |
| `pitch_midi` | `integer` [0, 127] / `null` | MIDI note number if pitched. |
| `pitch_name` | `string` / `null` | Standard pitch notation with octave (e.g., `C4`, `A#2`). |
| `frequency_hz` | `number` / `null` | Fundamental frequency in Hz. |
| `cents_deviation` | `number` / `null` | Deviation from standard equal temperament pitch in cents (-50 to +50). |
| `velocity` | `integer` [0, 127] | Dynamic intensity mapped to standard MIDI velocity range. |
| `confidence` | `number` [0.0, 1.0] | Detection confidence score. |
| `uncertainty_flags`| `array[string]` | Specific event-level warnings. |

### 3.7 `layer_interactions` (`array[LayerInteractionModel]`)
Cross-layer interaction metrics.

| Field | Type | Description |
| :--- | :--- | :--- |
| `layer_pair` | `string` | Name of interacting pair (e.g., `bass_percussion`). |
| `synchrony_score` | `number` [0.0, 1.0] | Temporal synchrony within 50ms tolerance. |
| `density_ratio` | `number` | Ratio of event count between layers. |
| `harmonic_support_overlap`| `number` [0.0, 1.0]| Harmonic alignment ratio. |

### 3.8 `provenance` (AnalysisProvenanceModel)
Metadata tracking analysis provenance and execution integrity.

| Field | Type | Description |
| :--- | :--- | :--- |
| `system_name` | `string` | Always `"Sonara"`. |
| `system_version` | `string` | Engine release version (e.g., `"0.1.0"`). |
| `smr_schema_version`| `string` | Specification version (`"1.0.0"`). |
| `timestamp_utc` | `string` | ISO 8601 UTC timestamp. |
| `execution_time_seconds`| `number` | Elapsed processing time. |
| `limitations` | `array[string]` | Stated methodological limitations. |
| `warnings` | `array[string]` | Runtime alerts and validation warnings. |

---

## 4. Versioning Strategy

SMR follows Semantic Versioning 2.0.0 (`MAJOR.MINOR.PATCH`):
- **MAJOR**: Incompatible schema modifications (removing fields, altering type requirements).
- **MINOR**: Backward-compatible additive enhancements (new optional descriptor fields, new layer types).
- **PATCH**: Backward-compatible bug fixes or documentation clarifications.

Parsers conforming to SMR v1.x MUST safely accept any document where `MAJOR == 1`, ignoring unrecognized keys.
