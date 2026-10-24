"""Sonara Instrumental Representation Package Container (.smr).

Provides a self-contained, machine-readable binary/archive package format (.smr)
bundling the master instrumental JSON specification, embedded audio sample assets,
and manifest verification tables.
"""

from __future__ import annotations

import io
import json
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import soundfile as sf

from sonara.representation.instrumental_schema import (
    SMRv2InstrumentalDocument,
    SMR_V2_SPEC_VERSION,
)


class SMRPackage:
    """Master archive container for SMR v2.0 Instrumental Documents and audio assets.
    
    Structure of a .smr package:
        manifest.json          - Package metadata, asset manifest, and checksums
        instrumental.json      - SMRv2InstrumentalDocument JSON document
        assets/
            samples/           - Audio slices, vocoders, and non-symbolic instruments (.wav)
            patterns/          - Optional standard MIDI export files (.mid)
    """

    MANIFEST_FILE = "manifest.json"
    DOCUMENT_FILE = "instrumental.json"
    ASSETS_DIR = "assets"
    SAMPLES_DIR = "assets/samples"
    PATTERNS_DIR = "assets/patterns"

    def __init__(self, document: Optional[SMRv2InstrumentalDocument] = None):
        self.document: Optional[SMRv2InstrumentalDocument] = document
        self.samples: Dict[str, Tuple[np.ndarray, int]] = {}  # rel_path -> (audio_array, sample_rate)
        self.midi_patterns: Dict[str, bytes] = {}             # rel_path -> midi_bytes
        self.metadata: Dict[str, str] = {
            "format_version": SMR_V2_SPEC_VERSION,
            "generator": "Sonara SMR v2.0 Engine",
        }

    def add_sample(self, asset_rel_path: str, audio: np.ndarray, sample_rate: int = 44100) -> str:
        """Register an audio sample slice into the package.
        
        Args:
            asset_rel_path: Relative path inside container (e.g. 'assets/samples/chopped_riff.wav')
            audio: Audio waveform as 1D or 2D numpy float array
            sample_rate: Sample rate in Hz
            
        Returns:
            Normalized relative path registered.
        """
        norm_path = asset_rel_path.replace("\\", "/").lstrip("/")
        if not norm_path.startswith("assets/samples/"):
            norm_path = f"assets/samples/{norm_path}"
            
        self.samples[norm_path] = (audio.astype(np.float32), sample_rate)
        if self.document and norm_path not in self.document.embedded_sample_assets:
            self.document.embedded_sample_assets.append(norm_path)
            
        return norm_path

    def get_sample(self, asset_rel_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Retrieve an audio sample by relative path."""
        norm_path = asset_rel_path.replace("\\", "/").lstrip("/")
        if not norm_path.startswith("assets/"):
            norm_path = f"assets/samples/{norm_path}"
        return self.samples.get(norm_path)

    def add_midi_pattern(self, pattern_rel_path: str, midi_data: bytes) -> str:
        """Register a MIDI sequence export for a pattern."""
        norm_path = pattern_rel_path.replace("\\", "/").lstrip("/")
        if not norm_path.startswith("assets/patterns/"):
            norm_path = f"assets/patterns/{norm_path}"
        self.midi_patterns[norm_path] = midi_data
        return norm_path

    def get_midi_pattern(self, pattern_rel_path: str) -> Optional[bytes]:
        norm_path = pattern_rel_path.replace("\\", "/").lstrip("/")
        if not norm_path.startswith("assets/"):
            norm_path = f"assets/patterns/{norm_path}"
        return self.midi_patterns.get(norm_path)

    def save_package(self, output_path: Union[str, Path]) -> Path:
        """Serialize the document and all embedded assets into a .smr archive.
        
        Args:
            output_path: File path ending in .smr (or directory)
            
        Returns:
            Resolved Path of the saved archive.
        """
        if self.document is None:
            raise ValueError("Cannot save SMR package without an active SMRv2InstrumentalDocument")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare manifest
        manifest = {
            "format_version": self.metadata.get("format_version", SMR_V2_SPEC_VERSION),
            "generator": self.metadata.get("generator", "Sonara SMR v2.0 Engine"),
            "track_title": self.document.track_title,
            "master_bpm": self.document.master_bpm,
            "master_key": f"{self.document.master_key} {self.document.master_scale}",
            "duration_seconds": self.document.total_duration_seconds,
            "total_bars": self.document.total_bars,
            "lead_vocal_excluded": self.document.vocal_exclusion.is_lead_vocal_excluded,
            "asset_checksums": {},
        }

        # Build in-memory zip
        with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Write audio samples
            for rel_path, (sig, sr) in self.samples.items():
                buf = io.BytesIO()
                sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
                sample_bytes = buf.getvalue()
                zf.writestr(rel_path, sample_bytes)
                manifest["asset_checksums"][rel_path] = hashlib.sha256(sample_bytes).hexdigest()

            # 2. Write MIDI patterns
            for rel_path, midi_bytes in self.midi_patterns.items():
                zf.writestr(rel_path, midi_bytes)
                manifest["asset_checksums"][rel_path] = hashlib.sha256(midi_bytes).hexdigest()

            # 3. Write Document JSON
            doc_json_str = self.document.model_dump_json(indent=2)
            doc_bytes = doc_json_str.encode("utf-8")
            zf.writestr(self.DOCUMENT_FILE, doc_bytes)
            manifest["asset_checksums"][self.DOCUMENT_FILE] = hashlib.sha256(doc_bytes).hexdigest()

            # 4. Write Manifest JSON
            manifest_json_str = json.dumps(manifest, indent=2)
            zf.writestr(self.MANIFEST_FILE, manifest_json_str.encode("utf-8"))

        return out_path

    @classmethod
    def load_package(cls, package_path: Union[str, Path]) -> "SMRPackage":
        """Load an SMR package (.smr archive or folder) into memory.
        
        Args:
            package_path: Path to the .smr file
            
        Returns:
            Loaded SMRPackage instance with document and assets.
        """
        pkg_p = Path(package_path)
        if not pkg_p.exists():
            raise FileNotFoundError(f"SMR package not found: {pkg_p}")

        package = cls()

        with zipfile.ZipFile(pkg_p, mode="r") as zf:
            namelist = zf.namelist()
            if cls.DOCUMENT_FILE not in namelist:
                raise ValueError(f"Corrupt SMR package: missing {cls.DOCUMENT_FILE}")

            # Load document
            doc_data = json.loads(zf.read(cls.DOCUMENT_FILE).decode("utf-8"))
            package.document = SMRv2InstrumentalDocument.model_validate(doc_data)

            # Load manifest
            if cls.MANIFEST_FILE in namelist:
                manifest_data = json.loads(zf.read(cls.MANIFEST_FILE).decode("utf-8"))
                package.metadata.update(manifest_data)

            # Load sample assets
            for name in namelist:
                if name.startswith("assets/samples/") and name.endswith(".wav"):
                    sample_bytes = zf.read(name)
                    buf = io.BytesIO(sample_bytes)
                    sig, sr = sf.read(buf, dtype="float32")
                    package.samples[name] = (sig, sr)
                elif name.startswith("assets/patterns/") and name.endswith(".mid"):
                    midi_bytes = zf.read(name)
                    package.midi_patterns[name] = midi_bytes

        return package
