"""Audio loading, validation, and metadata extraction module."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded or validated."""
    pass


@dataclass
class AudioMetadata:
    """Metadata extracted from an audio file."""
    file_path: str
    file_name: str
    file_size_bytes: int
    sha256_hash: str
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    is_silent: bool = False
    is_clipped: bool = False
    peak_amplitude: float = 0.0
    rms_amplitude: float = 0.0


class AudioLoader:
    """Handles loading, resampling, and metadata inspection of audio files."""

    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}

    def __init__(self, default_sr: int = 22050, mono: bool = True):
        """
        Initialize AudioLoader.

        Args:
            default_sr: Target sample rate for resampling. Default 22050 Hz.
            mono: If True, downmixes multi-channel audio to mono.
        """
        self.default_sr = default_sr
        self.mono = mono

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA256 checksum of an audio file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def inspect(self, file_path: str | Path) -> AudioMetadata:
        """
        Inspect an audio file and extract metadata without loading entire waveform if possible.

        Args:
            file_path: Path to the target audio file.

        Returns:
            AudioMetadata instance.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise AudioLoadError(f"File not found: {path}")
        if not path.is_file():
            raise AudioLoadError(f"Path is not a regular file: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise AudioLoadError(f"Audio file is empty (0 bytes): {path}")

        file_hash = self.compute_sha256(path)
        ext = path.suffix.lower()

        # Try inspection via soundfile first (fast for WAV/FLAC/OGG)
        try:
            with sf.SoundFile(str(path)) as sfile:
                channels = sfile.channels
                sr = sfile.samplerate
                duration = float(sfile.frames) / float(sr) if sr > 0 else 0.0
                fmt = sfile.format
        except Exception:
            # Fall back to librosa for formats like MP3/M4A
            try:
                duration = librosa.get_duration(path=str(path))
                sr = self.default_sr
                channels = 1
                fmt = ext.lstrip(".").upper()
            except Exception as e:
                raise AudioLoadError(f"Failed to inspect audio file {path}: {str(e)}") from e

        # Quick check for amplitude stats by reading a small chunk if possible
        peak, rms, is_silent, is_clipped = 0.0, 0.0, False, False
        try:
            with sf.SoundFile(str(path)) as sfile:
                # Read at most 1 second to inspect amplitude
                chunk_frames = min(sfile.frames, sfile.samplerate)
                if chunk_frames > 0:
                    chunk = sfile.read(chunk_frames, dtype="float32")
                    peak = float(np.max(np.abs(chunk)))
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    is_silent = bool(peak < 1e-4)
                    is_clipped = bool(peak >= 0.999)
        except Exception:
            pass

        return AudioMetadata(
            file_path=str(path),
            file_name=path.name,
            file_size_bytes=file_size,
            sha256_hash=file_hash,
            duration_seconds=duration,
            sample_rate=sr,
            channels=channels,
            format=fmt,
            is_silent=is_silent,
            is_clipped=is_clipped,
            peak_amplitude=peak,
            rms_amplitude=rms,
        )

    def load(
        self,
        file_path: str | Path,
        sr: Optional[int] = None,
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> Tuple[np.ndarray, AudioMetadata]:
        """
        Load audio file into a NumPy floating point array.

        Args:
            file_path: Path to the audio file.
            sr: Target sample rate (defaults to self.default_sr if None).
            duration: Maximum duration to load in seconds.
            offset: Start offset in seconds.

        Returns:
            Tuple of (waveform_numpy_array, AudioMetadata).
        """
        target_sr = sr or self.default_sr
        meta = self.inspect(file_path)

        try:
            waveform, actual_sr = librosa.load(
                meta.file_path,
                sr=target_sr,
                mono=self.mono,
                offset=offset,
                duration=duration,
            )
        except Exception as e:
            raise AudioLoadError(f"Failed to load audio array from {file_path}: {str(e)}") from e

        if waveform.size == 0:
            raise AudioLoadError(f"Loaded audio waveform is empty: {file_path}")

        # Update amplitude stats with complete loaded array
        peak = float(np.max(np.abs(waveform)))
        rms = float(np.sqrt(np.mean(waveform**2)))
        meta.peak_amplitude = peak
        meta.rms_amplitude = rms
        meta.is_silent = bool(peak < 1e-4)
        meta.is_clipped = bool(peak >= 0.999)

        return waveform, meta
