"""Tests for SMRSynthesizer module."""

from pathlib import Path
import soundfile as sf
import pytest

from sonara.generation.generator import StyleConditionedGenerator
from sonara.generation.synthesizer import SMRSynthesizer


def test_smr_synthesizer_render(temp_dir: Path) -> None:
    generator = StyleConditionedGenerator(seed=99)
    doc = generator.generate(duration_seconds=4.0, target_bpm=120.0, target_key="A")

    synth = SMRSynthesizer(sample_rate=22050)
    wav_path = temp_dir / "rendered_test.wav"
    rendered_path = synth.render(doc, wav_path)

    assert rendered_path.exists()
    assert rendered_path.stat().st_size > 1000

    # Verify audio properties with soundfile
    with sf.SoundFile(str(rendered_path)) as sfile:
        assert sfile.samplerate == 22050
        assert sfile.channels == 2
        assert sfile.frames >= int(4.0 * 22050)
        data = sfile.read()
        peak = float(max(abs(data.min()), abs(data.max())))
        assert 0.1 <= peak <= 1.0  # Proper amplitude without hard clipping
