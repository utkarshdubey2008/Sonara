"""Tests for end-to-end music reconstruction and web application."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from sonara.analysis.comparator import ReconstructionComparator
from sonara.app import app
from sonara.generation.generator import StyleConditionedGenerator
from sonara.generation.synthesizer import SMRSynthesizer
from sonara.pipeline import SonaraPipeline


def test_reconstruction_comparator_identical_files(sine_audio_file: Path) -> None:
    comparator = ReconstructionComparator(sample_rate=22050)
    report = comparator.compare(sine_audio_file, sine_audio_file)

    assert report.composite_fidelity_score >= 90.0
    assert report.harmonic_alignment >= 0.95
    assert report.envelope_correlation >= 0.95
    assert len(report.metrics) >= 4
    assert len(report.reproduced_capabilities) > 0


def test_end_to_end_music_reconstruction_roundtrip(temp_dir: Path) -> None:
    # 1. Synthesize a ground truth 6-second musical song in D major at 120 BPM
    gen = StyleConditionedGenerator(seed=42)
    original_smr = gen.generate(duration_seconds=6.0, target_bpm=120.0, target_key="D")

    synth = SMRSynthesizer(sample_rate=22050)
    orig_wav = temp_dir / "ground_truth.wav"
    synth.render(original_smr, orig_wav)

    # 2. Ingest and decode into SMR via SonaraPipeline
    pipeline = SonaraPipeline(target_sr=22050)
    decoded_smr = pipeline.analyze(orig_wav)

    assert len(decoded_smr.events) > 10
    assert decoded_smr.rhythm.estimated_bpm > 100.0

    # 3. Render reconstructed audio strictly from the decoded SMR
    recon_wav = temp_dir / "reconstructed.wav"
    synth.render(decoded_smr, recon_wav)
    assert recon_wav.exists()
    assert recon_wav.stat().st_size > 10000

    # 4. Objective mathematical comparison
    comparator = ReconstructionComparator(sample_rate=22050)
    report = comparator.compare(orig_wav, recon_wav)

    assert report.composite_fidelity_score > 60.0
    assert report.harmonic_alignment > 0.60
    assert report.tempo_error_bpm < 15.0


def test_web_application_endpoints(sine_audio_file: Path) -> None:
    client = TestClient(app)

    # Test GET index UI
    resp_index = client.get("/")
    assert resp_index.status_code == 200
    assert "Sonara Reconstruction Engine" in resp_index.text

    # Test POST upload
    with open(sine_audio_file, "rb") as f:
        resp_upload = client.post("/api/upload", files={"file": ("test.wav", f, "audio/wav")})
    assert resp_upload.status_code == 200
    data_upload = resp_upload.json()
    assert data_upload["status"] == "success"
    session_id = data_upload["session_id"]

    # Test POST analyze
    resp_analyze = client.post(f"/api/analyze/{session_id}")
    assert resp_analyze.status_code == 200
    data_analyze = resp_analyze.json()
    assert data_analyze["status"] == "success"
    assert "summary" in data_analyze
    assert "smr" in data_analyze
    assert "comparison" in data_analyze

    # Test GET SMR
    resp_smr = client.get(f"/api/smr/{session_id}")
    assert resp_smr.status_code == 200
    assert resp_smr.json()["smr_version"] == "1.0.0"

    # Test GET original audio stream
    resp_orig_audio = client.get(f"/api/audio/{session_id}/original")
    assert resp_orig_audio.status_code == 200

    # Test GET reconstructed audio stream
    resp_recon_audio = client.get(f"/api/audio/{session_id}/reconstructed")
    assert resp_recon_audio.status_code == 200

    # Test GET MIDI stream
    resp_midi = client.get(f"/api/audio/{session_id}/midi")
    assert resp_midi.status_code == 200

    # Test GET download SMR
    resp_dl = client.get(f"/api/download/smr/{session_id}")
    assert resp_dl.status_code == 200
