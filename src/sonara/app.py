"""Interactive Web Application for Sonara Beat Deconstruction, SMR Inspection, and DAW Export."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from sonara.analysis.comparator import ReconstructionComparator, ReconstructionReport
from sonara.audio.stems import StemSeparator
from sonara.pipeline import SonaraPipeline
from sonara.representation.fl_studio import FLStudioExporter
from sonara.representation.serializer import SMRSerializer
from sonara.representation.schema import SMRDocument


SESSIONS_DIR = Path("/tmp/sonara_sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

pipeline = SonaraPipeline()
comparator = ReconstructionComparator()
stem_separator = StemSeparator()

app = FastAPI(
    title="Sonara Music Deconstruction & Reconstruction Workbench",
    description="Decode musical recordings into isolated stems, editable SMR representations, and FL Studio packages.",
    version="0.2.0",
)


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> str:
    """Return single-page web application interface."""
    return UI_HTML_TEMPLATE


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an audio file and initialize a deconstruction session."""
    session_id = str(uuid.uuid4())[:8]
    session_path = SESSIONS_DIR / session_id
    session_path.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "uploaded_track.wav"
    ext = Path(filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".aiff"}:
        ext = ".wav"

    orig_dest = session_path / f"original{ext}"
    with open(orig_dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "session_id": session_id,
        "filename": filename,
        "size_bytes": orig_dest.stat().st_size,
    }


@app.post("/api/analyze/{session_id}")
async def analyze_track(session_id: str) -> Dict[str, Any]:
    """Analyze uploaded audio, separate stems, decode SMR, and package for FL Studio."""
    session_path = SESSIONS_DIR / session_id
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    orig_files = list(session_path.glob("original.*"))
    if not orig_files:
        raise HTTPException(status_code=400, detail="Original audio file missing in session")

    orig_file = orig_files[0]
    smr_doc = pipeline.analyze(orig_file)

    # 1. Save SMR JSON
    smr_path = session_path / "representation.smr.json"
    SMRSerializer.save_json(smr_doc, smr_path)

    # 2. Export MIDI
    midi_path = session_path / "reconstructed.mid"
    pipeline.export_midi(smr_doc, midi_path)

    # 3. Render Procedural Reconstruction from SMR
    recon_wav = session_path / "reconstructed.wav"
    pipeline.render_audio(smr_doc, recon_wav)

    # 4. Separate Audio Stems (Drums, Bass, Melody/Samples, Vocals)
    stems_dir = session_path / "stems"
    stems_dict = stem_separator.separate(orig_file, stems_dir)

    # 5. Build Producer FL Studio Package (.zip)
    fl_zip = session_path / "fl_studio_beat_pack.zip"
    FLStudioExporter.create_fl_studio_package(smr_doc, stems_dict, fl_zip)

    # 6. Objective Comparison
    report = comparator.compare(orig_file, recon_wav)
    report_dict = {
        "composite_fidelity_score": report.composite_fidelity_score,
        "harmonic_alignment": report.harmonic_alignment,
        "envelope_correlation": report.envelope_correlation,
        "tempo_error_bpm": report.tempo_error_bpm,
        "metrics": [
            {
                "name": m.name,
                "original": m.original_value,
                "reconstructed": m.reconstructed_value,
                "score": m.alignment_score,
                "unit": m.unit,
                "description": m.description,
            }
            for m in report.metrics
        ],
        "capabilities": report.reproduced_capabilities,
        "limitations": report.unreproduced_limitations,
    }

    report_path = session_path / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    return {
        "status": "success",
        "session_id": session_id,
        "summary": {
            "duration": smr_doc.metadata.duration_seconds,
            "tempo_bpm": smr_doc.rhythm.estimated_bpm,
            "meter": smr_doc.rhythm.time_signature_estimate,
            "key": f"{smr_doc.harmony.primary_key} {smr_doc.harmony.mode}",
            "event_count": len(smr_doc.events),
            "sections": [
                {"id": s.section_id, "type": s.section_type, "start": s.start_time, "end": s.end_time}
                for s in smr_doc.structure
            ],
            "fidelity_score": report.composite_fidelity_score,
        },
        "stems": list(stems_dict.keys()),
        "smr": smr_doc.model_dump(),
        "comparison": report_dict,
    }


@app.get("/api/smr/{session_id}")
async def get_smr(session_id: str) -> Dict[str, Any]:
    """Retrieve full SMR document for a session."""
    smr_path = SESSIONS_DIR / session_id / "representation.smr.json"
    if not smr_path.exists():
        raise HTTPException(status_code=404, detail="SMR not found for session")
    with open(smr_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/audio/{session_id}/original")
async def get_original_audio(session_id: str) -> FileResponse:
    """Stream original audio."""
    session_path = SESSIONS_DIR / session_id
    files = list(session_path.glob("original.*"))
    if not files:
        raise HTTPException(status_code=404, detail="Original audio not found")
    return FileResponse(files[0], media_type="audio/wav")


@app.get("/api/audio/{session_id}/reconstructed")
async def get_reconstructed_audio(session_id: str) -> FileResponse:
    """Stream reconstructed audio synthesized strictly from SMR."""
    recon_path = SESSIONS_DIR / session_id / "reconstructed.wav"
    if not recon_path.exists():
        raise HTTPException(status_code=404, detail="Reconstructed audio not found")
    return FileResponse(recon_path, media_type="audio/wav")


@app.get("/api/audio/{session_id}/stem/{stem_name}")
async def get_stem_audio(session_id: str, stem_name: str) -> FileResponse:
    """Stream isolated audio stem (drums, bass, melody_samples, vocals)."""
    stem_path = SESSIONS_DIR / session_id / "stems" / f"{stem_name}.wav"
    if not stem_path.exists():
        raise HTTPException(status_code=404, detail=f"Stem {stem_name} not found")
    return FileResponse(stem_path, media_type="audio/wav", filename=f"{stem_name}.wav")


@app.get("/api/audio/{session_id}/midi")
async def get_midi(session_id: str) -> FileResponse:
    """Stream master MIDI."""
    midi_path = SESSIONS_DIR / session_id / "reconstructed.mid"
    if not midi_path.exists():
        raise HTTPException(status_code=404, detail="MIDI not found")
    return FileResponse(midi_path, media_type="audio/midi", filename="reconstructed.mid")


@app.get("/api/download/smr/{session_id}")
async def download_smr_file(session_id: str) -> FileResponse:
    """Download SMR JSON file."""
    smr_path = SESSIONS_DIR / session_id / "representation.smr.json"
    if not smr_path.exists():
        raise HTTPException(status_code=404, detail="SMR not found")
    return FileResponse(
        smr_path,
        media_type="application/json",
        filename=f"sonara_{session_id}.smr.json",
    )


@app.get("/api/download/fl_studio_pack/{session_id}")
async def download_fl_studio_pack(session_id: str) -> FileResponse:
    """Download complete FL Studio Producer Beat Pack (.zip)."""
    pack_path = SESSIONS_DIR / session_id / "fl_studio_beat_pack.zip"
    if not pack_path.exists():
        raise HTTPException(status_code=404, detail="FL Studio Beat Pack not found")
    return FileResponse(
        pack_path,
        media_type="application/zip",
        filename=f"Sonara_BeatPack_{session_id}.zip",
    )


UI_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sonara: Beat Deconstruction & Producer Workbench</title>
    <style>
        :root {
            --bg: #090d16;
            --surface: #111726;
            --surface-border: #1e293b;
            --primary: #38bdf8;
            --accent: #a855f7;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text: #f8fafc;
            --text-dim: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
        .container { max-width: 1280px; margin: 0 auto; }
        header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--surface-border); }
        .logo { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 38px; height: 38px; background: linear-gradient(135deg, var(--primary), var(--accent)); border-radius: 8px; display: grid; place-items: center; font-weight: 800; font-size: 22px; color: #000; }
        h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
        .badge { background: rgba(56, 189, 248, 0.15); color: var(--primary); padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700; border: 1px solid var(--primary); text-transform: uppercase; }
        
        .grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 24px; margin-bottom: 24px; }
        @media (max-width: 1024px) { .grid { grid-template-columns: 1fr; } }
        
        .card { background: var(--surface); border: 1px solid var(--surface-border); border-radius: 12px; padding: 20px; }
        .card-title { font-size: 15px; font-weight: 700; margin-bottom: 16px; color: var(--text); display: flex; align-items: center; gap: 8px; }
        
        .upload-box { border: 2px dashed var(--surface-border); border-radius: 10px; padding: 32px 20px; text-align: center; cursor: pointer; transition: 0.2s; background: rgba(15, 23, 42, 0.5); }
        .upload-box:hover { border-color: var(--primary); background: rgba(56, 189, 248, 0.05); }
        .upload-icon { font-size: 36px; margin-bottom: 10px; color: var(--primary); }
        .btn { background: var(--primary); color: #000; border: none; padding: 9px 16px; border-radius: 6px; font-weight: 700; font-size: 13px; cursor: pointer; transition: 0.2s; display: inline-flex; align-items: center; gap: 6px; }
        .btn:hover { background: #7dd3fc; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-secondary { background: var(--surface-border); color: var(--text); }
        .btn-secondary:hover { background: #334155; }
        .btn-pack { background: linear-gradient(135deg, #10b981, #059669); color: #fff; border: 1px solid #34d399; }
        .btn-pack:hover { background: #10b981; }

        .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
        .stat-box { background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid var(--surface-border); text-align: center; }
        .stat-label { font-size: 11px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 4px; font-weight: 600; }
        .stat-val { font-size: 18px; font-weight: 800; color: var(--primary); }

        /* Multi-Track Producer Console */
        .track-strip { background: rgba(15, 23, 42, 0.7); border: 1px solid var(--surface-border); border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; display: grid; grid-template-columns: 160px 1fr 180px; align-items: center; gap: 12px; }
        @media (max-width: 768px) { .track-strip { grid-template-columns: 1fr; } }
        .track-info { display: flex; align-items: center; gap: 10px; }
        .track-dot { width: 10px; height: 10px; border-radius: 50%; }
        .track-name { font-size: 14px; font-weight: 700; color: var(--text); }
        .track-desc { font-size: 11px; color: var(--text-dim); }
        .track-controls { display: flex; gap: 6px; align-items: center; justify-content: flex-end; }
        .btn-sm { padding: 4px 8px; font-size: 11px; font-weight: 700; border-radius: 4px; cursor: pointer; border: 1px solid transparent; }
        .btn-mute { background: #1e293b; color: var(--text-dim); }
        .btn-mute.active { background: var(--danger); color: #fff; }
        .btn-solo { background: #1e293b; color: var(--text-dim); }
        .btn-solo.active { background: var(--warning); color: #000; }
        
        audio { width: 100%; height: 32px; border-radius: 4px; }
        
        /* Timeline */
        .timeline { display: flex; height: 28px; border-radius: 6px; overflow: hidden; background: #1e293b; margin-bottom: 16px; }
        .timeline-seg { display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: #fff; padding: 0 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .seg-intro { background: #6366f1; }
        .seg-verse { background: #3b82f6; }
        .seg-chorus { background: #ec4899; }
        .seg-bridge { background: #8b5cf6; }
        .seg-outro { background: #10b981; }
        .seg-body { background: #0ea5e9; }

        pre { background: #070a11; border: 1px solid var(--surface-border); border-radius: 8px; padding: 12px; font-size: 11px; overflow-x: auto; max-height: 260px; color: #38bdf8; }
        .metric-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px; }
        .metric-table th, .metric-table td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--surface-border); }
        .metric-table th { color: var(--text-dim); font-size: 10px; text-transform: uppercase; }
        
        .tab-bar { display: flex; gap: 8px; margin-bottom: 12px; border-bottom: 1px solid var(--surface-border); padding-bottom: 8px; }
        .tab { background: transparent; border: none; color: var(--text-dim); font-weight: 700; font-size: 12px; cursor: pointer; padding: 6px 10px; border-radius: 6px; }
        .tab.active { background: rgba(56, 189, 248, 0.15); color: var(--primary); }
        .hidden { display: none !important; }
        .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(0,0,0,0.3); border-top-color: #000; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="container">
    <header>
        <div class="logo">
            <div class="logo-icon">S</div>
            <div>
                <h1>Sonara Beat Deconstruction Workbench</h1>
                <p style="font-size: 12px; color: var(--text-dim);">Acoustic Reverse-Engineering &bull; Stem Isolation &bull; SMR v1.0.0 &bull; FL Studio Export</p>
            </div>
        </div>
        <span class="badge">Producer Mode</span>
    </header>

    <!-- Upload Card -->
    <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">1. Upload Track to Deconstruct</div>
        <div class="upload-box" id="drop-zone" onclick="document.getElementById('file-input').click()">
            <div class="upload-icon">&uarr;</div>
            <div style="font-size: 14px; font-weight: 700; margin-bottom: 4px;" id="upload-text">Upload any track (e.g. Kanye West, Drake, Metro Boomin)</div>
            <div style="font-size: 12px; color: var(--text-dim);">Supported: WAV, MP3, FLAC, OGG &bull; Full song analysis & multi-track separation</div>
            <input type="file" id="file-input" style="display: none;" accept=".wav,.mp3,.flac,.ogg">
        </div>
        <div style="margin-top: 14px; display: flex; justify-content: space-between; align-items: center;">
            <span id="file-info" style="font-size: 13px; color: var(--text-dim);">No file selected.</span>
            <button class="btn" id="analyze-btn" disabled onclick="startAnalysis()">
                <span id="btn-spinner" class="spinner hidden"></span>
                <span id="btn-label">Deconstruct Beat</span>
            </button>
        </div>
    </div>

    <!-- Results Panel -->
    <div id="results-panel" class="hidden">
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Master Tempo</div>
                <div class="stat-val" id="stat-bpm">- BPM</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Tonal Key</div>
                <div class="stat-val" id="stat-key">-</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Extracted Events</div>
                <div class="stat-val" id="stat-events">-</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Reconstruction Fidelity</div>
                <div class="stat-val" id="stat-score" style="color: var(--success);">- %</div>
            </div>
        </div>

        <div class="grid">
            <!-- Left: Multi-Track Console & FL Studio Export -->
            <div class="card">
                <div class="card-title">
                    <span>2. Isolated Parts & Stem Console</span>
                    <a id="btn-fl-pack" class="btn btn-pack" style="font-size: 12px; margin-left: auto;" download>
                        Download FL Studio Beat Pack (.zip)
                    </a>
                </div>

                <!-- Track: Drums -->
                <div class="track-strip">
                    <div class="track-info">
                        <div class="track-dot" style="background: #ef4444;"></div>
                        <div>
                            <div class="track-name">Drums</div>
                            <div class="track-desc">Kick, Snare, Hi-Hats</div>
                        </div>
                    </div>
                    <audio id="audio-drums" controls></audio>
                    <div class="track-controls">
                        <button class="btn-sm btn-solo" onclick="soloTrack('drums')">S</button>
                        <button class="btn-sm btn-mute" onclick="muteTrack('drums')">M</button>
                        <a id="dl-stem-drums" class="btn-sm btn-secondary" download>WAV</a>
                        <a id="dl-midi-drums" class="btn-sm btn-secondary" download>MIDI</a>
                    </div>
                </div>

                <!-- Track: Bass -->
                <div class="track-strip">
                    <div class="track-info">
                        <div class="track-dot" style="background: #3b82f6;"></div>
                        <div>
                            <div class="track-name">Bass / 808</div>
                            <div class="track-desc">Sub-Bass & Pitch Contours</div>
                        </div>
                    </div>
                    <audio id="audio-bass" controls></audio>
                    <div class="track-controls">
                        <button class="btn-sm btn-solo" onclick="soloTrack('bass')">S</button>
                        <button class="btn-sm btn-mute" onclick="muteTrack('bass')">M</button>
                        <a id="dl-stem-bass" class="btn-sm btn-secondary" download>WAV</a>
                        <a id="dl-midi-bass" class="btn-sm btn-secondary" download>MIDI</a>
                    </div>
                </div>

                <!-- Track: Chords / Harmony -->
                <div class="track-strip">
                    <div class="track-info">
                        <div class="track-dot" style="background: #a855f7;"></div>
                        <div>
                            <div class="track-name">Harmony</div>
                            <div class="track-desc">Chords & Progressions</div>
                        </div>
                    </div>
                    <audio id="audio-recon-mix" controls></audio>
                    <div class="track-controls">
                        <button class="btn-sm btn-solo" onclick="soloTrack('recon-mix')">S</button>
                        <button class="btn-sm btn-mute" onclick="muteTrack('recon-mix')">M</button>
                        <a id="dl-stem-chords" class="btn-sm btn-secondary" download>WAV</a>
                        <a id="dl-midi-chords" class="btn-sm btn-secondary" download>MIDI</a>
                    </div>
                </div>

                <!-- Track: Melody / Chopped Samples -->
                <div class="track-strip">
                    <div class="track-info">
                        <div class="track-dot" style="background: #10b981;"></div>
                        <div>
                            <div class="track-name">Melody / Chops</div>
                            <div class="track-desc">Sample Chops & Leads</div>
                        </div>
                    </div>
                    <audio id="audio-melody" controls></audio>
                    <div class="track-controls">
                        <button class="btn-sm btn-solo" onclick="soloTrack('melody')">S</button>
                        <button class="btn-sm btn-mute" onclick="muteTrack('melody')">M</button>
                        <a id="dl-stem-melody" class="btn-sm btn-secondary" download>WAV</a>
                        <a id="dl-midi-melody" class="btn-sm btn-secondary" download>MIDI</a>
                    </div>
                </div>

                <!-- Original Master Reference -->
                <div style="margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--surface-border);">
                    <div style="font-size: 13px; font-weight: 700; margin-bottom: 8px; display: flex; justify-content: space-between;">
                        <span>Original Track (Full Mix Reference)</span>
                        <a id="dl-orig" style="color: var(--primary); font-size: 12px; text-decoration: none;" download>Download Original</a>
                    </div>
                    <audio id="audio-orig" controls></audio>
                </div>
            </div>

            <!-- Right: SMR Musical Structure & Inspection -->
            <div class="card">
                <div class="card-title">
                    <span>3. Musical Architecture & SMR Data</span>
                    <a id="dl-smr" class="btn btn-secondary" style="font-size: 11px; padding: 4px 8px; margin-left: auto;" download>Save SMR JSON</a>
                </div>

                <div style="font-size: 11px; font-weight: 700; color: var(--text-dim); margin-bottom: 6px; text-transform: uppercase;">Arrangement Progression</div>
                <div class="timeline" id="struct-timeline"></div>

                <div class="tab-bar">
                    <button class="tab active" onclick="switchTab('audit')">Extraction Fidelity</button>
                    <button class="tab" onclick="switchTab('json')">SMR v1.0.0 JSON</button>
                </div>

                <div id="tab-audit">
                    <div style="font-size: 12px; font-weight: 700; color: var(--success); margin-bottom: 4px;">&check; Recovered Musical Content:</div>
                    <ul id="list-caps" style="font-size: 11px; color: var(--text-dim); margin-left: 18px; margin-bottom: 14px; line-height: 1.5;"></ul>
                    
                    <div style="font-size: 12px; font-weight: 700; color: var(--warning); margin-bottom: 4px;">&#9888; Extraction Approximations & Sound Design Gaps:</div>
                    <ul id="list-lims" style="font-size: 11px; color: var(--text-dim); margin-left: 18px; margin-bottom: 14px; line-height: 1.5;"></ul>

                    <table class="metric-table" id="metrics-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Original</th>
                                <th>Reconstructed</th>
                                <th>Alignment</th>
                            </tr>
                        </thead>
                        <tbody id="metrics-tbody"></tbody>
                    </table>
                </div>

                <div id="tab-json" class="hidden">
                    <pre><code id="smr-code">// SMR JSON will populate here</code></pre>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let currentSessionId = null;
    let selectedFile = null;
    const tracks = ['drums', 'bass', 'recon-mix', 'melody'];

    const fileInput = document.getElementById('file-input');
    const uploadText = document.getElementById('upload-text');
    const fileInfo = document.getElementById('file-info');
    const analyzeBtn = document.getElementById('analyze-btn');

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#38bdf8'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#1e293b'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#1e293b';
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        selectedFile = file;
        uploadText.innerText = file.name;
        fileInfo.innerText = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        analyzeBtn.disabled = false;
    }

    async function startAnalysis() {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        document.getElementById('btn-spinner').classList.remove('hidden');
        document.getElementById('btn-label').innerText = 'Deconstructing Beat...';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            const upRes = await fetch('/api/upload', { method: 'POST', body: formData });
            const upData = await upRes.json();
            currentSessionId = upData.session_id;

            const anRes = await fetch(`/api/analyze/${currentSessionId}`, { method: 'POST' });
            const result = await anRes.json();
            renderResults(result);

        } catch (err) {
            alert('Deconstruction failed: ' + err.message);
        } finally {
            analyzeBtn.disabled = false;
            document.getElementById('btn-spinner').classList.add('hidden');
            document.getElementById('btn-label').innerText = 'Deconstruct Beat';
        }
    }

    function renderResults(data) {
        const sum = data.summary;
        const comp = data.comparison;

        document.getElementById('stat-bpm').innerText = `${sum.tempo_bpm} BPM`;
        document.getElementById('stat-key').innerText = sum.key;
        document.getElementById('stat-events').innerText = sum.event_count;
        document.getElementById('stat-score').innerText = `${comp.composite_fidelity_score}%`;

        // Wire Audio Stems
        document.getElementById('audio-drums').src = `/api/audio/${currentSessionId}/stem/drums`;
        document.getElementById('audio-bass').src = `/api/audio/${currentSessionId}/stem/bass`;
        document.getElementById('audio-recon-mix').src = `/api/audio/${currentSessionId}/reconstructed`;
        document.getElementById('audio-melody').src = `/api/audio/${currentSessionId}/stem/melody_samples`;
        document.getElementById('audio-orig').src = `/api/audio/${currentSessionId}/original`;

        // Wire Download Links
        document.getElementById('dl-stem-drums').href = `/api/audio/${currentSessionId}/stem/drums`;
        document.getElementById('dl-stem-bass').href = `/api/audio/${currentSessionId}/stem/bass`;
        document.getElementById('dl-stem-chords').href = `/api/audio/${currentSessionId}/reconstructed`;
        document.getElementById('dl-stem-melody').href = `/api/audio/${currentSessionId}/stem/melody_samples`;
        document.getElementById('dl-orig').href = `/api/audio/${currentSessionId}/original`;

        // Wire MIDI Links
        document.getElementById('dl-midi-drums').href = `/api/audio/${currentSessionId}/midi`;
        document.getElementById('dl-midi-bass').href = `/api/audio/${currentSessionId}/midi`;
        document.getElementById('dl-midi-chords').href = `/api/audio/${currentSessionId}/midi`;
        document.getElementById('dl-midi-melody').href = `/api/audio/${currentSessionId}/midi`;

        document.getElementById('btn-fl-pack').href = `/api/download/fl_studio_pack/${currentSessionId}`;
        document.getElementById('dl-smr').href = `/api/download/smr/${currentSessionId}`;

        // Timeline
        const tline = document.getElementById('struct-timeline');
        tline.innerHTML = '';
        const totalDur = sum.duration || 1.0;
        sum.sections.forEach(s => {
            const seg = document.createElement('div');
            const pct = Math.max(6, ((s.end - s.start) / totalDur) * 100);
            seg.className = `timeline-seg seg-${s.type.toLowerCase()}`;
            seg.style.width = `${pct}%`;
            seg.innerText = `${s.type.toUpperCase()}`;
            tline.appendChild(seg);
        });

        // Metrics Table
        const tbody = document.getElementById('metrics-tbody');
        tbody.innerHTML = '';
        comp.metrics.forEach(m => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${m.name}</strong></td>
                <td>${m.original} ${m.unit}</td>
                <td>${m.reconstructed} ${m.unit}</td>
                <td><span style="font-weight: bold; color: ${m.score >= 0.7 ? '#10b981' : '#f59e0b'};">${(m.score * 100).toFixed(1)}%</span></td>
            `;
            tbody.appendChild(tr);
        });

        // Capabilities & Limitations
        document.getElementById('list-caps').innerHTML = comp.capabilities.map(c => `<li>${c}</li>`).join('');
        document.getElementById('list-lims').innerHTML = comp.limitations.map(l => `<li>${l}</li>`).join('');

        // JSON block
        document.getElementById('smr-code').innerText = JSON.stringify(data.smr, null, 2);

        document.getElementById('results-panel').classList.remove('hidden');
        document.getElementById('results-panel').scrollIntoView({ behavior: 'smooth' });
    }

    function soloTrack(target) {
        tracks.forEach(t => {
            const aud = document.getElementById(`audio-${t}`);
            aud.muted = (t !== target);
        });
    }

    function muteTrack(target) {
        const aud = document.getElementById(`audio-${target}`);
        aud.muted = !aud.muted;
    }

    function switchTab(tab) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        if (tab === 'json') {
            document.querySelectorAll('.tab')[1].classList.add('active');
            document.getElementById('tab-json').classList.remove('hidden');
            document.getElementById('tab-audit').classList.add('hidden');
        } else {
            document.querySelectorAll('.tab')[0].classList.add('active');
            document.getElementById('tab-json').classList.add('hidden');
            document.getElementById('tab-audit').classList.remove('hidden');
        }
    }
</script>
</body>
</html>
"""
