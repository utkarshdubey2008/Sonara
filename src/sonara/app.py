"""Interactive Web Application for Sonara Audio Analysis, SMR Inspection, and Reconstruction."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from sonara.analysis.comparator import ReconstructionComparator, ReconstructionReport
from sonara.pipeline import SonaraPipeline
from sonara.representation.serializer import SMRSerializer
from sonara.representation.schema import SMRDocument


SESSIONS_DIR = Path("/tmp/sonara_sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

pipeline = SonaraPipeline()
comparator = ReconstructionComparator()

app = FastAPI(
    title="Sonara Music Reconstruction Application",
    description="Analyze audio into SMR representation, inspect musical structure, and reconstruct music side-by-side.",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> str:
    """Return single-page web application interface."""
    return UI_HTML_TEMPLATE


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an audio file and initialize a reconstruction session."""
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
    """Analyze uploaded audio into SMR representation and cache results."""
    session_path = SESSIONS_DIR / session_id
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    orig_files = list(session_path.glob("original.*"))
    if not orig_files:
        raise HTTPException(status_code=400, detail="Original audio file missing in session")

    orig_file = orig_files[0]
    smr_doc = pipeline.analyze(orig_file)

    # Save SMR JSON
    smr_path = session_path / "representation.smr.json"
    SMRSerializer.save_json(smr_doc, smr_path)

    # Export MIDI
    midi_path = session_path / "reconstructed.mid"
    pipeline.export_midi(smr_doc, midi_path)

    # Render Reconstructed Audio from SMR
    recon_wav = session_path / "reconstructed.wav"
    pipeline.render_audio(smr_doc, recon_wav)

    # Compute Objective Comparison
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
    """Stream reconstructed audio synthesized from SMR."""
    recon_path = SESSIONS_DIR / session_id / "reconstructed.wav"
    if not recon_path.exists():
        raise HTTPException(status_code=404, detail="Reconstructed audio not found")
    return FileResponse(recon_path, media_type="audio/wav")


@app.get("/api/audio/{session_id}/midi")
async def get_midi(session_id: str) -> FileResponse:
    """Stream reconstructed MIDI."""
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
        filename=f"sonara_representation_{session_id}.smr.json",
    )


UI_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sonara: Music Representation & Reconstruction</title>
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #131a2a;
            --surface-border: #1e293b;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.25);
            --accent: #818cf8;
            --success: #34d399;
            --warning: #fbbf24;
            --text: #f1f5f9;
            --text-dim: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px; padding-bottom: 16px; border-bottom: 1px solid var(--surface-border); }
        .logo { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 36px; height: 36px; background: linear-gradient(135deg, var(--primary), var(--accent)); border-radius: 8px; display: grid; place-items: center; font-weight: bold; font-size: 20px; color: #000; }
        h1 { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
        .badge { background: rgba(56, 189, 248, 0.15); color: var(--primary); padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; border: 1px solid var(--primary); }
        
        /* Grid Layout */
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        
        .card { background: var(--surface); border: 1px solid var(--surface-border); border-radius: 12px; padding: 20px; }
        .card-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: var(--text); display: flex; align-items: center; gap: 8px; }
        
        /* Upload Area */
        .upload-box { border: 2px dashed var(--surface-border); border-radius: 10px; padding: 36px 20px; text-align: center; cursor: pointer; transition: 0.2s; background: rgba(15, 23, 42, 0.4); }
        .upload-box:hover { border-color: var(--primary); background: rgba(56, 189, 248, 0.05); }
        .upload-icon { font-size: 40px; margin-bottom: 12px; color: var(--primary); }
        .btn { background: var(--primary); color: #000; border: none; padding: 10px 18px; border-radius: 6px; font-weight: 600; font-size: 14px; cursor: pointer; transition: 0.2s; display: inline-flex; align-items: center; gap: 8px; }
        .btn:hover { background: #7dd3fc; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-secondary { background: var(--surface-border); color: var(--text); }
        .btn-secondary:hover { background: #334155; }
        
        /* Metric Badges & Cards */
        .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
        .stat-box { background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid var(--surface-border); text-align: center; }
        .stat-label { font-size: 11px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 4px; }
        .stat-val { font-size: 18px; font-weight: 700; color: var(--primary); }
        
        /* Timeline */
        .timeline { display: flex; height: 32px; border-radius: 6px; overflow: hidden; background: #1e293b; margin-bottom: 16px; }
        .timeline-seg { display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; color: #fff; padding: 0 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .seg-intro { background: #6366f1; }
        .seg-verse { background: #3b82f6; }
        .seg-chorus { background: #ec4899; }
        .seg-bridge { background: #8b5cf6; }
        .seg-outro { background: #10b981; }
        .seg-body { background: #0ea5e9; }

        /* A/B Comparison Player */
        .player-card { background: rgba(15, 23, 42, 0.7); border: 1px solid var(--surface-border); border-radius: 10px; padding: 16px; margin-bottom: 16px; }
        .player-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .player-label { font-size: 14px; font-weight: 600; color: var(--text); display: flex; align-items: center; gap: 8px; }
        audio { width: 100%; border-radius: 6px; height: 36px; outline: none; }
        
        /* Metric Table */
        .metric-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .metric-table th, .metric-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--surface-border); }
        .metric-table th { color: var(--text-dim); font-size: 11px; text-transform: uppercase; }
        .score-bar { height: 6px; border-radius: 3px; background: #334155; overflow: hidden; margin-top: 4px; }
        .score-fill { height: 100%; background: var(--success); }
        
        /* Code / JSON block */
        pre { background: #070a11; border: 1px solid var(--surface-border); border-radius: 8px; padding: 12px; font-size: 12px; overflow-x: auto; max-height: 280px; color: #38bdf8; }
        
        .tab-bar { display: flex; gap: 8px; margin-bottom: 12px; border-bottom: 1px solid var(--surface-border); padding-bottom: 8px; }
        .tab { background: transparent; border: none; color: var(--text-dim); font-weight: 600; font-size: 13px; cursor: pointer; padding: 6px 12px; border-radius: 6px; }
        .tab.active { background: rgba(56, 189, 248, 0.15); color: var(--primary); }
        
        .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(0,0,0,0.3); border-top-color: #000; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .hidden { display: none !important; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <div class="logo">
            <div class="logo-icon">S</div>
            <div>
                <h1>Sonara Reconstruction Engine</h1>
                <p style="font-size: 12px; color: var(--text-dim);">Acoustic Analysis &bull; SMR v1.0.0 &bull; Procedural Resynthesis</p>
            </div>
        </div>
        <span class="badge">SMR Verification Mode</span>
    </header>

    <!-- Top Workflow Card: Upload & Trigger -->
    <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">1. Audio Input & SMR Decoding</div>
        <div class="upload-box" id="drop-zone" onclick="document.getElementById('file-input').click()">
            <div class="upload-icon">&uarr;</div>
            <div style="font-size: 15px; font-weight: 600; margin-bottom: 4px;" id="upload-text">Click or drag & drop an audio file here</div>
            <div style="font-size: 12px; color: var(--text-dim);">Supports WAV, MP3, FLAC, OGG (Maximum 10 MB recommended)</div>
            <input type="file" id="file-input" style="display: none;" accept=".wav,.mp3,.flac,.ogg">
        </div>
        <div style="margin-top: 16px; display: flex; justify-content: space-between; align-items: center;">
            <span id="file-info" style="font-size: 13px; color: var(--text-dim);">No file selected.</span>
            <button class="btn" id="analyze-btn" disabled onclick="startAnalysis()">
                <span id="btn-spinner" class="spinner hidden"></span>
                <span id="btn-label">Analyze & Reconstruct</span>
            </button>
        </div>
    </div>

    <!-- Main Results (Hidden initially) -->
    <div id="results-panel" class="hidden">
        <!-- Overview Stats -->
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Estimated Tempo</div>
                <div class="stat-val" id="stat-bpm">- BPM</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Tonal Center</div>
                <div class="stat-val" id="stat-key">-</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Decoded Events</div>
                <div class="stat-val" id="stat-events">-</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Fidelity Score</div>
                <div class="stat-val" id="stat-score" style="color: var(--success);">- %</div>
            </div>
        </div>

        <div class="grid">
            <!-- Left Column: Audio A/B Side-by-Side Comparison -->
            <div class="card">
                <div class="card-title">
                    <span>2. Side-by-Side Audio Comparison</span>
                    <button class="btn btn-secondary" style="font-size: 12px; padding: 4px 10px; margin-left: auto;" onclick="toggleSyncPlay()">
                        Sync Play/Pause
                    </button>
                </div>

                <div class="player-card">
                    <div class="player-header">
                        <span class="player-label"><span style="color: var(--primary);">&bull;</span> Original Audio</span>
                        <a id="dl-orig" class="btn btn-secondary" style="font-size: 11px; padding: 2px 8px;" download>Download</a>
                    </div>
                    <audio id="audio-orig" controls></audio>
                </div>

                <div class="player-card">
                    <div class="player-header">
                        <span class="player-label"><span style="color: var(--success);">&bull;</span> Reconstructed Audio (Rendered from SMR)</span>
                        <div style="display: flex; gap: 6px;">
                            <a id="dl-midi" class="btn btn-secondary" style="font-size: 11px; padding: 2px 8px;" download>MIDI</a>
                            <a id="dl-recon" class="btn btn-secondary" style="font-size: 11px; padding: 2px 8px;" download>WAV</a>
                        </div>
                    </div>
                    <audio id="audio-recon" controls></audio>
                </div>

                <div style="font-size: 13px; font-weight: 600; margin: 16px 0 8px 0;">Objective Reconstruction Metrics</div>
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

            <!-- Right Column: SMR Representation & Structural Timeline -->
            <div class="card">
                <div class="card-title">
                    <span>3. Decoded SMR Representation</span>
                    <div style="margin-left: auto; display: flex; gap: 6px;">
                        <button class="btn btn-secondary" style="font-size: 11px; padding: 2px 8px;" onclick="copySMR()">Copy JSON</button>
                        <a id="dl-smr" class="btn btn-secondary" style="font-size: 11px; padding: 2px 8px;" download>Save .smr.json</a>
                    </div>
                </div>

                <div style="font-size: 12px; font-weight: 600; color: var(--text-dim); margin-bottom: 6px;">Structural Arrangement Timeline</div>
                <div class="timeline" id="struct-timeline"></div>

                <div class="tab-bar">
                    <button class="tab active" onclick="switchTab('json')">SMR JSON</button>
                    <button class="tab" onclick="switchTab('audit')">Capabilities & Limitations</button>
                </div>

                <div id="tab-json">
                    <pre><code id="smr-code">// SMR JSON will populate here</code></pre>
                </div>

                <div id="tab-audit" class="hidden">
                    <div style="font-size: 13px; font-weight: 600; color: var(--success); margin-bottom: 6px;">&check; Successfully Decoded from Signal:</div>
                    <ul id="list-caps" style="font-size: 12px; color: var(--text-dim); margin-left: 20px; margin-bottom: 16px; line-height: 1.6;"></ul>
                    
                    <div style="font-size: 13px; font-weight: 600; color: var(--warning); margin-bottom: 6px;">&#9888; Current Reconstruction Limitations:</div>
                    <ul id="list-lims" style="font-size: 12px; color: var(--text-dim); margin-left: 20px; line-height: 1.6;"></ul>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let currentSessionId = null;
    let selectedFile = null;
    let smrData = null;

    const fileInput = document.getElementById('file-input');
    const uploadText = document.getElementById('upload-text');
    const fileInfo = document.getElementById('file-info');
    const analyzeBtn = document.getElementById('analyze-btn');

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#38bdf8'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#1e293b'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#1e293b';
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
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
        document.getElementById('btn-label').innerText = 'Uploading...';

        try {
            // 1. Upload
            const formData = new FormData();
            formData.append('file', selectedFile);
            const upRes = await fetch('/api/upload', { method: 'POST', body: formData });
            const upData = await upRes.json();
            currentSessionId = upData.session_id;

            // 2. Analyze & Reconstruct
            document.getElementById('btn-label').innerText = 'Analyzing & Reconstructing...';
            const anRes = await fetch(`/api/analyze/${currentSessionId}`, { method: 'POST' });
            const result = await anRes.json();

            smrData = result.smr;
            renderResults(result);

        } catch (err) {
            alert('Analysis failed: ' + err.message);
        } finally {
            analyzeBtn.disabled = false;
            document.getElementById('btn-spinner').classList.add('hidden');
            document.getElementById('btn-label').innerText = 'Analyze & Reconstruct';
        }
    }

    function renderResults(data) {
        const sum = data.summary;
        const comp = data.comparison;

        document.getElementById('stat-bpm').innerText = `${sum.tempo_bpm} BPM`;
        document.getElementById('stat-key').innerText = sum.key;
        document.getElementById('stat-events').innerText = sum.event_count;
        document.getElementById('stat-score').innerText = `${comp.composite_fidelity_score}%`;

        // Audio sources
        const origAud = document.getElementById('audio-orig');
        const reconAud = document.getElementById('audio-recon');
        origAud.src = `/api/audio/${currentSessionId}/original`;
        reconAud.src = `/api/audio/${currentSessionId}/reconstructed`;

        document.getElementById('dl-orig').href = `/api/audio/${currentSessionId}/original`;
        document.getElementById('dl-recon').href = `/api/audio/${currentSessionId}/reconstructed`;
        document.getElementById('dl-midi').href = `/api/audio/${currentSessionId}/midi`;
        document.getElementById('dl-smr').href = `/api/download/smr/${currentSessionId}`;

        // Metrics Table
        const tbody = document.getElementById('metrics-tbody');
        tbody.innerHTML = '';
        comp.metrics.forEach(m => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${m.name}</strong><br><span style="font-size: 11px; color: var(--text-dim);">${m.description}</span></td>
                <td>${m.original} ${m.unit}</td>
                <td>${m.reconstructed} ${m.unit}</td>
                <td>
                    <span style="font-weight: bold; color: ${m.score >= 0.7 ? '#34d399' : '#fbbf24'};">${(m.score * 100).toFixed(1)}%</span>
                    <div class="score-bar"><div class="score-fill" style="width: ${m.score * 100}%;"></div></div>
                </td>
            `;
            tbody.appendChild(tr);
        });

        // Timeline
        const tline = document.getElementById('struct-timeline');
        tline.innerHTML = '';
        const totalDur = sum.duration || 1.0;
        sum.sections.forEach(s => {
            const seg = document.createElement('div');
            const pct = Math.max(8, ((s.end - s.start) / totalDur) * 100);
            seg.className = `timeline-seg seg-${s.type.toLowerCase()}`;
            seg.style.width = `${pct}%`;
            seg.innerText = `${s.type.toUpperCase()}`;
            tline.appendChild(seg);
        });

        // JSON block
        document.getElementById('smr-code').innerText = JSON.stringify(data.smr, null, 2);

        // Capabilities & Limitations
        const listCaps = document.getElementById('list-caps');
        listCaps.innerHTML = comp.capabilities.map(c => `<li>${c}</li>`).join('');
        const listLims = document.getElementById('list-lims');
        listLims.innerHTML = comp.limitations.map(l => `<li>${l}</li>`).join('');

        document.getElementById('results-panel').classList.remove('hidden');
        document.getElementById('results-panel').scrollIntoView({ behavior: 'smooth' });
    }

    function toggleSyncPlay() {
        const a1 = document.getElementById('audio-orig');
        const a2 = document.getElementById('audio-recon');
        if (a1.paused) {
            a1.currentTime = a2.currentTime = Math.min(a1.currentTime, a2.currentTime);
            a1.play();
            a2.play();
        } else {
            a1.pause();
            a2.pause();
        }
    }

    function switchTab(tab) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        if (tab === 'json') {
            document.querySelectorAll('.tab')[0].classList.add('active');
            document.getElementById('tab-json').classList.remove('hidden');
            document.getElementById('tab-audit').classList.add('hidden');
        } else {
            document.querySelectorAll('.tab')[1].classList.add('active');
            document.getElementById('tab-json').classList.add('hidden');
            document.getElementById('tab-audit').classList.remove('hidden');
        }
    }

    function copySMR() {
        if (!smrData) return;
        navigator.clipboard.writeText(JSON.stringify(smrData, null, 2));
        alert('SMR JSON copied to clipboard!');
    }
</script>
</body>
</html>
"""
