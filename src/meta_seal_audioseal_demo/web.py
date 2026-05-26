from __future__ import annotations

import argparse
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf
import torch
import uvicorn
from audioseal import AudioSeal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample_poly

from .cli import (
    MESSAGE_BITS,
    SAMPLE_RATE,
    add_noise_for_snr,
    build_speech_like_waveform,
    detect,
    resample_roundtrip,
    tensor_to_wav,
)


APP_DIR = Path(__file__).resolve().parents[2]
WEB_OUTPUT_DIR = APP_DIR / "outputs" / "web"
MAX_UPLOAD_SECONDS = 20
_GENERATOR: torch.nn.Module | None = None
_DETECTOR: torch.nn.Module | None = None


def app_factory() -> FastAPI:
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Meta Seal AudioSeal Web Demo")
    app.mount("/outputs", StaticFiles(directory=APP_DIR / "outputs"), name="outputs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.post("/api/sample")
    def sample() -> dict[str, Any]:
        clean = build_speech_like_waveform()
        return process_waveform(clean, "sample")

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Choose an audio file.")

        data = await file.read()
        try:
            wav = bytes_to_waveform(data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read audio: {exc}") from exc

        return process_waveform(wav, Path(file.filename).stem or "upload")

    return app


def get_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    global _GENERATOR, _DETECTOR
    if _GENERATOR is None:
        _GENERATOR = AudioSeal.load_generator("audioseal_wm_16bits").eval()
    if _DETECTOR is None:
        _DETECTOR = AudioSeal.load_detector("audioseal_detector_16bits").eval()
    return _GENERATOR, _DETECTOR


def bytes_to_waveform(data: bytes) -> torch.Tensor:
    audio, sample_rate = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)

    max_samples = sample_rate * MAX_UPLOAD_SECONDS
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sample_rate).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1:
        audio = audio / peak
    if audio.size < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - audio.size))

    return torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def process_waveform(clean: torch.Tensor, label: str) -> dict[str, Any]:
    run_id = f"{safe_stem(label)}-{uuid.uuid4().hex[:8]}"
    run_dir = WEB_OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    message = torch.tensor([MESSAGE_BITS], dtype=torch.int64)
    with torch.inference_mode():
        generator, detector = get_models()
        watermark = generator.get_watermark(clean, sample_rate=SAMPLE_RATE, message=message)
        watermarked = (clean + watermark).clamp(-1, 1)
        noisy = add_noise_for_snr(watermarked, 30)
        resampled = resample_roundtrip(watermarked)
        cropped = watermarked[:, :, SAMPLE_RATE:-SAMPLE_RATE] if clean.shape[-1] > SAMPLE_RATE * 2 else watermarked

        noise = watermarked - clean
        snr_db = 10 * torch.log10(torch.mean(clean**2).clamp_min(1e-12) / torch.mean(noise**2).clamp_min(1e-12))
        results = {
            "run_id": run_id,
            "duration_seconds": round(clean.shape[-1] / SAMPLE_RATE, 3),
            "message_bits": MESSAGE_BITS,
            "watermark_snr_db": round(float(snr_db), 3),
            "clean": detect(detector, clean),
            "watermarked": detect(detector, watermarked),
            "watermarked_plus_30db_noise": detect(detector, noisy),
            "watermarked_resampled_16k_to_8k_to_16k": detect(detector, resampled),
            "watermarked_center_crop": detect(detector, cropped),
        }

    tensor_to_wav(run_dir / "clean.wav", clean)
    tensor_to_wav(run_dir / "watermarked.wav", watermarked)
    tensor_to_wav(run_dir / "watermarked_plus_30db_noise.wav", noisy)
    tensor_to_wav(run_dir / "watermarked_resampled.wav", resampled)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    results["audio"] = {
        "clean": f"/outputs/web/{run_id}/clean.wav",
        "watermarked": f"/outputs/web/{run_id}/watermarked.wav",
        "watermarked_plus_30db_noise": f"/outputs/web/{run_id}/watermarked_plus_30db_noise.wav",
        "watermarked_resampled": f"/outputs/web/{run_id}/watermarked_resampled.wav",
        "results": f"/outputs/web/{run_id}/results.json",
    }
    return results


def safe_stem(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:40] or "audio"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Meta Seal / AudioSeal web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()
    uvicorn.run("meta_seal_audioseal_demo.web:app_factory", factory=True, host=args.host, port=args.port)


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meta Seal AudioSeal Demo</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dde6;
      --panel: #f7f8fa;
      --accent: #0b7285;
      --accent-strong: #09535f;
      --warn: #b7791f;
      --good: #177245;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #ffffff;
      color: var(--ink);
    }
    main {
      width: min(1160px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 18px;
    }
    h1 {
      font-size: 28px;
      line-height: 1.12;
      margin: 0 0 8px;
      letter-spacing: 0;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      max-width: 760px;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .panel h2 {
      font-size: 15px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    button, .file-label {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 10px 12px;
      font-weight: 650;
      cursor: pointer;
      min-height: 42px;
    }
    button.secondary, .file-label {
      background: white;
      color: var(--accent-strong);
    }
    button:disabled {
      opacity: .58;
      cursor: wait;
    }
    input[type="file"] {
      width: 1px;
      height: 1px;
      opacity: 0;
      position: absolute;
    }
    .filename {
      color: var(--muted);
      font-size: 14px;
      min-width: 160px;
      overflow-wrap: anywhere;
    }
    .status {
      padding: 11px 12px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      color: var(--muted);
      margin-bottom: 16px;
      min-height: 44px;
    }
    .results {
      display: grid;
      grid-template-columns: 1.05fr .95fr;
      gap: 16px;
      align-items: start;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: white;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      letter-spacing: 0;
      margin-bottom: 2px;
    }
    .metric span {
      color: var(--muted);
      font-size: 13px;
    }
    .score-list {
      display: grid;
      gap: 10px;
    }
    .score-row {
      display: grid;
      grid-template-columns: minmax(165px, 1fr) 4fr 70px;
      gap: 10px;
      align-items: center;
    }
    .bar {
      height: 12px;
      border-radius: 999px;
      background: #e8ebf0;
      overflow: hidden;
    }
    .bar span {
      display: block;
      height: 100%;
      width: 0%;
      background: var(--good);
    }
    .score-row[data-low="true"] .bar span { background: var(--warn); }
    audio {
      width: 100%;
      height: 42px;
    }
    .audio-grid {
      display: grid;
      gap: 12px;
    }
    .audio-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: white;
    }
    .audio-item h3 {
      font-size: 14px;
      margin: 0 0 8px;
    }
    .bits {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 12px;
    }
    .bit {
      width: 26px;
      height: 26px;
      display: inline-grid;
      place-items: center;
      border-radius: 5px;
      border: 1px solid var(--line);
      background: white;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    .bit.one {
      background: #dff3e9;
      border-color: #aad7c0;
    }
    pre {
      max-height: 280px;
      overflow: auto;
      background: #101828;
      color: #eef4ff;
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      line-height: 1.45;
    }
    .hidden { display: none; }

    @media (max-width: 820px) {
      header, .controls, .results, .metric-grid {
        grid-template-columns: 1fr;
      }
      .score-row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Meta Seal AudioSeal Demo</h1>
        <p>Embed an invisible watermark into audio, then verify how strongly the detector sees it after noise, resampling, and cropping.</p>
      </div>
      <button id="sampleBtn">Run sample audio</button>
    </header>

    <section class="controls">
      <div class="panel">
        <h2>Synthetic trial</h2>
        <p>Uses the deterministic five-second speech-like waveform from the repository.</p>
      </div>
      <div class="panel">
        <h2>Upload audio</h2>
        <form id="uploadForm" class="row">
          <label class="file-label" for="audioFile">Choose file</label>
          <input id="audioFile" name="file" type="file" accept="audio/*,.wav,.flac,.ogg">
          <span id="fileName" class="filename">No file selected</span>
          <button class="secondary" id="uploadBtn" type="submit">Watermark upload</button>
        </form>
      </div>
    </section>

    <div id="status" class="status">Ready. First run may take a few seconds while AudioSeal loads the model.</div>

    <section id="results" class="results hidden">
      <div class="panel">
        <h2>Detection</h2>
        <div class="metric-grid">
          <div class="metric"><strong id="scoreWatermarked">-</strong><span>watermarked score</span></div>
          <div class="metric"><strong id="scoreClean">-</strong><span>clean score</span></div>
          <div class="metric"><strong id="snr">-</strong><span>watermark SNR</span></div>
        </div>
        <div id="scoreList" class="score-list"></div>
        <div id="bits" class="bits"></div>
      </div>

      <div class="panel">
        <h2>Audio outputs</h2>
        <div id="audioGrid" class="audio-grid"></div>
        <pre id="jsonOut"></pre>
      </div>
    </section>
  </main>

  <script>
    const sampleBtn = document.getElementById("sampleBtn");
    const uploadForm = document.getElementById("uploadForm");
    const uploadBtn = document.getElementById("uploadBtn");
    const audioFile = document.getElementById("audioFile");
    const fileName = document.getElementById("fileName");
    const statusBox = document.getElementById("status");
    const resultsEl = document.getElementById("results");

    audioFile.addEventListener("change", () => {
      fileName.textContent = audioFile.files[0]?.name || "No file selected";
    });

    sampleBtn.addEventListener("click", async () => {
      await callApi("/api/sample", { method: "POST" });
    });

    uploadForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!audioFile.files[0]) {
        setStatus("Choose an audio file first.", true);
        return;
      }
      const form = new FormData();
      form.append("file", audioFile.files[0]);
      await callApi("/api/upload", { method: "POST", body: form });
    });

    async function callApi(url, options) {
      setBusy(true);
      setStatus("Running AudioSeal. This can take a moment on CPU.", false);
      try {
        const response = await fetch(url, options);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Request failed");
        render(data);
        setStatus(`Done. Run ID: ${data.run_id}`, false);
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        setBusy(false);
      }
    }

    function setBusy(value) {
      sampleBtn.disabled = value;
      uploadBtn.disabled = value;
    }

    function setStatus(message, isError) {
      statusBox.textContent = message;
      statusBox.style.borderColor = isError ? "#d92d20" : "";
      statusBox.style.color = isError ? "#b42318" : "";
    }

    function render(data) {
      resultsEl.classList.remove("hidden");
      document.getElementById("scoreWatermarked").textContent = fmt(data.watermarked.score);
      document.getElementById("scoreClean").textContent = fmt(data.clean.score);
      document.getElementById("snr").textContent = `${data.watermark_snr_db} dB`;

      const scoreList = document.getElementById("scoreList");
      scoreList.innerHTML = "";
      [
        ["Clean", data.clean.score],
        ["Watermarked", data.watermarked.score],
        ["+ 30 dB noise", data.watermarked_plus_30db_noise.score],
        ["Resampled", data.watermarked_resampled_16k_to_8k_to_16k.score],
        ["Center crop", data.watermarked_center_crop.score],
      ].forEach(([label, score]) => {
        const row = document.createElement("div");
        row.className = "score-row";
        row.dataset.low = score < 0.8;
        row.innerHTML = `<span>${label}</span><div class="bar"><span style="width:${Math.max(0, Math.min(1, score)) * 100}%"></span></div><strong>${fmt(score)}</strong>`;
        scoreList.appendChild(row);
      });

      const bits = document.getElementById("bits");
      bits.innerHTML = "";
      data.watermarked.decoded_message.forEach((bit) => {
        const item = document.createElement("span");
        item.className = `bit ${bit ? "one" : ""}`;
        item.textContent = bit;
        bits.appendChild(item);
      });

      const audioGrid = document.getElementById("audioGrid");
      audioGrid.innerHTML = "";
      [
        ["Clean input", data.audio.clean],
        ["Watermarked", data.audio.watermarked],
        ["Watermarked + noise", data.audio.watermarked_plus_30db_noise],
        ["Watermarked resampled", data.audio.watermarked_resampled],
      ].forEach(([label, src]) => {
        const item = document.createElement("div");
        item.className = "audio-item";
        item.innerHTML = `<h3>${label}</h3><audio controls src="${src}"></audio>`;
        audioGrid.appendChild(item);
      });

      document.getElementById("jsonOut").textContent = JSON.stringify(data, null, 2);
    }

    function fmt(value) {
      return Number(value).toFixed(3);
    }
  </script>
</body>
</html>
"""


app = app_factory()
