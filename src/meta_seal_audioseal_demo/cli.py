from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

# On Windows CPU environments, AudioSeal's torch.compile path can require the
# MSVC C++ compiler. Disable it before importing audioseal.
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf
import torch
from audioseal import AudioSeal
from scipy.signal import resample_poly


SAMPLE_RATE = 16_000
MESSAGE_BITS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]


def build_speech_like_waveform(duration_s: float = 5.0, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Create deterministic, speech-like synthetic audio with pauses."""
    sample_count = int(duration_s * sample_rate)
    t = torch.arange(sample_count, dtype=torch.float32) / sample_rate

    fundamental = 170 + 35 * torch.sin(2 * math.pi * 0.8 * t)
    phase = 2 * math.pi * torch.cumsum(fundamental / sample_rate, dim=0)
    voiced = 0.18 * (
        torch.sin(phase)
        + 0.45 * torch.sin(2 * phase)
        + 0.25 * torch.sin(3 * phase)
    )

    envelope = (0.55 + 0.45 * torch.sin(2 * math.pi * 2.7 * t)).clamp(0, 1)
    pause_mask = ((t % 1.25) > 0.18).float() * ((t % 1.25) < 1.08).float()
    wav = voiced * envelope * pause_mask
    return wav.unsqueeze(0).unsqueeze(0)


def detect(detector: torch.nn.Module, wav: torch.Tensor) -> dict[str, Any]:
    score, message = detector.detect_watermark(wav)
    frame_scores, _ = detector(wav)
    positive = frame_scores[:, 1, :]
    return {
        "score": round(float(score), 6),
        "positive_frame_ratio_over_0_5": round(float((positive > 0.5).float().mean()), 6),
        "positive_frame_mean": round(float(positive.mean()), 6),
        "decoded_message": message.detach().cpu().numpy().astype(int).tolist()[0],
    }


def add_noise_for_snr(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(wav.shape, generator=generator, dtype=wav.dtype, device=wav.device)
    signal_power = torch.mean(wav**2)
    noise_power = torch.mean(noise**2).clamp_min(1e-12)
    scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return (wav + scale * noise).clamp(-1, 1)


def resample_roundtrip(wav: torch.Tensor, from_sr: int = SAMPLE_RATE, via_sr: int = 8_000) -> torch.Tensor:
    audio = wav.squeeze().detach().cpu().numpy()
    down = resample_poly(audio, via_sr, from_sr)
    up = resample_poly(down, from_sr, via_sr)
    up = up[: wav.shape[-1]]
    if up.shape[0] < wav.shape[-1]:
        up = np.pad(up, (0, wav.shape[-1] - up.shape[0]))
    return torch.from_numpy(up.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def tensor_to_wav(path: Path, wav: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.squeeze().detach().cpu().numpy(), SAMPLE_RATE)


def run(output_dir: Path) -> dict[str, Any]:
    torch.manual_seed(7)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = build_speech_like_waveform()
    message = torch.tensor([MESSAGE_BITS], dtype=torch.int64)

    with torch.inference_mode():
        generator = AudioSeal.load_generator("audioseal_wm_16bits").eval()
        detector = AudioSeal.load_detector("audioseal_detector_16bits").eval()

        watermark = generator.get_watermark(clean, sample_rate=SAMPLE_RATE, message=message)
        watermarked = (clean + watermark).clamp(-1, 1)

        noise = watermarked - clean
        snr_db = 10 * torch.log10(torch.mean(clean**2) / torch.mean(noise**2))

        noisy = add_noise_for_snr(watermarked, 30)
        resampled = resample_roundtrip(watermarked)
        cropped = watermarked[:, :, SAMPLE_RATE:-SAMPLE_RATE]

        results = {
            "environment": {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "no_torch_compile": os.environ.get("NO_TORCH_COMPILE"),
                "sample_rate": SAMPLE_RATE,
            },
            "message_bits": MESSAGE_BITS,
            "watermark_snr_db": round(float(snr_db), 3),
            "clean": detect(detector, clean),
            "watermarked": detect(detector, watermarked),
            "watermarked_plus_30db_noise": detect(detector, noisy),
            "watermarked_resampled_16k_to_8k_to_16k": detect(detector, resampled),
            "watermarked_center_crop_3s": detect(detector, cropped),
        }

    tensor_to_wav(output_dir / "clean.wav", clean)
    tensor_to_wav(output_dir / "watermarked.wav", watermarked)
    tensor_to_wav(output_dir / "watermarked_plus_30db_noise.wav", noisy)
    tensor_to_wav(output_dir / "watermarked_resampled.wav", resampled)

    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Meta Seal / AudioSeal watermarking trial.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    results = run(args.output_dir)
    print(json.dumps(results, indent=2))
