from __future__ import annotations

import math
from typing import Any, Dict, Sequence

from .linalgutil import fft_radix2


def rmse(a: Sequence[float], b: Sequence[float], indices: Sequence[int] | None = None) -> float:
    if indices is None:
        indices = range(min(len(a), len(b)))  # type: ignore[assignment]
    vals = list(indices)
    if not vals:
        return 0.0
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in vals) / len(vals))


def snr_db(clean: Sequence[float], estimate: Sequence[float]) -> float:
    signal = sum(float(x) * float(x) for x in clean)
    noise = sum((float(estimate[i]) - float(clean[i])) ** 2 for i in range(min(len(clean), len(estimate))))
    if noise <= 1e-20:
        return 120.0
    return 10.0 * math.log10(max(1e-20, signal) / noise)


def log_spectral_distance_db(
    clean: Sequence[float], restored: Sequence[float], sample_rate: int, max_frames: int = 48
) -> float:
    frame = 256
    hop = 128
    n = min(len(clean), len(restored))
    positions = list(range(0, max(1, n - frame), hop))[:max_frames]
    if not positions:
        return 0.0
    total = 0.0
    count = 0
    for start in positions:
        re_c = [0.0] * frame
        im_c = [0.0] * frame
        re_r = [0.0] * frame
        im_r = [0.0] * frame
        for k in range(frame):
            w = 0.5 - 0.5 * math.cos(2.0 * math.pi * k / (frame - 1))
            ic = start + k
            ir = start + k
            re_c[k] = (float(clean[ic]) if ic < len(clean) else 0.0) * w
            re_r[k] = (float(restored[ir]) if ir < len(restored) else 0.0) * w
        fft_radix2(re_c, im_c)
        fft_radix2(re_r, im_r)
        for k in range(1, frame // 2):
            mc = math.sqrt(re_c[k] ** 2 + im_c[k] ** 2) + 1e-9
            mr = math.sqrt(re_r[k] ** 2 + im_r[k] ** 2) + 1e-9
            d = 20.0 * math.log10(mr / mc)
            total += d * d
            count += 1
    return math.sqrt(total / max(1, count))


def evaluate_restoration(
    clean: Sequence[float],
    clipped: Sequence[float],
    restoration: Dict[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    restored = restoration["samples"]
    clipped_indices = [
        i for i, v in enumerate(clipped) if abs(float(v)) >= threshold - 2e-5
    ]
    repair_indices = []
    context_lo = len(clean)
    context_hi = 0
    for report in restoration.get("runs", []):
        run = report["run"]
        for i in range(int(run["start"]), int(run["end"]) + 1):
            repair_indices.append(i)
        context_lo = min(context_lo, max(0, int(run["start"]) - 96))
        context_hi = max(context_hi, min(len(clean), int(run["end"]) + 97))
    if not repair_indices:
        repair_indices = clipped_indices
        context_lo, context_hi = 0, len(clean)
    true_peak = max(abs(float(v)) for v in clean)
    restored_peak = max(abs(float(v)) for v in restored)
    repair_true_peak = max(abs(float(clean[i])) for i in repair_indices)
    repair_restored_peak = max(abs(float(restored[i])) for i in repair_indices)
    lo, hi = restoration.get("prior_peak_range", [threshold, None])
    covers = hi is not None and lo <= true_peak <= float(hi)
    region_covers = hi is not None and threshold <= repair_true_peak <= float(hi)
    clipped_rmse = rmse(clean, restored, repair_indices)
    all_clipped_rmse_after = rmse(clean, restored, clipped_indices)
    all_clipped_rmse_before = rmse(clean, clipped, clipped_indices)
    all_rmse = rmse(clean, restored)
    naive_clipped_rmse = rmse(clean, clipped, repair_indices)
    ctx_c = clean[context_lo:context_hi]
    ctx_r = restored[context_lo:context_hi]
    return {
        "true_peak": true_peak,
        "restored_peak": restored_peak,
        "peak_abs_error": abs(restored_peak - true_peak),
        "repair_region_true_peak": repair_true_peak,
        "repair_region_restored_peak": repair_restored_peak,
        "repair_region_peak_abs_error": abs(repair_restored_peak - repair_true_peak),
        "prior_peak_range": [lo, hi],
        "true_peak_covered_by_prior_range": bool(covers),
        "repair_region_covered_by_prior_range": bool(region_covers),
        "repair_rmse_before": naive_clipped_rmse,
        "repair_rmse_after": clipped_rmse,
        "repair_rmse_improvement_ratio": naive_clipped_rmse / max(1e-12, clipped_rmse),
        "repair_snr_db": snr_db([clean[i] for i in repair_indices], [restored[i] for i in repair_indices]),
        "all_clipped_rmse_before": all_clipped_rmse_before,
        "all_clipped_rmse_after": all_clipped_rmse_after,
        "clipped_sample_rmse_before": naive_clipped_rmse,
        "clipped_sample_rmse_after": clipped_rmse,
        "rmse_improvement_ratio": naive_clipped_rmse / max(1e-12, clipped_rmse),
        "all_sample_rmse": all_rmse,
        "snr_db": snr_db(clean, restored),
        "context_log_spectral_distance_db": log_spectral_distance_db(
            ctx_c, ctx_r, int(restoration.get("sample_rate", 8000))
        ),
        "log_spectral_distance_db": log_spectral_distance_db(
            clean, restored, int(restoration.get("sample_rate", 8000))
        ),
        "reclip": restoration["errors"],
        "clean_truth_used_by_solver": False,
    }
