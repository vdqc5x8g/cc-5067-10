from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .solver import estimate_local_period


def _shift_score(
    primary: Sequence[float],
    secondary: Sequence[float],
    threshold: float,
    shift: int,
    lo: int,
    hi: int,
    level_tolerance: float,
) -> Optional[Dict[str, float | int]]:
    n = len(primary)
    m = len(secondary)
    pp = ss = ps = 0.0
    count = 0
    for i in range(max(0, lo), min(n, hi)):
        if abs(float(primary[i])) >= threshold - level_tolerance:
            continue
        j = i + shift
        if j < 0 or j >= m:
            continue
        p = float(primary[i])
        s = float(secondary[j])
        pp += p * p
        ss += s * s
        ps += p * s
        count += 1
    if count < 48 or pp <= 1e-12 or ss <= 1e-12:
        return None
    gain = ps / ss
    ncc = ps / math.sqrt(pp * ss)
    residual = 0.0
    for i in range(max(0, lo), min(n, hi)):
        if abs(float(primary[i])) >= threshold - level_tolerance:
            continue
        j = i + shift
        if j < 0 or j >= m:
            continue
        d = float(primary[i]) - gain * float(secondary[j])
        residual += d * d
    rmse = math.sqrt(residual / count)
    return {
        "shift": shift,
        "ncc": ncc,
        "gain": gain,
        "rmse": rmse,
        "pairs": count,
    }


def align_secondary(
    primary: Sequence[float],
    secondary: Sequence[float],
    threshold: float,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    sample_rate: int = 8000,
    max_shift: int = 400,
    radius: int = 420,
    level_tolerance: float = 2e-5,
    ambiguity_margin: float = 0.025,
    min_ncc: float = 0.72,
) -> Dict[str, Any]:
    n = len(primary)
    center = n // 2 if start is None else (int(start) + (int(end) if end is not None else int(start))) // 2
    lo = max(0, center - radius)
    hi = min(n, center + radius)
    scores: List[Dict[str, float | int]] = []
    for shift in range(-max_shift, max_shift + 1):
        item = _shift_score(
            primary, secondary, threshold, shift, lo, hi, level_tolerance
        )
        if item is not None:
            scores.append(item)
    if not scores:
        return {
            "status": "not_identified",
            "reason": "overlap_or_unclipped_pairs_insufficient",
            "shift": None,
            "certified": False,
            "candidates": [],
        }

    by_ncc = sorted(scores, key=lambda d: float(d["ncc"]), reverse=True)
    top: List[Dict[str, float | int]] = []
    for item in by_ncc:
        shift = int(item["shift"])
        if all(abs(shift - int(c["shift"])) > 6 for c in top):
            top.append(item)
        if len(top) >= 8:
            break
    best = top[0]
    best_ncc = abs(float(best["ncc"]))
    period_info = estimate_local_period(primary, center, sample_rate)
    period = period_info.get("period_samples")

    ambiguous_with: Optional[Dict[str, float | int]] = None
    for cand in top[1:]:
        if abs(abs(float(cand["ncc"])) - best_ncc) <= ambiguity_margin:
            if period is None or abs(abs(int(cand["shift"]) - int(best["shift"])) - int(period)) <= 4:
                ambiguous_with = cand
                break
    if ambiguous_with is None and len(top) > 1:
        second = top[1]
        if abs(abs(float(second["ncc"])) - best_ncc) <= ambiguity_margin * 0.5:
            ambiguous_with = second

    if best_ncc < min_ncc:
        status = "not_identified"
        certified = False
        reason = "best_correlation_too_low"
    elif ambiguous_with is not None:
        status = "ambiguous_delay"
        certified = False
        reason = "periodic_or_near_periodic_delay_alternatives"
    else:
        status = "identified"
        certified = True
        reason = "unique_correlation_peak"

    gain = float(best["gain"])
    aligned = [0.0] * n
    coverage = 0
    shift = int(best["shift"])
    for i in range(n):
        j = i + shift
        if 0 <= j < len(secondary):
            aligned[i] = gain * float(secondary[j])
            coverage += 1

    return {
        "status": status,
        "reason": reason,
        "shift": shift,
        "gain": gain,
        "certified": certified,
        "best": best,
        "ambiguous_with": ambiguous_with,
        "period": period_info,
        "candidates": top[:5],
        "aligned_template": aligned,
        "coverage": coverage / n,
        "analysis_window": [lo, hi],
    }


def forced_alignment(
    primary: Sequence[float],
    secondary: Sequence[float],
    threshold: float,
    shift: int,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    level_tolerance: float = 2e-5,
) -> Dict[str, Any]:
    n = len(primary)
    center = n // 2 if start is None else (int(start) + (int(end or start))) // 2
    item = _shift_score(
        primary,
        secondary,
        threshold,
        int(shift),
        max(0, center - 512),
        min(n, center + 512),
        level_tolerance,
    )
    gain = float(item["gain"]) if item else 1.0
    aligned = [0.0] * n
    valid = 0
    for i in range(n):
        j = i + int(shift)
        if 0 <= j < len(secondary):
            aligned[i] = gain * float(secondary[j])
            valid += 1
    return {
        "status": "forced_unverified",
        "reason": "caller_forced_shift_peak_is_not_certified",
        "shift": int(shift),
        "gain": gain,
        "certified": False,
        "best": item,
        "candidates": [item] if item else [],
        "aligned_template": aligned,
        "coverage": valid / n,
    }
