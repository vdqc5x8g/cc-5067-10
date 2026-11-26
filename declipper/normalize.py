from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from .audio import rms_db


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def gain_to_db(gain: float) -> float:
    return 20.0 * math.log10(max(1e-12, gain))


def plan_loudness_normalization(
    samples: Sequence[float],
    *,
    target_rms_db: float = -18.0,
    ceiling: float = 0.98,
    peak_estimate: Optional[float] = None,
    peak_upper_bound: Optional[float] = None,
    certified_peak: bool = False,
    policy: str = "guarded",
) -> Dict[str, Any]:
    """Plan RMS loudness normalization without treating an uncertified peak as headroom."""
    if ceiling <= 0:
        raise ValueError("ceiling 必须为正数")
    current_rms_db = rms_db(samples)
    loudness_gain = db_to_gain(target_rms_db - current_rms_db)
    candidate_peak = max(abs(float(x)) for x in samples) if samples else 0.0
    believed_peak = float(peak_estimate if peak_estimate is not None else candidate_peak)

    naive_ceiling_gain = ceiling / max(1e-12, believed_peak)
    naive_gain = min(loudness_gain, naive_ceiling_gain)
    believed_headroom_db = gain_to_db(ceiling / max(1e-12, believed_peak))

    if policy == "naive_point_estimate":
        gain = naive_gain
        return {
            "policy": policy,
            "input_rms_db": current_rms_db,
            "target_rms_db": target_rms_db,
            "loudness_gain": loudness_gain,
            "gain": gain,
            "gain_db": gain_to_db(gain),
            "peak_estimate_used": believed_peak,
            "believed_headroom_db": believed_headroom_db,
            "certified_headroom_db": believed_headroom_db,
            "uses_uncertain_peak_as_headroom": True,
            "peak_certified": False,
            "warning": "仅用于对照：把单点先验峰高误当成已证明的真实余量",
        }

    if certified_peak:
        upper = float(peak_upper_bound if peak_upper_bound is not None else believed_peak)
        certified_gain = ceiling / max(1e-12, upper)
        gain = min(loudness_gain, certified_gain)
        return {
            "policy": "guarded",
            "status": "certified_peak_bound",
            "input_rms_db": current_rms_db,
            "target_rms_db": target_rms_db,
            "loudness_gain": loudness_gain,
            "gain": gain,
            "gain_db": gain_to_db(gain),
            "peak_estimate_used": believed_peak,
            "certified_peak_upper": upper,
            "certified_headroom_db": gain_to_db(ceiling / max(1e-12, upper)),
            "uses_uncertain_peak_as_headroom": False,
            "peak_certified": True,
            "boost_blocked_db": 0.0,
        }

    # The candidate WAV itself has a measurable peak; attenuate if that actual
    # exported candidate would hit the ceiling. Never use a speculative low peak
    # to calculate positive makeup gain.
    candidate_safety_gain = min(1.0, ceiling / max(1e-12, candidate_peak))
    gain = min(1.0, loudness_gain, candidate_safety_gain)
    requested_boost_db = max(0.0, gain_to_db(loudness_gain))
    applied_boost_db = max(0.0, gain_to_db(gain))
    return {
        "policy": "guarded",
        "status": "peak_uncertain_boost_blocked",
        "input_rms_db": current_rms_db,
        "target_rms_db": target_rms_db,
        "loudness_gain": loudness_gain,
        "gain": gain,
        "gain_db": gain_to_db(gain),
        "candidate_peak": candidate_peak,
        "speculative_peak_estimate": believed_peak,
        "naive_believed_headroom_db": believed_headroom_db,
        "certified_headroom_db": None,
        "uses_uncertain_peak_as_headroom": False,
        "peak_certified": False,
        "requested_boost_db": requested_boost_db,
        "boost_blocked_db": max(0.0, requested_boost_db - applied_boost_db),
        "warning": "峰高未被观测或已认证副麦证明；禁止把先验峰高对应的余量用于正向增益",
    }


def apply_gain(samples: Sequence[float], gain: float) -> List[float]:
    return [float(x) * gain for x in samples]


def normalize_samples(
    samples: Sequence[float], **kwargs: Any
) -> tuple[List[float], Dict[str, Any]]:
    plan = plan_loudness_normalization(samples, **kwargs)
    return apply_gain(samples, float(plan["gain"])), plan
