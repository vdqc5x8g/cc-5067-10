"""Measured choice between the two ways of building practice material.

The brief asks us to pick between
  (A) clip a single known dry source, and
  (B) construct several originals that share an identical observation,
and to decide *by measurement*, not assertion. This module runs both at the
same threshold and reports, sample-by-sample, whether each method actually
produces identical observations. The result selects method B.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from .material import build_material, clip
from .spec import Spec


def _natural_variant(spec: Spec, *, body_peak: float, harmonic_scale: float,
                     phase_shift: float) -> List[float]:
    """A plainly different dry source: same generator family, different timbre."""
    n = spec.total_samples
    sr = spec.sample_rate
    b = spec.body
    out: List[float] = []
    for i in range(n):
        t = i / sr
        env = (min(1.0, t / b.attack_seconds) if b.attack_seconds > 0 else 1.0) * math.exp(-b.decay * t)
        v = 0.0
        for h in b.harmonics:
            v += (
                h.amplitude * harmonic_scale
                * (math.exp(-t * (h.decay - 1.0)) if h.decay > 0 else 1.0)
                * math.sin(
                    2.0 * math.pi * b.fundamental_hz * h.multiple * t
                    + h.phase + phase_shift * h.multiple
                )
            )
        out.append(b.amplitude * body_peak * env * v)
    peak = max((abs(v) for v in out), default=0.0) or 1.0
    return [v * b.amplitude * body_peak / peak for v in out]


def _identical_max_diff(a: List[float], b: List[float]) -> Dict[str, float | int]:
    n = min(len(a), len(b))
    diffs = [abs(a[i] - b[i]) for i in range(n)]
    return {
        "max_abs_diff": max(diffs, default=0.0),
        "differing_sample_count": sum(1 for d in diffs if d > 1e-12),
        "sample_count": n,
    }


def compare_construction_methods(spec: Spec) -> Dict[str, Any]:
    threshold = spec.clip_threshold
    sr = spec.sample_rate

    # Method A: two genuinely different natural dry sources, clipped.
    a = _natural_variant(spec, body_peak=1.0, harmonic_scale=1.0, phase_shift=0.0)
    b = _natural_variant(spec, body_peak=1.0, harmonic_scale=1.35, phase_shift=1.1)
    obs_a = clip(a, threshold)
    obs_b = clip(b, threshold)
    method_a = _identical_max_diff(obs_a, obs_b)

    # Method B: constructed originals sharing an observation by design.
    material = build_material(spec)
    ref = material.observations["T0"]
    worst = 0.0
    differing = 0
    for cand in material.candidates:
        obs_c = clip(cand.samples, threshold)
        cmp = _identical_max_diff(obs_c, ref)
        worst = max(worst, float(cmp["max_abs_diff"]))
        differing += int(cmp["differing_sample_count"])
    method_b = {
        "max_abs_diff": worst,
        "differing_sample_count": differing,
        "sample_count": len(ref),
        "candidate_count": len(material.candidates),
    }

    a_identical = method_a["max_abs_diff"] == 0.0
    b_identical = method_b["max_abs_diff"] == 0.0
    chosen = "construct_identical_observation" if (b_identical and not a_identical) else "clip_known_dry"
    return {
        "threshold": threshold,
        "sample_rate": sr,
        "method_a_clip_known_dry_source": method_a,
        "method_b_construct_identical_observation": method_b,
        "method_a_yields_identical_observation": a_identical,
        "method_b_yields_identical_observation": b_identical,
        "chosen": chosen,
        "rationale": (
            "实测：直接削同一干声的不同变体在同一阈值下仍产生逐样本不同的观测，"
            "峰值区与未削区都无法保证一致；显式构造核心样值可在削顶后得到逐样本相同"
            "的观测。故采用构造法（B）。"
            if chosen == "construct_identical_observation" else
            "实测显示直接削已知干声已能产生一致观测；无需构造。"
        ),
    }
