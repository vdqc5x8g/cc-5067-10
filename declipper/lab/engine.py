"""Blind reconstruction engine for the teaching lab.

Every public function here takes an *observation* array and a threshold. Truth
arrays are never accepted. The engine exposes two hypotheses (conservative
minimum-slope repair and harmonic-prior continuation), re-clips both, and
records the residual against the supplied observation. A truth audit lives in
``evaluation`` and is applied only after the fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..solver import (
    ClippingRestoreError,
    detect_runs,
    difference_bounds,
    harmonic_template_for_run,
    reclipping_errors,
    solve_run_qp,
)


@dataclass
class HypothesisResult:
    name: str
    samples: List[float] = field(repr=False)
    peak: float
    stop_reason: str
    iterations: int
    constraint_violation: float
    reclip_max_error: float
    unclipped_max_error: float
    description: str

    def public(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "peak": self.peak,
            "stop_reason": self.stop_reason,
            "iterations": self.iterations,
            "constraint_violation": self.constraint_violation,
            "reclip_max_error": self.reclip_max_error,
            "unclipped_max_error": self.unclipped_max_error,
            "description": self.description,
        }


@dataclass
class BlindResult:
    threshold: float
    run: Dict[str, int]
    hypotheses: List[HypothesisResult]
    plateau_observation: List[float]
    notes: List[str]


def _select_run(obs: Sequence[float], threshold: float,
                start: Optional[int], end: Optional[int]):
    runs = detect_runs(obs, threshold)
    if start is not None and end is not None:
        chosen = [r for r in runs if r.start <= end and r.end >= start]
        if not chosen:
            raise ClippingRestoreError(
                f"框选 [{start}, {end}] 内没有阈值 ±{threshold:g} 的削波平台"
            )
        return max(chosen, key=lambda r: r.length)
    if not runs:
        raise ClippingRestoreError("观测中没有任何削波平台")
    return max(runs, key=lambda r: r.length)


def blind_reconstruct(
    observation: Sequence[float],
    sample_rate: int,
    threshold: float,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    prior_strength: float = 0.34,
    slope_weight: float = 0.04,
    curvature_weight: float = 1.0,
) -> BlindResult:
    """Reconstruct an observation without any access to the ground truth."""
    obs = [float(v) for v in observation]
    if len(obs) < 8:
        raise ClippingRestoreError("观测太短，无法建立边界约束")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ClippingRestoreError("削波阈值必须是大于 0 的有限数")

    run = _select_run(obs, threshold, start, end)
    run_info = {"start": run.start, "end": run.end, "sign": run.sign, "length": run.length}

    conservative = solve_run_qp(
        obs, run, threshold,
        slope_weight=slope_weight,
        curvature_weight=curvature_weight,
    )
    notes: List[str] = []
    cont_hyp: Optional[HypothesisResult] = None
    try:
        template, info = harmonic_template_for_run(obs, run, sample_rate, threshold, 2e-5)
        cont = solve_run_qp(
            obs, run, threshold,
            template=template,
            template_weight=prior_strength,
            slope_weight=slope_weight,
            curvature_weight=curvature_weight,
        )
        cont_hyp = HypothesisResult(
            name="continuation",
            samples=cont["samples"],
            peak=float(cont["peak_estimate"]),
            stop_reason=str(cont["optimizer"]["stop_reason"]),
            iterations=int(cont["optimizer"]["iterations"]),
            constraint_violation=float(cont["optimizer"]["constraint_violation_max"]),
            reclip_max_error=float(reclipping_errors(cont["samples"], obs, threshold)["reclip_max_error"]),
            unclipped_max_error=float(reclipping_errors(cont["samples"], obs, threshold)["unclipped_max_error"]),
            description=f"谐波先验延拓（权重 {prior_strength:g}，拟合 RMSE {info.get('fit_rmse_unclipped', 0):.4g}）",
        )
    except Exception as exc:  # template fit is a prior, not a guarantee
        notes.append(f"延拓先验不可用: {type(exc).__name__}: {exc}")

    cons_hyp = HypothesisResult(
        name="conservative",
        samples=conservative["samples"],
        peak=float(conservative["peak_estimate"]),
        stop_reason=str(conservative["optimizer"]["stop_reason"]),
        iterations=int(conservative["optimizer"]["iterations"]),
        constraint_violation=float(conservative["optimizer"]["constraint_violation_max"]),
        reclip_max_error=float(reclipping_errors(conservative["samples"], obs, threshold)["reclip_max_error"]),
        unclipped_max_error=float(reclipping_errors(conservative["samples"], obs, threshold)["unclipped_max_error"]),
        description="保守最小斜率/曲率修补：只改动饱和区并取最小运动",
    )

    hypotheses = [cons_hyp]
    if cont_hyp is not None:
        hypotheses.append(cont_hyp)
        notes.append("两种假设都满足观测；两者必须并列试听，不能把其一当作唯一真值")
    else:
        notes.append("仅有保守解可用；不能把圆滑顶端冒充唯一真值")

    m = run.length
    plateau = [float(obs[run.start + k]) for k in range(m)]
    return BlindResult(
        threshold=threshold,
        run=run_info,
        hypotheses=hypotheses,
        plateau_observation=plateau,
        notes=notes,
    )


def residual(reconstruction: Sequence[float], observation: Sequence[float],
             threshold: float) -> Dict[str, Any]:
    """Re-clip a reconstruction and compare to the observation sample-by-sample."""
    m = min(len(reconstruction), len(observation))
    re_clipped = [max(-threshold, min(threshold, float(reconstruction[i]))) for i in range(m)]
    per_sample = [abs(re_clipped[i] - float(observation[i])) for i in range(m)]
    return {
        "max_abs_residual": max(per_sample, default=0.0),
        "sum_abs_residual": float(sum(per_sample)),
        "mismatch_count": sum(1 for d in per_sample if d > 2e-5),
        "per_sample": per_sample,
        "re_clipped": re_clipped,
    }


def injected_peak(reconstruction: Sequence[float], threshold: float) -> List[float]:
    """The part of a reconstruction that re-clipping would remove.

    This is the audible "guess" the hypothesis injects into the saturated
    region: ``x - clip(x)``. A feasible reconstruction re-clips back to the
    observation, so this signal is exactly what the observation cannot show.
    """
    return [
        float(v) - max(-threshold, min(threshold, float(v)))
        for v in reconstruction
    ]


def hypothesis_difference(a: Sequence[float], b: Sequence[float]) -> List[float]:
    m = min(len(a), len(b))
    return [float(a[i]) - float(b[i]) for i in range(m)]


def hypothesis_table(result: BlindResult, observation: Sequence[float]) -> List[Dict[str, Any]]:
    rows = []
    for h in result.hypotheses:
        res = residual(h.samples, observation, result.threshold)
        obs_indices = [result.run["start"] + k for k in range(result.run["length"])]
        peak_region = max(abs(h.samples[i]) for i in obs_indices)
        rows.append({
            **h.public(),
            "peak_in_region": peak_region,
            "reclip_from_residual": res["max_abs_residual"],
            "residual_mismatch_count": res["mismatch_count"],
            "feasible": res["max_abs_residual"] <= 2e-5,
        })
    return rows


def prior_envelope_for(observation: Sequence[float], threshold: float,
                       run: Dict[str, int]) -> Dict[str, Any]:
    from ..solver import Run, prior_envelope
    r = Run(run["start"], run["end"], run["sign"])
    bounds = difference_bounds(observation, threshold)
    env = prior_envelope(observation, r, threshold, bounds)
    return {
        "peak_range_observable": [threshold, None],
        "peak_range_prior_envelope": env["peak_range_prior_envelope"],
        "hard_prior_envelope_feasible": env["hard_prior_envelope_feasible"],
        "slew_bound": env["slew_bound"],
        "curvature_bound": env["curvature_bound"],
    }
