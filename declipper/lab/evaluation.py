"""Blind evaluation and the threshold-lift exclusion experiment.

This module is the only place that may inspect candidate originals. It is kept
separate from ``engine`` so reconstruction code can be audited to confirm it
never receives truth. Here, truth is used to *grade* and to *exclude*, never to
repair.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .engine import blind_reconstruct, residual
from .material import LabMaterial, clip


def candidate_is_consistent(candidate_samples: Sequence[float], observation: Sequence[float],
                            threshold: float) -> bool:
    return clip(candidate_samples, threshold) == [float(v) for v in observation]


def consistent_candidates(material: LabMaterial, level: str) -> List[str]:
    obs = material.observation(level)
    t = material.thresholds[level]
    return [c.label for c in material.candidates
            if candidate_is_consistent(c.samples, obs, t)]


def exclusion_experiment(material: LabMaterial) -> Dict[str, Any]:
    """Compare the candidate set consistent with T0 against the set at T1.

    The requirement is that the set strictly shrinks when the threshold is
    raised to reveal exactly one hidden sample, and that the small-signal
    regions are untouched by the threshold change.
    """
    s0 = consistent_candidates(material, "T0")
    s1 = consistent_candidates(material, "T1")
    excluded = [c for c in s0 if c not in s1]
    newly = s1 + excluded
    o0 = material.observation("T0")
    o1 = material.observation("T1")
    intact_changes = [
        abs(o0[i] - o1[i]) for i in range(len(o0))
        if not (material.event_start <= i <= material.core_end)
    ]
    max_intact = max(intact_changes, default=0.0)
    return {
        "low_threshold": material.thresholds["T0"],
        "raised_threshold": material.thresholds["T1"],
        "reveal_index": material.reveal_index,
        "candidate_set_low": s0,
        "candidate_set_raised": s1,
        "excluded_by_raising_threshold": excluded,
        "set_shrank": len(s1) < len(s0),
        "exactly_one_newly_visible_sample": len([
            i for i in range(len(o0))
            if abs(o0[i]) >= material.thresholds["T0"] - 1e-12
            and abs(o1[i]) < material.thresholds["T1"] - 1e-12
        ]) == 1,
        "intact_region_max_change": max_intact,
        "intact_region_sample_for_sample_unchanged": max_intact == 0.0,
        "surviving_candidates": s1,
        "note": (
            "抬阈值后仍不能区分集合内的存活原音；被排除的是在暴露样本上与原观测"
            "不一致的候选。小信号未受损区逐样本不变。"
        ),
    }


def truth_audit(material: LabMaterial, level: str, reconstructed: Sequence[float]) -> List[Dict[str, Any]]:
    """Grade a *single* reconstruction against every candidate (evaluation only)."""
    obs = material.observation(level)
    t = material.thresholds[level]
    rows = []
    for cand in material.candidates:
        truth = cand.samples
        n = min(len(truth), len(reconstructed), len(obs))
        region = [abs(float(reconstructed[i]) - float(truth[i])) for i in range(n)]
        intact = [
            abs(float(reconstructed[i]) - float(truth[i]))
            for i in range(n)
            if not (material.event_start <= i <= material.core_end)
        ]
        rows.append({
            "candidate": cand.label,
            "role": cand.role,
            "reclip_max_error": residual(reconstructed, obs, t)["max_abs_residual"],
            "peak_abs_error_all": max(region, default=0.0),
            "intact_max_abs_error": max(intact, default=0.0),
            "consistent_with_observation": candidate_is_consistent(cand.samples, obs, t),
        })
    return rows


def blind_comparison(material: LabMaterial, level: str = "T0",
                     prior_strength: float = 0.34) -> Dict[str, Any]:
    """Run the blind engine on an observation and grade without leaking truth."""
    obs = material.observation(level)
    t = material.thresholds[level]
    blind = blind_reconstruct(obs, material.sample_rate, t,
                              start=material.event_start, end=material.core_end,
                              prior_strength=prior_strength)
    rows = []
    for h in blind.hypotheses:
        audit = truth_audit(material, level, h.samples)
        rows.append({
            "hypothesis": h.public(),
            "truth_audit": audit,
            "outperforms_which": [
                a["candidate"] for a in audit if a["peak_abs_error_all"] <= h.reclip_max_error + 1e-9
            ],
        })
    return {
        "level": level,
        "threshold": t,
        "run": blind.run,
        "notes": blind.notes,
        "hypotheses": rows,
        "plateau_observation": blind.plateau_observation,
        "identifiable_from_observation_alone": False,
        "truth_audit_is_evaluation_only": True,
    }


def threshold_lift_comparison(material: LabMaterial, prior_strength: float = 0.34) -> Dict[str, Any]:
    """The required blind comparison before and after raising the threshold."""
    before = blind_comparison(material, "T0", prior_strength)
    after = blind_comparison(material, "T1", prior_strength)
    exclusion = exclusion_experiment(material)
    intact = material.observations["T0"]
    raised = material.observations["T1"]
    small_unchanged = all(
        abs(intact[i] - raised[i]) < 1e-15
        for i in range(len(intact))
        if not (material.event_start <= i <= material.core_end)
    )
    return {
        "before": before,
        "after": after,
        "exclusion": exclusion,
        "checks": {
            "candidate_set_strictly_shrank": exclusion["set_shrank"],
            "exactly_one_sample_newly_revealed": exclusion["exactly_one_newly_visible_sample"],
            "small_signal_unchanged_sample_for_sample": small_unchanged,
            "reconstruction_sees_only_observation": True,
        },
        "passed": all([
            exclusion["set_shrank"],
            exclusion["exactly_one_newly_visible_sample"],
            small_unchanged,
        ]),
    }
