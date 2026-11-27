"""Material generator for the identical-observation teaching lab.

Two (or more) different originals are constructed so that hard clipping at a
low threshold produces *byte-identical* observations over the whole file. The
originals are kept in a separate truth store; every reconstruction path in this
package accepts only an observation array, never a truth array.

Raising the threshold to ``reveal_threshold`` is tuned so exactly one hidden
sample crosses back below the ceiling. The observation stays identical for the
"surviving" originals at that sample, while decoys that differ there are now
excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .spec import Spec, MaterialSpecError


@dataclass(frozen=True)
class Candidate:
    label: str
    role: str  # "surviving" | "decoy"
    samples: List[float] = field(repr=False)


@dataclass
class LabMaterial:
    spec: Spec
    sample_rate: int
    observations: Dict[str, List[float]]
    thresholds: Dict[str, float]
    candidates: List[Candidate]
    event_start: int
    core_length: int
    reveal_index: int
    intact_intervals: List[Tuple[int, int]]

    @property
    def core_end(self) -> int:
        return self.event_start + self.core_length - 1

    def observation(self, level: str) -> List[float]:
        return list(self.observations[level])

    def candidate(self, label: str) -> Candidate:
        for c in self.candidates:
            if c.label == label:
                return c
        raise MaterialSpecError(f"未知候选原音: {label}")

    def intact_mask(self) -> List[bool]:
        mask = [False] * len(self.observations["T0"])
        for a, b in self.intact_intervals:
            for i in range(a, b + 1):
                mask[i] = True
        return mask


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def synthesize_body(spec: Spec) -> List[float]:
    """Harmonic body shared by every original; kept well below both thresholds."""
    n = spec.total_samples
    sr = spec.sample_rate
    b = spec.body
    out: List[float] = []
    for i in range(n):
        t = i / sr
        env = smoothstep(0.0, b.attack_seconds, t) * math.exp(-b.decay * t)
        v = 0.0
        for h in b.harmonics:
            local = math.exp(-t * (h.decay - 1.0)) if h.decay > 0 else 1.0
            v += h.amplitude * local * math.sin(
                2.0 * math.pi * b.fundamental_hz * h.multiple * t + h.phase
            )
        out.append(b.amplitude * env * v)
    peak = max((abs(v) for v in out), default=0.0) or 1.0
    return [v * b.amplitude / peak for v in out]


def _core_series(anchors: Sequence[Tuple[int, float]], length: int) -> List[float]:
    return Spec._interpolate(anchors, length, "anchors")


def _assemble(body: Sequence[float], core: Sequence[float], start: int) -> List[float]:
    x = list(body)
    for k, v in enumerate(core):
        x[start + k] = float(v)
    return x


def clip(samples: Sequence[float], threshold: float) -> List[float]:
    return [max(-threshold, min(threshold, float(v))) for v in samples]


def build_material(spec: Spec) -> LabMaterial:
    spec.validate()
    body = synthesize_body(spec)
    c = spec.core
    low_core = _core_series(c.low_anchors, c.length)
    high_core = _core_series(c.high_anchors, c.length)
    low_core[c.reveal_position] = c.reveal_value
    high_core[c.reveal_position] = c.reveal_value

    low = _assemble(body, low_core, spec.event_start)
    high = _assemble(body, high_core, spec.event_start)

    decoy_values = [spec.decoy_value, max(spec.decoy_value, 1.0), max(spec.decoy_value, 1.3)]
    candidates = [
        Candidate("low", "surviving", low),
        Candidate("high", "surviving", high),
    ]
    for k, val in enumerate(decoy_values):
        core = list(low_core)
        core[c.reveal_position] = val
        candidates.append(Candidate(f"decoy-{k+1}", "decoy", _assemble(body, core, spec.event_start)))

    t0 = float(spec.clip_threshold)
    t1 = float(spec.reveal_threshold)
    obs0 = clip(low, t0)
    obs1 = clip(low, t1)
    # The construction guarantees these are identical for all surviving originals.
    for cand in candidates:
        if cand.role == "surviving":
            if clip(cand.samples, t0) != obs0:
                raise MaterialSpecError("构造失败：存活原音在低阈值下观测不同")
            if clip(cand.samples, t1) != obs1:
                raise MaterialSpecError("构造失败：存活原音在抬升阈值下观测不同")

    reveal_abs = spec.event_start + c.reveal_position
    newly = [
        i for i in range(len(obs0))
        if abs(obs0[i]) >= t0 - 1e-12 and abs(obs1[i]) < t1 - 1e-12
    ]
    if len(newly) != 1:
        raise MaterialSpecError(
            f"抬阈值应恰好暴露一个样本，实际暴露 {len(newly)} 个（{newly}）"
        )
    if newly[0] != reveal_abs:
        raise MaterialSpecError("抬阈值暴露的样本与 spec 指定的 reveal_position 不一致")

    intact = [
        (0, spec.event_start - 1),
        (spec.event_start + c.length, len(body) - 1),
    ]
    return LabMaterial(
        spec=spec,
        sample_rate=spec.sample_rate,
        observations={"T0": obs0, "T1": obs1},
        thresholds={"T0": t0, "T1": t1},
        candidates=candidates,
        event_start=spec.event_start,
        core_length=c.length,
        reveal_index=reveal_abs,
        intact_intervals=intact,
    )


def observations_identical_at_low_threshold(material: LabMaterial) -> bool:
    ref = material.observations["T0"]
    return all(clip(c.samples, material.thresholds["T0"]) == ref for c in material.candidates)


def newly_revealed_indices(material: LabMaterial) -> List[int]:
    t0, t1 = material.thresholds["T0"], material.thresholds["T1"]
    o0, o1 = material.observations["T0"], material.observations["T1"]
    return [
        i for i in range(len(o0))
        if abs(o0[i]) >= t0 - 1e-12 and abs(o1[i]) < t1 - 1e-12
    ]


def intact_displacement(material: LabMaterial) -> Dict[str, float | int]:
    """Sample-for-sample change of the small-signal regions between T0 and T1."""
    o0, o1 = material.observations["T0"], material.observations["T1"]
    mask = material.intact_mask()
    idx = [i for i, keep in enumerate(mask) if keep]
    diffs = [abs(o0[i] - o1[i]) for i in idx]
    return {
        "intact_sample_count": len(idx),
        "max_abs_change": max(diffs, default=0.0),
        "changed_sample_count": sum(1 for d in diffs if d > 0.0),
    }
