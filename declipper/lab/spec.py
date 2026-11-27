"""Editable material specification for the identical-observation teaching lab.

The spec is a plain JSON document so a learner can edit harmonic content,
transient shape and clipping thresholds without touching solver code. All
values are validated with typed errors before any synthesis happens.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


class MaterialSpecError(ValueError):
    """Raised when a material spec is malformed or cannot produce the paired lab."""


def _finite(name: str, value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise MaterialSpecError(f"{name} 必须是数字，收到 {value!r}") from exc
    if not math.isfinite(v):
        raise MaterialSpecError(f"{name} 必须是有限数")
    return v


@dataclass
class Harmonic:
    multiple: int
    amplitude: float
    phase: float
    decay: float


@dataclass
class Body:
    fundamental_hz: float = 196.0
    amplitude: float = 0.13
    decay: float = 1.45
    attack_seconds: float = 0.012
    harmonics: List[Harmonic] = field(
        default_factory=lambda: [
            Harmonic(1, 1.0, 0.40, 1.70),
            Harmonic(2, 0.55, 1.30, 2.20),
            Harmonic(3, 0.32, 2.60, 2.90),
            Harmonic(4, 0.18, 0.80, 3.40),
            Harmonic(5, 0.10, 2.10, 3.90),
        ]
    )


@dataclass
class Core:
    """Explicit samples placed inside the event; the reconstruction cannot see them."""

    length: int = 9
    # (position from core start, value) anchors; raised-cosine interpolation between.
    low_anchors: List[Tuple[int, float]] = field(
        default_factory=lambda: [(0, 0.55), (1, 0.70), (4, 0.95), (8, 0.80)]
    )
    high_anchors: List[Tuple[int, float]] = field(
        default_factory=lambda: [(0, 0.55), (1, 0.72), (4, 1.50), (8, 0.80)]
    )
    reveal_position: int = 0
    reveal_value: float = 0.55


@dataclass
class Spec:
    sample_rate: int = 8000
    duration_seconds: float = 0.20
    event_start: int = 240
    clip_threshold: float = 0.30
    reveal_threshold: float = 0.65
    body: Body = field(default_factory=Body)
    core: Core = field(default_factory=Core)
    decoy_value: float = 0.95
    pre_roll: float = 0.16
    post_roll: float = 0.12

    # ---- validation -------------------------------------------------
    def validate(self) -> None:
        if not 4000 <= self.sample_rate <= 192000:
            raise MaterialSpecError("sample_rate 需在 4000..192000 之间")
        if self.duration_seconds <= 0 or self.duration_seconds > 10:
            raise MaterialSpecError("duration_seconds 需在 (0, 10] 之间")
        if not 0 < self.clip_threshold < self.reveal_threshold:
            raise MaterialSpecError(
                "必须满足 0 < clip_threshold < reveal_threshold（抬阈值必须向上）"
            )
        c = self.core
        if c.length < 4 or c.length > 256:
            raise MaterialSpecError("core.length 需在 4..256 之间")
        if not 0 <= c.reveal_position < c.length:
            raise MaterialSpecError("core.reveal_position 必须落在核心内")
        lo = self._interpolate(c.low_anchors, c.length, "core.low_anchors")
        hi = self._interpolate(c.high_anchors, c.length, "core.high_anchors")
        # Every un-revealed core sample must sit at or above the raised threshold,
        # otherwise more than one sample would change when T is lifted.
        for k in range(c.length):
            if k == c.reveal_position:
                continue
            for name, series in (("low", lo), ("high", hi)):
                if series[k] < self.reveal_threshold - 1e-12:
                    raise MaterialSpecError(
                        f"{name} 核心样本 {k}={series[k]:.4f} 低于抬升阈值 "
                        f"{self.reveal_threshold}；抬阈值会暴露多于一个样本"
                    )
        if not (self.clip_threshold < c.reveal_value < self.reveal_threshold):
            raise MaterialSpecError(
                "core.reveal_value 必须严格落在 clip_threshold 与 reveal_threshold 之间，"
                "这样抬阈值恰好暴露它"
            )
        if self.decoy_value <= self.reveal_threshold:
            raise MaterialSpecError(
                "decoy_value 必须高于 reveal_threshold，这样诱饵在抬阈值后才会被排除"
            )
        if self.event_start < 32:
            raise MaterialSpecError("event_start 过早，事件左侧缺少上下文")
        total = self.total_samples
        if self.event_start + c.length + 32 >= total:
            raise MaterialSpecError("音频太短，事件右侧缺少上下文；请增大时长")
        self._interpolate(c.low_anchors, c.length, "core.low_anchors")
        hp = max(abs(self._interpolate(c.high_anchors, c.length, "core.high_anchors")[k])
                 for k in range(c.length))
        lp = max(abs(v) for v in lo)
        if not hp > lp + 1e-6:
            raise MaterialSpecError("high 核心必须明显高于 low 核心，否则无法体现不同原音")

    @staticmethod
    def _interpolate(anchors: Sequence[Tuple[int, float]], length: int, name: str) -> List[float]:
        pts = sorted((int(p), _finite(f"{name}.position", v)) for p, v in anchors)
        if not pts:
            raise MaterialSpecError(f"{name} 不能为空")
        if pts[0][0] != 0 or pts[-1][0] != length - 1:
            raise MaterialSpecError(f"{name} 首尾锚点必须分别在 0 与 {length - 1}")
        out = [0.0] * length
        for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
            if p1 <= p0:
                raise MaterialSpecError(f"{name} 锚点位置必须严格递增")
            for k in range(p0, p1 + 1):
                u = (k - p0) / (p1 - p0)
                w = 0.5 - 0.5 * math.cos(math.pi * u)
                out[k] = v0 + (v1 - v0) * w
        return out

    @property
    def total_samples(self) -> int:
        return int(round(self.sample_rate * self.duration_seconds))

    # ---- (de)serialisation -----------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "event_start": self.event_start,
            "clip_threshold": self.clip_threshold,
            "reveal_threshold": self.reveal_threshold,
            "decoy_value": self.decoy_value,
            "pre_roll": self.pre_roll,
            "post_roll": self.post_roll,
            "body": {
                "fundamental_hz": self.body.fundamental_hz,
                "amplitude": self.body.amplitude,
                "decay": self.body.decay,
                "attack_seconds": self.body.attack_seconds,
                "harmonics": [
                    {
                        "multiple": h.multiple,
                        "amplitude": h.amplitude,
                        "phase": h.phase,
                        "decay": h.decay,
                    }
                    for h in self.body.harmonics
                ],
            },
            "core": {
                "length": self.core.length,
                "low_anchors": [list(a) for a in self.core.low_anchors],
                "high_anchors": [list(a) for a in self.core.high_anchors],
                "reveal_position": self.core.reveal_position,
                "reveal_value": self.core.reveal_value,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Spec":
        if not isinstance(data, dict):
            raise MaterialSpecError("spec 顶层必须是 JSON 对象")
        b = data.get("body", {}) or {}
        c = data.get("core", {}) or {}
        try:
            body = Body(
                fundamental_hz=float(b.get("fundamental_hz", 196.0)),
                amplitude=float(b.get("amplitude", 0.13)),
                decay=float(b.get("decay", 1.45)),
                attack_seconds=float(b.get("attack_seconds", 0.012)),
                harmonics=[
                    Harmonic(
                        int(h["multiple"]), float(h["amplitude"]),
                        float(h["phase"]), float(h["decay"]),
                    )
                    for h in b.get("harmonics", [])
                ] or Body().harmonics,
            )
            core = Core(
                length=int(c.get("length", 9)),
                low_anchors=[(int(p), float(v)) for p, v in c.get("low_anchors", Core().low_anchors)],
                high_anchors=[(int(p), float(v)) for p, v in c.get("high_anchors", Core().high_anchors)],
                reveal_position=int(c.get("reveal_position", 0)),
                reveal_value=float(c.get("reveal_value", 0.55)),
            )
            spec = cls(
                sample_rate=int(data.get("sample_rate", 8000)),
                duration_seconds=float(data.get("duration_seconds", 0.20)),
                event_start=int(data.get("event_start", 240)),
                clip_threshold=float(data.get("clip_threshold", 0.30)),
                reveal_threshold=float(data.get("reveal_threshold", 0.65)),
                body=body,
                core=core,
                decoy_value=float(data.get("decoy_value", 0.95)),
                pre_roll=float(data.get("pre_roll", 0.16)),
                post_roll=float(data.get("post_roll", 0.12)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MaterialSpecError(f"spec 字段无法解析: {exc}") from exc
        spec.validate()
        return spec

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Spec":
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise MaterialSpecError(f"找不到 spec 文件: {path}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MaterialSpecError(f"spec JSON 解析失败: {exc}") from exc
        return cls.from_dict(data)
