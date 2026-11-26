from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

from .audio import Audio, write_wav
from .solver import detect_runs


def synthesize_chord(
    *,
    sample_rate: int = 8000,
    duration: float = 1.35,
    freqs: Sequence[float] = (220.0, 277.18, 329.63, 440.0),
    peak: float = 0.82,
    seed: int = 7,
    decay: float = 2.1,
    harmonics: int = 7,
    brightness: float = 1.25,
) -> List[float]:
    n = int(round(sample_rate * duration))
    rng = random.Random(seed)
    phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in freqs for _ in range(harmonics)]
    out: List[float] = []
    phase_index = 0
    for i in range(n):
        t = i / sample_rate
        attack = min(1.0, t / 0.012)
        env = attack * math.exp(-decay * t)
        v = 0.0
        pi = phase_index
        for fi, f0 in enumerate(freqs):
            note_weight = 1.0 - 0.08 * fi
            for h in range(1, harmonics + 1):
                inharm = math.sqrt(1.0 + 0.00012 * h * h)
                f = f0 * h * inharm
                amp = note_weight / (h**brightness)
                # A small pluck/body modulation keeps the delay peak identifiable.
                body = 1.0 + 0.08 * math.sin(2.0 * math.pi * 5.2 * t + fi)
                v += amp * body * math.sin(2.0 * math.pi * f * t + phases[pi])
                pi += 1
        v *= env
        out.append(v)
    current_peak = max(abs(v) for v in out) or 1.0
    scale = peak / current_peak
    return [v * scale for v in out]


def hard_clip(x: Sequence[float], threshold: float) -> List[float]:
    return [max(-threshold, min(threshold, float(v))) for v in x]


def periodic_tone(
    *,
    sample_rate: int = 8000,
    duration: float = 1.0,
    fundamental: float = 200.0,
    peak: float = 0.9,
    harmonics: int = 5,
) -> List[float]:
    n = int(round(sample_rate * duration))
    out = []
    for i in range(n):
        t = i / sample_rate
        v = sum((1.0 / h) * math.sin(2.0 * math.pi * fundamental * h * t) for h in range(1, harmonics + 1))
        out.append(v)
    current_peak = max(abs(v) for v in out) or 1.0
    return [v * peak / current_peak for v in out]


def delayed_secondary(
    clean: Sequence[float],
    shift: int,
    *,
    gain: float = 0.72,
    noise: float = 0.006,
    seed: int = 19,
) -> List[float]:
    rng = random.Random(seed)
    out: List[float] = []
    for i in range(len(clean)):
        v = gain * clean[i - shift] if i >= shift else 0.0
        # Tiny deterministic microphone noise.
        n = rng.uniform(-noise, noise)
        out.append(v + n)
    return out


def best_demo_run(
    y: Sequence[float], threshold: float, target_length: int = 10
) -> Tuple[int, int]:
    runs = detect_runs(y, threshold)
    if not runs:
        raise RuntimeError("合成样例没有产生削波平台")
    suitable = [r for r in runs if 6 <= r.length <= 24]
    pool = suitable or runs

    def score(r):
        return abs(r.length - target_length) + 0.001 * abs((r.start + r.end) / 2 - len(y) * 0.35)

    r = min(pool, key=score)
    # Return a target_length identical-value subplateau centered in the run.
    if r.length <= target_length:
        return r.start, r.end
    extra = r.length - target_length
    a = r.start + extra // 2
    return a, a + target_length - 1
