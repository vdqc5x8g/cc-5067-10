from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def fft_radix2(re: List[float], im: List[float], inverse: bool = False) -> None:
    n = len(re)
    if n & (n - 1):
        raise ValueError("FFT 长度必须是 2 的幂")
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]

    length = 2
    sign = 1.0 if inverse else -1.0
    while length <= n:
        half = length >> 1
        angle = sign * 2.0 * math.pi / length
        wr = math.cos(angle)
        wi = math.sin(angle)
        for start in range(0, n, length):
            cur_r, cur_i = 1.0, 0.0
            for k in range(half):
                a = start + k
                b = a + half
                tr = cur_r * re[b] - cur_i * im[b]
                ti = cur_r * im[b] + cur_i * re[b]
                re[b] = re[a] - tr
                im[b] = im[a] - ti
                re[a] += tr
                im[a] += ti
                new_r = cur_r * wr - cur_i * wi
                cur_i = cur_r * wi + cur_i * wr
                cur_r = new_r
        length <<= 1

    if inverse:
        for i in range(n):
            re[i] /= n
            im[i] /= n


def solve_linear(a: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    n = len(b)
    m = [list(a[i]) + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise ValueError("线性方程组奇异")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            if abs(factor) < 1e-15:
                continue
            for j in range(col, n + 1):
                m[row][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_sinusoids(
    samples: Sequence[float],
    mask: Sequence[bool],
    sample_rate: int,
    center: int,
    *,
    window: int = 1024,
    max_freqs: int = 8,
) -> Tuple[List[float], List[float], List[float], float]:
    """Fit a constant plus cos/sin terms to observed (unclipped) samples."""
    n_total = len(samples)
    half = window // 2
    lo = max(0, center - half)
    hi = min(n_total, center + half)
    n_fft = 1
    while n_fft < max(64, hi - lo):
        n_fft <<= 1
    n_fft = min(n_fft, 4096)

    re = [0.0] * n_fft
    im = [0.0] * n_fft
    actual = 0
    for i in range(lo, hi):
        w = 0.5 - 0.5 * math.cos(2.0 * math.pi * actual / max(1, hi - lo - 1))
        re[actual] = samples[i] * w
        actual += 1
    fft_radix2(re, im)

    nyquist_bin = n_fft // 2 - 1
    bins = []
    for k in range(2, nyquist_bin):
        freq = k * sample_rate / n_fft
        if freq < 45 or freq > sample_rate * 0.42:
            continue
        mag = math.hypot(re[k], im[k])
        bins.append((mag, k, freq))
    bins.sort(reverse=True)

    selected: List[Tuple[int, float]] = []
    for mag, k, freq in bins:
        if all(abs(k - kk) > 3 for kk, _ in selected):
            selected.append((k, freq))
        if len(selected) >= max_freqs:
            break
    freqs = [f for _, f in selected]
    terms = 1 + 2 * len(freqs)

    ata = [[0.0 for _ in range(terms)] for _ in range(terms)]
    atb = [0.0] * terms
    count = 0
    for idx in range(lo, hi):
        if not mask[idx]:
            continue
        row = [1.0]
        t = idx / sample_rate
        for f in freqs:
            row.append(math.cos(2.0 * math.pi * f * t))
            row.append(math.sin(2.0 * math.pi * f * t))
        x = float(samples[idx])
        for i in range(terms):
            atb[i] += row[i] * x
            for j in range(i, terms):
                ata[i][j] += row[i] * row[j]
        count += 1

    if count < terms + 4:
        raise ValueError("可用于谐波延拓的未削波样本不足")

    ridge = 1e-7 * max(1.0, count)
    for i in range(terms):
        ata[i][i] += ridge
        for j in range(i):
            ata[i][j] = ata[j][i]
    coeff = solve_linear(ata, atb)

    err2 = 0.0
    for idx in range(lo, hi):
        if not mask[idx]:
            continue
        pred = coeff[0]
        t = idx / sample_rate
        q = 1
        for f in freqs:
            pred += coeff[q] * math.cos(2.0 * math.pi * f * t) + coeff[q + 1] * math.sin(
                2.0 * math.pi * f * t
            )
            q += 2
        d = pred - samples[idx]
        err2 += d * d
    rmse = math.sqrt(err2 / count)
    return freqs, coeff, [lo, hi], rmse


def sinusoid_value(index: int, sample_rate: int, freqs: Sequence[float], coeff: Sequence[float]) -> float:
    v = coeff[0]
    t = index / sample_rate
    q = 1
    for f in freqs:
        v += coeff[q] * math.cos(2.0 * math.pi * f * t) + coeff[q + 1] * math.sin(
            2.0 * math.pi * f * t
        )
        q += 2
    return v
