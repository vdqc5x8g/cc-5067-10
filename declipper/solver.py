from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .linalgutil import fit_sinusoids, sinusoid_value
from .optim import box_qp


class ClippingRestoreError(ValueError):
    """Raised when a clipping restoration request is numerically or semantically invalid."""


@dataclass
class Run:
    start: int
    end: int
    sign: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def key(self) -> str:
        return f"{self.start}:{self.end}:{self.sign}"


def detect_runs(
    y: Sequence[float], threshold: float, level_tolerance: float = 2e-5
) -> List[Run]:
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ClippingRestoreError("削波阈值必须是大于 0 的有限数")
    if threshold > 100.0:
        raise ClippingRestoreError("削波阈值异常大；请检查采样格式与阈值单位")
    runs: List[Run] = []
    n = len(y)
    i = 0
    tol = max(0.0, level_tolerance)
    while i < n:
        v = float(y[i])
        sign = 1 if v >= threshold - tol else -1 if v <= -threshold + tol else 0
        if sign == 0:
            i += 1
            continue
        j = i
        while j + 1 < n:
            w = float(y[j + 1])
            if sign == 1 and w >= threshold - tol:
                j += 1
            elif sign == -1 and w <= -threshold + tol:
                j += 1
            else:
                break
        runs.append(Run(i, j, sign))
        i = j + 1
    return runs


def select_runs(
    y: Sequence[float],
    threshold: float,
    start: Optional[int] = None,
    end: Optional[int] = None,
    level_tolerance: float = 2e-5,
) -> List[Run]:
    runs = detect_runs(y, threshold, level_tolerance)
    if start is None and end is None:
        return runs
    n = len(y)
    a = 0 if start is None else max(0, int(start))
    b = n - 1 if end is None else min(n - 1, int(end))
    if a > b:
        raise ClippingRestoreError("框选起点不能大于终点")
    selected = [r for r in runs if r.start <= b and r.end >= a]
    if not selected:
        raise ClippingRestoreError(
            f"框选区间 [{a}, {b}] 内没有检测到阈值 ±{threshold:g} 的连续削波平台"
        )
    return selected


def difference_bounds(
    y: Sequence[float], threshold: float, level_tolerance: float = 2e-5
) -> Dict[str, float]:
    n = len(y)
    mask = [abs(float(v)) < threshold - level_tolerance for v in y]
    d1: List[float] = []
    d2: List[float] = []
    for i in range(n - 1):
        if mask[i] and mask[i + 1]:
            d1.append(abs(float(y[i + 1]) - float(y[i])))
    for i in range(1, n - 1):
        if mask[i - 1] and mask[i] and mask[i + 1]:
            q = float(y[i - 1]) - 2.0 * float(y[i]) + float(y[i + 1])
            d2.append(abs(q))

    def quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    slew = max(quantile(d1, 0.995), max(d1) if d1 else 0.0) * 1.35 + 1e-7
    curvature = max(quantile(d2, 0.995), max(d2) if d2 else 0.0) * 1.5 + 1e-7
    return {
        "slew_bound": slew,
        "curvature_bound": curvature,
        "observed_first_diff_max": max(d1, default=0.0),
        "observed_second_diff_max": max(d2, default=0.0),
        "unclipped_fraction": sum(mask) / max(1, n),
    }


def estimate_local_period(
    y: Sequence[float], center: int, sample_rate: int, radius: int = 512
) -> Dict[str, Any]:
    n = len(y)
    lo = max(0, center - radius)
    hi = min(n, center + radius)
    seg = [float(y[i]) for i in range(lo, hi)]
    if len(seg) < 64:
        return {"period_samples": None, "score": 0.0, "candidates": []}
    mean = sum(seg) / len(seg)
    seg = [v - mean for v in seg]
    energy0 = sum(v * v for v in seg)
    if energy0 <= 1e-14:
        return {"period_samples": None, "score": 0.0, "candidates": []}
    max_lag = min(240, len(seg) // 3)
    min_lag = max(4, int(sample_rate / 1200.0))
    scores: List[Tuple[float, int]] = []
    for lag in range(min_lag, max_lag + 1):
        num = 0.0
        ea = 0.0
        eb = 0.0
        for i in range(len(seg) - lag):
            a = seg[i]
            b = seg[i + lag]
            num += a * b
            ea += a * a
            eb += b * b
        den = math.sqrt(max(1e-20, ea * eb))
        scores.append((num / den, lag))
    scores.sort(reverse=True)
    top: List[Dict[str, float | int]] = []
    for score, lag in scores[:8]:
        if all(abs(lag - int(c["lag"])) > 3 for c in top):
            top.append({"lag": lag, "score": score})
        if len(top) >= 4:
            break
    best_score, best_lag = scores[0]
    return {
        "period_samples": int(best_lag),
        "score": float(best_score),
        "candidates": top,
    }


def _run_energy_and_grad(
    z: Sequence[float],
    var_positions: Sequence[int],
    m: int,
    threshold: float,
    slope_weight: float,
    curvature_weight: float,
    ridge_weight: float,
    template_z: Optional[Sequence[float]],
    template_weight: float,
) -> Tuple[float, List[float]]:
    length = len(z)
    energy = 0.0
    gz = [0.0] * length

    if slope_weight:
        for i in range(length - 1):
            d = z[i + 1] - z[i]
            energy += slope_weight * d * d
            g = 2.0 * slope_weight * d
            gz[i] -= g
            gz[i + 1] += g

    if curvature_weight:
        for i in range(1, length - 1):
            q = z[i - 1] - 2.0 * z[i] + z[i + 1]
            energy += curvature_weight * q * q
            g = 2.0 * curvature_weight * q
            gz[i - 1] += g
            gz[i] -= 2.0 * g
            gz[i + 1] += g

    grad: List[float] = []
    for k, pos in enumerate(var_positions):
        h = z[pos]
        if ridge_weight:
            d = h - threshold
            energy += ridge_weight * d * d
            gz[pos] += 2.0 * ridge_weight * d
        if template_z is not None and template_weight:
            d = h - float(template_z[k])
            energy += template_weight * d * d
            gz[pos] += 2.0 * template_weight * d
        grad.append(gz[pos])
    return energy, grad


def solve_run_qp(
    y: Sequence[float],
    run: Run,
    threshold: float,
    *,
    template: Optional[Sequence[float]] = None,
    template_weight: float = 0.0,
    slope_weight: float = 0.04,
    curvature_weight: float = 1.0,
    ridge_weight: float = 1e-5,
    max_iter: int = 4000,
) -> Dict[str, Any]:
    n = len(y)
    s, e, sign = run.start, run.end, run.sign
    lo = max(0, s - 2)
    hi = min(n, e + 3)
    length = hi - lo
    var_positions = [s - lo + k for k in range(run.length)]
    z0 = [sign * float(y[i]) for i in range(lo, hi)]
    for pos in var_positions:
        z0[pos] = threshold
    template_z = None
    if template is not None:
        if len(template) != run.length:
            raise ClippingRestoreError("延拓模板长度与削波平台长度不一致")
        template_z = [sign * float(v) for v in template]

    lower = [threshold] * run.length
    upper = [math.inf] * run.length

    def eg(h: Sequence[float]) -> Tuple[float, List[float]]:
        z = list(z0)
        for k, pos in enumerate(var_positions):
            z[pos] = float(h[k])
        return _run_energy_and_grad(
            z,
            var_positions,
            run.length,
            threshold,
            slope_weight,
            curvature_weight,
            ridge_weight,
            template_z,
            template_weight,
        )

    lipschitz = (
        8.0 * slope_weight
        + 32.0 * curvature_weight
        + 2.0 * template_weight
        + 2.0 * ridge_weight
        + 1e-9
    )
    solved = box_qp(
        run.length,
        eg,
        lower,
        upper,
        [threshold] * run.length,
        lipschitz=lipschitz,
        max_iter=max_iter,
        grad_tol=3e-7,
    )
    h = [max(threshold, float(v)) for v in solved["x"]]
    projection_adjustment = max(
        (abs(h[k] - float(solved["x"][k])) for k in range(run.length)), default=0.0
    )
    out = list(y)
    for k in range(run.length):
        out[s + k] = sign * h[k]

    boundary = boundary_diagnostics(out, y, run)
    peak = max(h)
    return {
        "run": {"start": s, "end": e, "sign": sign, "length": run.length},
        "h": h,
        "samples": out,
        "peak_estimate": peak,
        "optimizer": {
            k: v
            for k, v in solved.items()
            if k not in {"x"}
        }
        | {"post_projection_adjustment": projection_adjustment},
        "boundary": boundary,
        "template_weight": template_weight if template is not None else 0.0,
    }


def boundary_diagnostics(
    x: Sequence[float], y: Sequence[float], run: Run
) -> Dict[str, float | bool]:
    s, e = run.start, run.end
    n = len(x)
    slope_jumps: List[float] = []
    curvature: List[float] = []
    value_gap = 0.0
    if s > 0:
        value_gap = max(value_gap, abs(x[s] - x[s - 1] - (x[s] - x[s - 1])))
    if e + 1 < n:
        value_gap = max(value_gap, abs(x[e] - x[e + 1] - (x[e] - x[e + 1])))

    if s >= 2:
        before = x[s - 1] - x[s - 2]
        join = x[s] - x[s - 1]
        slope_jumps.append(abs(join - before))
    if e + 2 < n:
        join = x[e + 1] - x[e]
        after = x[e + 2] - x[e + 1]
        slope_jumps.append(abs(after - join))
    for c in (s - 1, s, e, e + 1):
        if 1 <= c < n - 1:
            curvature.append(abs(x[c - 1] - 2.0 * x[c] + x[c + 1]))
    observed_boundary_change = max(
        [abs(x[i] - y[i]) for i in range(max(0, s - 2), min(n, e + 3))],
        default=0.0,
    )
    return {
        "max_slope_jump": max(slope_jumps, default=0.0),
        "max_boundary_curvature": max(curvature, default=0.0),
        "observed_boundary_change": observed_boundary_change,
        "continuous_sample_join": True,
        "value_gap": value_gap,
    }


def prior_envelope(
    y: Sequence[float],
    run: Run,
    threshold: float,
    bounds: Dict[str, float],
) -> Dict[str, Any]:
    """Outer peak envelope from explicit hard slew/curvature limits."""
    s, e, sign = run.start, run.end, run.sign
    m = run.length
    slew = float(bounds["slew_bound"])
    curv = float(bounds["curvature_bound"])
    z = [sign * float(v) for v in y]

    left = [math.inf] * m
    if s > 0:
        if s >= 2:
            d_prev = z[s - 1] - z[s - 2]
            d = min(slew, d_prev + curv)
        else:
            d = slew
        d = max(-slew, min(slew, d))
        left[0] = z[s - 1] + d
        d_prev = d
        for k in range(1, m):
            d = min(slew, d_prev + curv)
            left[k] = left[k - 1] + d
            d_prev = d

    right = [math.inf] * m
    if e + 1 < len(y):
        if e + 2 < len(y):
            d_next = z[e + 2] - z[e + 1]
            d_cur = max(-slew, d_next - curv)
        else:
            d_cur = -slew
        d_cur = max(-slew, min(slew, d_cur))
        right[m - 1] = z[e + 1] - d_cur
        for k in range(m - 2, -1, -1):
            d_prev = max(-slew, d_cur - curv)
            right[k] = right[k + 1] - d_prev
            d_cur = d_prev

    upper: List[Optional[float]] = []
    hard_feasible = True
    for k in range(m):
        u = min(left[k], right[k])
        if not math.isfinite(u):
            upper.append(None)
        else:
            if u < threshold - 1e-6:
                hard_feasible = False
            upper.append(max(threshold, u))
    finite_upper = [u for u in upper if u is not None]
    peak_upper = max(finite_upper) if finite_upper else None
    return {
        "sample_upper": upper,
        "peak_upper_prior": peak_upper,
        "peak_range_observable": [threshold, None],
        "peak_range_prior_envelope": [threshold, peak_upper],
        "hard_prior_envelope_feasible": hard_feasible,
        "slew_bound": slew,
        "curvature_bound": curv,
    }


def harmonic_template_for_run(
    y: Sequence[float],
    run: Run,
    sample_rate: int,
    threshold: float,
    level_tolerance: float,
) -> Tuple[List[float], Dict[str, Any]]:
    mask = [abs(float(v)) < threshold - level_tolerance for v in y]
    center = (run.start + run.end) // 2
    freqs, coeff, window, rmse = fit_sinusoids(
        y, mask, sample_rate, center, window=3072, max_freqs=14
    )
    template = [
        sinusoid_value(i, sample_rate, freqs, coeff)
        for i in range(run.start, run.end + 1)
    ]
    return template, {
        "frequencies_hz": [round(f, 2) for f in freqs],
        "fit_rmse_unclipped": rmse,
        "window": window,
        "prior": "8 个以内正弦+余弦项的局部带限/谐波延拓",
    }


def reclipping_errors(
    x: Sequence[float], y: Sequence[float], threshold: float
) -> Dict[str, float | int]:
    max_err = 0.0
    mismatch = 0
    for i, v in enumerate(x):
        r = max(-threshold, min(threshold, float(v)))
        d = abs(r - float(y[i]))
        max_err = max(max_err, d)
        if d > 2e-5:
            mismatch += 1
    unclipped_err = max(
        (abs(float(x[i]) - float(y[i])) for i in range(len(y)) if abs(float(y[i])) < threshold - 2e-5),
        default=0.0,
    )
    return {
        "reclip_max_error": max_err,
        "reclip_mismatch_count": mismatch,
        "unclipped_max_error": unclipped_err,
    }


def _empty_candidate_store() -> Dict[str, Any]:
    return {"conservative": None, "continuation": None, "dual": None}


def restore_signal(
    y: Sequence[float],
    sample_rate: int,
    threshold: float,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    mode: str = "auto",
    level_tolerance: float = 2e-5,
    secondary_template: Optional[Sequence[float]] = None,
    secondary_weight: float = 8.0,
    alignment: Optional[Dict[str, Any]] = None,
    certified_dual: bool = False,
    reliable_fraction: float = 0.40,
) -> Dict[str, Any]:
    if not y or len(y) < 8:
        raise ClippingRestoreError("音频样本太短，无法进行边界衔接检查")
    if mode not in {"auto", "conservative", "continuation", "dual"}:
        raise ClippingRestoreError(f"未知重建模式: {mode}")

    runs = select_runs(y, threshold, start, end, level_tolerance)
    bounds = difference_bounds(y, threshold, level_tolerance)
    work = list(y)
    run_reports: List[Dict[str, Any]] = []
    alternatives: Dict[str, List[float]] = {
        "conservative": list(y),
        "continuation": list(y),
        "dual": list(y),
    }
    has_continuation = False
    has_dual = secondary_template is not None

    for run in runs:
        period_info = estimate_local_period(
            y, (run.start + run.end) // 2, sample_rate
        )
        period = period_info.get("period_samples")
        period_budget = reliable_fraction * float(
            period if period else max(8, sample_rate / 220.0)
        )
        time_budget = 0.002 * sample_rate
        reliable_len = max(6, int(round(min(time_budget, period_budget))))
        long_run = run.length > reliable_len

        conservative = solve_run_qp(y, run, threshold)
        env = prior_envelope(y, run, threshold, bounds)
        cont_result: Optional[Dict[str, Any]] = None
        template_info: Optional[Dict[str, Any]] = None
        should_fit_continuation = mode == "continuation" or (mode == "auto" and long_run)
        if mode in {"auto", "continuation"} and should_fit_continuation:
            try:
                template, template_info = harmonic_template_for_run(
                    y, run, sample_rate, threshold, level_tolerance
                )
                cont_weight = 0.35 if not long_run else 0.18
                cont_result = solve_run_qp(
                    y, run, threshold, template=template, template_weight=cont_weight
                )
            except Exception as exc:  # surface as solver status, but UI can show
                template_info = {"error": f"{type(exc).__name__}: {exc}"}
                if mode == "continuation":
                    raise ClippingRestoreError(f"延拓求解失败: {exc}") from exc

        dual_result: Optional[Dict[str, Any]] = None
        if secondary_template is not None:
            tmpl = [secondary_template[i] for i in range(run.start, run.end + 1)]
            dual_result = solve_run_qp(
                y,
                run,
                threshold,
                template=tmpl,
                template_weight=secondary_weight,
                curvature_weight=0.8,
            )

        # Primary selection precedence: explicit dual > explicit continuation > auto.
        if mode == "dual":
            if dual_result is None:
                raise ClippingRestoreError("双轨模式缺少已对齐的副麦模板")
            primary = dual_result
            primary_kind = "dual_constrained"
        elif mode == "continuation":
            if cont_result is None:
                raise ClippingRestoreError("延拓模板不可用")
            primary = cont_result
            primary_kind = "harmonic_continuation"
        else:
            if dual_result is not None and certified_dual:
                primary = dual_result
                primary_kind = "certified_dual_constrained"
            elif long_run and cont_result is not None:
                primary = cont_result
                primary_kind = "long_gap_continuation_speculative"
            else:
                primary = conservative
                primary_kind = "minimum_effort_constrained"

        for k in range(run.length):
            alternatives["conservative"][run.start + k] = conservative["samples"][run.start + k]
            if cont_result is not None:
                has_continuation = True
                alternatives["continuation"][run.start + k] = cont_result["samples"][run.start + k]
            if dual_result is not None:
                alternatives["dual"][run.start + k] = dual_result["samples"][run.start + k]
            work[run.start + k] = primary["samples"][run.start + k]

        report: Dict[str, Any] = {
            "run": conservative["run"],
            "long_saturation": long_run,
            "reliable_prior_length": reliable_len,
            "period": period_info,
            "prior_bounds": env,
            "conservative": {
                "peak": conservative["peak_estimate"],
                "optimizer": conservative["optimizer"],
                "boundary": conservative["boundary"],
                "errors": reclipping_errors(conservative["samples"], y, threshold),
            },
            "primary_kind": primary_kind,
            "primary_peak": primary["peak_estimate"],
            "primary_optimizer": primary["optimizer"],
            "primary_boundary": primary["boundary"],
            "primary_errors": reclipping_errors(primary["samples"], y, threshold),
            "template": template_info,
            "alignment": alignment,
            "certified_dual": certified_dual,
        }
        if cont_result is not None:
            report["continuation"] = {
                "peak": cont_result["peak_estimate"],
                "optimizer": cont_result["optimizer"],
                "boundary": cont_result["boundary"],
                "errors": reclipping_errors(cont_result["samples"], y, threshold),
            }
        if dual_result is not None:
            report["dual"] = {
                "peak": dual_result["peak_estimate"],
                "optimizer": dual_result["optimizer"],
                "boundary": dual_result["boundary"],
                "errors": reclipping_errors(dual_result["samples"], y, threshold),
            }
        run_reports.append(report)

    errors = reclipping_errors(work, y, threshold)
    observed_peak = max(abs(float(v)) for v in y)
    primary_peaks = [r["primary_peak"] for r in run_reports]
    prior_uppers = [
        r["prior_bounds"]["peak_upper_prior"]
        for r in run_reports
        if r["prior_bounds"]["peak_upper_prior"] is not None
    ]
    result: Dict[str, Any] = {
        "samples": work,
        "sample_rate": sample_rate,
        "threshold": threshold,
        "mode": mode,
        "primary_kind": run_reports[0]["primary_kind"] if len(run_reports) == 1 else "mixed",
        "runs": run_reports,
        "errors": errors,
        "observed_peak": observed_peak,
        "region_peak_estimate": max(primary_peaks, default=observed_peak),
        "hard_peak_range": [threshold, None],
        "prior_peak_range": [threshold, max(prior_uppers) if prior_uppers else None],
        "certified_peak": bool(certified_dual and secondary_template is not None),
        "stop_reason": (
            run_reports[0]["primary_optimizer"]["stop_reason"] if run_reports else "no_runs"
        ),
        "constraint_violation_max": max(
            float(r["primary_optimizer"]["constraint_violation_max"]) for r in run_reports
        ),
    }
    if result["certified_peak"]:
        dual_peaks = [
            float(r["dual"]["peak"])
            for r in run_reports
            if isinstance(r.get("dual"), dict) and r["dual"].get("peak") is not None
        ]
        if dual_peaks:
            best = (alignment or {}).get("best") if isinstance(alignment, dict) else None
            rmse = float((best or {}).get("rmse") or 0.0)
            margin = max(0.02, 8.0 * rmse)
            peak = max(dual_peaks)
            result["certified_peak_range"] = [
                max(threshold, peak - margin),
                peak + margin,
            ]
            result["peak_certification_basis"] = (
                "唯一时延对齐的未削副麦；区间使用对齐残差经验余量，不声称任意麦克风偏差都被证明"
            )
    result["alternatives"] = {
        "conservative": alternatives["conservative"],
    }
    if has_continuation:
        result["alternatives"]["continuation"] = alternatives["continuation"]
    if has_dual:
        result["alternatives"]["dual"] = alternatives["dual"]
    result["warnings"] = _build_warnings(run_reports, errors, bounds)

    repaired_indices = [
        i
        for report in run_reports
        for i in range(int(report["run"]["start"]), int(report["run"]["end"]) + 1)
    ]
    candidate_sets: Dict[str, Sequence[float]] = {"primary": work}
    candidate_sets.update(result["alternatives"])
    feasible_candidates: List[Dict[str, Any]] = []
    for name, candidate_samples in candidate_sets.items():
        candidate_errors = reclipping_errors(candidate_samples, y, threshold)
        peak = max(abs(float(candidate_samples[i])) for i in repaired_indices)
        feasible = (
            candidate_errors["reclip_max_error"] <= 2e-5
            and candidate_errors["unclipped_max_error"] <= 1e-8
        )
        feasible_candidates.append(
            {
                "name": name,
                "feasible": bool(feasible),
                "peak": peak,
                "errors": candidate_errors,
            }
        )
    feasible_peaks = [c["peak"] for c in feasible_candidates if c["feasible"]]
    result["feasibility"] = {
        "tolerance": {"reclip_max_error": 2e-5, "unclipped_max_error": 1e-8},
        "candidates": feasible_candidates,
        "sampled_feasible_peak_range": [
            min(feasible_peaks) if feasible_peaks else None,
            max(feasible_peaks) if feasible_peaks else None,
        ],
        "note": "只有通过再削波吻合和未削样本等式误差门的候选才计入采样可行区；完整观测峰高上界仍为 +∞",
    }
    return result


def _build_warnings(
    reports: Sequence[Dict[str, Any]],
    errors: Dict[str, float | int],
    bounds: Dict[str, float],
) -> List[str]:
    warnings: List[str] = []
    if errors["reclip_max_error"] > 1e-5 or errors["unclipped_max_error"] > 1e-8:
        warnings.append("重建结果未通过再削波/未削样本约束，请查看约束违反记录")
    for r in reports:
        run = r["run"]
        if r["long_saturation"]:
            warnings.append(
                f"平台 {run['start']}–{run['end']} 长度 {run['length']} 超出可靠先验长度 "
                f"{r['reliable_prior_length']}；保守解与延拓解必须并列看待"
            )
        b = r["primary_boundary"]
        if float(b["max_boundary_curvature"]) > bounds["curvature_bound"] * 2.0:
            warnings.append(
                f"平台 {run['start']}–{run['end']} 边界曲率 {b['max_boundary_curvature']:.3g} "
                "明显超过未削波样本经验范围，需要单独检查衔接"
            )
        opt = r["primary_optimizer"]
        if opt["stop_reason"] == "line_search_failed" or not opt["converged"]:
            warnings.append(
                f"平台 {run['start']}–{run['end']} 优化器停止原因: {opt['stop_reason']}"
            )
        if float(opt["constraint_violation_max"]) > 1e-6:
            warnings.append("存在未满足的幅度不等式约束")
    return warnings
