from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .audio import write_wav
from .dual import align_secondary, forced_alignment
from .evaluation import evaluate_restoration
from .normalize import apply_gain, plan_loudness_normalization
from .solver import detect_runs, restore_signal
from .synth import (
    delayed_secondary,
    hard_clip,
    periodic_tone,
    synthesize_chord,
)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _url(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _alignment_brief(al: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in al.items()
        if k not in {"aligned_template"}
    }


def run_period_offset_check(root: Path, sample_rate: int = 8000, threshold: float = 0.34) -> Dict[str, Any]:
    out = root / "period_offset"
    out.mkdir(parents=True, exist_ok=True)
    clean = synthesize_chord(peak=1.18, freqs=(220.0, 277.18, 329.63, 440.0), seed=9)
    clipped = hard_clip(clean, threshold)
    runs = [r for r in detect_runs(clipped, threshold) if 8 <= r.length <= 12]
    run = runs[min(3, len(runs) - 1)]
    secondary = delayed_secondary(clean, 57, gain=0.8, noise=0.003)

    correct = align_secondary(
        clipped, secondary, threshold, start=run.start, end=run.end, sample_rate=sample_rate
    )
    period = int(correct["period"]["period_samples"] or 36)
    wrong_shift = 57 + period
    wrong_align = forced_alignment(
        clipped, secondary, threshold, wrong_shift, start=run.start, end=run.end
    )
    wrong_result = restore_signal(
        clipped,
        sample_rate,
        threshold,
        start=run.start,
        end=run.end,
        mode="dual",
        secondary_template=wrong_align["aligned_template"],
        secondary_weight=8.0,
        certified_dual=False,
        alignment={"status": "forced_one_period_offset"},
    )
    peak = float(wrong_result["region_peak_estimate"])
    target_rms = -16.0
    naive = plan_loudness_normalization(
        wrong_result["samples"],
        target_rms_db=target_rms,
        peak_estimate=peak,
        policy="naive_point_estimate",
    )
    guarded = plan_loudness_normalization(
        wrong_result["samples"],
        target_rms_db=target_rms,
        peak_estimate=peak,
        certified_peak=False,
    )
    naive_gain = float(naive["gain"])
    guarded_gain = float(guarded["gain"])
    true_peak = max(abs(v) for v in clean)
    true_gained_peak = true_peak * naive_gain
    ceiling = 0.98
    gained_clean = apply_gain(clean, naive_gain)
    over_ceiling = sum(1 for v in gained_clean if abs(v) > ceiling)
    digital_clip_samples = sum(1 for v in gained_clean if abs(v) > 1.0)

    write_wav(out / "wrong_period_alignment_restored.f32.wav", wrong_result["samples"], sample_rate)
    write_wav(out / "naive_loudness_master.f32.wav", apply_gain(wrong_result["samples"], naive_gain), sample_rate)
    write_wav(out / "guarded_loudness_master.f32.wav", apply_gain(wrong_result["samples"], guarded_gain), sample_rate)

    report = {
        "scenario": "强制选择偏移一个弦音宏观周期的副麦时延，并把该错误选择传给响度归一器",
        "selection": {"start": run.start, "end": run.end, "length": run.length},
        "correct_alignment": _alignment_brief(correct),
        "wrong_alignment": _alignment_brief(wrong_align),
        "period_samples_added": period,
        "wrong_reconstructed_peak": peak,
        "true_peak_evaluation_only": true_peak,
        "naive_normalizer": naive,
        "guarded_normalizer": guarded,
        "evaluation_of_naive_gain_on_truth": {
            "gained_true_peak": true_gained_peak,
            "samples_over_ceiling": over_ceiling,
            "samples_over_digital_full_scale": digital_clip_samples,
            "naive_mistook_estimate_for_real_margin": bool(
                naive["believed_headroom_db"] is not None and true_gained_peak > ceiling
            ),
        },
        "checks": {
            "guarded_did_not_certify_headroom": guarded["certified_headroom_db"] is None,
            "guarded_blocked_positive_makeup": float(guarded["gain"]) <= 1.0 + 1e-12,
            "guarded_did_not_use_unknown_peak_as_margin": guarded[
                "uses_uncertain_peak_as_headroom"
            ]
            is False,
        },
        "audio": {
            "wrong_restored": "period_offset/wrong_period_alignment_restored.f32.wav",
            "naive_master": "period_offset/naive_loudness_master.f32.wav",
            "guarded_master": "period_offset/guarded_loudness_master.f32.wav",
        },
    }
    write_json(out / "period_offset_check.json", report)
    return report


def run_ambiguous_delay_example(root: Path, sample_rate: int = 8000, threshold: float = 0.30) -> Dict[str, Any]:
    out = root / "failure_ambiguous_delay"
    out.mkdir(parents=True, exist_ok=True)
    clean = periodic_tone(sample_rate=sample_rate, fundamental=200.0, peak=0.92)
    clipped = hard_clip(clean, threshold)
    secondary = delayed_secondary(clean, 61, gain=0.9, noise=0.0008, seed=41)
    alignment = align_secondary(clipped, secondary, threshold, max_shift=160, radius=500)
    write_wav(out / "periodic_clipped.f32.wav", clipped, sample_rate)
    write_wav(out / "unclipped_secondary.f32.wav", secondary, sample_rate)
    report = {
        "failure_type": "ambiguous_delay_one_period_alternatives",
        "policy": "不得导出认证的双轨单解；只能返回单轨峰高范围，或由用户强制时延并标记未认证",
        "alignment": _alignment_brief(alignment),
        "single_track_peak_range_observable": [threshold, None],
    }
    if alignment["status"] == "identified":
        # The synthetic exact-period signal should be ambiguous, but do not fake failure.
        report["warning"] = "此随机/参数下相关峰可区分；未构造出预期失败"
    write_json(out / "failure_ambiguous_delay.json", report)
    return report


def run_long_plateau_example(root: Path, sample_rate: int = 8000) -> Dict[str, Any]:
    out = root / "failure_long_plateau"
    out.mkdir(parents=True, exist_ok=True)
    threshold = 0.16
    clean = synthesize_chord(
        sample_rate=sample_rate,
        duration=1.5,
        freqs=(82.41, 123.47, 164.81, 220.0),
        peak=1.45,
        seed=31,
        decay=1.35,
        harmonics=6,
        brightness=1.15,
    )
    clipped = hard_clip(clean, threshold)
    from .solver import detect_runs

    run = max(detect_runs(clipped, threshold), key=lambda r: r.length)
    result = restore_signal(
        clipped, sample_rate, threshold, start=run.start, end=run.end, mode="auto"
    )
    conservative = result["alternatives"]["conservative"]
    continuation = result["alternatives"].get("continuation", result["samples"])
    cons_eval = evaluate_restoration(clean, clipped, {**result, "samples": conservative}, threshold)
    cont_eval = evaluate_restoration(clean, clipped, {**result, "samples": continuation}, threshold)
    primary_eval = evaluate_restoration(clean, clipped, result, threshold)

    write_wav(out / "long_clipped.f32.wav", clipped, sample_rate)
    write_wav(out / "long_clean_evaluation_only.f32.wav", clean, sample_rate)
    write_wav(out / "long_conservative.f32.wav", conservative, sample_rate)
    write_wav(out / "long_continuation_speculative.f32.wav", continuation, sample_rate)
    solver_report = {
        key: value
        for key, value in result.items()
        if key not in {"samples", "alternatives"}
    }
    report = {
        "failure_type": "plateau_longer_than_reliable_prior",
        "selection": {"start": run.start, "end": run.end, "length": run.length},
        "policy": "同时导出保守最小先验解和带谐波延拓的推测解，不用单一圆滑顶端冒充真值",
        "solver": solver_report,
        "evaluation_only": {
            "primary": primary_eval,
            "conservative": cons_eval,
            "continuation": cont_eval,
        },
        "audio": {
            "clipped": "failure_long_plateau/long_clipped.f32.wav",
            "conservative": "failure_long_plateau/long_conservative.f32.wav",
            "continuation": "failure_long_plateau/long_continuation_speculative.f32.wav",
            "clean_evaluation_only": "failure_long_plateau/long_clean_evaluation_only.f32.wav",
        },
    }
    write_json(out / "failure_long_plateau.json", report)
    return report


def build_evaluation_pack(root: Path, sample_rate: int = 8000) -> Dict[str, Any]:
    eval_dir = root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    clean = synthesize_chord(peak=1.18, freqs=(220.0, 277.18, 329.63, 440.0), seed=9)
    degrees = [0.55, 0.40, 0.27, 0.16]
    items = []
    for threshold in degrees:
        clipped = hard_clip(clean, threshold)
        # Keep the same musical peak/cycle across severity for comparable metrics.
        a, b = 204, 211
        result = restore_signal(
            clipped, sample_rate, threshold, start=a, end=b, mode="auto"
        )
        metrics = evaluate_restoration(clean, clipped, result, threshold)
        stem = f"degree_t{threshold:.2f}"
        write_wav(eval_dir / f"{stem}_clipped.f32.wav", clipped, sample_rate)
        write_wav(eval_dir / f"{stem}_restored.f32.wav", result["samples"], sample_rate)
        write_wav(eval_dir / f"{stem}_clean_evaluation_only.f32.wav", clean, sample_rate)
        if "conservative" in result["alternatives"]:
            write_wav(
                eval_dir / f"{stem}_conservative.f32.wav",
                result["alternatives"]["conservative"],
                sample_rate,
            )
        if "continuation" in result["alternatives"]:
            write_wav(
                eval_dir / f"{stem}_continuation.f32.wav",
                result["alternatives"]["continuation"],
                sample_rate,
            )
        item = {
            "threshold": threshold,
            "selection": {"start": a, "end": b},
            "metrics": metrics,
            "solver_summary": {
                "primary_kind": result["primary_kind"],
                "warnings": result["warnings"],
                "errors": result["errors"],
                "prior_peak_range": result["prior_peak_range"],
                "stop_reason": result["stop_reason"],
            },
            "audio": {
                "clipped": f"evaluation/{stem}_clipped.f32.wav",
                "restored": f"evaluation/{stem}_restored.f32.wav",
                "clean_evaluation_only": f"evaluation/{stem}_clean_evaluation_only.f32.wav",
            },
        }
        items.append(item)
    report = {
        "note": "干净波形仅在 evaluate_restoration 中用于事后评估，不进入 restore_signal",
        "items": items,
    }
    write_json(eval_dir / "evaluation_report.json", report)
    return report


def build_demo_pack(out_dir: str | Path = "examples/generated") -> Dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    threshold = 0.34
    specs = [
        {
            "id": "amber-low-peak",
            "title": "Amber 低峰值弦音",
            "peak": 0.72,
            "freqs": (196.0, 246.94, 293.66, 392.0),
            "seed": 3,
            "secondary": False,
            "selection": (204, 211),
        },
        {
            "id": "silver-high-peak",
            "title": "Silver 高峰值弦音（含未削副麦）",
            "peak": 1.18,
            "freqs": (220.0, 277.18, 329.63, 440.0),
            "seed": 9,
            "secondary": True,
            "selection": (204, 211),
        },
    ]

    tracks: List[Dict[str, Any]] = []
    plateaus: List[List[float]] = []
    for spec in specs:
        clean = synthesize_chord(
            sample_rate=sample_rate,
            peak=float(spec["peak"]),
            freqs=tuple(spec["freqs"]),  # type: ignore[arg-type]
            seed=int(spec["seed"]),
        )
        clipped = hard_clip(clean, threshold)
        a, b = spec["selection"]
        plateau = clipped[a : b + 1]
        plateaus.append([float(v) for v in plateau])

        clean_path = root / f"{spec['id']}.clean_evaluation_only.f32.wav"
        clipped_path = root / f"{spec['id']}.clipped.f32.wav"
        restored_path = root / f"{spec['id']}.single_track_restored.f32.wav"
        write_wav(clean_path, clean, sample_rate)
        write_wav(clipped_path, clipped, sample_rate)

        single = restore_signal(
            clipped, sample_rate, threshold, start=a, end=b, mode="auto"
        )
        write_wav(restored_path, single["samples"], sample_rate)

        track: Dict[str, Any] = {
            "id": spec["id"],
            "title": spec["title"],
            "clipped_url": _url(clipped_path, root),
            "restored_url": _url(restored_path, root),
            "clean_evaluation_url": _url(clean_path, root),
            "selection": {"start": a, "end": b},
            "plateau_values": [round(float(v), 8) for v in plateau],
            "single_track": {
                "region_peak_estimate": single["region_peak_estimate"],
                "hard_peak_range": single["hard_peak_range"],
                "prior_peak_range": single["prior_peak_range"],
                "primary_kind": single["primary_kind"],
                "warnings": single["warnings"],
                "errors": single["errors"],
                "stop_reason": single["stop_reason"],
            },
        }

        if spec["secondary"]:
            secondary = delayed_secondary(clean, 57, gain=0.8, noise=0.003)
            sec_path = root / f"{spec['id']}.unclipped_secondary.f32.wav"
            dual_path = root / f"{spec['id']}.dual_restored.f32.wav"
            write_wav(sec_path, secondary, sample_rate)
            alignment = align_secondary(
                clipped, secondary, threshold, start=a, end=b, sample_rate=sample_rate
            )
            if alignment["status"] == "identified":
                dual = restore_signal(
                    clipped,
                    sample_rate,
                    threshold,
                    start=a,
                    end=b,
                    mode="dual",
                    secondary_template=alignment["aligned_template"],
                    certified_dual=True,
                    alignment={"status": alignment["status"], "shift": alignment["shift"]},
                )
                write_wav(dual_path, dual["samples"], sample_rate)
                track["dual_url"] = _url(dual_path, root)
                track["dual"] = {
                    "region_peak_estimate": dual["region_peak_estimate"],
                    "certified_peak": dual["certified_peak"],
                    "stop_reason": dual["stop_reason"],
                }
            track["secondary_url"] = _url(sec_path, root)
            track["alignment"] = _alignment_brief(alignment)
        tracks.append(track)

    identical = len(plateaus) == 2 and len(plateaus[0]) == len(plateaus[1]) and all(
        abs(plateaus[0][i] - plateaus[1][i]) < 1e-12 for i in range(len(plateaus[0]))
    )
    manifest = {
        "sample_rate": sample_rate,
        "threshold": threshold,
        "graphical_entry": "python -m declipper panel  # http://127.0.0.1:8765",
        "identical_plateau_constructed": identical,
        "plateau_sample_count": len(plateaus[0]) if plateaus else 0,
        "tracks": tracks,
        "evaluation_report_url": "generated/evaluation/evaluation_report.json",
        "period_offset_report_url": "generated/period_offset/period_offset_check.json",
        "failure_ambiguous_delay_url": "generated/failure_ambiguous_delay/failure_ambiguous_delay.json",
        "failure_long_plateau_url": "generated/failure_long_plateau/failure_long_plateau.json",
    }
    write_json(root / "manifest.json", manifest)

    evaluation = build_evaluation_pack(root)
    period = run_period_offset_check(root)
    ambiguous = run_ambiguous_delay_example(root)
    long_case = run_long_plateau_example(root)
    write_json(
        root / "demo_report.json",
        {
            "manifest": manifest,
            "evaluation": evaluation,
            "period_offset_check": period,
            "ambiguous_delay": ambiguous,
            "long_plateau": {
                k: v for k, v in long_case.items() if k != "solver"
            },
        },
    )
    return manifest
