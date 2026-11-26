from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .audio import read_wav, write_wav
from .dual import align_secondary, forced_alignment
from .pipeline import build_demo_pack, jsonable, run_period_offset_check
from .server import run_server
from .solver import ClippingRestoreError, restore_signal


def _parse_range(args: argparse.Namespace, sample_rate: int):
    start = args.start_sample
    end = args.end_sample
    if args.start_time is not None:
        start = int(round(float(args.start_time) * sample_rate))
    if args.end_time is not None:
        end = int(round(float(args.end_time) * sample_rate))
    return start, end


def restore_command(args: argparse.Namespace) -> int:
    audio = read_wav(args.input)
    start, end = _parse_range(args, audio.sample_rate)
    secondary_template = None
    certified = False
    alignment: Optional[Dict[str, Any]] = None
    mode = args.mode

    if args.secondary:
        sec = read_wav(args.secondary)
        if args.shift is not None:
            alignment = forced_alignment(
                audio.samples,
                sec.samples,
                args.threshold,
                args.shift,
                start=start,
                end=end,
            )
        else:
            alignment = align_secondary(
                audio.samples,
                sec.samples,
                args.threshold,
                start=start,
                end=end,
                sample_rate=audio.sample_rate,
            )
        if alignment.get("status") == "identified":
            secondary_template = alignment["aligned_template"]
            certified = True
            mode = "dual" if mode == "auto" else mode
        elif args.shift is not None and args.force_unverified_dual:
            secondary_template = alignment["aligned_template"]
            certified = False
            mode = "dual"
        elif not args.fallback_single:
            print(json.dumps({
                "error": "副麦时延不可唯一识别，不能导出认证双轨单解",
                "alignment": {k: v for k, v in alignment.items() if k != "aligned_template"},
            }, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        else:
            mode = "conservative" if mode in {"auto", "dual"} else mode

    result = restore_signal(
        audio.samples,
        audio.sample_rate,
        args.threshold,
        start=start,
        end=end,
        mode=mode,
        secondary_template=secondary_template,
        certified_dual=certified,
    )
    out = Path(args.output)
    if args.pcm16:
        from .audio import write_pcm16_preview

        peak = max(abs(v) for v in result["samples"]) or 1.0
        gain = min(1.0, 0.99 / peak) if args.safe_preview_gain else 1.0
        write_pcm16_preview(str(out), result["samples"], audio.sample_rate, gain)
    else:
        write_wav(str(out), result["samples"], audio.sample_rate, "float32")

    report = {
        "output": str(out),
        "sample_rate": audio.sample_rate,
        "threshold": args.threshold,
        "primary_kind": result["primary_kind"],
        "region_peak_estimate": result["region_peak_estimate"],
        "hard_peak_range_observable": result["hard_peak_range"],
        "prior_peak_range": result["prior_peak_range"],
        "certified_peak": result["certified_peak"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "stop_reason": result["stop_reason"],
        "alignment": None if alignment is None else {
            k: v for k, v in alignment.items() if k != "aligned_template"
        },
        "runs": result["runs"],
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(jsonable({k: v for k, v in report.items() if k != "runs"}), ensure_ascii=False, indent=2))
    return 0


def demo_command(args: argparse.Namespace) -> int:
    manifest = build_demo_pack(args.out)
    print(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2))
    return 0


def period_check_command(args: argparse.Namespace) -> int:
    report = run_period_offset_check(Path(args.out))
    print(json.dumps(jsonable(report["checks"]), ensure_ascii=False, indent=2))
    return 0 if all(report["checks"].values()) else 3


def panel_command(args: argparse.Namespace) -> int:
    run_server(host=args.host, port=args.port, demo_dir=args.demo_dir, open_browser=args.open_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="declipper",
        description="已知对称硬削波阈值的约束重建原型；CLI 仅辅助图形母带面板。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="生成合成弦音、WAV、评估和失败样例")
    p_demo.add_argument("--out", default="examples/generated")
    p_demo.set_defaults(func=demo_command)

    p_restore = sub.add_parser("restore", help="重建一个或多个框选削波平台并导出 WAV")
    p_restore.add_argument("input")
    p_restore.add_argument("--threshold", type=float, required=True)
    p_restore.add_argument("--start-sample", type=int)
    p_restore.add_argument("--end-sample", type=int)
    p_restore.add_argument("--start-time", type=float)
    p_restore.add_argument("--end-time", type=float)
    p_restore.add_argument("--secondary")
    p_restore.add_argument("--shift", type=int, help="强制副麦偏移样本数；结果标记为未认证")
    p_restore.add_argument(
        "--force-unverified-dual",
        action="store_true",
        help="与时延不可识别/--shift 一起使用：仍运行双轨模板，但不认证峰高",
    )
    p_restore.add_argument(
        "--fallback-single",
        action="store_true",
        help="副麦不可对齐时回退单轨峰高范围",
    )
    p_restore.add_argument("--mode", choices=["auto", "conservative", "continuation", "dual"], default="auto")
    p_restore.add_argument("--output", "-o", required=True)
    p_restore.add_argument("--report")
    p_restore.add_argument("--pcm16", action="store_true")
    p_restore.add_argument("--safe-preview-gain", action="store_true")
    p_restore.set_defaults(func=restore_command)

    p_period = sub.add_parser("period-check", help="运行偏移一个周期的响度归一安全检查")
    p_period.add_argument("--out", default="examples/generated")
    p_period.set_defaults(func=period_check_command)

    p_panel = sub.add_parser("panel", aliases=["serve"], help="启动图形化母带试听/削波重建面板")
    p_panel.add_argument("--host", default="127.0.0.1")
    p_panel.add_argument("--port", type=int, default=8765)
    p_panel.add_argument("--demo-dir", default="examples/generated")
    p_panel.add_argument("--open-browser", action="store_true")
    p_panel.set_defaults(func=panel_command)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ClippingRestoreError as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(json.dumps({"error": "FileNotFoundError", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
