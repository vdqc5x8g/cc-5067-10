from __future__ import annotations

import base64
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from .audio import read_wav, read_wav_bytes, wav_bytes
from .dual import align_secondary, forced_alignment
from .lab import Spec, build_lab_pack, lab_payload
from .lab.spec import MaterialSpecError
from .normalize import normalize_samples
from .pipeline import build_demo_pack, jsonable
from .solver import ClippingRestoreError, restore_signal

WEB_DIR = Path(__file__).parent / "web"
_REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_SPEC_PATH = _REPO_ROOT / "labs" / "analog" / "paired_clip.spec.json"


def _lab_spec_path() -> Path:
    return LAB_SPEC_PATH


def _b64_wav(samples, sample_rate: int) -> str:
    return base64.b64encode(wav_bytes(samples, sample_rate, "float32")).decode("ascii")


def create_handler(demo_dir: Path):
    demo_dir = demo_dir.resolve()

    class PanelHandler(BaseHTTPRequestHandler):
        server_version = "DeclipperPanel/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(jsonable(payload), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                self._send_json(404, {"error": "FileNotFoundError", "message": str(path)})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if path == "/api/health":
                    self._send_json(200, {"ok": True, "service": "declipper-panel"})
                    return
                if path == "/api/demo":
                    manifest_path = demo_dir / "manifest.json"
                    if not manifest_path.exists():
                        build_demo_pack(demo_dir)
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    self._send_json(200, manifest)
                    return
                if path == "/api/lab":
                    # Serve the live lab payload from the editable spec.
                    # Truth fields are included for display only and never
                    # flow into the reconstruction engine.
                    spec = Spec.from_json_file(_lab_spec_path())
                    payload = lab_payload(spec)
                    self._send_json(200, payload)
                    return
                if path == "/api/lab-spec":
                    self._send_json(200, Spec.from_json_file(_lab_spec_path()).to_dict())
                    return
                if path == "/api/track-samples":
                    qs = parse_qs(parsed.query)
                    track_id = qs.get("track_id", [""])[0]
                    kind = qs.get("kind", ["clipped"])[0]
                    track = self._load_track(track_id)
                    if kind == "secondary":
                        rel = track.get("secondary_url")
                    else:
                        rel = track.get("clipped_url")
                    if not rel:
                        raise ClippingRestoreError("该轨道没有请求的音频类型")
                    audio = read_wav(str(demo_dir / rel))
                    root_manifest = json.loads((demo_dir / "manifest.json").read_text(encoding="utf-8"))
                    self._send_json(200, {
                        "samples": audio.samples,
                        "sample_rate": audio.sample_rate,
                        "threshold": root_manifest.get("threshold"),
                        "selection": track.get("selection"),
                    })
                    return
                if path.startswith("/assets/"):
                    rel = path[len("/assets/"):]
                    safe = (demo_dir / rel).resolve()
                    if demo_dir not in safe.parents and safe != demo_dir:
                        self._send_json(403, {"error": "Forbidden"})
                        return
                    if safe.suffix == ".wav":
                        self._send_file(safe, "audio/wav")
                    elif safe.suffix == ".json":
                        self._send_file(safe, "application/json; charset=utf-8")
                    else:
                        self._send_file(safe, "application/octet-stream")
                    return
                if path in {"/", "/index.html"}:
                    self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
                    return
                if path in {"/lab", "/lab.html"}:
                    self._send_file(WEB_DIR / "lab.html", "text/html; charset=utf-8")
                    return
                if path == "/lab.js":
                    self._send_file(WEB_DIR / "lab.js", "text/javascript; charset=utf-8")
                    return
                if path == "/app.js":
                    self._send_file(WEB_DIR / "app.js", "text/javascript; charset=utf-8")
                    return
                if path == "/styles.css":
                    self._send_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
                    return
                self._send_json(404, {"error": "NotFound", "message": path})
            except Exception as exc:
                self._send_json(500, {"error": type(exc).__name__, "message": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    req = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    raise ClippingRestoreError(f"请求 JSON 无法解析: {exc}") from exc
                if parsed.path == "/api/parse":
                    try:
                        wav = base64.b64decode(req["wav_base64"])
                    except Exception as exc:
                        raise ClippingRestoreError("WAV base64 数据无法解码") from exc
                    audio = read_wav_bytes(wav)
                    self._send_json(200, {
                        "samples": audio.samples,
                        "sample_rate": audio.sample_rate,
                        "duration_seconds": audio.n / audio.sample_rate,
                    })
                    return
                if parsed.path == "/api/lab":
                    spec_data = req.get("spec")
                    if spec_data is None:
                        spec = Spec.from_json_file(_lab_spec_path())
                    else:
                        spec = Spec.from_dict(spec_data)
                    prior = float(req.get("prior_strength", 0.34))
                    self._send_json(200, lab_payload(
                        spec, prior_strength=prior,
                        reveal_truth=bool(req.get("reveal_truth", False)),
                    ))
                    return
                if parsed.path != "/api/restore":
                    self._send_json(404, {"error": "NotFound"})
                    return
                payload = self._handle_restore(req)
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(400, {
                    "error": type(exc).__name__,
                    "message": str(exc),
                })

        def _load_track(self, track_id: str) -> Dict[str, Any]:
            manifest_path = demo_dir / "manifest.json"
            if not manifest_path.exists():
                build_demo_pack(demo_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for track in manifest.get("tracks", []):
                if track.get("id") == track_id:
                    return track
            raise ClippingRestoreError(f"找不到演示轨道: {track_id}")

        def _handle_restore(self, req: Dict[str, Any]) -> Dict[str, Any]:
            threshold = float(req.get("threshold"))
            sample_rate = int(req.get("sample_rate", 8000))
            track: Optional[Dict[str, Any]] = None
            track_id = req.get("track_id")
            if track_id and "samples" not in req:
                track = self._load_track(str(track_id))
                audio = read_wav(str(demo_dir / track["clipped_url"]))
                samples = audio.samples
                sample_rate = audio.sample_rate
                if req.get("start") is None and track.get("selection"):
                    req["start"] = track["selection"]["start"]
                    req["end"] = track["selection"]["end"]
            else:
                raw_samples = req.get("samples")
                if not isinstance(raw_samples, list) or not raw_samples:
                    raise ClippingRestoreError("请求中没有音频样本")
                samples = [float(v) for v in raw_samples]

            start = req.get("start")
            end = req.get("end")
            start_i = None if start is None else int(start)
            end_i = None if end is None else int(end)

            secondary = None
            if isinstance(req.get("secondary"), list) and req["secondary"]:
                secondary = [float(v) for v in req["secondary"]]
            elif req.get("use_secondary") and track and track.get("secondary_url"):
                secondary = read_wav(str(demo_dir / track["secondary_url"])).samples

            template = None
            certified = False
            alignment: Optional[Dict[str, Any]] = None
            mode = str(req.get("mode", "auto"))
            alignment_mode = str(req.get("alignment_mode", "auto"))

            if secondary is not None:
                if alignment_mode == "force" and req.get("forced_shift") is not None:
                    alignment = forced_alignment(
                        samples,
                        secondary,
                        threshold,
                        int(req["forced_shift"]),
                        start=start_i,
                        end=end_i,
                    )
                    template = alignment["aligned_template"]
                    certified = False
                    if mode == "auto":
                        mode = "dual"
                else:
                    alignment = align_secondary(
                        samples,
                        secondary,
                        threshold,
                        start=start_i,
                        end=end_i,
                        sample_rate=sample_rate,
                    )
                    if alignment["status"] == "identified":
                        template = alignment["aligned_template"]
                        certified = True
                        if mode == "auto":
                            mode = "dual"
                    elif alignment_mode == "force" and alignment.get("shift") is not None:
                        # Auto produced a best ambiguous shift; caller explicitly forces it.
                        template = alignment["aligned_template"]
                        certified = False
                        if mode == "auto":
                            mode = "dual"
                    else:
                        # Do not claim a dual single solution when delay is not identifiable.
                        mode = "conservative" if mode in {"auto", "dual"} else mode

            result = restore_signal(
                samples,
                sample_rate,
                threshold,
                start=start_i,
                end=end_i,
                mode=mode,
                secondary_template=template,
                certified_dual=certified,
            )

            normalized = None
            normalized_wav = None
            if req.get("normalize"):
                certified_range = result.get("certified_peak_range")
                if result["certified_peak"] and certified_range:
                    upper = certified_range[1]
                else:
                    upper = result["prior_peak_range"][1]
                normalized_samples, plan = normalize_samples(
                    result["samples"],
                    target_rms_db=float(req.get("target_rms_db", -18.0)),
                    ceiling=float(req.get("ceiling", 0.98)),
                    peak_estimate=float(result["region_peak_estimate"]),
                    peak_upper_bound=upper,
                    certified_peak=bool(result["certified_peak"]),
                )
                normalized = plan
                normalized_wav = _b64_wav(normalized_samples, sample_rate)

            alignment_public = None
            if alignment is not None:
                alignment_public = {k: v for k, v in alignment.items() if k != "aligned_template"}

            return {
                "track_id": track_id,
                "sample_rate": sample_rate,
                "threshold": threshold,
                "start": start_i,
                "end": end_i,
                "mode_used": mode,
                "samples": result["samples"],
                "wav_base64": _b64_wav(result["samples"], sample_rate),
                "alternatives": result.get("alternatives", {}),
                "alternative_wav_base64": {
                    name: _b64_wav(values, sample_rate)
                    for name, values in result.get("alternatives", {}).items()
                },
                "normalized": normalized,
                "normalized_wav_base64": normalized_wav,
                "result": {k: v for k, v in result.items() if k != "samples" and k != "alternatives"},
                "alignment": alignment_public,
            }

    return PanelHandler


def run_server(host: str = "127.0.0.1", port: int = 8765, demo_dir: str = "examples/generated", open_browser: bool = False) -> None:
    demo_path = Path(demo_dir)
    manifest = demo_path / "manifest.json"
    if not manifest.exists():
        build_demo_pack(demo_path)
    httpd = ThreadingHTTPServer((host, port), create_handler(demo_path))
    url = f"http://{host}:{port}/"
    print(f"削波重建母带面板: {url}")
    print("按 Ctrl+C 停止")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
