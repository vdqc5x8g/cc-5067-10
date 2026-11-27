"""Build the on-disk lab pack: observations, truth, reference, and reports.

Directory policy (enforced here and documented in the lab README):

    pack/
      observation/      <- safe to hand to the reconstruction engine
        T0.observation.f32.wav
        T1.observation.f32.wav
      truth/            <- NEVER read by the engine; sealed for evaluation
        low.truth.f32.wav
        high.truth.f32.wav
        decoy-1.truth.f32.wav ...
      reference/
        intact_intervals.json   (sample intervals untouched by clipping)
        pair_observations.json  (byte-level equality proof at T0/T1)
      spec.json                 (editable generator input)
      lab_report.json
      blind/                    (engine outputs, observation-only inputs)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from ..audio import write_wav
from .engine import blind_reconstruct, hypothesis_table, prior_envelope_for
from .evaluation import (
    blind_comparison,
    candidate_is_consistent,
    exclusion_experiment,
    threshold_lift_comparison,
)
from .methods import compare_construction_methods
from .material import (
    LabMaterial,
    build_material,
    intact_displacement,
    observations_identical_at_low_threshold,
)
from .spec import Spec


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def build_lab_pack(out_dir: str | Path, spec: Spec) -> Dict[str, Any]:
    root = Path(out_dir)
    clean_dirs = ["observation", "truth", "reference", "blind"]
    for name in clean_dirs:
        d = root / name
        if d.exists():
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
    (root / "observation").mkdir(parents=True, exist_ok=True)
    (root / "truth").mkdir(parents=True, exist_ok=True)
    (root / "reference").mkdir(parents=True, exist_ok=True)
    (root / "blind").mkdir(parents=True, exist_ok=True)

    material = build_material(spec)
    sr = material.sample_rate

    # ---- observations (engine-safe) ----
    for level, key in (("low", "T0"), ("raised", "T1")):
        write_wav(str(root / "observation" / f"{key}.observation.f32.wav"),
                  material.observations[key], sr)

    # ---- truth (sealed) ----
    for cand in material.candidates:
        write_wav(str(root / "truth" / f"{cand.label}.truth.f32.wav"), cand.samples, sr)

    # ---- reference intervals ----
    intact_disp = intact_displacement(material)
    _write_json(root / "reference" / "intact_intervals.json", {
        "sample_rate": sr,
        "event_start": material.event_start,
        "event_end": material.core_end,
        "reveal_index": material.reveal_index,
        "intact_intervals": [list(iv) for iv in material.intact_intervals],
        "policy": "reference 区间仅用于事后核对与评分；重建引擎不得读取",
        "intact_displacement_T0_to_T1": intact_disp,
    })
    _write_json(root / "reference" / "pair_observations.json", {
        "low_threshold": material.thresholds["T0"],
        "raised_threshold": material.thresholds["T1"],
        "identical_at_low_threshold_across_all_candidates": observations_identical_at_low_threshold(material),
        "identical_at_raised_threshold_for_survivors": all(
            candidate_is_consistent(c.samples, material.observations["T1"], material.thresholds["T1"])
            for c in material.candidates if c.role == "surviving"
        ),
        "plateau_observation_T0": material.observations["T0"][material.event_start:material.core_end + 1],
        "plateau_observation_T1": material.observations["T1"][material.event_start:material.core_end + 1],
    })

    # ---- editable spec ----
    _write_json(root / "spec.json", spec.to_dict())

    # ---- blind runs at both thresholds, engine sees observation only ----
    blind_runs: Dict[str, Any] = {}
    for level, key in (("low", "T0"), ("raised", "T1")):
        obs = material.observations[key]
        t = material.thresholds[key]
        result = blind_reconstruct(obs, sr, t,
                                   start=material.event_start, end=material.core_end)
        blind_runs[key] = {
            "threshold": t,
            "run": result.run,
            "notes": result.notes,
            "hypotheses": hypothesis_table(result, obs),
            "prior_envelope": prior_envelope_for(obs, t, result.run),
            "identifiable_from_observation_alone": False,
        }
        for h in result.hypotheses:
            write_wav(str(root / "blind" / f"{key}.{h.name}.f32.wav"), h.samples, sr)

    # ---- comparison reports ----
    exclusion = exclusion_experiment(material)
    lift = threshold_lift_comparison(material)
    compare = blind_comparison(material, "T0")

    report = {
        "title": "同观测不同原音的削顶教学音轨",
        "sample_rate": sr,
        "thresholds": material.thresholds,
        "event": {
            "start": material.event_start,
            "end": material.core_end,
            "core_length": material.core_length,
            "reveal_index": material.reveal_index,
        },
        "construction": {
            "identical_low_threshold_observation": True,
            "candidates": [
                {"label": c.label, "role": c.role} for c in material.candidates
            ],
            "intact_displacement": intact_disp,
        },
        "blind_runs": blind_runs,
        "exclusion_experiment": exclusion,
        "threshold_lift": lift,
        "construction_method_measurement": compare_construction_methods(spec),
        "truth_isolation": {
            "engine_api_accepts": ["observation", "sample_rate", "threshold", "start", "end", "prior_strength"],
            "engine_api_rejects": ["truth", "candidate", "clean"],
            "truth_directory": "truth/",
            "observation_directory": "observation/",
            "generator_proof": "build_material 用 clip(original) 构造观测；engine.blind_reconstruct 只接收 observation 数组",
        },
        "audio": {
            "observations": {
                "T0": "observation/T0.observation.f32.wav",
                "T1": "observation/T1.observation.f32.wav",
            },
            "truth": {c.label: f"truth/{c.label}.truth.f32.wav" for c in material.candidates},
            "blind": {
                f"{key}.{h}": f"blind/{key}.{h}.f32.wav"
                for key in ("T0", "T1") for h in ("conservative", "continuation")
            },
        },
        "compare": compare,
        "checks": lift["checks"],
        "passed": bool(lift["passed"]),
    }
    _write_json(root / "lab_report.json", report)
    return report


def clip_eq(material: LabMaterial, label: str) -> bool:
    from .evaluation import candidate_is_consistent
    cand = material.candidate(label)
    return candidate_is_consistent(
        cand.samples, material.observations["T1"], material.thresholds["T1"]
    )


def lab_payload(spec: Spec, *, prior_strength: float = 0.34,
                base64_audio: bool = True, reveal_truth: bool = False) -> Dict[str, Any]:
    """In-memory lab report for the live panel.

    By default the payload carries *no* truth waveforms and *no* per-candidate
    scores: only the exclusion verdict (which candidates survive a threshold
    lift) and the observation-only reconstruction. Passing ``reveal_truth``
    additionally attaches sealed truth audio and per-candidate grading under
    ``truth_audit`` — used only by the explicit "reveal" control, never by the
    reconstruction engine.
    """
    import base64

    from ..audio import wav_bytes
    from .engine import blind_reconstruct, hypothesis_table, injected_peak, prior_envelope_for, hypothesis_difference
    from .evaluation import exclusion_experiment, threshold_lift_comparison

    material = build_material(spec)
    sr = material.sample_rate
    obs0 = material.observations["T0"]
    obs1 = material.observations["T1"]

    blind: Dict[str, Any] = {}
    audio: Dict[str, Any] = {"observation": {}, "blind": {}, "injected": {}, "difference": {}}
    waveforms: Dict[str, Any] = {"observation": {}, "blind": {}, "injected": {}, "difference": {}}
    hypothesis_samples: Dict[str, List[float]] = {}
    for key, obs in (("T0", obs0), ("T1", obs1)):
        t = material.thresholds[key]
        result = blind_reconstruct(obs, sr, t, start=material.event_start,
                                   end=material.core_end, prior_strength=prior_strength)
        rows = hypothesis_table(result, obs)
        blind[key] = {
            "threshold": t,
            "run": result.run,
            "notes": result.notes,
            "hypotheses": rows,
            "prior_envelope": prior_envelope_for(obs, t, result.run),
            "identifiable_from_observation_alone": False,
        }
        waveforms["observation"][key] = obs
        if base64_audio:
            audio["observation"][key] = base64.b64encode(wav_bytes(obs, sr, "float32")).decode("ascii")
        for h in result.hypotheses:
            name = f"{key}.{h.name}"
            hypothesis_samples[name] = h.samples
            waveforms["blind"][name] = h.samples
            inj = injected_peak(h.samples, t)
            waveforms["injected"][name] = inj
            if base64_audio:
                audio["blind"][name] = base64.b64encode(
                    wav_bytes(h.samples, sr, "float32")).decode("ascii")
                audio["injected"][name] = base64.b64encode(
                    wav_bytes(inj, sr, "float32")).decode("ascii")
        # Compare the two hypotheses where both exist: the audible gap.
        if len(result.hypotheses) == 2:
            diff = hypothesis_difference(result.hypotheses[0].samples,
                                         result.hypotheses[1].samples)
            waveforms["difference"][key] = diff
            if base64_audio:
                audio["difference"][key] = base64.b64encode(
                    wav_bytes(diff, sr, "float32")).decode("ascii")

    lift = threshold_lift_comparison(material, prior_strength)
    intact_disp = intact_displacement(material)
    payload: Dict[str, Any] = {
        "sample_rate": sr,
        "thresholds": material.thresholds,
        "event": {
            "start": material.event_start,
            "end": material.core_end,
            "core_length": material.core_length,
            "reveal_index": material.reveal_index,
        },
        "spec": spec.to_dict(),
        "construction": {
            "identical_low_threshold_observation": observations_identical_at_low_threshold(material),
            "candidates": [{"label": c.label, "role": c.role} for c in material.candidates],
            "plateau_observation_T0": obs0[material.event_start:material.core_end + 1],
            "plateau_observation_T1": obs1[material.event_start:material.core_end + 1],
            "intact_displacement": intact_disp,
        },
        "blind": blind,
        "exclusion_experiment": exclusion_experiment(material),
        "construction_method_measurement": compare_construction_methods(spec),
        "checks": lift["checks"],
        "passed": bool(lift["passed"]),
        "waveforms": waveforms,
        "audio_base64": audio,
        "truth_revealed": bool(reveal_truth),
    }
    if reveal_truth:
        payload["truth"] = {
            "intact_intervals": [list(iv) for iv in material.intact_intervals],
            "intact_mask": material.intact_mask(),
            "candidates": {
                c.label: {
                    "role": c.role,
                    "samples": c.samples,
                    "audio_base64": (
                        base64.b64encode(wav_bytes(c.samples, sr, "float32")).decode("ascii")
                        if base64_audio else None
                    ),
                }
                for c in material.candidates
            },
            "policy": "仅用于揭示后的核对与评分；重建引擎从未读取这些数组",
        }
        payload["truth_audit"] = {
            name: _truth_rows(material, name.split(".")[0], samples)
            for name, samples in hypothesis_samples.items()
        }
    return payload


def _truth_rows(material: LabMaterial, level: str, reconstructed) -> List[Dict[str, Any]]:
    from .evaluation import truth_audit
    return truth_audit(material, level, reconstructed)
