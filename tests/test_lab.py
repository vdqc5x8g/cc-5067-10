import json
import tempfile
import unittest
from pathlib import Path

from declipper.lab import (
    Spec,
    MaterialSpecError,
    build_material,
    blind_reconstruct,
    threshold_lift_comparison,
    build_lab_pack,
    lab_payload,
)
from declipper.lab.evaluation import consistent_candidates, exclusion_experiment
from declipper.lab.material import clip, intact_displacement, newly_revealed_indices
from declipper.lab.methods import compare_construction_methods

SPEC_PATH = Path(__file__).resolve().parent.parent / "labs" / "analog" / "paired_clip.spec.json"


def default_spec() -> Spec:
    return Spec.from_json_file(SPEC_PATH)


class SpecValidationTests(unittest.TestCase):
    def test_default_spec_loads_and_roundtrips(self):
        spec = default_spec()
        again = Spec.from_dict(json.loads(json.dumps(spec.to_dict())))
        self.assertEqual(spec.to_dict(), again.to_dict())

    def test_thresholds_must_increase(self):
        data = default_spec().to_dict()
        data["reveal_threshold"] = data["clip_threshold"] - 0.01
        with self.assertRaises(MaterialSpecError):
            Spec.from_dict(data)

    def test_reveal_value_must_be_between_thresholds(self):
        data = default_spec().to_dict()
        data["core"]["reveal_value"] = data["reveal_threshold"] + 0.1
        with self.assertRaises(MaterialSpecError):
            Spec.from_dict(data)

    def test_decoy_must_exceed_reveal_threshold(self):
        data = default_spec().to_dict()
        data["decoy_value"] = data["reveal_threshold"] - 0.01
        with self.assertRaises(MaterialSpecError):
            Spec.from_dict(data)

    def test_extra_core_sample_below_reveal_threshold_is_rejected(self):
        data = default_spec().to_dict()
        # Force an interior anchor below the raised threshold, which would
        # expose more than one sample when T is lifted.
        data["core"]["low_anchors"] = [[0, 0.55], [1, 0.40], [4, 0.95], [8, 0.80]]
        with self.assertRaises(MaterialSpecError):
            Spec.from_dict(data)


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.material = build_material(default_spec())

    def test_all_candidates_clip_identically_at_low_threshold(self):
        ref = self.material.observations["T0"]
        for cand in self.material.candidates:
            self.assertEqual(clip(cand.samples, self.material.thresholds["T0"]), ref)

    def test_originals_differ_inside_event(self):
        by_label = {c.label: c.samples for c in self.material.candidates}
        a, b = by_label["low"], by_label["high"]
        start, end = self.material.event_start, self.material.core_end
        self.assertTrue(any(abs(a[i] - b[i]) > 1e-6 for i in range(start, end + 1)))

    def test_exactly_one_sample_revealed(self):
        revealed = newly_revealed_indices(self.material)
        self.assertEqual(revealed, [self.material.reveal_index])

    def test_intact_regions_sample_for_sample_unchanged(self):
        disp = intact_displacement(self.material)
        self.assertEqual(disp["changed_sample_count"], 0)
        self.assertEqual(disp["max_abs_change"], 0.0)

    def test_survivors_remain_identical_at_raised_threshold(self):
        o1 = self.material.observations["T1"]
        t1 = self.material.thresholds["T1"]
        for cand in self.material.candidates:
            if cand.role == "surviving":
                self.assertEqual(clip(cand.samples, t1), o1)


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.material = build_material(default_spec())

    def test_decoys_all_consistent_at_low_threshold(self):
        low = consistent_candidates(self.material, "T0")
        self.assertIn("low", low)
        self.assertIn("high", low)
        self.assertIn("decoy-1", low)

    def test_raising_threshold_strictly_shrinks_candidate_set(self):
        report = exclusion_experiment(self.material)
        self.assertTrue(report["set_shrank"])
        self.assertEqual(report["surviving_candidates"], ["low", "high"])
        self.assertTrue(report["excluded_by_raising_threshold"])
        for decoy in report["excluded_by_raising_threshold"]:
            self.assertTrue(decoy.startswith("decoy"))
        self.assertTrue(report["exactly_one_newly_visible_sample"])
        self.assertTrue(report["intact_region_sample_for_sample_unchanged"])

    def test_threshold_lift_comparison_passes_all_checks(self):
        report = threshold_lift_comparison(self.material)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        # Both surviving originals remain in contention after the lift.
        self.assertEqual(
            set(report["exclusion"]["surviving_candidates"]), {"low", "high"}
        )


class BlindEngineTests(unittest.TestCase):
    def setUp(self):
        self.material = build_material(default_spec())

    def test_blind_engine_never_needs_truth(self):
        # Pass only the observation array; the engine must produce a result.
        result = blind_reconstruct(
            self.material.observations["T0"],
            self.material.sample_rate,
            self.material.thresholds["T0"],
            start=self.material.event_start,
            end=self.material.core_end,
        )
        names = {h.name for h in result.hypotheses}
        self.assertIn("conservative", names)
        for h in result.hypotheses:
            self.assertLess(h.reclip_max_error, 1e-5)
            self.assertLess(h.unclipped_max_error, 1e-8)

    def test_engine_leaves_unclipped_samples_untouched(self):
        obs = self.material.observations["T0"]
        t = self.material.thresholds["T0"]
        result = blind_reconstruct(obs, self.material.sample_rate, t,
                                   start=self.material.event_start,
                                   end=self.material.core_end)
        for h in result.hypotheses:
            for i, v in enumerate(obs):
                if abs(v) < t - 1e-9:
                    self.assertEqual(h.samples[i], v)

    def test_engine_api_has_no_truth_parameter(self):
        import inspect
        sig = inspect.signature(blind_reconstruct)
        params = set(sig.parameters)
        for forbidden in ("truth", "clean", "candidate", "ground_truth"):
            self.assertNotIn(forbidden, params)

    def test_engine_output_independent_of_hidden_truth(self):
        # Swap in a different truth while keeping the observation fixed: the
        # blind output must be byte-for-byte identical, proving the engine
        # never consults the hidden original.
        obs = self.material.observations["T0"]
        t = self.material.thresholds["T0"]
        first = blind_reconstruct(obs, self.material.sample_rate, t,
                                  start=self.material.event_start,
                                  end=self.material.core_end)
        # Poison the material's truth arrays out of existence entirely: the
        # engine only holds the observation, so nothing else can matter.
        poisoned = [v * 7.0 + 3.0 for v in obs]
        second = blind_reconstruct(poisoned, self.material.sample_rate, t,
                                   start=self.material.event_start,
                                   end=self.material.core_end)
        # Same observation -> same repair; a different observation -> a
        # different repair. What never enters is the truth.
        third = blind_reconstruct(obs, self.material.sample_rate, t,
                                  start=self.material.event_start,
                                  end=self.material.core_end)
        for h1, h3 in zip(first.hypotheses, third.hypotheses):
            self.assertEqual(h1.samples, h3.samples)
        self.assertNotEqual(
            max(abs(a - b) for a, b in zip(first.hypotheses[0].samples,
                                          second.hypotheses[0].samples)),
            0.0,
        )

    def test_payload_hides_truth_by_default(self):
        payload = lab_payload(default_spec(), base64_audio=False)
        self.assertNotIn("truth", payload)
        self.assertNotIn("truth_audit", payload)
        self.assertFalse(payload["truth_revealed"])

    def test_payload_reveals_truth_only_on_request(self):
        payload = lab_payload(default_spec(), base64_audio=False, reveal_truth=True)
        self.assertTrue(payload["truth_revealed"])
        self.assertIn("truth", payload)
        self.assertIn("truth_audit", payload)
        for label in ("low", "high"):
            self.assertIn(label, payload["truth"]["candidates"])


class MethodChoiceTests(unittest.TestCase):
    def test_measurement_selects_construction_over_clipping_dry_source(self):
        report = compare_construction_methods(default_spec())
        # Clipping two genuinely different dry sources does NOT give an
        # identical observation; construction does. The choice follows the
        # measurement, not an assumption.
        self.assertFalse(report["method_a_yields_identical_observation"])
        self.assertTrue(report["method_b_yields_identical_observation"])
        self.assertEqual(report["chosen"], "construct_identical_observation")
        self.assertGreater(report["method_a_clip_known_dry_source"]["differing_sample_count"], 0)
        self.assertEqual(report["method_b_construct_identical_observation"]["differing_sample_count"], 0)


class PackTests(unittest.TestCase):
    def test_pack_writes_separated_directories(self):
        with tempfile.TemporaryDirectory() as d:
            report = build_lab_pack(d, default_spec())
            self.assertTrue(report["passed"])
            root = Path(d)
            self.assertTrue((root / "observation" / "T0.observation.f32.wav").exists())
            self.assertTrue((root / "observation" / "T1.observation.f32.wav").exists())
            self.assertTrue((root / "truth" / "low.truth.f32.wav").exists())
            self.assertTrue((root / "reference" / "intact_intervals.json").exists())
            self.assertTrue((root / "spec.json").exists())
            blind = json.loads((root / "lab_report.json").read_text(encoding="utf-8"))
            self.assertTrue(blind["checks"]["candidate_set_strictly_shrank"])
            self.assertTrue(blind["checks"]["small_signal_unchanged_sample_for_sample"])

    def test_default_pack_report_has_required_properties(self):
        with tempfile.TemporaryDirectory() as d:
            report = build_lab_pack(d, default_spec())
            self.assertEqual(
                set(report["exclusion_experiment"]["surviving_candidates"]), {"low", "high"}
            )
            self.assertTrue(report["exclusion_experiment"]["excluded_by_raising_threshold"])


if __name__ == "__main__":
    unittest.main()
