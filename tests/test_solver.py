import math
import unittest

from declipper.solver import (
    ClippingRestoreError,
    detect_runs,
    restore_signal,
    select_runs,
)
from declipper.synth import hard_clip, synthesize_chord


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.clean = synthesize_chord(
            sample_rate=8000,
            duration=0.8,
            peak=0.95,
            freqs=(220.0, 330.0, 440.0),
            seed=5,
        )
        self.threshold = 0.38
        self.clipped = hard_clip(self.clean, self.threshold)

    def test_detects_symmetric_runs(self):
        runs = detect_runs(self.clipped, self.threshold)
        self.assertGreater(len(runs), 2)
        self.assertTrue(all(abs(self.clipped[r.start]) >= self.threshold for r in runs))

    def test_restoration_keeps_observations_and_reclips(self):
        run = max(detect_runs(self.clipped, self.threshold), key=lambda r: r.length)
        result = restore_signal(
            self.clipped, 8000, self.threshold, start=run.start, end=run.end
        )
        self.assertLess(result["errors"]["unclipped_max_error"], 1e-8)
        self.assertLess(result["errors"]["reclip_max_error"], 1e-8)
        self.assertEqual(result["errors"]["reclip_mismatch_count"], 0)
        self.assertLess(result["constraint_violation_max"], 1e-6)
        self.assertIn("feasibility", result)
        self.assertTrue(all(c["feasible"] for c in result["feasibility"]["candidates"]))
        for i in range(run.start, run.end + 1):
            if result["runs"][0]["run"]["sign"] == 1:
                self.assertGreaterEqual(result["samples"][i], self.threshold - 1e-6)
            else:
                self.assertLessEqual(result["samples"][i], -self.threshold + 1e-6)

    def test_prior_range_covers_demo_peak_and_hard_range_is_infinite(self):
        result = restore_signal(
            self.clipped, 8000, self.threshold, start=204, end=211
        )
        true_peak = max(abs(v) for v in self.clean[204:213])
        lo, hi = result["prior_peak_range"]
        self.assertEqual(lo, self.threshold)
        self.assertIsNone(result["hard_peak_range"][1])
        if hi is not None:
            self.assertLessEqual(lo, true_peak)
            self.assertGreaterEqual(hi, self.threshold)

    def test_selection_without_plateau_raises_typed_error(self):
        with self.assertRaises(ClippingRestoreError):
            select_runs([0.0, 0.1, -0.2, 0.05], 0.9, 0, 3)

    def test_invalid_threshold(self):
        with self.assertRaises(ClippingRestoreError):
            restore_signal(self.clipped, 8000, 0.0)

    def test_long_plateau_produces_two_candidates(self):
        clean = synthesize_chord(
            sample_rate=8000,
            duration=1.2,
            peak=1.5,
            freqs=(82.4, 164.8),
            seed=31,
            decay=1.2,
        )
        clipped = hard_clip(clean, 0.14)
        run = max(detect_runs(clipped, 0.14), key=lambda r: r.length)
        result = restore_signal(
            clipped, 8000, 0.14, start=run.start, end=run.end, mode="auto"
        )
        self.assertTrue(result["runs"][0]["long_saturation"])
        self.assertIn("conservative", result["alternatives"])
        self.assertIn("continuation", result["alternatives"])
        self.assertTrue(any("超出可靠先验" in w for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
