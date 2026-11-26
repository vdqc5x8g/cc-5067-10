import unittest

from declipper.dual import align_secondary
from declipper.normalize import plan_loudness_normalization
from declipper.synth import delayed_secondary, hard_clip, periodic_tone, synthesize_chord


class DualAndNormalizerTests(unittest.TestCase):
    def test_unique_delay_is_identified(self):
        clean = synthesize_chord(peak=1.1, seed=9)
        clipped = hard_clip(clean, 0.34)
        secondary = delayed_secondary(clean, 57, gain=0.8, noise=0.002)
        al = align_secondary(clipped, secondary, 0.34, start=204, end=211)
        self.assertEqual(al["status"], "identified")
        self.assertEqual(al["shift"], 57)
        self.assertTrue(al["certified"])

    def test_exact_periodic_signal_is_ambiguous(self):
        clean = periodic_tone()
        clipped = hard_clip(clean, 0.30)
        secondary = delayed_secondary(clean, 61, gain=0.9, noise=0.0005)
        al = align_secondary(clipped, secondary, 0.30, max_shift=140, radius=450)
        self.assertEqual(al["status"], "ambiguous_delay")
        self.assertFalse(al["certified"])
        self.assertGreaterEqual(len(al["candidates"]), 2)

    def test_guarded_normalizer_does_not_boost_unknown_peak(self):
        plan = plan_loudness_normalization(
            [0.2] * 100 + [-0.2] * 100,
            target_rms_db=-10,
            peak_estimate=0.31,
            certified_peak=False,
        )
        self.assertIsNone(plan["certified_headroom_db"])
        self.assertFalse(plan["uses_uncertain_peak_as_headroom"])
        self.assertLessEqual(plan["gain"], 1.0 + 1e-12)
        self.assertGreater(plan["boost_blocked_db"], 0)

    def test_naive_policy_is_only_a_diagnostic_contrast(self):
        plan = plan_loudness_normalization(
            [0.2] * 100,
            target_rms_db=-10,
            peak_estimate=0.31,
            policy="naive_point_estimate",
        )
        self.assertTrue(plan["uses_uncertain_peak_as_headroom"])
        self.assertIsNotNone(plan["believed_headroom_db"])


if __name__ == "__main__":
    unittest.main()
