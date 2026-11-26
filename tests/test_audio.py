import tempfile
import unittest
from pathlib import Path

from declipper.audio import read_wav, wav_bytes, write_wav


class WavTests(unittest.TestCase):
    def test_float_wav_roundtrip(self):
        samples = [0.0, 0.25, -0.5, 0.99, 1.2]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.wav"
            write_wav(str(path), samples, 8000)
            audio = read_wav(str(path))
            self.assertEqual(audio.sample_rate, 8000)
            self.assertEqual(len(audio.samples), len(samples))
            for a, b in zip(samples, audio.samples):
                self.assertAlmostEqual(a, b, places=6)

    def test_wav_bytes_have_riff_header(self):
        data = wav_bytes([0.1, -0.1], 8000)
        self.assertEqual(data[:4], b"RIFF")
        self.assertEqual(data[8:12], b"WAVE")


if __name__ == "__main__":
    unittest.main()
