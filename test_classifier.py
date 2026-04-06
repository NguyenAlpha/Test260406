import unittest

import numpy as np

from classifier import _group_from_reactions, predict_blood_group


class TestClassifier(unittest.TestCase):
    def test_group_mapping(self):
        self.assertEqual(_group_from_reactions(True, False, True), "A+")
        self.assertEqual(_group_from_reactions(False, True, False), "B-")
        self.assertEqual(_group_from_reactions(True, True, True), "AB+")
        self.assertEqual(_group_from_reactions(False, False, False), "O-")

    def test_prediction_structure(self):
        img = np.full((90, 300, 3), 150, dtype=np.uint8)
        # Add noisy texture in Anti-A and Anti-D regions to simulate positive reactions.
        rng = np.random.default_rng(1)
        img[:, :100] = np.clip(img[:, :100] + rng.integers(-60, 60, size=(90, 100, 3)), 0, 255)
        img[:, 200:] = np.clip(img[:, 200:] + rng.integers(-70, 70, size=(90, 100, 3)), 0, 255)

        # Lower threshold is intentional here so synthetic noise reliably crosses positive cutoff.
        result = predict_blood_group(img, threshold=0.2)
        self.assertIn("blood_group", result)
        self.assertIn("reactions", result)
        self.assertIn("confidence", result)
        self.assertIn(result["blood_group"], {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"})

    def test_invalid_image(self):
        with self.assertRaises(ValueError):
            predict_blood_group(np.array([]))


if __name__ == "__main__":
    unittest.main()
