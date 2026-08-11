import unittest

import numpy as np
import torch

from sped.metrics import evaluate_all
from sped.model import AdditiveOnlyModel, InteractionPerturbationModel


class ModelShapeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.gene1 = torch.tensor([1, 2])
        self.gene2 = torch.tensor([2, 3])
        self.kind = torch.tensor([0, 1])
        self.reference = torch.zeros(2, 5)

    def test_additive_model_output_shape(self):
        model = AdditiveOnlyModel(4, 5, emb_dim=3, effect_hidden=7)
        output = model(self.gene1, self.gene2, self.kind, self.reference)
        self.assertEqual(tuple(output.shape), (2, 5))

    def test_interaction_model_is_order_invariant_for_double(self):
        model = InteractionPerturbationModel(
            4, 5, emb_dim=3, effect_hidden=7, interaction_hidden=6
        )
        kind = torch.ones(2, dtype=torch.long)
        first = model(self.gene1, self.gene2, kind, self.reference)
        second = model(self.gene2, self.gene1, kind, self.reference)
        torch.testing.assert_close(first, second)


class MetricTests(unittest.TestCase):
    def test_perfect_prediction_has_perfect_delta_correlation(self):
        reference = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        truth = np.array(
            [[2.0, 0.0, 1.5, 0.5], [0.5, 1.5, 0.0, 2.0]], dtype=np.float32
        )
        metrics = evaluate_all(truth, truth.copy(), reference, top_k=3)
        self.assertAlmostEqual(metrics["mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["pearson_delta"], 1.0, places=6)
        self.assertAlmostEqual(metrics["deg_recall"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
