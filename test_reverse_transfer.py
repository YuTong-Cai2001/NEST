"""Tests for reverse-transfer sampling and unchanged source training."""

import contextlib
import io
import unittest

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_reverse_transfer import (
    AnomalyGFMTrainer, TemporalGraphDataset, normalize_target_graph,
    retained_windows, select_target_nodes, sparse_collate_fn, training_config,
)
from evaluate_reference_scoring import score_metrics


class ReverseTransferTests(unittest.TestCase):
    def test_label_blind_deterministic_selection(self):
        edges = np.array([[0, 1], [0, 2], [1, 2], [3, 4]])
        selected, degree = select_target_nodes(edges, 7, 2)
        np.testing.assert_array_equal(selected, [0, 1])
        np.testing.assert_array_equal(degree, [2, 2, 2, 1, 1, 0, 0])
        np.testing.assert_array_equal(select_target_nodes(edges[::-1], 7, 2)[0], selected)

    def test_window_sampling_matches_original(self):
        timestamps = np.arange(117) * 7 + 1
        origin, bins, ids = retained_windows(timestamps)
        self.assertEqual(origin, 1)
        np.testing.assert_array_equal(ids, np.arange(117)[::117 // 20][:20])
        np.testing.assert_array_equal(bins, np.arange(117))

    def test_directed_subgraph_normalization(self):
        edges = np.array([[0, 1], [0, 1], [1, 2], [3, 4]])
        mapping = np.array([0, 1, 2, -1, -1])
        graph, count = normalize_target_graph(edges, mapping, 3)
        self.assertEqual(count, 3)
        a = np.array([[1, 2, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
        inv = a.sum(axis=1) ** -.5
        np.testing.assert_allclose(graph.toarray(), a * inv[:, None] * inv[None, :], rtol=1e-6)

    def test_background_mask_and_direction(self):
        labels = np.array([0, 1, 2, 3])
        scores = np.array([.2, .9, 1., -2.])
        metrics = score_metrics(labels, scores, np.isin(labels, (0, 1)))
        self.assertEqual(metrics["auc"], 1)
        self.assertEqual(metrics["ap"], 1)

    def test_original_epoch_and_partial_source_labels(self):
        # This uses the REAL imported training epoch, not a rewritten loss.
        from pathlib import Path
        import tempfile
        from unittest.mock import patch

        features = [np.random.default_rng(7 + t).normal(size=(6, 24)).astype(np.float32)
                    for t in range(5)]
        graphs = [sp.csr_matrix(np.eye(6, dtype=np.float32) + np.roll(np.eye(6, dtype=np.float32), 1, axis=1))
                  for _ in range(5)]
        labels = np.array([0, 1, -1, -1, 0, -1], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            for variant in ("full", "no_proto"):
                cfg = training_config(Path(temporary), variant, 42)
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer = AnomalyGFMTrainer(cfg)
                    trainer.feature_dim = 24
                    dataset = TemporalGraphDataset({"temporal_sequences": [features],
                        "temporal_adjacencies": [graphs], "valid_timestamps": [0]}, 5, labels)
                    trainer.train_loader = DataLoader(dataset, batch_size=1,
                        collate_fn=sparse_collate_fn, num_workers=0)
                    trainer.create_model()
                    original = F.binary_cross_entropy_with_logits
                    counts = []

                    def checked(logits, y, **kwargs):
                        counts.append(y.numel())
                        return original(logits, y, **kwargs)

                    with patch("train_anomaly_gfm.F.binary_cross_entropy_with_logits", side_effect=checked):
                        metrics = trainer.train_epoch_sparse()
                self.assertEqual(counts, [3])
                self.assertTrue(np.isfinite(metrics["total_loss"]))
                expected = (metrics["anomaly_loss"] + trainer.lambda_proto * metrics["prototype_loss"]
                            + trainer.lambda_self * metrics["consistency_loss"])
                self.assertAlmostEqual(metrics["total_loss"], expected, places=5)
                self.assertIsNotNone(trainer.model.base_model.normal_prototype)
                del trainer
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
