"""Synthetic checks before running the fixed Elliptic target experiment."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_elliptic_transfer import (
    fixed_protocol, join_incidence, map_target_labels, training_config, wallet_table,
)


class EllipticTransferTests(unittest.TestCase):
    def test_time_specific_projection_and_full_neighborhoods(self):
        tx = pd.DataFrame({"txId": [10, 11, 12], "Time step": [45, 49, 44]})
        incoming = pd.DataFrame({"input_address": ["a", "b", "c", "q"], "txId": [10, 10, 11, 12]})
        outgoing = pd.DataFrame({"txId": [10, 10, 11, 12], "output_address": ["c", "d", "e", "r"]})
        presence, pairs = join_incidence(incoming, outgoing, tx)
        self.assertEqual(set(presence.address), set("abcde"))
        self.assertEqual(len(pairs[pairs.time == 45]), 4)
        self.assertEqual(len(pairs[pairs.time == 49]), 1)
        self.assertEqual(set(zip(pairs.input_address, pairs.output_address)),
                         {("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "e")})
        pd.testing.assert_frame_equal(join_incidence(incoming, outgoing, tx)[0], presence)

    def test_missing_transaction_fails(self):
        with self.assertRaises(ValueError):
            join_incidence(pd.DataFrame({"input_address": ["a"], "txId": [2]}),
                           pd.DataFrame({"output_address": ["b"], "txId": [1]}),
                           pd.DataFrame({"txId": [1], "Time step": [45]}))

    def test_labels_unknown_and_final_presence(self):
        ids = np.array(["a", "b", "c", "d"])
        classes = pd.DataFrame({"address": ids, "class": [1, 2, 3, 1]})
        labels, mask, active = map_target_labels(classes, ids, ["a", "b", "c"])
        np.testing.assert_array_equal(labels, [1, 0, -1, 1])
        np.testing.assert_array_equal(mask, [True, True, False, False])
        np.testing.assert_array_equal(active, [True, True, True, False])

    def test_feature_duplicates_are_not_samples(self):
        frame = pd.DataFrame({"address": ["b", "a", "a", "a"],
                              "Time step": [45, 45, 45, 49], "f": [2., 1., 1., 1.]})
        values, columns = wallet_table(frame, ["a", "b"])
        np.testing.assert_array_equal(values, [[1], [2]])
        self.assertEqual(columns, ["f"])
        with self.assertRaises(ValueError):
            wallet_table(frame.assign(**{"class": 1}), ["a", "b"])
        frame.loc[3, "f"] = 7
        with self.assertRaises(ValueError):
            wallet_table(frame, ["a", "b"])

    def test_main_experiment_not_ablation(self):
        self.assertEqual(fixed_protocol()["target_time_steps"], [45, 46, 47, 48, 49])
        for seed in (42, 43, 44):
            config = training_config(Path("temporary_test_not_created"), seed)
            self.assertEqual(config["seed"], seed)
            self.assertEqual(config["num_epochs"], 50)
            self.assertEqual(config["lambda_proto"], .3)
            self.assertEqual(config["lambda_self"], .5)
            self.assertEqual(config["lambda_reg"], 0.)
            self.assertEqual(config["prototype_mode"], "single")
            self.assertTrue(config["model_config"]["use_local_evolution_score"])
            self.assertTrue(config["use_structural_temporal_descriptors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
