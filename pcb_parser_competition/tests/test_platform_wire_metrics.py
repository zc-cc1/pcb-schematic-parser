import copy
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import platform_metrics as old
import platform_metrics_v2 as v2
import platform_metrics_v3 as v3
from platform_wire_metrics import logical_nets, score_wire, _stable_assignment
from validate_platform_wire_metrics import permuted
import random


def edge(x=0, y=0):
    return [{"x": x, "y": y}, {"x": x + 100, "y": y}]


def net(members="R1.pin_1", *edges):
    return {"hyperGraph": f"({members})", "edges": {str(i): e for i, e in enumerate(edges)}}


def doc(nets=None, components=None):
    return {"nets": nets or {}, "components": components or {}, "pins": {}}


class WireMetricsTests(unittest.TestCase):
    def assert_counts(self, pred, truth, expected, tolerance=20):
        result = score_wire(pred, truth, tolerance)
        self.assertEqual(tuple(result[k] for k in ("tp", "fp", "fn")), expected)
        self.assertGreaterEqual(result["fn"], 0)
        self.assertTrue(0 <= result["f1"] <= 1)

    def test_duplicate_edges_cannot_reuse_one_truth(self):
        self.assert_counts(doc({"n": net("R1.pin_1", edge(), edge())}),
                           doc({"n": net("R1.pin_1", edge())}), (1, 1, 0))

    def test_zero_hit_net_cannot_consume_truth(self):
        self.assert_counts(doc({"first": net("R2.pin_1", edge(1000)),
                               "second": net("R1.pin_1", edge())}),
                           doc({"n": net("R1.pin_1", edge())}), (1, 1, 0))

    def test_coincident_wrong_network_gets_no_credit(self):
        self.assert_counts(doc({"n": net("R2.pin_1", edge())}),
                           doc({"n": net("R1.pin_1", edge())}), (0, 1, 1))

    def test_reversed_endpoints(self):
        self.assert_counts(doc({"n": net("R1.pin_1", edge()[::-1])}),
                           doc({"n": net("R1.pin_1", edge())}), (1, 0, 0))

    def test_edge_assignment_maximizes_eligible_pairs(self):
        self.assert_counts(doc({"n": net("R1.pin_1", edge(), edge(0, -20))}),
                           doc({"n": net("R1.pin_1", edge(), edge(0, 20))}), (2, 0, 0))

    def test_threshold_is_euclidean_and_inclusive(self):
        truth = doc({"n": net("R1.pin_1", edge())})
        self.assert_counts(doc({"n": net("R1.pin_1", edge(0, 20))}), truth, (1, 0, 0))
        self.assert_counts(doc({"n": net("R1.pin_1", edge(0, 20.001))}), truth, (0, 1, 1))
        self.assert_counts(doc({"n": net("R1.pin_1", edge(15, 15))}), truth, (0, 1, 1))

    def test_ordinary_pin_numbers_ignored_but_ic_numbers_preserved(self):
        self.assert_counts(doc({"n": net("R1.pin_2", edge())}),
                           doc({"n": net("R1.pin_1", edge())}), (1, 0, 0))
        components = {"U1": {"type": "box"}}
        self.assert_counts(doc({"n": net("U1.pin_2", edge())}, components),
                           doc({"n": net("U1.pin_1", edge())}, components), (0, 1, 1))

    def test_rail_branches_pool_edges(self):
        components = {"G1": {"type": "gnd"}, "G2": {"type": "gnd"}}
        pred = doc({"n1": net("G1.pin_1,R1.pin_1", edge()),
                    "n2": net("G2.pin_1,R2.pin_1", edge(0, 100))}, components)
        truth = doc({"n": net("G1.pin_1,R1.pin_1,R2.pin_1", edge(), edge(0, 100))}, components)
        self.assert_counts(pred, truth, (2, 0, 0))

    def test_ground_and_supply_named_gnd_are_separate(self):
        components = {"G": {"type": "gnd"}, "V": {"type": "v", "value": "GND"}}
        self.assert_counts(doc({"n": net("V.pin_1", edge())}, components),
                           doc({"n": net("G.pin_1", edge())}, components), (0, 1, 1))

    def test_transitive_rail_merges_and_default_supply(self):
        components = {"A": {"type": "v", "value": None},
                      "B": {"type": "v", "value": ""}, "G": {"type": "gnd"}}
        document = doc({"n1": net("A.pin_1", edge()),
                        "n2": net("B.pin_1,G.pin_1", edge(0, 100)),
                        "n3": net("G.pin_1", edge(0, 200))}, components)
        self.assertEqual(len(logical_nets(document)), 1)
        self.assert_counts(document, document, (3, 0, 0))

    def test_different_supply_values_remain_separate(self):
        components = {"A": {"type": "v", "value": "3V3"},
                      "B": {"type": "v", "value": "5V"}}
        self.assertEqual(len(logical_nets(doc({"a": net("A.pin_1"),
                                              "b": net("B.pin_1")}, components))), 2)

    def test_invalid_segments_count_as_errors(self):
        for invalid in [[], edge() + [{"x": 1, "y": 2}],
                        [{"x": float("nan"), "y": 0}, {"x": 100, "y": 0}],
                        [{"x": True, "y": 0}, {"x": 100, "y": 0}]]:
            with self.subTest(invalid=invalid):
                bad = doc({"n": net("R1.pin_1", invalid)})
                good = doc({"n": net("R1.pin_1", edge())})
                self.assert_counts(bad, good, (0, 1, 1))
                self.assert_counts(good, bad, (0, 1, 1))

    def test_empty_and_one_sided_input(self):
        good = doc({"n": net("R1.pin_1", edge())})
        self.assert_counts(doc(), doc(), (0, 0, 0))
        self.assert_counts(good, doc(), (0, 1, 0))
        self.assert_counts(doc(), good, (0, 0, 1))

    def test_invalid_tolerance(self):
        for tolerance in [-1, float("inf"), float("nan"), True]:
            with self.assertRaises(ValueError):
                score_wire(doc(), doc(), tolerance)

    def test_json_order_and_endpoint_direction_invariance(self):
        truth = doc({"net_10": net("R2.pin_1,R3.pin_1", edge(0, 100)),
                     "net_2": net("R1.pin_1,R2.pin_1", edge(), edge(100))})
        pred = copy.deepcopy(truth)
        pred["nets"]["net_3"] = net("R1.pin_1", edge(500))
        expected = score_wire(pred, truth)
        rng = random.Random(7)
        for _ in range(20):
            self.assertEqual(score_wire(permuted(pred, rng), permuted(truth, rng)), expected)

    def test_hungarian_objective_matches_independent_solver(self):
        rng = np.random.default_rng(7)
        for n, m in [(1, 5), (5, 1), (5, 7), (7, 5), (8, 8)]:
            for _ in range(20):
                cost = rng.integers(-3, 4, (n, m)).astype(float)
                pairs = _stable_assignment(cost)
                rows, cols = linear_sum_assignment(cost)
                self.assertEqual(len(pairs), min(n, m))
                self.assertEqual(len({i for i, _ in pairs}), len(pairs))
                self.assertEqual(len({j for _, j in pairs}), len(pairs))
                self.assertEqual(sum(cost[i, j] for i, j in pairs), cost[rows, cols].sum())

    def test_all_old_entry_points_use_fixed_wire_and_20px_default(self):
        truth = doc({"n": net("R1.pin_1", edge())})
        pred = doc({"n": net("R1.pin_1", edge(0, 10), edge(0, 10))})
        expected = score_wire(pred, truth)
        self.assertEqual(old.score_wire(pred, truth), expected)
        for module in [old, v2, v3]:
            self.assertEqual(module.score_case(pred, truth)["wire_f1"], expected)
        self.assertEqual(old.score_wire(pred, truth, 5)["tp"], 0)


if __name__ == "__main__":
    unittest.main()
