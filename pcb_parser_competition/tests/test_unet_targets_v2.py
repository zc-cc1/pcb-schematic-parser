"""Regression tests for the v2 supervision builder.

Pinned behaviours:
  1. `edge.coordinates` is used ONLY when `keyPointPaths.wire` is empty.
  2. An edge that already has keyPointPaths never falls through to coordinates,
     so filled net polygons cannot leak into the wire mask.
  3. Unowned root pins produce ignore pixels instead of rejecting the page.
  4. Out-of-bounds geometry produces ignore pixels and is never drawn as wire.
"""
import unittest

import numpy as np
from PIL import Image

from pcb_parser.unet_targets_v2 import build_targets


def page(edges=None, components=None, pins=None, width=32, height=24):
    return {"canvas": {"width": width, "height": height},
            "edges": edges or {},
            "components": components or {},
            "pins": pins or {}}


class TargetsV2Tests(unittest.TestCase):
    def test_coordinates_recover_edge_without_key_point_paths(self):
        source = page(edges={"E_1": {"category": "wire", "isFlywire": False,
                                     "keyPointPaths": {"wire": []},
                                     "coordinates": [[4, 6], [5, 6], [6, 6], [7, 6]]}},
                      components={"C_1": {"pins": {"P_1": {"tip": [7, 6]}}}})
        wire, _, wire_ignore, _, _, stats = build_targets(source, 32, 24, line_width=1)
        mask = np.asarray(wire)
        self.assertEqual(mask[6, 5], 255)
        self.assertEqual(stats["edges_recovered_from_coordinates"], 1)
        self.assertEqual(stats["edges_without_geometry"], 0)
        # recovered geometry is real supervision, so it must not be marked ignored
        self.assertEqual(int((np.asarray(wire_ignore) > 0).sum()), 0)

    def test_edge_with_key_paths_never_uses_filled_coordinates(self):
        # `coordinates` here describes a large filled area; it must be ignored
        filled = [[x, y] for y in range(2, 20) for x in range(2, 30)]
        source = page(edges={"E_1": {"category": "wire", "isFlywire": False,
                                     "keyPointPaths": {"wire": [[[4, 6], [20, 6]]]},
                                     "coordinates": filled}},
                      components={"C_1": {"pins": {"P_1": {"tip": [20, 6]}}}})
        wire, _, _, _, _, stats = build_targets(source, 32, 24, line_width=1)
        mask = np.asarray(wire)
        self.assertEqual(mask[6, 10], 255)
        self.assertEqual(mask[15, 10], 0)
        self.assertEqual(stats["edges_recovered_from_coordinates"], 0)

    def test_unowned_root_pin_is_ignored_not_page_rejection(self):
        source = page(edges={"E_1": {"category": "wire", "isFlywire": False,
                                     "keyPointPaths": {"wire": [[[4, 6], [20, 6]]]}}},
                      components={"C_1": {"pins": {"P_1": {"tip": [20, 6]}}}},
                      pins={"P_1": {"tip": [20, 6]}, "P_9": {"tip": [8, 18]}})
        _, _, wire_ignore, pin_ignore, tips, stats = build_targets(source, 32, 24, line_width=1)
        self.assertEqual(stats["unowned_root_pins_ignored"], 1)
        self.assertEqual(len(tips), 1)
        self.assertGreater(int((np.asarray(pin_ignore) > 0).sum()), 0)
        self.assertGreater(int((np.asarray(wire_ignore) > 0).sum()), 0)

    def test_out_of_bounds_path_is_ignored_not_drawn(self):
        source = page(edges={"E_1": {"category": "wire", "isFlywire": False,
                                     "keyPointPaths": {"wire": [[[4, 6], [-9, 6]]]}}},
                      components={"C_1": {"pins": {"P_1": {"tip": [4, 6]}}}})
        wire, _, wire_ignore, _, _, stats = build_targets(source, 32, 24, line_width=1)
        self.assertEqual(stats["edges_skipped_out_of_bounds"], 1)
        self.assertEqual(int((np.asarray(wire) > 0).sum()), 0)
        self.assertGreater(int((np.asarray(wire_ignore) > 0).sum()), 0)

    def test_flywires_and_deleted_edges_stay_out(self):
        source = page(edges={"E_1": {"category": "wire", "isFlywire": False,
                                     "keyPointPaths": {"wire": [[[4, 6], [20, 6]]]}},
                             "E_2": {"category": "wire", "isFlywire": True,
                                     "keyPointPaths": {"wire": [[[4, 18], [20, 18]]]}},
                             "E_3": {"category": "wire", "state": "deleted",
                                     "keyPointPaths": {"wire": [[[4, 21], [20, 21]]]}}},
                      components={"C_1": {"pins": {"P_1": {"tip": [20, 6]}}}})
        wire, _, _, _, _, stats = build_targets(source, 32, 24, line_width=1)
        mask = np.asarray(wire)
        self.assertEqual(mask[6, 10], 255)
        self.assertEqual(mask[18, 10], 0)
        self.assertEqual(mask[21, 10], 0)
        self.assertEqual(stats["flywires_excluded"], 1)

    def test_pin_heatmap_peaks_at_tip_not_base(self):
        source = page(edges={"E_1": {"category": "wire", "isFlywire": False,
                                     "keyPointPaths": {"wire": [[[4, 6], [20, 6]]]}}},
                      components={"C_1": {"pins": {"P_1": {"tip": [20, 6], "base": [14, 6]}}}},
                      pins={"P_1": {"tip": [20, 6], "base": [14, 6]}})
        _, heat, _, _, tips, _ = build_targets(source, 32, 24, line_width=1, pin_sigma=2.0)
        values = np.asarray(heat, dtype=np.float32) / 255.0
        self.assertEqual(tips[0]["point"], [20.0, 6.0])
        self.assertGreater(values[6, 20], 0.9)
        self.assertLess(values[6, 14], 0.1)

    def test_canvas_mismatch_rejected(self):
        source = page(width=30, height=24)
        with self.assertRaises(ValueError):
            build_targets(source, 32, 24)


if __name__ == "__main__":
    unittest.main()
