import unittest
import numpy as np
from pcb_parser.unet_path_recovery import recover_contiguous_path


class PathRecoverySafetyTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((12, 12, 3), 255, np.uint8)

    def test_unordered_straight_pixels_can_be_recovered(self):
        self.image[4, 2:6] = 0
        self.assertEqual(recover_contiguous_path([[4,4],[2,4],[5,4],[3,4]],12,12,self.image), [[2,4],[5,4]])

    def test_keeps_turns_and_backtracking(self):
        trace = [[2,2],[3,2],[4,2],[4,3],[4,4],[4,3]]
        for x,y in trace:
            self.image[y,x] = 0
        self.assertEqual(recover_contiguous_path(trace,12,12,self.image), trace)

    def test_two_ink_endpoints_do_not_authorize_a_bridge(self):
        self.image[2,2] = self.image[8,8] = 0
        with self.assertRaisesRegex(ValueError, "Sparse endpoints"):
            recover_contiguous_path([[2,2],[8,8]],12,12,self.image)

    def test_rejects_blank_pixel_in_source_trace(self):
        self.image[2,2:4] = 0
        with self.assertRaisesRegex(ValueError, "original-image ink"):
            recover_contiguous_path([[2,2],[3,2],[4,2]],12,12,self.image)

    def test_rejects_coordinate_on_exclusive_image_boundary(self):
        with self.assertRaisesRegex(ValueError, "integer image pixels"):
            recover_contiguous_path([[11,2],[12,2]],12,12,self.image)


if __name__ == "__main__":
    unittest.main()
