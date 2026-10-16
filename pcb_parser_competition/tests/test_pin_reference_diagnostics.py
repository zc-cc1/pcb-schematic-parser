import copy
import unittest

from pin_reference_diagnostics import diagnose_fields


def pin(cid='P1', point=(10, 10), owner=0, status='assigned', number='2', name=None):
    return {'candidate_id': cid, 'point': list(point), 'owner_detection_index': owner,
            'owner_status': status, 'owner': 'U1', 'number': number, 'pinname': name}


def ref(point=(10, 10), owner='C1', numbers=None, name=''):
    return {'point': list(point), 'component_source_id': owner,
            'expected_numbers': ['2'] if numbers is None else numbers, 'expected_name': name}


class DiagnosticTests(unittest.TestCase):
    def test_matched_field_and_missing_reference_field_separate(self):
        rows = diagnose_fields([pin(name='F')], [ref()], {0: 'C1'})
        self.assertEqual([r['reason'] for r in rows], ['agrees_with_draft', 'reference_field_missing'])

    def test_geometry_near_miss_does_not_relax_radius(self):
        row = diagnose_fields([pin(point=(15.01, 10))], [ref()], {0: 'C1'})[0]
        self.assertEqual(row['reason'], 'same_owner_point_outside_5px')
        self.assertIsNone(row['matched_reference'])
        self.assertAlmostEqual(row['nearest_same_owner_references'][0]['distance_px'], 5.01)
        self.assertEqual(row['additional_match_count'], 0)

    def test_radius_boundary_retains_original_match(self):
        self.assertEqual(diagnose_fields([pin(point=(15, 10))], [ref()], {0: 'C1'})[0]['reason'], 'agrees_with_draft')

    def test_competing_prediction_cannot_reuse_reference(self):
        rows = diagnose_fields([pin(), pin('P2', (12, 10))], [ref()], {0: 'C1'})
        self.assertEqual(rows[1]['reason'], 'one_to_one_reference_competition')
        self.assertEqual(rows[1]['eligible_references'][0]['reserved_by_candidate_id'], 'P1')
        self.assertIsNone(rows[1]['matched_reference'])

    def test_nearby_other_owner_is_not_same_owner(self):
        row = diagnose_fields([pin()], [ref(owner='C2')], {0: 'C1'})[0]
        self.assertEqual(row['reason'], 'mapped_owner_has_no_reference_pins')
        self.assertEqual(row['nearest_same_owner_references'], [])

    def test_unmapped_owner_is_not_missing_pin_reference(self):
        self.assertEqual(diagnose_fields([pin()], [ref()], {})[0]['reason'], 'detected_owner_not_matched_to_draft')

    def test_ambiguous_prediction_owner_reported(self):
        self.assertEqual(diagnose_fields([pin(owner=None, status='ambiguous')], [ref()], {})[0]['reason'], 'prediction_owner_unresolved')

    def test_unresolved_owner_with_stale_index_cannot_reserve_reference(self):
        rows = diagnose_fields([pin(status='ambiguous'), pin('P2')], [ref()], {0: 'C1'})
        self.assertEqual(rows[0]['reason'], 'prediction_owner_unresolved')
        self.assertIsNone(rows[0]['matched_reference'])
        self.assertEqual(rows[1]['reason'], 'agrees_with_draft')

    def test_same_text_far_away_does_not_override_geometry_match(self):
        rows = diagnose_fields([pin()], [ref(numbers=['3']), ref((10, 25))], {0: 'C1'})
        self.assertEqual(rows[0]['reason'], 'field_disagrees_with_draft')
        self.assertEqual(rows[0]['nearest_same_text_reference']['distance_px'], 15)

    def test_empty_lists_inputs_and_missing_fields(self):
        self.assertEqual(diagnose_fields([], [], {}), [])
        self.assertEqual(diagnose_fields([pin(number=None)], [ref()], {0: 'C1'}), [])
        self.assertEqual(diagnose_fields([pin()], [], {0: 'C1'})[0]['reason'], 'mapped_owner_has_no_reference_pins')

    def test_input_payloads_are_preserved(self):
        payload = [[pin()], [ref()], {0: 'C1'}]
        old = copy.deepcopy(payload)
        diagnose_fields(*payload)
        self.assertEqual(payload, old)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            diagnose_fields([pin(), pin()], [ref()], {0: 'C1'})


if __name__ == '__main__':
    unittest.main()
