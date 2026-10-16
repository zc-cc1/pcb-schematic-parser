"""Contract checks for the submission adapter, without loading image models."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('platform_adapter', ROOT/'submission/platform_adapter/adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
old_adapter = sys.modules.get('adapter')
sys.modules['adapter'] = adapter
agent_spec = importlib.util.spec_from_file_location('deployment_agent', ROOT/'submission/platform_adapter/agent.py')
agent = importlib.util.module_from_spec(agent_spec)
agent_spec.loader.exec_module(agent)
if old_adapter is None:
    del sys.modules['adapter']
else:
    sys.modules['adapter'] = old_adapter


class PlatformAdapterTests(unittest.TestCase):
    def test_selected_detector_provenance_uses_actual_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights = root/'weights/selected.pt'
            weights.parent.mkdir()
            weights.write_bytes(b'selected detector checkpoint')
            config = root/agent.DETECTOR_CONFIG
            config.parent.mkdir()
            config.write_text(json.dumps({'weights': 'weights/selected.pt',
                'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                'recommended_mode': 'fused', 'detector_provenance': {'calibration_epoch': 76}}), encoding='utf-8')
            with patch.object(agent, 'ROOT', root):
                profile = agent.detector_profile()
            self.assertEqual(profile['weights'], 'weights/selected.pt')
            self.assertEqual(profile['weights_sha256'], hashlib.sha256(weights.read_bytes()).hexdigest())
            self.assertEqual(profile['config_sha256'], hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertEqual(profile['calibration_epoch'], 76)

    def test_selected_detector_rejects_replaced_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights = root/'weights/selected.pt'
            weights.parent.mkdir()
            weights.write_bytes(b'another detector')
            config = root/agent.DETECTOR_CONFIG
            config.parent.mkdir()
            config.write_text(json.dumps({'weights': 'weights/selected.pt',
                'weights_sha256': hashlib.sha256(b'expected detector').hexdigest()}), encoding='utf-8')
            with patch.object(agent, 'ROOT', root), self.assertRaisesRegex(ValueError, 'checkpoint'):
                agent.detector_profile()

    def fixture(self):
        return {'components': {'R_1': {'type': 'resistor', 'Name': None, 'value': '10K', 'bbox': [1, 2, 11, 22]}},
                'pins': {'R_1': {'pin_A5': {'pinname': 'IN', 'point': {'x': 5.5, 'y': 6.25}},
                                 'pin_2': {'pinname': '2', 'point': {'x': 15, 'y': 6.25}}}},
                'nets': {'n1': {'hyperGraph': '(R_1.A5,R_1.2)',
                                'edges': {'e1': [{'x': 5.5, 'y': 6.25}, {'x': 15, 'y': 6.25}]}}}}

    def test_geometry_text_and_membership_are_preserved(self):
        source = self.fixture(); before = deepcopy(source)
        result = adapter.to_sample(source)
        self.assertEqual(result['components']['R_1']['center'], [6, 12])
        self.assertEqual(result['components']['R_1']['value'], '10K')
        self.assertEqual(result['pins']['R_1_A5']['position'], [5.5, 6.25])
        self.assertEqual(result['pins']['R_1_A5']['pin_number'], 'A5')
        self.assertEqual(result['nets']['n1']['pins'], ['R_1_A5', 'R_1_2'])
        self.assertEqual(result['nets']['n1']['wire_segments'][0]['points'], [[5.5, 6.25], [15, 6.25]])
        self.assertEqual(source, before)

    def test_unresolved_net_member_fails(self):
        source = self.fixture(); source['nets']['n1']['hyperGraph'] = '(R_1.A5,R_1.3)'
        with self.assertRaisesRegex(ValueError, 'Unresolved net member'):
            adapter.to_sample(source)

    def test_duplicate_net_member_fails(self):
        source = self.fixture(); source['nets']['n1']['hyperGraph'] = '(R_1.A5,R_1.A5)'
        with self.assertRaisesRegex(ValueError, 'Duplicate net members'):
            adapter.to_sample(source)

    def test_nonfinite_geometry_fails(self):
        source = self.fixture(); source['pins']['R_1']['pin_A5']['point']['x'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid point'):
            adapter.to_sample(source)

    def test_flattened_pin_collision_fails(self):
        source = self.fixture()
        source['components']['R'] = deepcopy(source['components']['R_1'])
        source['pins']['R'] = {'pin_1_A5': {'pinname': 'IN', 'point': {'x': 1, 'y': 2}}}
        with self.assertRaisesRegex(ValueError, 'collision'):
            adapter.to_sample(source)

    def test_target_preserves_pins_nets_values_and_raw_input(self):
        source = self.fixture(); before = deepcopy(source)
        target = adapter.to_target(source)
        self.assertEqual(target['components']['R_1']['type'], 'r')
        self.assertEqual(target['pins'], source['pins'])
        self.assertEqual(target['nets'], source['nets'])
        self.assertEqual(target['components']['R_1']['Name'], 'R_1')
        self.assertEqual(target['components']['R_1']['bbox'], [2, 2, 10, 22])
        self.assertEqual(target['components']['R_1']['value'], '10K')
        self.assertEqual(source, before)

    def test_public_box_convention_aligns_real_cached_resistor(self):
        source = {'components': {'R24': {'type': 'resistor', 'Name': None, 'value': '100K',
                   'bbox': [177.754, 194.913, 193.162, 242.173]}}, 'pins': {}, 'nets': {}}
        actual = adapter.to_target(source)['components']['R24']
        golden = [181, 207, 189, 227]  # Public 0042 training annotation.
        self.assertLessEqual(max(abs(a-b) for a, b in zip(actual['bbox'], golden)), 5)
        self.assertEqual(actual['Name'], 'R24')
        self.assertEqual(actual['value'], '100K')

    def test_observed_ic_and_diode_names_are_preserved(self):
        source = {'components': {
            'U1': {'type': 'ic', 'Name': 'MSP432E401YTPDT', 'value': None, 'bbox': [1, 2, 99, 200]},
            'D1': {'type': 'diode', 'Name': '1N4007', 'value': None, 'bbox': [10, 10, 70, 30]},
        }, 'pins': {}, 'nets': {}}
        target = adapter.to_target(source)
        self.assertEqual(target['components']['U1'], {**source['components']['U1'], 'type': 'box'})
        self.assertEqual(target['components']['D1']['Name'], '1N4007')

    def test_duplicate_passive_names_keep_base_without_changing_keys(self):
        source = self.fixture()
        source['components'] = {
            'C6_1': {'type': 'crystal_2pin', 'Name': 'C6', 'value': '22p', 'bbox': [1, 2, 11, 22]},
            'C6_2': {'type': 'capacitor', 'Name': None, 'value': None, 'bbox': [50, 50, 100, 100]},
        }
        source['pins'] = {};source['nets'] = {}
        target = adapter.to_target(source)
        self.assertEqual(set(target['components']), {'C6_1', 'C6_2'})
        self.assertEqual(target['components']['C6_2']['Name'], 'C6')

    def test_public_component_projection_is_idempotent_and_legacy_replay_available(self):
        source = self.fixture()
        target = adapter.to_target(source)
        self.assertEqual(adapter.to_target(target), target)
        legacy = adapter.to_target(source, normalize_components=False)
        self.assertEqual(legacy['components']['R_1']['Name'], None)
        self.assertEqual(legacy['components']['R_1']['bbox'], source['components']['R_1']['bbox'])

    def test_live_target_rejects_legacy_flat_pins(self):
        with self.assertRaisesRegex(ValueError, 'nested pin owner'):
            adapter.validate_target(adapter.to_sample(self.fixture()))

    def test_live_target_rejects_legacy_net_layout(self):
        source = self.fixture()
        source['nets'] = {'n1': {'pins': ['R_1_A5', 'R_1_2'], 'wire_segments': []}}
        with self.assertRaisesRegex(ValueError, 'hyperGraph/edges'):
            adapter.to_target(source)

    def test_target_rejects_unresolved_references(self):
        source = self.fixture(); source['nets']['n1']['hyperGraph'] = '(R_1.A5,MISSING.1)'
        with self.assertRaisesRegex(ValueError, 'Unresolved net member'):
            adapter.to_target(source)

    def test_target_rejects_nonfinite_tip(self):
        source = self.fixture(); source['pins']['R_1']['pin_A5']['point']['x'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'Invalid point'):
            adapter.to_target(source)

    def test_target_handles_owner_names_that_collide_when_flattened(self):
        source = self.fixture()
        source['components']['R'] = deepcopy(source['components']['R_1'])
        source['pins']['R'] = {'pin_1_A5': {'pinname': 'IN', 'point': {'x': 1, 'y': 2}}}
        adapter.validate_target(adapter.to_target(source))

    def test_worker_policy_respects_small_gpu_and_case_count(self):
        self.assertEqual(agent.choose_workers(0, 'cuda:0', 10, 8*1024**3), 1)
        self.assertEqual(agent.choose_workers(0, 'cuda:0', 10, 80*1024**3), 3)
        self.assertEqual(agent.choose_workers(0, 'cpu', 10), 1)
        self.assertEqual(agent.choose_workers(3, 'cuda:0', 2, 80*1024**3), 2)
        with self.assertRaises(ValueError):
            agent.choose_workers(4, 'cpu', 10)

    def test_auto_device_requires_a_successful_cuda_kernel(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.ones.return_value.__matmul__.return_value.sum.return_value.item.return_value = 64
        torch.cuda.get_device_name.return_value = 'Test GPU'
        torch.cuda.get_device_capability.return_value = (8, 0)
        torch.cuda.get_arch_list.return_value = ['sm_80']
        device, probe = agent.probe_runtime_device(torch, 'auto')
        self.assertEqual(device, 'cuda:0')
        self.assertTrue(probe['success'])
        torch.cuda.synchronize.assert_called_once_with('cuda:0')

    def test_missing_gpu_retains_initialization_error_and_falls_back(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.cuda.init.side_effect = RuntimeError('No NVIDIA driver exposed')
        device, probe = agent.probe_runtime_device(torch, 'auto')
        self.assertEqual(device, 'cpu')
        self.assertFalse(probe['success'])
        self.assertIn('No NVIDIA driver exposed', probe['error'])
        torch.ones.assert_not_called()

    def test_visible_but_incompatible_gpu_falls_back_with_kernel_error(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.ones.side_effect = RuntimeError('no kernel image is available')
        device, probe = agent.probe_runtime_device(torch, 'auto')
        self.assertEqual(device, 'cpu')
        self.assertIn('no kernel image', probe['error'])

    def test_explicit_cuda_failure_is_retained_for_preflight_to_reject(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.cuda.init.side_effect = RuntimeError('driver unavailable')
        device, probe = agent.probe_runtime_device(torch, 'cuda:1')
        self.assertEqual(device, 'cuda:1')
        self.assertFalse(probe['success'])
        self.assertIn('driver unavailable', probe['error'])

    def test_explicit_cpu_does_not_initialize_cuda(self):
        torch = MagicMock()
        device, probe = agent.probe_runtime_device(torch, 'cpu')
        self.assertEqual(device, 'cpu')
        self.assertFalse(probe['attempted'])
        torch.cuda.init.assert_not_called()
        torch.ones.assert_not_called()


if __name__ == '__main__':
    unittest.main()
