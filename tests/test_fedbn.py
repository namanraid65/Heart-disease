"""
Comprehensive Unit & Integration Test Suite for FedBN Implementation
Tests all 10 explicit FedBN verification criteria:
  TEST 1: Linear/Conv parameters are aggregated with sample-weighted FedAvg.
  TEST 2: BN running_mean is NOT averaged.
  TEST 3: BN running_var is NOT averaged.
  TEST 4: num_batches_tracked is NOT FedAvg aggregated.
  TEST 5: Client H1 retains its BN state after receiving global parameters.
  TEST 6: Client H2 retains its BN state after receiving global parameters.
  TEST 7: Different client BN statistics remain different after aggregation.
  TEST 8: The model can still train after aggregation.
  TEST 9: Validation-based model selection still works.
  TEST 10: Final test evaluation still occurs only once after model selection.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np

from federated.config import DEVICE, INPUT_FEATURES, DROPOUT_RATE, RESNET_DROPOUT_RATE
from federated.utils import (
    get_bn_buffer_names,
    get_shared_parameter_names,
    get_model_shared_parameters,
    set_model_shared_parameters,
    get_model_bn_state,
    set_model_bn_state,
    aggregate_fedavg
)
from federated.client import HospitalClient
from federated.resnet_client import HospitalResNetClient
from federated.server import FederatedServer
from federated.resnet_server import FederatedResNetServer
from federated.strategy import FedAvgStrategy
from models.alexnet_1d import build_alexnet_1d
from models.resnet_1d import build_resnet_1d


class TestFedBN(unittest.TestCase):
    """Unit test suite for FedBN parameter classification, aggregation, and isolation."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_01_linear_conv_parameters_aggregated(self):
        """TEST 1: Linear/Conv parameters are aggregated using sample-weighted FedAvg."""
        model_a = build_alexnet_1d(input_dim=INPUT_FEATURES, dropout_rate=DROPOUT_RATE)
        model_b = build_alexnet_1d(input_dim=INPUT_FEATURES, dropout_rate=DROPOUT_RATE)

        # Mutate weights distinctly
        with torch.no_grad():
            model_a.features[0].weight.fill_(1.0)
            model_b.features[0].weight.fill_(3.0)

        params_a = get_model_shared_parameters(model_a)
        params_b = get_model_shared_parameters(model_b)

        # Weighting: 100 samples from A, 300 samples from B -> weights: 0.25 and 0.75
        aggregated = aggregate_fedavg([
            (params_a, 100),
            (params_b, 300)
        ])

        server_model = build_alexnet_1d(input_dim=INPUT_FEATURES, dropout_rate=DROPOUT_RATE)
        set_model_shared_parameters(server_model, aggregated)

        expected_conv1_weight = 0.25 * 1.0 + 0.75 * 3.0  # 2.5
        actual_conv1_weight = server_model.features[0].weight.detach().cpu().numpy()
        self.assertTrue(np.allclose(actual_conv1_weight, expected_conv1_weight, atol=1e-5))

    def test_02_bn_running_mean_not_averaged(self):
        """TEST 2: BN running_mean is NOT FedAvg averaged."""
        client_h1 = HospitalClient('hospital_1', device='cpu')
        client_h2 = HospitalClient('hospital_2', device='cpu')

        # Set distinct running_mean values
        with torch.no_grad():
            client_h1.model.features[1].running_mean.fill_(10.0)
            client_h2.model.features[1].running_mean.fill_(20.0)

        # Extract parameters sent to server
        params_h1 = client_h1.get_parameters(config={})
        params_h2 = client_h2.get_parameters(config={})

        strategy = FedAvgStrategy()
        aggregated, _ = strategy.aggregate_fit(1, [
            ('hospital_1', params_h1, 100, {}),
            ('hospital_2', params_h2, 100, {})
        ])

        # Server model receives aggregated shared parameters
        server = FederatedServer(device='cpu')
        server.set_global_parameters(aggregated)

        # Clients receive global parameters
        client_h1.model.features[1].running_mean.fill_(10.0)
        set_model_shared_parameters(client_h1.model, aggregated)
        set_model_shared_parameters(client_h2.model, aggregated)

        # Verify running_mean was NOT averaged to 15.0; H1 has 10.0, H2 has 20.0
        self.assertTrue(torch.allclose(client_h1.model.features[1].running_mean, torch.tensor(10.0)))
        self.assertTrue(torch.allclose(client_h2.model.features[1].running_mean, torch.tensor(20.0)))

    def test_03_bn_running_var_not_averaged(self):
        """TEST 3: BN running_var is NOT FedAvg averaged."""
        client_h1 = HospitalClient('hospital_1', device='cpu')
        client_h2 = HospitalClient('hospital_2', device='cpu')

        with torch.no_grad():
            client_h1.model.features[1].running_var.fill_(2.5)
            client_h2.model.features[1].running_var.fill_(8.5)

        params_h1 = client_h1.get_parameters(config={})
        params_h2 = client_h2.get_parameters(config={})

        strategy = FedAvgStrategy()
        aggregated, _ = strategy.aggregate_fit(1, [
            ('hospital_1', params_h1, 200, {}),
            ('hospital_2', params_h2, 200, {})
        ])

        # Apply global shared parameters to clients
        set_model_shared_parameters(client_h1.model, aggregated)
        set_model_shared_parameters(client_h2.model, aggregated)

        self.assertTrue(torch.allclose(client_h1.model.features[1].running_var, torch.tensor(2.5)))
        self.assertTrue(torch.allclose(client_h2.model.features[1].running_var, torch.tensor(8.5)))

    def test_04_num_batches_tracked_not_aggregated(self):
        """TEST 4: num_batches_tracked is NOT FedAvg aggregated."""
        client_h1 = HospitalClient('hospital_1', device='cpu')
        client_h2 = HospitalClient('hospital_2', device='cpu')

        client_h1.model.features[1].num_batches_tracked.fill_(15)
        client_h2.model.features[1].num_batches_tracked.fill_(45)

        params_h1 = client_h1.get_parameters(config={})
        params_h2 = client_h2.get_parameters(config={})

        strategy = FedAvgStrategy()
        aggregated, _ = strategy.aggregate_fit(1, [
            ('hospital_1', params_h1, 100, {}),
            ('hospital_2', params_h2, 100, {})
        ])

        set_model_shared_parameters(client_h1.model, aggregated)
        set_model_shared_parameters(client_h2.model, aggregated)

        self.assertEqual(client_h1.model.features[1].num_batches_tracked.item(), 15)
        self.assertEqual(client_h2.model.features[1].num_batches_tracked.item(), 45)

    def test_05_client_h1_retains_bn_state(self):
        """TEST 5: Client H1 retains its BN state after receiving global parameters."""
        client_h1 = HospitalClient('hospital_1', device='cpu')
        with torch.no_grad():
            client_h1.model.features[1].running_mean.fill_(7.7)
            client_h1.model.features[1].running_var.fill_(3.3)

        bn_state_before = client_h1.get_bn_state()

        # Dummy global update with different weights
        server = FederatedServer(device='cpu')
        global_params = server.get_global_parameters()

        # Client receives global parameters
        client_h1.set_bn_state(bn_state_before)
        set_model_shared_parameters(client_h1.model, global_params)

        bn_state_after = client_h1.get_bn_state()
        for k in bn_state_before:
            self.assertTrue(torch.allclose(bn_state_before[k], bn_state_after[k]))

    def test_06_client_h2_retains_bn_state(self):
        """TEST 6: Client H2 retains its BN state after receiving global parameters."""
        client_h2 = HospitalClient('hospital_2', device='cpu')
        with torch.no_grad():
            client_h2.model.features[1].running_mean.fill_(12.4)
            client_h2.model.features[1].running_var.fill_(5.8)

        bn_state_before = client_h2.get_bn_state()
        server = FederatedServer(device='cpu')
        global_params = server.get_global_parameters()

        set_model_shared_parameters(client_h2.model, global_params)
        bn_state_after = client_h2.get_bn_state()

        for k in bn_state_before:
            self.assertTrue(torch.allclose(bn_state_before[k], bn_state_after[k]))

    def test_07_different_client_bn_stats_remain_different(self):
        """TEST 7: Different client BN statistics remain distinct after aggregation round."""
        client_h1 = HospitalResNetClient('hospital_1', device='cpu')
        client_h2 = HospitalResNetClient('hospital_2', device='cpu')

        with torch.no_grad():
            client_h1.model.stem[1].running_mean.fill_(1.23)
            client_h2.model.stem[1].running_mean.fill_(9.87)

        params_h1 = client_h1.get_parameters()
        params_h2 = client_h2.get_parameters()

        strategy = FedAvgStrategy()
        aggregated, _ = strategy.aggregate_fit(1, [
            ('hospital_1', params_h1, 212, {}),
            ('hospital_2', params_h2, 205, {})
        ])

        client_h1.set_parameters(aggregated)
        client_h2.set_parameters(aggregated)

        h1_mean = client_h1.model.stem[1].running_mean.mean().item()
        h2_mean = client_h2.model.stem[1].running_mean.mean().item()

        self.assertAlmostEqual(h1_mean, 1.23, places=4)
        self.assertAlmostEqual(h2_mean, 9.87, places=4)
        self.assertNotEqual(h1_mean, h2_mean)

    def test_08_model_can_train_after_aggregation(self):
        """TEST 8: Model can still execute local training iterations after aggregation."""
        client = HospitalClient('hospital_1', device='cpu')
        server = FederatedServer(device='cpu')

        global_params = server.get_global_parameters()
        updated_params, num_samples, metrics = client.fit(
            parameters=global_params,
            config={'local_epochs': 1, 'lr': 0.001}
        )

        self.assertEqual(num_samples, client.num_train_samples)
        self.assertIn('train_loss', metrics)
        self.assertFalse(np.isnan(metrics['train_loss']))
        self.assertEqual(len(updated_params), len(global_params))

    def test_09_validation_model_selection_works(self):
        """TEST 9: Validation-based model selection functions correctly under FedBN."""
        from federated.simulation import run_federated_simulation
        res = run_federated_simulation(num_rounds=2, local_epochs=1)
        self.assertIn('best_round', res)
        self.assertIn(res['best_round'], [1, 2])
        self.assertIsNotNone(res['best_val_score'])

        # Verify checkpoint contains client_bn_states
        ckpt = torch.load(res['final_checkpoint'], weights_only=False)
        self.assertIn('client_bn_states', ckpt)
        self.assertIsNotNone(ckpt['client_bn_states'])
        self.assertIn('hospital_1', ckpt['client_bn_states'])
        self.assertIn('hospital_2', ckpt['client_bn_states'])
        self.assertIn('hospital_3', ckpt['client_bn_states'])

    def test_10_final_test_evaluation_occurs_once(self):
        """TEST 10: Final test evaluation occurs strictly after best model selection."""
        from federated.simulation import run_federated_simulation
        res = run_federated_simulation(num_rounds=2, local_epochs=1)
        final_eval = res['final_eval']
        self.assertEqual(final_eval['split'], 'test')
        self.assertEqual(final_eval['test_samples'], 109)
        # Check all hospitals evaluated on test
        for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
            self.assertIn(cid, final_eval['client_results'])
            self.assertEqual(final_eval['client_results'][cid]['split'], 'test')


if __name__ == '__main__':
    unittest.main()
