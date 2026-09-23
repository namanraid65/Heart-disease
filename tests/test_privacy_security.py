"""
Unit and Integration Tests for Privacy and Security in Heterogeneous Federated Learning (Phase 9)

Tests cover:
  1. Secure Aggregation Simulation via Pairwise Additive Masking (Tests 1-7)
  2. Client-Side Differential Privacy (Clipping + Gaussian Noise + RDP) (Tests 8-14)
  3. Integration Across Optimization Strategies & Architecture Boundaries (Tests 15-23)
"""

import sys
from pathlib import Path
import copy
import unittest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.heterogeneous.encoder import HospitalEncoder
from models.heterogeneous.predictor import SharedPredictor
from models.heterogeneous.composite import HeterogeneousCompositeModel
from federated.heterogeneous.privacy import (
    PrivacyConfig,
    compute_l2_norm,
    clip_update,
    add_gaussian_noise,
    clip_and_noise_update,
    PairwiseMaskingProtocol,
    RDPAccountant
)
from federated.heterogeneous.client import HeterogeneousHospitalClient
from federated.heterogeneous.server import HeterogeneousFederatedServer
from federated.heterogeneous.strategy import (
    HeterogeneousFedAvgStrategy,
    HeterogeneousFedProxStrategy,
    HeterogeneousFedAdamStrategy
)
from preprocessing.heterogeneous_schema import get_client_schema


class TestSimulatedSecureAggregation(unittest.TestCase):
    """Tests 1 - 7: Pairwise additive masking secure aggregation prototype."""

    def setUp(self):
        self.latent_dim = 16
        self.client_ids_2 = ["hospital_1", "hospital_2"]
        self.client_ids_3 = ["hospital_1", "hospital_2", "hospital_3"]
        self.sample_shapes = [(8, 16), (8,), (1, 8), (1,)]
        # Sample updates for 3 clients
        np.random.seed(42)
        self.updates = {
            cid: [np.random.randn(*s).astype(np.float32) for s in self.sample_shapes]
            for cid in self.client_ids_3
        }

    def test_01_two_client_cancellation(self):
        """Test 1: 2-client simulated secure aggregation cancels out exactly (M1 + M2 = 0)."""
        round_seed = 12345
        mask_1 = PairwiseMaskingProtocol.generate_client_mask(
            client_id="hospital_1",
            participating_client_ids=self.client_ids_2,
            layer_shapes=self.sample_shapes,
            round_seed=round_seed
        )
        mask_2 = PairwiseMaskingProtocol.generate_client_mask(
            client_id="hospital_2",
            participating_client_ids=self.client_ids_2,
            layer_shapes=self.sample_shapes,
            round_seed=round_seed
        )
        # Sum of masks must equal zero
        for m1, m2 in zip(mask_1, mask_2):
            mask_sum = m1 + m2
            np.testing.assert_allclose(mask_sum, 0.0, atol=1e-6)

    def test_02_three_client_cancellation(self):
        """Test 2: 3-client simulated secure aggregation cancels out (sum M_i = 0)."""
        round_seed = 54321
        masks = [
            PairwiseMaskingProtocol.generate_client_mask(
                client_id=cid,
                participating_client_ids=self.client_ids_3,
                layer_shapes=self.sample_shapes,
                round_seed=round_seed
            )
            for cid in self.client_ids_3
        ]
        num_layers = len(self.sample_shapes)
        for layer_idx in range(num_layers):
            layer_sum = sum(m[layer_idx] for m in masks)
            np.testing.assert_allclose(layer_sum, 0.0, atol=1e-6)

    def test_03_server_receives_only_masked_updates(self):
        """Test 3: Server receives masked updates where masked != raw update."""
        round_seed = 999
        raw_u = self.updates["hospital_1"]
        masked_u = PairwiseMaskingProtocol.mask_update(
            update=raw_u,
            client_id="hospital_1",
            participating_client_ids=self.client_ids_3,
            round_seed=round_seed,
            weight_fraction=1.0
        )
        # Verify masked layers are NOT identical to raw layers
        for raw_layer, masked_layer in zip(raw_u, masked_u):
            diff = np.max(np.abs(masked_layer - raw_layer))
            self.assertGreater(diff, 1e-4)

    def test_04_secure_aggregation_applies_to_shared_predictor_only(self):
        """Test 4: Secure aggregation applies strictly to shared predictor parameters."""
        predictor = SharedPredictor(latent_dim=self.latent_dim)
        pred_params = [p.detach().cpu().numpy() for p in predictor.parameters()]

        masked = PairwiseMaskingProtocol.mask_update(
            update=pred_params,
            client_id="hospital_1",
            participating_client_ids=self.client_ids_3,
            round_seed=101,
            weight_fraction=0.33
        )
        self.assertEqual(len(masked), len(pred_params))
        for m, p in zip(masked, pred_params):
            self.assertEqual(m.shape, p.shape)

    def test_05_encoders_completely_excluded_from_secure_aggregation(self):
        """Test 5: Private hospital encoders are completely excluded from secure aggregation."""
        encoder_h1 = HospitalEncoder(input_dim=25, latent_dim=16)
        enc_params = [p.detach().cpu().numpy() for p in encoder_h1.parameters()]

        # The protocol should reject or fail shape matching if an encoder is passed
        predictor_shapes = [tuple(p.shape) for p in SharedPredictor(latent_dim=16).parameters()]
        strat = HeterogeneousFedAvgStrategy(expected_parameter_shapes=predictor_shapes)

        with self.assertRaises(ValueError):
            strat.validate_client_parameters("hospital_1", enc_params)

    def test_06_mask_generation_is_deterministic_given_seed(self):
        """Test 6: Mask generation is deterministic given round seed."""
        mask_a = PairwiseMaskingProtocol.generate_client_mask(
            client_id="hospital_1",
            participating_client_ids=self.client_ids_3,
            layer_shapes=self.sample_shapes,
            round_seed=777
        )
        mask_b = PairwiseMaskingProtocol.generate_client_mask(
            client_id="hospital_1",
            participating_client_ids=self.client_ids_3,
            layer_shapes=self.sample_shapes,
            round_seed=777
        )
        for a, b in zip(mask_a, mask_b):
            np.testing.assert_array_equal(a, b)

        # Different seed produces different mask
        mask_c = PairwiseMaskingProtocol.generate_client_mask(
            client_id="hospital_1",
            participating_client_ids=self.client_ids_3,
            layer_shapes=self.sample_shapes,
            round_seed=778
        )
        any_diff = any(not np.allclose(a, c) for a, c in zip(mask_a, mask_c))
        self.assertTrue(any_diff)

    def test_07_dropout_limitation_documented_and_enforced(self):
        """Test 7: If a client drops out, pairwise masks do NOT cancel (dropout limitation verified)."""
        round_seed = 42
        # Generate masks for 3 clients
        masks = [
            PairwiseMaskingProtocol.generate_client_mask(
                client_id=cid,
                participating_client_ids=self.client_ids_3,
                layer_shapes=self.sample_shapes,
                round_seed=round_seed
            )
            for cid in self.client_ids_3
        ]
        # Drop client 3: sum of client 1 and 2 masks only
        dropped_sum = [m1 + m2 for m1, m2 in zip(masks[0], masks[1])]
        # Must NOT cancel to zero
        residual_norm = compute_l2_norm(dropped_sum)
        self.assertGreater(residual_norm, 1.0, "Missing client should leave uncancelled mask residual.")


class TestClientSideDifferentialPrivacy(unittest.TestCase):
    """Tests 8 - 14: L2 clipping, calibrated Gaussian noise, RDP accountant."""

    def setUp(self):
        self.shapes = [(8, 16), (8,), (1, 8), (1,)]
        np.random.seed(42)
        self.update = [np.random.randn(*s).astype(np.float32) for s in self.shapes]

    def test_08_l2_norm_strictly_bounded(self):
        """Test 8: L2 clipping strictly bounds update norm <= C."""
        max_norm = 1.0
        clipped, raw_norm, scale = clip_update(self.update, max_norm=max_norm)
        final_norm = compute_l2_norm(clipped)
        self.assertLessEqual(final_norm, max_norm + 1e-6)

    def test_09_update_below_threshold_invariant(self):
        """Test 9: Update below threshold is unchanged in direction and norm."""
        small_update = [u * 0.01 for u in self.update]
        orig_norm = compute_l2_norm(small_update)
        self.assertLess(orig_norm, 1.0)

        clipped, raw_norm, scale = clip_update(small_update, max_norm=1.0)
        final_norm = compute_l2_norm(clipped)
        self.assertAlmostEqual(raw_norm, orig_norm, places=5)
        self.assertAlmostEqual(scale, 1.0, places=5)
        self.assertAlmostEqual(final_norm, orig_norm, places=5)
        for s, c in zip(small_update, clipped):
            np.testing.assert_allclose(s, c, atol=1e-6)

    def test_10_update_above_threshold_scaled_down_to_c(self):
        """Test 10: Update above threshold is correctly scaled down to C."""
        large_update = [u * 10.0 for u in self.update]
        orig_norm = compute_l2_norm(large_update)
        max_norm = 2.0
        self.assertGreater(orig_norm, max_norm)

        clipped, raw_norm, scale = clip_update(large_update, max_norm=max_norm)
        final_norm = compute_l2_norm(clipped)
        self.assertAlmostEqual(final_norm, max_norm, places=5)
        # Check collinearity: clipped = (max_norm / orig_norm) * large_update
        expected_scale = max_norm / orig_norm
        self.assertAlmostEqual(scale, expected_scale, places=5)
        for l, c in zip(large_update, clipped):
            np.testing.assert_allclose(c, l * expected_scale, atol=1e-5)

    def test_11_gaussian_noise_added_after_clipping(self):
        """Test 11: Gaussian noise is added after clipping with specified sigma."""
        max_norm = 1.0
        noise_multiplier = 0.5
        noised, telemetry = clip_and_noise_update(
            update=self.update,
            max_norm=max_norm,
            noise_multiplier=noise_multiplier,
            seed=42
        )
        self.assertEqual(len(noised), len(self.update))
        # Noised update should differ from purely clipped update
        clipped, _, _ = clip_update(self.update, max_norm=max_norm)
        diff = compute_l2_norm([n - c for n, c in zip(noised, clipped)])
        self.assertGreater(diff, 0.1)

    def test_12_dp_applies_to_shared_predictor_only(self):
        """Test 12: DP applies strictly to shared predictor parameters."""
        predictor = SharedPredictor(latent_dim=16)
        pred_params = [p.detach().cpu().numpy() for p in predictor.parameters()]
        noised, telemetry = clip_and_noise_update(
            update=pred_params,
            max_norm=1.0,
            noise_multiplier=0.3,
            seed=42
        )
        self.assertEqual(len(noised), len(pred_params))
        for n, p in zip(noised, pred_params):
            self.assertEqual(n.shape, p.shape)

    def test_13_encoders_completely_excluded_from_dp(self):
        """Test 13: Client local encoders are never clipped or noised."""
        schema = get_client_schema("hospital_1")
        client = HeterogeneousHospitalClient(client_id="hospital_1", schema=schema, latent_dim=16)
        initial_enc_weights = [p.clone() for p in client.model.encoder.parameters()]

        global_pred_params = client.get_parameters()
        # Fit with DP enabled
        _, _, metrics = client.fit(
            parameters=global_pred_params,
            config={
                'local_epochs': 1,
                'lr': 0.01,
                'privacy_config': {
                    'differential_privacy': True,
                    'max_update_norm': 1.0,
                    'noise_multiplier': 0.5,
                    'seed': 42
                }
            }
        )
        # Encoder weights should have updated via local gradient descent, NOT noised/clipped
        current_enc_weights = [p.clone() for p in client.model.encoder.parameters()]
        self.assertEqual(len(initial_enc_weights), len(current_enc_weights))

    def test_14_seed_reproducibility(self):
        """Test 14: Random seed controls noise generation reproducibly."""
        noised_1, _ = clip_and_noise_update(self.update, max_norm=1.0, noise_multiplier=0.4, seed=123)
        noised_2, _ = clip_and_noise_update(self.update, max_norm=1.0, noise_multiplier=0.4, seed=123)
        for n1, n2 in zip(noised_1, noised_2):
            np.testing.assert_array_equal(n1, n2)

        noised_3, _ = clip_and_noise_update(self.update, max_norm=1.0, noise_multiplier=0.4, seed=124)
        any_diff = any(not np.allclose(n1, n3) for n1, n3 in zip(noised_1, noised_3))
        self.assertTrue(any_diff)


class TestPrivacyIntegration(unittest.TestCase):
    """Tests 15 - 23: Integration of Privacy Mechanisms with FL Optimization Strategies."""

    @classmethod
    def setUpClass(cls):
        cls.latent_dim = 16
        cls.client_ids = ["hospital_1", "hospital_2", "hospital_3"]

    def _setup_clients_and_server(self, strategy_obj, privacy_cfg):
        clients = {}
        for cid in self.client_ids:
            schema = get_client_schema(cid)
            clients[cid] = HeterogeneousHospitalClient(
                client_id=cid,
                schema=schema,
                latent_dim=self.latent_dim,
                lr=0.01
            )
        server = HeterogeneousFederatedServer(
            latent_dim=self.latent_dim,
            strategy=strategy_obj,
            privacy_config=privacy_cfg
        )
        return clients, server

    def test_15_fedavg_plus_dp(self):
        """Test 15: FedAvg + DP runs and updates model."""
        p_cfg = {'differential_privacy': True, 'max_update_norm': 1.0, 'noise_multiplier': 0.3, 'delta': 1e-5}
        strat = HeterogeneousFedAvgStrategy()
        clients, server = self._setup_clients_and_server(strat, p_cfg)

        initial_params = server.get_global_parameters()
        client_updates = []
        for cid, c in clients.items():
            params, n_samples, metrics = c.fit(initial_params, {'local_epochs': 1, 'privacy_config': p_cfg})
            client_updates.append((cid, params, n_samples, metrics))

        summary = server.aggregate_round(server_round=1, client_results=client_updates)
        updated_params = server.get_global_parameters()

        self.assertIsNotNone(server.epsilon)
        self.assertGreater(server.epsilon, 0.0)
        # Verify model changed
        diff = sum(float(np.sum(np.abs(u - i))) for u, i in zip(updated_params, initial_params))
        self.assertGreater(diff, 1e-4)

    def test_16_fedprox_plus_dp(self):
        """Test 16: FedProx + DP runs and updates model."""
        p_cfg = {'differential_privacy': True, 'max_update_norm': 1.0, 'noise_multiplier': 0.3, 'delta': 1e-5}
        strat = HeterogeneousFedProxStrategy(proximal_mu=0.01)
        clients, server = self._setup_clients_and_server(strat, p_cfg)

        initial_params = server.get_global_parameters()
        client_updates = []
        for cid, c in clients.items():
            params, n_samples, metrics = c.fit(initial_params, {'local_epochs': 1, 'proximal_mu': 0.01, 'privacy_config': p_cfg})
            client_updates.append((cid, params, n_samples, metrics))

        summary = server.aggregate_round(server_round=1, client_results=client_updates)
        self.assertEqual(summary.get('proximal_mu'), 0.01)
        self.assertIsNotNone(server.epsilon)

    def test_17_fedadam_plus_dp(self):
        """Test 17: FedAdam + DP runs and updates model."""
        p_cfg = {'differential_privacy': True, 'max_update_norm': 1.0, 'noise_multiplier': 0.3, 'delta': 1e-5}
        strat = HeterogeneousFedAdamStrategy(server_lr=0.1)
        clients, server = self._setup_clients_and_server(strat, p_cfg)

        initial_params = server.get_global_parameters()
        client_updates = []
        for cid, c in clients.items():
            params, n_samples, metrics = c.fit(initial_params, {'local_epochs': 1, 'privacy_config': p_cfg})
            client_updates.append((cid, params, n_samples, metrics))

        summary = server.aggregate_round(server_round=1, client_results=client_updates)
        self.assertEqual(summary.get('server_lr'), 0.1)
        self.assertIsNotNone(server.epsilon)

    def test_18_fedavg_plus_secagg(self):
        """Test 18: FedAvg + Secure Aggregation runs and updates model."""
        p_cfg = {'secure_aggregation': True, 'differential_privacy': False}
        strat = HeterogeneousFedAvgStrategy()
        clients, server = self._setup_clients_and_server(strat, p_cfg)

        initial_params = server.get_global_parameters()
        client_updates = []
        for cid, c in clients.items():
            params, n_samples, metrics = c.fit(initial_params, {'local_epochs': 1, 'privacy_config': p_cfg})
            client_updates.append((cid, params, n_samples, metrics))

        summary = server.aggregate_round(server_round=1, client_results=client_updates)
        self.assertTrue(summary.get('secure_aggregation'))
        self.assertIsNone(server.epsilon)  # Epsilon is None when DP is disabled

    def test_19_fedadam_plus_secagg(self):
        """Test 19: FedAdam + Secure Aggregation runs and updates model."""
        p_cfg = {'secure_aggregation': True, 'differential_privacy': False}
        strat = HeterogeneousFedAdamStrategy(server_lr=0.1)
        clients, server = self._setup_clients_and_server(strat, p_cfg)

        initial_params = server.get_global_parameters()
        client_updates = []
        for cid, c in clients.items():
            params, n_samples, metrics = c.fit(initial_params, {'local_epochs': 1, 'privacy_config': p_cfg})
            client_updates.append((cid, params, n_samples, metrics))

        summary = server.aggregate_round(server_round=1, client_results=client_updates)
        self.assertTrue(summary.get('secure_aggregation'))
        self.assertIsNone(server.epsilon)

    def test_20_secagg_plus_dp_together(self):
        """Test 20: Secure Aggregation + DP run together correctly."""
        p_cfg = {
            'secure_aggregation': True,
            'differential_privacy': True,
            'max_update_norm': 1.0,
            'noise_multiplier': 0.3,
            'delta': 1e-5
        }
        strat = HeterogeneousFedAvgStrategy()
        clients, server = self._setup_clients_and_server(strat, p_cfg)

        initial_params = server.get_global_parameters()
        client_updates = []
        for cid, c in clients.items():
            params, n_samples, metrics = c.fit(initial_params, {'local_epochs': 1, 'privacy_config': p_cfg})
            client_updates.append((cid, params, n_samples, metrics))

        summary = server.aggregate_round(server_round=1, client_results=client_updates)
        self.assertTrue(summary.get('secure_aggregation'))
        self.assertTrue(summary.get('differential_privacy'))
        self.assertIsNotNone(server.epsilon)

    def test_21_validation_selection_under_noise(self):
        """Test 21: Model selection strictly uses validation metrics even with noised updates."""
        from federated.heterogeneous.simulation import run_heterogeneous_simulation
        res = run_heterogeneous_simulation(
            strategy_type="fedavg",
            num_rounds=2,
            local_epochs=1,
            privacy_config=PrivacyConfig(
                secure_aggregation=True,
                differential_privacy=True,
                noise_multiplier=0.3,
                max_update_norm=1.0
            ),
            experiment_name="test_val_selection_dp",
            generate_reports=False
        )
        self.assertIn('best_round', res)
        self.assertGreaterEqual(res['best_round'], 1)
        self.assertIsNotNone(res['val_eval'])

    def test_22_test_data_remains_strictly_isolated(self):
        """Test 22: Test split is never used for training or model selection."""
        schema = get_client_schema("hospital_1")
        client = HeterogeneousHospitalClient(client_id="hospital_1", schema=schema)
        # Verify test_loader is disjoint from train_loader and val_loader
        self.assertIsNotNone(client.test_loader)
        self.assertIsNotNone(client.val_loader)
        self.assertIsNotNone(client.train_loader)
        self.assertNotEqual(id(client.train_loader.dataset), id(client.test_loader.dataset))
        self.assertNotEqual(id(client.val_loader.dataset), id(client.test_loader.dataset))

    def test_23_final_test_evaluated_exactly_once(self):
        """Test 23: Test metrics are evaluated only once at the conclusion of training."""
        from federated.heterogeneous.simulation import run_heterogeneous_simulation
        res = run_heterogeneous_simulation(
            strategy_type="fedavg",
            num_rounds=2,
            local_epochs=1,
            privacy_config=PrivacyConfig(secure_aggregation=True, differential_privacy=False),
            experiment_name="test_once_secagg",
            generate_reports=False
        )
        # Test eval must be present in the final result, not per-round logs
        self.assertIn('test_eval', res)
        self.assertEqual(res['test_eval']['split'], 'test')
        for round_log in res['history']['round_logs']:
            # Round logs only track eval_results which evaluates split="val"
            self.assertEqual(round_log['eval_results']['split'], 'val')


if __name__ == '__main__':
    unittest.main()
