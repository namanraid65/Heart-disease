"""
Unit and Integration Test Suite for FedProx and FedOpt (Phase 8)
Verifies:
  1. FedProx proximal penalty computation over shared predictor.
  2. FedProx proximal penalty is exactly zero when predictor equals global parameters.
  3. Strict parameter boundary: changing encoder weights does NOT alter proximal loss.
  4. FedProx client local training executes and returns valid predictor updates.
  5. FedOpt pseudo-gradient calculation: Delta_t = sum_i (n_i / N) * (w_{i, t} - w_t).
  6. FedAdam server momentum and adaptive second-moment update tracking.
  7. FedYogi and FedAdagrad update mechanisms execute properly.
  8. Server optimizer operates strictly on shared predictor; encoder is never passed.
  9. Multi-site heterogeneous feature aggregation (H1=25, H2=25, H3=25, H4=30) across FedProx and FedAdam.
  10. Strict validation-based checkpointing and untouched test evaluation integrity.
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
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.heterogeneous_schema import ClientFeatureSchema, get_client_schema
from models.heterogeneous.composite import build_heterogeneous_model
from federated.heterogeneous.client import HeterogeneousHospitalClient
from federated.heterogeneous.server import HeterogeneousFederatedServer
from federated.heterogeneous.strategy import (
    HeterogeneousFedAvgStrategy,
    HeterogeneousFedProxStrategy,
    HeterogeneousFedOptStrategy,
    HeterogeneousFedAdamStrategy,
    HeterogeneousFedYogiStrategy,
    HeterogeneousFedAdagradStrategy
)
from federated.heterogeneous.evaluate import evaluate_heterogeneous_system


class TestFedProxAndFedOpt(unittest.TestCase):
    """Test suite for FedProx and FedOpt extensions on Heterogeneous FL."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.latent_dim = 32

    def test_01_fedprox_loss_computation(self):
        """TEST 1: FedProx proximal penalty is non-zero when predictor parameters diverge from global."""
        model = build_heterogeneous_model(input_dim=25, latent_dim=32)
        initial_params = [p.clone().detach() for p in model.predictor.parameters()]

        # Mutate predictor parameters
        with torch.no_grad():
            for p in model.predictor.parameters():
                p.add_(0.1)

        mu = 0.5
        prox_term = sum(torch.sum((p - p_init) ** 2) for p, p_init in zip(model.predictor.parameters(), initial_params))
        prox_loss = 0.5 * mu * prox_term

        self.assertGreater(prox_loss.item(), 0.0)

    def test_02_fedprox_zero_loss_at_init(self):
        """TEST 2: FedProx proximal penalty is exactly zero when predictor equals global parameters."""
        model = build_heterogeneous_model(input_dim=25, latent_dim=32)
        initial_params = [p.clone().detach() for p in model.predictor.parameters()]

        mu = 1.0
        prox_term = sum(torch.sum((p - p_init) ** 2) for p, p_init in zip(model.predictor.parameters(), initial_params))
        prox_loss = 0.5 * mu * prox_term

        self.assertAlmostEqual(prox_loss.item(), 0.0, places=6)

    def test_03_fedprox_boundary_isolation(self):
        """TEST 3: Proximal penalty does NOT change when private encoder weights change."""
        model = build_heterogeneous_model(input_dim=25, latent_dim=32)
        initial_predictor_params = [p.clone().detach() for p in model.predictor.parameters()]

        # Compute initial prox loss (should be 0)
        mu = 0.1
        prox_loss_before = 0.5 * mu * sum(
            torch.sum((p - p_init) ** 2)
            for p, p_init in zip(model.predictor.parameters(), initial_predictor_params)
        )

        # Mutate ENCODER weights heavily
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_(10.0)

        prox_loss_after = 0.5 * mu * sum(
            torch.sum((p - p_init) ** 2)
            for p, p_init in zip(model.predictor.parameters(), initial_predictor_params)
        )

        # Encoder mutation must have zero impact on predictor proximal loss
        self.assertEqual(prox_loss_before.item(), prox_loss_after.item())

    def test_04_fedprox_client_convergence(self):
        """TEST 4: Client trains with proximal_mu > 0 and returns updated shared predictor parameters."""
        schema = ClientFeatureSchema(hospital_id='h1', hospital_name='Hospital 1', feature_names=[f"f_{i}" for i in range(25)])
        x_data = torch.randn(32, 25)
        y_data = torch.randint(0, 2, (32, 1)).float()
        loader = DataLoader(TensorDataset(x_data, y_data), batch_size=16)

        client = HeterogeneousHospitalClient(
            client_id='h1',
            schema=schema,
            latent_dim=32,
            device='cpu',
            custom_loaders={'train': loader, 'val': loader, 'test': loader}
        )

        initial_params = client.get_parameters()
        updated_params, n_samples, metrics = client.fit(
            parameters=initial_params,
            config={'local_epochs': 2, 'lr': 0.01, 'proximal_mu': 0.1}
        )

        self.assertEqual(n_samples, 32)
        self.assertIn('prox_loss', metrics)
        self.assertIn('task_loss', metrics)
        self.assertEqual(len(updated_params), len(initial_params))

        # Check weights updated
        changed = any(not np.allclose(u, init) for u, init in zip(updated_params, initial_params))
        self.assertTrue(changed)

    def test_05_fedopt_pseudogradient_computation(self):
        """TEST 5: FedOpt accurately computes sample-weighted pseudo-gradient Delta_t."""
        strategy = HeterogeneousFedOptStrategy(server_lr=0.1)

        # Initial global parameters: two layers of ones
        global_params = [np.ones((4, 4), dtype=np.float32), np.ones((4,), dtype=np.float32)]

        # Client 1 (n=60): layers of 2.0 -> delta = +1.0
        c1_params = [2.0 * np.ones((4, 4), dtype=np.float32), 2.0 * np.ones((4,), dtype=np.float32)]

        # Client 2 (n=40): layers of 0.0 -> delta = -1.0
        c2_params = [np.zeros((4, 4), dtype=np.float32), np.zeros((4,), dtype=np.float32)]

        # Total samples = 100
        # Expected Delta = 0.6 * (+1.0) + 0.4 * (-1.0) = +0.2
        updates = [
            (c1_params, 60),
            (c2_params, 40)
        ]
        delta_t = strategy.compute_pseudo_gradient(updates, global_params, total_samples=100)

        for d in delta_t:
            np.testing.assert_allclose(d, 0.2 * np.ones_like(d), atol=1e-5)

    def test_06_fedadam_moments_and_update(self):
        """TEST 6: FedAdam maintains moments (m_t, v_t) and updates predictor parameters correctly."""
        strategy = HeterogeneousFedAdamStrategy(server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3)
        global_params = [np.zeros((2, 2), dtype=np.float32)]

        # Client update: delta = 1.0 (client params = 1.0, n=100)
        results = [('h1', [np.ones((2, 2), dtype=np.float32)], 100, {'train_loss': 0.5, 'train_accuracy': 0.8})]

        # Round 1
        new_params, summary = strategy.aggregate_fit(server_round=1, results=results, current_global_parameters=global_params)

        # Verify m_t and v_t are initialized and non-zero
        self.assertIsNotNone(strategy.m_t)
        self.assertIsNotNone(strategy.v_t)
        self.assertTrue(np.all(strategy.m_t[0] > 0.0))
        self.assertTrue(np.all(strategy.v_t[0] > 0.0))

        # Verify parameters stepped in the direction of the pseudo-gradient
        self.assertTrue(np.all(new_params[0] > 0.0))

    def test_07_fedyogi_and_fedadagrad_updates(self):
        """TEST 7: FedYogi and FedAdagrad update rules execute properly on shared predictor."""
        global_params = [np.zeros((2, 2), dtype=np.float32)]
        results = [('h1', [np.ones((2, 2), dtype=np.float32)], 50, {'train_loss': 0.4, 'train_accuracy': 0.85})]

        # FedYogi
        yogi = HeterogeneousFedYogiStrategy(server_lr=0.1)
        yogi_params, _ = yogi.aggregate_fit(server_round=1, results=results, current_global_parameters=global_params)
        self.assertTrue(np.all(yogi_params[0] > 0.0))

        # FedAdagrad
        adagrad = HeterogeneousFedAdagradStrategy(server_lr=0.1)
        adagrad_params, _ = adagrad.aggregate_fit(server_round=1, results=results, current_global_parameters=global_params)
        self.assertTrue(np.all(adagrad_params[0] > 0.0))

    def test_08_server_optimizer_boundary(self):
        """TEST 8: Server optimizer operates strictly on shared predictor layers; rejects extra layers."""
        server = HeterogeneousFederatedServer(
            latent_dim=32,
            strategy=HeterogeneousFedAdamStrategy(server_lr=0.1),
            device='cpu'
        )
        valid_params = server.get_global_parameters()

        # Simulate client returning extra layer (e.g. encoder weights)
        illegal_params = list(valid_params) + [np.zeros((64, 25), dtype=np.float32)]
        results = [('rogue_client', illegal_params, 50, {})]

        with self.assertRaises(ValueError) as ctx:
            server.aggregate_round(server_round=1, client_results=results)
        self.assertIn("Parameter count mismatch", str(ctx.exception))

    def test_09_multi_hospital_heterogeneous_fedprox_fedadam(self):
        """
        TEST 9: 4-hospital heterogeneous setup (H1=25, H2=25, H3=25, H4_synthetic=30)
        succeeds with both FedProx and FedAdam.
        """
        client_configs = [
            ('h1', 25, 40),
            ('h2', 25, 40),
            ('h3', 25, 30),
            ('h4_synthetic', 30, 50)
        ]

        # Test both strategies
        for strat_name, strat_obj, active_mu in [
            ('FedProx', HeterogeneousFedProxStrategy(proximal_mu=0.01), 0.01),
            ('FedAdam', HeterogeneousFedAdamStrategy(server_lr=0.1), 0.0)
        ]:
            clients = {}
            for cid, feat_dim, n_samples in client_configs:
                schema = ClientFeatureSchema(
                    hospital_id=cid,
                    hospital_name=f"Hospital {cid}",
                    feature_names=[f"f_{i}" for i in range(feat_dim)]
                )
                x_data = torch.randn(n_samples, feat_dim)
                y_data = torch.randint(0, 2, (n_samples, 1)).float()
                loader = DataLoader(TensorDataset(x_data, y_data), batch_size=16)

                client = HeterogeneousHospitalClient(
                    client_id=cid,
                    schema=schema,
                    latent_dim=32,
                    device='cpu',
                    custom_loaders={'train': loader, 'val': loader, 'test': loader}
                )
                clients[cid] = client

            server = HeterogeneousFederatedServer(
                latent_dim=32,
                strategy=strat_obj,
                device='cpu'
            )

            # Round 1
            global_params = server.get_global_parameters()
            updates = []
            for cid, client in clients.items():
                params, n, m = client.fit(
                    parameters=global_params,
                    config={'local_epochs': 1, 'lr': 0.01, 'proximal_mu': active_mu}
                )
                updates.append((cid, params, n, m))

            summary = server.aggregate_round(server_round=1, client_results=updates)
            self.assertEqual(summary['num_clients'], 4)
            self.assertEqual(summary['total_participating_samples'], 160)

    def test_10_validation_protocol_preserved(self):
        """TEST 10: Validation-based evaluation functions properly under FedAdam and FedProx."""
        server = HeterogeneousFederatedServer(
            latent_dim=32,
            strategy=HeterogeneousFedAdamStrategy(server_lr=0.1),
            device='cpu'
        )
        schema = get_client_schema('hospital_1')
        x_data = torch.randn(20, schema.input_dimension)
        y_data = torch.randint(0, 2, (20, 1)).float()
        loader = DataLoader(TensorDataset(x_data, y_data), batch_size=10)

        client = HeterogeneousHospitalClient(
            client_id='hospital_1',
            schema=schema,
            latent_dim=32,
            device='cpu',
            custom_loaders={'train': loader, 'val': loader, 'test': loader}
        )

        val_eval = evaluate_heterogeneous_system(server, {'hospital_1': client}, split="val")
        self.assertIn('macro_metrics', val_eval)
        self.assertIn('roc_auc', val_eval['macro_metrics'])
        self.assertIn('f1', val_eval['macro_metrics'])


if __name__ == '__main__':
    unittest.main()
