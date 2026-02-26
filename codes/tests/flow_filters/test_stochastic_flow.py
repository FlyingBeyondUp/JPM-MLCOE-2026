import unittest
import tensorflow as tf
import tensorflow_probability as tfp

from codes.Filters.flow_filters.stochastic_flow import StochasticFlow

tfd = tfp.distributions


class DummyModel:
    def __init__(self, state_dim=3, obs_dim=2):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.observation_noise = tfd.MultivariateNormalDiag(
            loc=tf.zeros(obs_dim),
            scale_diag=tf.ones(obs_dim)
        )

    def observation_fn(self, x, noise):
        # y = Hx + noise, H = [I; 0]
        H = tf.concat(
            [tf.eye(self.obs_dim, dtype=tf.float32),
             tf.zeros((self.obs_dim, self.state_dim - self.obs_dim), dtype=tf.float32)],
            axis=1,
        )
        return tf.matmul(x, H, transpose_b=True) + noise



class StochasticFlowTestCase(unittest.TestCase):
    def _make_flow(self, stiff=0,rng_seed=None):
        model = DummyModel(state_dim=3, obs_dim=2)
        return StochasticFlow(
            model=model,
            num_particles=5,
            stiffness_weight=stiff,
            diffusion_q=tf.random.normal((3, 2)),
            diffusion_dim=2,
            rng_seed=rng_seed,
        )

    def test_flow_update_shapes(self):
        # test the shape of the outputs of _flow_update to ensure consistency.
        flow = self._make_flow(rng_seed=123)
        particles = tf.zeros((2, 5, 3), dtype=tf.float32)
        observations = tf.zeros((2, 2), dtype=tf.float32)

        out_particles, x_filt, P_filt = flow._flow_update(
            observations, particles, num_flow_steps=4
        )

        self.assertEqual(tuple(out_particles.shape), (2, 5, 3))
        self.assertEqual(tuple(x_filt.shape), (2, 3))
        self.assertEqual(tuple(P_filt.shape), (2, 3, 3))

    def test_deterministic_diffusion_noise(self):
        # test that setting rng_seed produces the same diffusion noise across runs,
        # ensuring reproducibility.

        flow = self._make_flow(rng_seed=42)
        particles = tf.random.stateless_normal(
            shape=(1, 5, 3),
            seed=tf.constant([123, 456], dtype=tf.int32),
            dtype=tf.float32
        )
        observations = tf.zeros((1, 2), dtype=tf.float32)

        out1, _, _ = flow._flow_update(observations, particles, num_flow_steps=20)
        out2, _, _ = flow._flow_update(observations, particles, num_flow_steps=20)

        max_diff = tf.reduce_max(tf.abs(out1 - out2))
        print(f"Max difference between runs with same seed: {max_diff.numpy()}")
        self.assertTrue(max_diff.numpy() < 1e-6)


    def test_optimal_path_execution(self):
        # stiffness_weight > 0 forces _get_optimal_path to run
        flow = self._make_flow(stiff=0.1)
        observations = tf.zeros((1, 2))
        particles = tf.random.normal((4, 10, 3))
        # This will trigger the RK4 solver and shooting method
        out, _, _ = flow._flow_update(observations, particles, num_flow_steps=20)
        self.assertEqual(out.shape, (4, 10, 3))

    def test_compute_flow_shapes(self):
        # test that compute_flow returns drift and diffusion terms with correct shapes,
        flow = self._make_flow(rng_seed=11)
        num_flow_steps = 20
        flow.betas = tf.linspace(0.0, 1.0, num_flow_steps)
        flow.beta_dots = tf.ones(num_flow_steps)

        particles = tf.random.uniform((2, 5, 3), dtype=tf.float32)
        observations = tf.zeros((2, 2), dtype=tf.float32)
        P_xx = tf.eye(3, dtype=tf.float32)[None, ...] * tf.ones((2, 1, 1), dtype=tf.float32)

        drift, q = flow.compute_flow(particles, observations, lam=0.5, P_xx=P_xx)
        self.assertEqual(tuple(drift.shape), (2, 5, 3))
        self.assertEqual(tuple(q.shape), (1,3, 2))


    def test_get_optimal_path_dimensions(self):
        # Test that _get_optimal_path produces correct dimensions for betas and beta_dots

        flow = self._make_flow(stiff=0.01)  # stiffness_weight > 0
        B = 3  # batch size
        D = 3  # state dimension
        num_flow_steps = 20

        # Create dummy covariance matrices
        # P_xx: prior covariance [B, D, D]
        P_xx = tf.eye(D)[tf.newaxis, :, :] + tf.random.normal((B, D, D)) * 0.1
        P_xx = tf.matmul(P_xx, P_xx, transpose_b=True)  # ensure positive definite

        # Hh: observation Hessian [B, D, D]
        obs_dim = flow.model.obs_dim
        H = tf.random.normal((B, obs_dim, D))
        Hh = -tf.linalg.matrix_transpose(H) @ flow.R_inv[tf.newaxis, :, :] @ H

        # Call _get_optimal_path
        flow._get_optimal_path(P_xx, Hh, num_flow_steps)

        # Check dimensions
        self.assertIsNotNone(flow.betas)
        self.assertIsNotNone(flow.beta_dots)
        self.assertEqual(tuple(flow.betas.shape), (B, num_flow_steps))
        self.assertEqual(tuple(flow.beta_dots.shape), (B, num_flow_steps))

        # Check boundary conditions for each batch
        # beta(0) ≈ 0
        max_beta_0 = tf.reduce_max(tf.abs(flow.betas[:, 0]))
        self.assertLess(max_beta_0.numpy(), 0.05,
                        f"beta(0) should be close to 0, got max {max_beta_0.numpy()}")

        # beta(1) ≈ 1
        max_beta_1_error = tf.reduce_max(tf.abs(flow.betas[:, -1] - 1.0))
        self.assertLess(max_beta_1_error.numpy(), 1e-3,
                        f"beta(1) should be close to 1, got max error {max_beta_1_error.numpy()}")

        # Check monotonicity: beta should be increasing
        beta_diffs = flow.betas[:, 1:] - flow.betas[:, :-1]
        min_diff = tf.reduce_min(beta_diffs)
        self.assertGreater(min_diff.numpy(), -1e-4,
                           "beta should be monotonically increasing")

        # Check beta_dot values are reasonable (should be positive)
        min_beta_dot = tf.reduce_min(flow.beta_dots)
        self.assertGreater(min_beta_dot.numpy(), 0.0,
                           "beta_dot should be positive")


    def test_get_optimal_path_linear_fallback(self):
        # Test that when mu=0, _get_optimal_path is not called and linear schedule is used

        flow_linear = self._make_flow(stiff=0)
        num_flow_steps = 20

        particles = tf.random.normal((1, 5, 3))
        observations = tf.zeros((1, 2))
        flow_linear._flow_update(observations, particles, num_flow_steps=num_flow_steps)

        # betas should already be initialized in __init__ for mu=0
        self.assertIsNotNone(flow_linear.betas)
        self.assertIsNotNone(flow_linear.beta_dots)
        self.assertEqual(tuple(flow_linear.betas.shape), (num_flow_steps,))
        self.assertEqual(tuple(flow_linear.beta_dots.shape), (num_flow_steps,))

        # Check linear schedule

        expected_betas = tf.cast(tf.linspace(0.0, 1.0, num_flow_steps), flow_linear.betas.dtype)
        max_diff = tf.reduce_max(tf.abs(tf.cast(flow_linear.betas, expected_betas.dtype) - expected_betas)).numpy()
        self.assertLess(max_diff, 1e-6)

        # beta_dots should all be 1.0
        expected_beta_dots = tf.ones(num_flow_steps)
        max_diff_dot = tf.reduce_max(tf.abs(flow_linear.beta_dots - expected_beta_dots))
        self.assertLess(max_diff_dot.numpy(), 1e-6)


    def test_get_optimal_path_consistency_across_batches(self):
        # Test that identical P_xx and Hh produce identical betas across batches

        flow = self._make_flow(stiff=0.1)
        B = 2
        D = 3
        num_flow_steps = 4

        # Create identical covariance matrices for all batches
        P_single = tf.eye(D) * 2.0
        P_xx = tf.stack([P_single] * B)  # [B, D, D]

        H_single = tf.random.normal((flow.model.obs_dim, D))
        Hh_single = -tf.linalg.matrix_transpose(H_single)[tf.newaxis, :, :] @ \
                    flow.R_inv[tf.newaxis, :, :] @ H_single[tf.newaxis, :, :]
        Hh = tf.tile(Hh_single, [B, 1, 1])  # [B, D, D]

        flow._get_optimal_path(P_xx, Hh, num_flow_steps)

        # Check that all batches have identical betas
        for b in range(1, B):
            max_diff = tf.reduce_max(tf.abs(flow.betas[b] - flow.betas[0]))
            self.assertLess(max_diff.numpy(), 1e-3,
                            f"Batch {b} should have same betas as batch 0")

            max_diff_dot = tf.reduce_max(tf.abs(flow.beta_dots[b] - flow.beta_dots[0]))
            self.assertLess(max_diff_dot.numpy(), 1e-3,
                            f"Batch {b} should have same beta_dots as batch 0")


if __name__ == "__main__":
    unittest.main()
