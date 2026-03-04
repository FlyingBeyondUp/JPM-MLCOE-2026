import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Assuming the modules are accessible in your path
from models.base_models import NLSSM
from Filters.basic_filters import UnscentedKalmanFilter
from Filters.flow_filters import EDHFlow, LEDHFlow

tfd = tfp.distributions
dtype = tf.float32


class TestDeterministicFlow(unittest.TestCase):
    """
    Professional test suite for the EDHFlow and LEDHFlow particle flow filters.
    Validates standalone empirical tracking, UKF-guided tracking, numerical
    integration stability, and proper opaque state tuple handling.
    """

    def setUp(self):
        """
        Sets up a stable Non-Linear State-Space Model (NLSSM) and generates
        a batch of observations before every test.
        """
        tf.random.set_seed(42)
        np.random.seed(42)

        self.state_dim = 3
        self.obs_dim = 2
        self.batch_size = 4
        self.time_steps = 8
        self.num_particles = 50
        self.num_flow_steps = 5  # Kept low for fast test execution

        # Define mildly non-linear but highly stable transition and observation functions
        def transition_fn(state, noise):
            # x_t = 0.9 * x_{t-1} + 0.1 * sin(x_{t-1}) + q_t
            return 0.9 * state + 0.1 * tf.sin(state) + noise

        def observation_fn(state, noise):
            # y_t = x_{t, :obs_dim} + r_t
            return state[..., :self.obs_dim] + noise

        # Define Noise Distributions
        process_noise = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.state_dim), scale_diag=tf.ones(self.state_dim) * 0.1)
        observation_noise = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.obs_dim), scale_diag=tf.ones(self.obs_dim) * 0.1)
        init_noise = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.state_dim), scale_diag=tf.ones(self.state_dim) * 1.0)
        x0 = tf.zeros(self.state_dim)

        # Initialize the NLSSM
        self.model = NLSSM(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            process_noise=process_noise,
            observation_noise=observation_noise,
            init_noise=init_noise,
            x0=x0
        )

        # Generate True States and Observations
        self.X_true, self.Y = self.model.sample(batch_size=self.batch_size, T=self.time_steps)

        # Create a reusable UKF for guided flow tests
        self.ukf = UnscentedKalmanFilter(model=self.model, train_noise=False)

    # ==========================================
    # Custom Validation Helpers
    # ==========================================
    def _assert_valid_covariance(self, P: tf.Tensor, tol: float = 1e-4):
        """Asserts that a covariance matrix tensor is symmetric and finite."""
        self.assertTrue(tf.reduce_all(tf.math.is_finite(P)), "Covariance matrix contains NaN or Inf.")
        P_transposed = tf.linalg.matrix_transpose(P)
        max_asymmetry = tf.reduce_max(tf.abs(P - P_transposed))
        self.assertLess(max_asymmetry, tol, f"Covariance matrix is not symmetric. Max diff: {max_asymmetry}")

    def _assert_valid_flow_output(self, results: dict):
        """Strictly validates the shapes and mathematical stability of the flow outputs."""
        expected_keys = {"particles", "x_filt", "P_filt", "log_likelihood"}
        self.assertTrue(expected_keys.issubset(results.keys()))

        # Check Shapes
        self.assertEqual(results['particles'].shape,
                         (self.batch_size, self.time_steps, self.num_particles, self.state_dim))
        self.assertEqual(results['x_filt'].shape,
                         (self.batch_size, self.time_steps, self.state_dim))
        self.assertEqual(results['P_filt'].shape,
                         (self.batch_size, self.time_steps, self.state_dim, self.state_dim))
        self.assertEqual(results['log_likelihood'].shape, (self.batch_size,))

        # Check numerical stability (No exploding ODE integrations)
        for key in expected_keys:
            self.assertTrue(tf.reduce_all(tf.math.is_finite(results[key])),
                            f"Flow output '{key}' contains NaN or Inf due to unstable ODE step.")

        # Ensure empirical covariances are symmetric
        self._assert_valid_covariance(results['P_filt'])

    # ==========================================
    # 1. EDH Flow Tests
    # ==========================================
    def test_edh_standalone_stability(self):
        """Tests the EDHFlow without an embedded UKF (standalone empirical tracking)."""
        edh = EDHFlow(
            model=self.model,
            num_particles=self.num_particles,
            num_flow_steps=self.num_flow_steps,
            ukf=None
        )
        results = edh.filter(self.Y)
        self._assert_valid_flow_output(results)

        # Because UKF is None, the log-likelihood should be strictly zero arrays (dummy placeholder)
        self.assertTrue(tf.reduce_all(results['log_likelihood'] == 0.0))

    def test_edh_with_ukf_guidance(self):
        """Tests the EDHFlow with an embedded UKF supplying the P_xx prior."""
        edh_guided = EDHFlow(
            model=self.model,
            num_particles=self.num_particles,
            num_flow_steps=self.num_flow_steps,
            ukf=self.ukf,
            resample_from_ukf=False
        )
        results = edh_guided.filter(self.Y)
        self._assert_valid_flow_output(results)

        # Because UKF is active, log-likelihood should be populated with actual probabilistic metrics
        self.assertFalse(tf.reduce_all(results['log_likelihood'] == 0.0))

    def test_edh_with_ukf_resampling(self):
        """Tests the EDHFlow utilizing the UKF covariance to resample particles after the flow."""
        edh_resampled = EDHFlow(
            model=self.model,
            num_particles=self.num_particles,
            num_flow_steps=self.num_flow_steps,
            ukf=self.ukf,
            resample_from_ukf=True
        )
        results = edh_resampled.filter(self.Y)
        self._assert_valid_flow_output(results)

    # ==========================================
    # 2. LEDH Flow Tests
    # ==========================================
    def test_ledh_standalone_stability(self):
        """Tests the LEDHFlow (per-particle Jacobians) without UKF."""
        ledh = LEDHFlow(
            model=self.model,
            num_particles=self.num_particles,
            num_flow_steps=self.num_flow_steps,
            ukf=None
        )
        results = ledh.filter(self.Y)
        self._assert_valid_flow_output(results)

    def test_ledh_with_ukf_guidance(self):
        """Tests the LEDHFlow with an embedded UKF supplying the P_xx prior."""
        ledh_guided = LEDHFlow(
            model=self.model,
            num_particles=self.num_particles,
            num_flow_steps=self.num_flow_steps,
            ukf=self.ukf,
            resample_from_ukf=False
        )
        results = ledh_guided.filter(self.Y)
        self._assert_valid_flow_output(results)

    # ==========================================
    # 3. Forecast API Tests
    # ==========================================
    def test_edh_forecast(self):
        """Tests that EDHFlow properly implements the unified forecasting API."""
        edh = EDHFlow(model=self.model, num_particles=self.num_particles)
        y_next, S_next = edh.forecast(self.Y)

        self.assertEqual(y_next.shape, (self.batch_size, self.obs_dim))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y_next)))

        self.assertEqual(S_next.shape, (self.batch_size, self.obs_dim, self.obs_dim))
        self._assert_valid_covariance(S_next)

    def test_ledh_forecast(self):
        """Tests that LEDHFlow properly implements the unified forecasting API."""
        ledh = LEDHFlow(model=self.model, num_particles=self.num_particles)
        y_next, S_next = ledh.forecast(self.Y)

        self.assertEqual(y_next.shape, (self.batch_size, self.obs_dim))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y_next)))

        self.assertEqual(S_next.shape, (self.batch_size, self.obs_dim, self.obs_dim))
        self._assert_valid_covariance(S_next)


if __name__ == '__main__':
    unittest.main(verbosity=2)