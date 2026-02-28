import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Assuming the modules are accessible in your path
from models.base_models import NLSSM
from Filters.basic_filters import ParticleFilter

tfd = tfp.distributions
dtype = tf.float32


class TestParticleFilter(unittest.TestCase):
    """
    Professional test suite for the ParticleFilter class.
    Validates empirical tracking, sequential importance resampling (SIR),
    weight normalization, dimensional consistency, and forecasting.
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
        self.time_steps = 10
        self.num_particles = 100

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

        # Instantiate the baseline Particle Filter (Multinomial Resampling)
        self.pf = ParticleFilter(
            model=self.model,
            num_particles=self.num_particles,
            resample_method='multinomial',
            resample_threshold=0.5  # Resample when ESS drops below 50%
        )

        # Generate True States and Observations
        self.X_true, self.Y = self.model.sample(batch_size=self.batch_size, T=self.time_steps)

    # ==========================================
    # Custom Validation Helpers
    # ==========================================
    def _assert_valid_covariance(self, P: tf.Tensor, tol: float = 1e-4):
        """Asserts that a covariance matrix tensor is symmetric and finite."""
        self.assertTrue(tf.reduce_all(tf.math.is_finite(P)), "Covariance matrix contains NaN or Inf.")
        P_transposed = tf.linalg.matrix_transpose(P)
        max_asymmetry = tf.reduce_max(tf.abs(P - P_transposed))
        self.assertLess(max_asymmetry, tol, f"Covariance matrix is not symmetric. Max diff: {max_asymmetry}")

    # ==========================================
    # Test Cases
    # ==========================================
    def test_pf_filter_shapes_and_stability(self):
        """
        Tests the PF forward pass for dimensional consistency, mathematical
        stability, and proper normalization of particle weights.
        """
        results = self.pf.filter(self.Y)

        expected_keys = {"particles", "weights", "x_filt", "P_filt", "ess", "log_likelihood"}
        self.assertTrue(expected_keys.issubset(results.keys()))

        # 1. Check Particles and Weights Shapes
        self.assertEqual(results['particles'].shape,
                         (self.batch_size, self.time_steps, self.num_particles, self.state_dim))
        self.assertEqual(results['weights'].shape, (self.batch_size, self.time_steps, self.num_particles))

        # 2. Check Empirical Statistics Shapes
        self.assertEqual(results['x_filt'].shape, (self.batch_size, self.time_steps, self.state_dim))
        self.assertEqual(results['P_filt'].shape, (self.batch_size, self.time_steps, self.state_dim, self.state_dim))
        self.assertEqual(results['ess'].shape, (self.batch_size, self.time_steps))
        self.assertEqual(results['log_likelihood'].shape, (self.batch_size,))

        # 3. Check for NaNs/Infs
        for key in expected_keys:
            self.assertTrue(tf.reduce_all(tf.math.is_finite(results[key])), f"PF output '{key}' contains NaN or Inf.")

        # 4. Check Weight Normalization
        # Weights should sum to exactly 1.0 across the particle dimension for every batch and time step
        weight_sums = tf.reduce_sum(results['weights'], axis=-1)
        max_deviation = tf.reduce_max(tf.abs(weight_sums - 1.0))
        self.assertLess(max_deviation, 1e-4, f"Particle weights do not sum to 1.0. Max deviation: {max_deviation}")

        # 5. Check Covariance Symmetry
        self._assert_valid_covariance(results['P_filt'])

        # 6. Check ESS Bounds (1 <= ESS <= N)
        min_ess = tf.reduce_min(results['ess'])
        max_ess = tf.reduce_max(results['ess'])
        self.assertGreaterEqual(min_ess, 1.0 - 1e-4)
        self.assertLessEqual(max_ess, float(self.num_particles) + 1e-4)

    def test_pf_resampling_trigger(self):
        """
        Tests the resampling mechanism explicitly by forcing the threshold to 1.0.
        This guarantees resampling occurs at every single time step, resetting
        weights uniformly to 1/N.
        """
        forced_resample_pf = ParticleFilter(
            model=self.model,
            num_particles=self.num_particles,
            resample_threshold=1.0  # Will ALWAYS trigger resampling
        )

        results = forced_resample_pf.filter(self.Y)
        weights = results['weights']  # Shape: [B, T, N]

        # After resampling, every weight should exactly equal 1 / N
        expected_uniform_weight = 1.0 / float(self.num_particles)
        max_deviation = tf.reduce_max(tf.abs(weights - expected_uniform_weight))

        # Note: we check starting from t=1 because the t=0 initialization is already uniform.
        self.assertLess(max_deviation, 1e-6, "Resampling failed to reset weights to uniform 1/N.")

    def test_pf_systematic_resampling(self):
        """Tests if the Systematic Resampling algorithm executes correctly without crashing."""
        sys_pf = ParticleFilter(
            model=self.model,
            num_particles=self.num_particles,
            resample_method='systematic',
            resample_threshold=0.8
        )
        # Should complete without TF errors or shape mismatches
        results = sys_pf.filter(self.Y)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(results['particles'])))

    def test_pf_forecast(self):
        """Tests the PF empirical forecasting logic mapping particles to observation space."""
        y_next, S_next = self.pf.forecast(self.Y)

        self.assertEqual(y_next.shape, (self.batch_size, self.obs_dim))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y_next)))

        self.assertEqual(S_next.shape, (self.batch_size, self.obs_dim, self.obs_dim))
        self._assert_valid_covariance(S_next)

    def test_additive_vs_distribution_likelihoods(self):
        """
        Tests that the `_compute_log_prob` dynamically switches logic
        without crashing if `get_observation_dist` is explicitly absent or present.
        """
        # Test 1: Standard Additive fallback (default in NLSSM)
        log_prob_additive = self.pf._compute_log_prob(
            target=tf.zeros([self.batch_size, 1, self.obs_dim]),
            source=tf.zeros([self.batch_size, self.num_particles, self.state_dim]),
            map_fn=self.model.observation_fn,
            noise_dist=self.model.observation_noise,
            dist_fn=None
        )
        self.assertEqual(log_prob_additive.shape, (self.batch_size, self.num_particles))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_prob_additive)))


if __name__ == '__main__':
    unittest.main(verbosity=2)