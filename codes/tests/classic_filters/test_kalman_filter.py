import unittest
import tensorflow as tf
import numpy as np

# Assuming the modules are in the same directory or Python path
from models.base_models import LGSSM
from Filters.basic_filters.kalman_filter import KalmanFilter, EM_initializer

dtype = tf.float32


class TestKalmanFilter(unittest.TestCase):
    """
    Professional test suite for the KalmanFilter class and associated routines.
    Validates dimensional consistency, numerical stability fallbacks, and EM learning.
    """

    def setUp(self):
        """
        Sets up the TRUE Linear Gaussian State-Space Model and generates
        true autoregressive observations before every test.
        """
        tf.random.set_seed(42)
        np.random.seed(42)

        self.state_dim = 3
        self.obs_dim = 2
        self.batch_size = 4
        self.time_steps = 15

        # 1. Initialize the TRUE LGSSM model using the default initialization
        self.true_model = LGSSM(state_dim=self.state_dim, obs_dim=self.obs_dim)

        # 2. Instantiate a baseline Kalman Filter for shape/mechanics testing
        self.kf = KalmanFilter(model=self.true_model, requires_stabilization=True)

        # 3. Generate the ground-truth data
        self.X_true, self.Y = self.true_model.sample(batch_size=self.batch_size, T=self.time_steps)

    def test_initialization(self):
        """Tests if the KalmanFilter initializes properties correctly."""
        self.assertTrue(self.kf.requires_stabilization)
        self.assertEqual(self.kf.model.state_dim, self.state_dim)
        self.assertEqual(self.kf.model.obs_dim, self.obs_dim)

    def test_filter_shapes_and_execution(self):
        """
        Tests the forward filter pass to ensure outputs maintain correct
        batch, time, and state/obs dimensions.
        """
        results = self.kf.filter(self.Y)

        expected_keys = {"x_filt", "P_filt", "x_pred", "P_pred", "log_likelihood"}
        self.assertTrue(expected_keys.issubset(results.keys()))

        expected_mean_shape = (self.batch_size, self.time_steps, self.state_dim)
        self.assertEqual(results['x_filt'].shape, expected_mean_shape)

        expected_cov_shape = (self.batch_size, self.time_steps, self.state_dim, self.state_dim)
        self.assertEqual(results['P_filt'].shape, expected_cov_shape)

        self.assertEqual(results['log_likelihood'].shape, (self.batch_size,))

    def test_forecast_shapes(self):
        """Tests the T+1 forecasting dimensions."""
        y_next, S_next = self.kf.forecast(self.Y)

        self.assertEqual(y_next.shape, (self.batch_size, self.obs_dim))
        self.assertEqual(S_next.shape, (self.batch_size, self.obs_dim, self.obs_dim))

    def test_smooth_filter_shapes(self):
        """Tests the backward RTS smoother execution and tensor dimensions."""
        x_smooth, P_smooth, J_ts, log_l = self.kf.smooth(self.Y)

        self.assertEqual(x_smooth.shape, (self.batch_size, self.time_steps, self.state_dim))
        self.assertEqual(P_smooth.shape, (self.batch_size, self.time_steps, self.state_dim, self.state_dim))
        self.assertEqual(J_ts.shape, (self.batch_size, self.time_steps - 1, self.state_dim, self.state_dim))
        self.assertEqual(log_l.shape, (self.batch_size,))

    def test_kalman_fit_em_convergence(self):
        """
        Tests the EM algorithm by initializing a filter with explicitly BAD
        parameters, fitting it to the true data, and verifying that the
        log-likelihood strictly increases as the model learns.
        """
        # 1. Create explicitly bad / naive parameter guesses
        bad_A = tf.eye(self.state_dim, dtype=dtype) * 0.1  # Wrong dynamics (assumes heavy decay)
        bad_C = tf.ones([self.obs_dim, self.state_dim], dtype=dtype) * 0.5  # Wrong observation map
        bad_Q = tf.eye(self.state_dim, dtype=dtype) * 5.0  # Massive overestimation of process noise
        bad_R = tf.eye(self.obs_dim, dtype=dtype) * 5.0  # Massive overestimation of measurement noise
        bad_x0 = tf.zeros([self.state_dim, 1], dtype=dtype)
        bad_P0 = tf.eye(self.state_dim, dtype=dtype) * 10.0

        guess_model = LGSSM(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            params=[bad_A, bad_C, bad_Q, bad_R, bad_x0, bad_P0]
        )

        # 2. Instantiate a learning filter with the bad guess
        kf_learning = KalmanFilter(model=guess_model, requires_stabilization=True)

        n_iters = 15

        # 3. Run the EM algorithm
        log_likelihoods = kf_learning.fit(self.Y, n_iter=n_iters)

        # 4. Verify output types (Checks the float casting fix)
        self.assertIsInstance(log_likelihoods, list)
        self.assertTrue(len(log_likelihoods) > 1)
        for ll in log_likelihoods:
            self.assertIsInstance(ll, float)
            self.assertTrue(tf.math.is_finite(ll))

        # 5. PROVE LEARNING: The final log-likelihood should be vastly superior to the initial guess
        initial_ll = log_likelihoods[0]
        final_ll = log_likelihoods[-1]

        print(f"\n--- EM Learning Test ---")
        print(f"Initial Bad-Guess Log-Likelihood: {initial_ll:.4f}")
        print(f"Final Learned Log-Likelihood:     {final_ll:.4f}")

        # Assert a substantial increase in likelihood
        self.assertGreater(final_ll, initial_ll + 5.0,
                           "EM algorithm failed to significantly improve the log-likelihood from a bad prior.")

        # Prove the Transition matrix actually shifted away from the bad guess
        A_learned = kf_learning.model.A
        max_diff = tf.reduce_max(tf.abs(bad_A - A_learned))
        self.assertGreater(max_diff, 0.1, "The A matrix did not update during the EM M-step.")

    def test_em_initializer(self):
        """
        Tests the SVD-based PCA initializer to ensure it returns a valid
        KalmanFilter instance configured to the data dimensions.
        """
        kf_init = EM_initializer(self.Y, self.state_dim)

        self.assertIsInstance(kf_init, KalmanFilter)
        self.assertEqual(kf_init.model.state_dim, self.state_dim)
        self.assertEqual(kf_init.model.obs_dim, self.obs_dim)

    def test_filter_without_stabilization(self):
        """
        Tests the filter with Joseph form disabled to ensure the standard
        P update branch functions correctly.
        """
        kf_unstable = KalmanFilter(model=self.true_model, requires_stabilization=False)
        results = kf_unstable.filter(self.Y)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(results['P_filt'])))


if __name__ == '__main__':
    unittest.main(verbosity=2)