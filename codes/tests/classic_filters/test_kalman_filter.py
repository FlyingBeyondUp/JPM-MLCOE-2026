import unittest
import tensorflow as tf
import numpy as np

# Assuming the modules are in the same directory or Python path
from models import LGSSM
from Filters.basic_filters.kalman_filter import KalmanFilter, EM_solver, EM_initializer


class TestKalmanFilter(unittest.TestCase):
    """
    Unittest for the KalmanFilter class and associated EM routines.
    Validates dimensional consistency, numerical stability fallbacks, and execution logic.
    """

    def setUp(self):
        """
        Sets up a generic Linear Gaussian State-Space Model and generates
        true autoregressive observations before every test.
        """
        tf.random.set_seed(42)
        np.random.seed(42)

        self.state_dim = 3
        self.obs_dim = 2
        self.batch_size = 4
        self.time_steps = 10

        # Initialize the base LGSSM model using the default initialization
        self.model = LGSSM(state_dim=self.state_dim, obs_dim=self.obs_dim)

        # Instantiate the Kalman Filter
        self.kf = KalmanFilter(model=self.model, requires_stabilization=True)

        # Generate observations using the model's own transition and observation equations
        # This returns true_states (X) and observations (Y)
        self.X_true, self.Y = self.model.sample(batch_size=self.batch_size, T=self.time_steps)

    def test_initialization(self):
        """Tests if the KalmanFilter initializes properties correctly."""
        self.assertTrue(self.kf.requires_stabilization)
        self.assertEqual(self.kf.model.state_dim, self.state_dim)
        self.assertEqual(self.kf.model.obs_dim, self.obs_dim)

    def test_filter_shapes_and_execution(self):
        """
        Tests the forward filter pass to ensure outputs maintain correct
        batch, time, and state/obs dimensions and that log-likelihoods are finite.
        """
        results = self.kf.filter(self.Y)

        # Check dictionary keys
        expected_keys = {"x_filt", "P_filt", "x_pred", "P_pred", "log_likelihood"}
        self.assertTrue(expected_keys.issubset(results.keys()))

        # Check Mean Shapes: [Batch, Time, State_Dim]
        expected_mean_shape = (self.batch_size, self.time_steps, self.state_dim)
        self.assertEqual(results['x_filt'].shape, expected_mean_shape)
        self.assertEqual(results['x_pred'].shape, expected_mean_shape)

        # Check Covariance Shapes: [Batch, Time, State_Dim, State_Dim]
        expected_cov_shape = (self.batch_size, self.time_steps, self.state_dim, self.state_dim)
        self.assertEqual(results['P_filt'].shape, expected_cov_shape)
        self.assertEqual(results['P_pred'].shape, expected_cov_shape)

        # Check Log-Likelihood Shape and finiteness: [Batch]
        self.assertEqual(results['log_likelihood'].shape, (self.batch_size,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(results['log_likelihood'])).numpy())

    def test_forecast_shapes(self):
        """Tests the T+1 forecasting dimensions."""
        y_next, S_next = self.kf.forecast(self.Y)

        # Check Predicted Observation Shape: [Batch, Obs_Dim]
        self.assertEqual(y_next.shape, (self.batch_size, self.obs_dim))

        # Check Innovation Covariance Shape: [Batch, Obs_Dim, Obs_Dim]
        self.assertEqual(S_next.shape, (self.batch_size, self.obs_dim, self.obs_dim))

    def test_smooth_filter_shapes(self):
        """Tests the backward RTS smoother execution and tensor dimensions and finite log-likelihood."""
        x_smooth, P_smooth, J_ts, log_l = self.kf.smooth_filter(self.Y)

        # Check Smoothed Mean Shape: [Batch, Time, State_Dim]
        self.assertEqual(x_smooth.shape, (self.batch_size, self.time_steps, self.state_dim))

        # Check Smoothed Covariance Shape: [Batch, Time, State_Dim, State_Dim]
        self.assertEqual(P_smooth.shape, (self.batch_size, self.time_steps, self.state_dim, self.state_dim))

        # Check Smoothing Gain Shape: [Batch, Time-1, State_Dim, State_Dim]
        self.assertEqual(J_ts.shape, (self.batch_size, self.time_steps - 1, self.state_dim, self.state_dim))

        # Check Log-Likelihood Shape and finiteness: [Batch]
        self.assertEqual(log_l.shape, (self.batch_size,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_l)).numpy())

    def test_em_solver_execution(self):
        """
        Tests the EM algorithm to ensure it can run iteratively without
        crashing (e.g., losing positive-definiteness) and returns a list of metrics.
        """
        max_iters = 5

        # Capture the initial A matrix to ensure it updates
        A_initial = tf.identity(self.kf.model.A)

        # Run EM Solver
        log_likelihoods = EM_solver(self.kf, self.Y, max_iters=max_iters)

        # Verify output is a list of scalars mapping to the iterations
        self.assertIsInstance(log_likelihoods, list)
        self.assertTrue(len(log_likelihoods) > 0)

        # Verify the model parameters were actually updated
        A_updated = self.kf.model.A
        self.assertFalse(tf.reduce_all(tf.equal(A_initial, A_updated)))

    def test_em_initializer(self):
        """
        Tests the SVD-based PCA initializer to ensure it returns a valid
        KalmanFilter instance configured to the data dimensions.
        """
        kf_init = EM_initializer(self.Y, self.state_dim)

        # Verify return type
        self.assertIsInstance(kf_init, KalmanFilter)

        # Verify the underlying model dimensions match the data
        self.assertEqual(kf_init.model.state_dim, self.state_dim)
        self.assertEqual(kf_init.model.obs_dim, self.obs_dim)

    def test_filter_without_stabilization(self):
        """
        Tests the filter with Joseph form disabled to ensure the standard
        P update branch functions correctly.
        """
        kf_unstable = KalmanFilter(model=self.model, requires_stabilization=False)
        results = kf_unstable.filter(self.Y)

        # Ensure it completed without NaN values
        self.assertFalse(tf.reduce_any(tf.math.is_nan(results['P_filt'])))


if __name__ == '__main__':
    unittest.main(verbosity=2)