import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Assuming the modules are accessible in your path
from models.base_models import NLSSM
from Filters.basic_filters import ExtendedKalmanFilter, UnscentedKalmanFilter

tfd = tfp.distributions
dtype = tf.float32


class TestNonlinearFilters(unittest.TestCase):
    """
    Professional test suite for the ExtendedKalmanFilter and UnscentedKalmanFilter.
    Validates batched execution, dimensional consistency, numerical stability,
    covariance symmetry, and parameter fitting convergence.
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

        # Instantiate Filters
        self.ekf = ExtendedKalmanFilter(model=self.model, requires_stabilization=True)
        self.ukf = UnscentedKalmanFilter(model=self.model, train_noise=True)

        # Generate True States and Observations using the base class sample method
        self.X_true, self.Y = self.model.sample(batch_size=self.batch_size, T=self.time_steps)

    # ==========================================
    # Custom Validation Helpers
    # ==========================================
    def _assert_valid_covariance(self, P: tf.Tensor, tol: float = 1e-5):
        """Asserts that a covariance matrix tensor is symmetric and contains no NaNs/Infs."""
        self.assertTrue(tf.reduce_all(tf.math.is_finite(P)), "Covariance matrix contains NaN or Inf.")

        # Check symmetry: max |P - P^T| < tol
        P_transposed = tf.linalg.matrix_transpose(P)
        max_asymmetry = tf.reduce_max(tf.abs(P - P_transposed))
        self.assertLess(max_asymmetry, tol, f"Covariance matrix is not symmetric. Max diff: {max_asymmetry}")

    def _assert_valid_filter_output(self, results: dict, check_jacobian: bool = False):
        """Strictly validates the numerical stability and shapes of filter outputs."""
        # 1. Check Log Likelihood
        log_l = results['log_likelihood']
        self.assertEqual(log_l.shape, (self.batch_size,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_l)), "Log-likelihood contains NaN or Inf.")

        # 2. Check States (Means)
        expected_mean_shape = (self.batch_size, self.time_steps, self.state_dim)
        for key in ['x_filt', 'x_pred']:
            self.assertEqual(results[key].shape, expected_mean_shape)
            self.assertTrue(tf.reduce_all(tf.math.is_finite(results[key])), f"{key} contains NaN or Inf.")

        # 3. Check Covariances
        expected_cov_shape = (self.batch_size, self.time_steps, self.state_dim, self.state_dim)
        for key in ['P_filt', 'P_pred']:
            self.assertEqual(results[key].shape, expected_cov_shape)
            self._assert_valid_covariance(results[key])

        # 4. Check Jacobians (EKF only)
        if check_jacobian:
            self.assertEqual(results['A_t'].shape, expected_cov_shape)
            self.assertTrue(tf.reduce_all(tf.math.is_finite(results['A_t'])), "Jacobian A_t contains NaN or Inf.")

    # ==========================================
    # 1. Extended Kalman Filter (EKF) Tests
    # ==========================================
    def test_ekf_filter_stability_and_shapes(self):
        """Tests the EKF forward pass for dimensional consistency and numerical stability."""
        results = self.ekf.filter(self.Y)
        expected_keys = {"x_filt", "P_filt", "x_pred", "P_pred", "A_t", "log_likelihood"}
        self.assertTrue(expected_keys.issubset(results.keys()))

        self._assert_valid_filter_output(results, check_jacobian=True)

    def test_ekf_smooth_stability(self):
        """Tests the EKF RTS Smoother backward pass for stability."""
        x_smooth, P_smooth, P_cross, log_l = self.ekf.smooth(self.Y)

        self.assertEqual(x_smooth.shape, (self.batch_size, self.time_steps, self.state_dim))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(x_smooth)))

        self.assertEqual(P_smooth.shape, (self.batch_size, self.time_steps, self.state_dim, self.state_dim))
        self._assert_valid_covariance(P_smooth)

        self.assertEqual(P_cross.shape, (self.batch_size, self.time_steps - 1, self.state_dim, self.state_dim))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(P_cross)))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_l)))

    def test_ekf_forecast(self):
        """Tests the EKF T+1 forecast logic."""
        y_next, S_next = self.ekf.forecast(self.Y)

        self.assertEqual(y_next.shape, (self.batch_size, self.obs_dim))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y_next)))

        self.assertEqual(S_next.shape, (self.batch_size, self.obs_dim, self.obs_dim))
        self._assert_valid_covariance(S_next)

    def test_ekf_fit_em_convergence(self):
        """Tests the EM algorithm for the EKF, ensuring valid loss history."""
        n_iters = 11
        log_likelihoods = self.ekf.fit(self.Y, n_iter=n_iters)

        self.assertIsInstance(log_likelihoods, list)
        self.assertEqual(len(log_likelihoods), n_iters)

        for ll in log_likelihoods:
            self.assertIsInstance(ll, float)
            self.assertFalse(np.isnan(ll) or np.isinf(ll), "EM produced NaN/Inf log-likelihoods.")

    # ==========================================
    # 2. Unscented Kalman Filter (UKF) Tests
    # ==========================================
    def test_ukf_filter_stability_and_shapes(self):
        """Tests the UKF forward pass for dimensional consistency and numerical stability."""
        results = self.ukf.filter(self.Y)
        expected_keys = {"x_filt", "P_filt", "x_pred", "P_pred", "log_likelihood"}
        self.assertTrue(expected_keys.issubset(results.keys()))

        self._assert_valid_filter_output(results, check_jacobian=False)

    def test_ukf_forecast(self):
        """Tests the UKF T+1 forecast logic utilizing Sigma Points."""
        y_next, S_next = self.ukf.forecast(self.Y)

        self.assertEqual(y_next.shape, (self.batch_size, self.obs_dim))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y_next)))

        self.assertEqual(S_next.shape, (self.batch_size, self.obs_dim, self.obs_dim))
        self._assert_valid_covariance(S_next)

    def test_ukf_fit_bptt_convergence(self):
        """Tests the BPTT algorithm for the UKF, ensuring valid gradients and loss."""
        n_iters = 11

        pre_train_alpha = float(self.ukf.alpha.numpy())
        pre_train_proc_scale = float(self.ukf.proc_log_scale[0].numpy())

        losses = self.ukf.fit(self.Y, n_iter=n_iters, learning_rate=0.05)

        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), n_iters)

        for loss in losses:
            self.assertIsInstance(loss, float)
            self.assertFalse(np.isnan(loss) or np.isinf(loss), "BPTT produced NaN/Inf loss.")

        # Verify that Backpropagation through time actually altered the trainable variables
        post_train_alpha = float(self.ukf.alpha.numpy())
        post_train_proc_scale = float(self.ukf.proc_log_scale[0].numpy())

        self.assertNotEqual(pre_train_alpha, post_train_alpha, "Alpha parameter did not update.")
        self.assertNotEqual(pre_train_proc_scale, post_train_proc_scale, "Process noise parameter did not update.")


if __name__ == '__main__':
    unittest.main(verbosity=2)