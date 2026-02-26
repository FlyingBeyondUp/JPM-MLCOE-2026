import os
import unittest
import tensorflow as tf
import tensorflow_probability as tfp

# Assuming your new class is saved in optimal_pfpf.py
from codes.Filters.flow_filters.optimal_path_invertible_flow import OptimalInvertiblePFPF
from codes.Filters.flow_filters.invertible_flow_ekf import ParticleUKF
from models import get1DLogSquaredSVM
from codes.Filters.basic_filters import UnscentedKalmanFilter
from codes.Filters.flow_filters import EDHFlow, LEDHFlow


class TestOptimalInvertiblePFPF(tf.test.TestCase):
    """
    Test suite for the OptimalInvertiblePFPF class, verifying the integration
    of BVP optimal paths into the deterministic particle flow framework.
    """

    def setUp(self):
        """
        Set up the testing environment before each test.
        Initializes a 1D Stochastic Volatility Model, a UKF, and the target filter.
        """
        super().setUp()
        tf.random.set_seed(42)

        # 1. Initialize the user's 1D SV Model
        self.state_dim = 1
        self.obs_dim = 1
        self.model = get1DLogSquaredSVM(alpha=0.9, beta=0.5, sigma=1.0)

        # 2. Initialize the standard UKF required for the InvertiblePFPF
        self.ukf = UnscentedKalmanFilter(model=self.model, alpha=1e-3, beta=2.0, kappa=0.0)

        # 3. Filter Hyperparameters
        self.num_particles = 20
        self.batch_size = 4
        self.bvp_grid_size = 15  # Small grid for fast testing
        self.stiffness_weight = 0.05

        # 4. Initialize the OptimalInvertiblePFPF (using EDHFlow by default)
        self.optimal_filter = OptimalInvertiblePFPF(
            model=self.model,
            num_particles=self.num_particles,
            ukf=self.ukf,
            flow_class=EDHFlow,
            stiffness_weight=self.stiffness_weight,
            bvp_grid_size=self.bvp_grid_size
        )

    def test_initialization(self):
        """Test if the filter initializes correctly and inherits properly."""
        # Use class name to avoid cross-module import mismatch assertions
        self.assertEqual(self.optimal_filter.ukf.__class__.__name__, "ParticleUKF")
        self.assertEqual(self.optimal_filter.mu, self.stiffness_weight)
        self.assertEqual(self.optimal_filter.bvp_grid_size, self.bvp_grid_size)

    def test_get_obs_hessian(self):
        """Test the computation of the observation Hessian (Hh)."""
        # Dummy auxiliary mean points [B, D]
        eta_aux_mean = tf.random.normal([self.batch_size, self.state_dim])

        Hh = self.optimal_filter._get_obs_hessian(eta_aux_mean)

        # Verify shape [B, D, D]
        self.assertEqual(Hh.shape, (self.batch_size, self.state_dim, self.state_dim))

        # Verify negative definiteness (or negative semi-definiteness)
        # Hh = -H^T R^-1 H, so its eigenvalues should be <= 0
        eigenvalues = tf.linalg.eigvalsh(Hh)
        self.assertAllLessEqual(eigenvalues, 1e-5)  # Account for numerical precision

    def test_solve_optimal_path_shapes_and_bounds(self):
        """
        Test the BVP solver to ensure it returns correctly shaped beta/velocity grids
        and satisfies the Dirichlet boundary conditions (beta(0)=0, beta(1)=1).
        """
        # Dummy prior covariance and observation Hessian
        P_xx = tf.eye(self.state_dim, batch_shape=[self.batch_size])
        Hh = -tf.eye(self.state_dim, batch_shape=[self.batch_size]) * 0.5

        betas, beta_dots = self.optimal_filter._solve_optimal_path(P_xx, Hh)

        # 1. Check Shapes: [B, bvp_grid_size]
        self.assertEqual(betas.shape, (self.batch_size, self.bvp_grid_size))
        self.assertEqual(beta_dots.shape, (self.batch_size, self.bvp_grid_size))

        # 2. Check Boundary Conditions
        # beta(0) == 0
        self.assertAllClose(betas[:, 0], tf.zeros([self.batch_size]), atol=1e-3)
        # beta(1) == 1
        self.assertAllClose(betas[:, -1], tf.ones([self.batch_size]), atol=1e-3)

        # 3. Check Monotonicity (betas should generally be increasing)
        beta_diffs = betas[:, 1:] - betas[:, :-1]
        self.assertAllGreaterEqual(beta_diffs, -1e-4)  # Allow tiny numerical dips

    def test_flow_mapping_edh_execution(self):
        """Test the core continuous adaptive flow mapping using EDHFlow."""
        eta_0 = tf.random.normal([self.batch_size, self.num_particles, self.state_dim])
        Y_t = tf.random.normal([self.batch_size, self.obs_dim])
        P_guide = tf.eye(self.state_dim, batch_shape=[self.batch_size])
        eta_aux = tf.random.normal([self.batch_size, self.state_dim])

        final_particles, final_log_det = self.optimal_filter._flow_with_det(
            eta_0, Y_t, num_flow_steps_unused=1, P_guide=P_guide, eta_aux=eta_aux
        )

        # Check output shapes
        self.assertEqual(final_particles.shape, (self.batch_size, self.num_particles, self.state_dim))
        self.assertEqual(final_log_det.shape, (self.batch_size, self.num_particles))

        # Ensure particles actually moved
        difference = tf.reduce_max(tf.abs(final_particles - eta_0))
        self.assertGreater(difference, 0.0)

    def test_flow_mapping_ledh_execution(self):
        """Test the core continuous adaptive flow mapping using LEDHFlow."""
        # Reinitialize filter with LEDHFlow
        ledh_filter = OptimalInvertiblePFPF(
            model=self.model, num_particles=self.num_particles,
            ukf=self.ukf, flow_class=LEDHFlow, stiffness_weight=self.stiffness_weight
        )

        eta_0 = tf.random.normal([self.batch_size, self.num_particles, self.state_dim])
        Y_t = tf.random.normal([self.batch_size, self.obs_dim])
        # LEDH gives per-particle covariances
        P_guide = tf.eye(self.state_dim, batch_shape=[self.batch_size, self.num_particles])
        eta_aux = tf.random.normal([self.batch_size, self.num_particles, self.state_dim])

        final_particles, final_log_det = ledh_filter._flow_with_det(
            eta_0, Y_t, num_flow_steps_unused=1, P_guide=P_guide, eta_aux=eta_aux
        )

        self.assertEqual(final_particles.shape, (self.batch_size, self.num_particles, self.state_dim))
        self.assertEqual(final_log_det.shape, (self.batch_size, self.num_particles))

    def test_zero_stiffness_fallback(self):
        """Test if the filter falls back to a linear beta schedule when mu = 0."""
        linear_filter = OptimalInvertiblePFPF(
            model=self.model, num_particles=self.num_particles,
            ukf=self.ukf, flow_class=EDHFlow, stiffness_weight=0.0  # Force linear schedule
        )

        eta_0 = tf.random.normal([self.batch_size, self.num_particles, self.state_dim])
        Y_t = tf.random.normal([self.batch_size, self.obs_dim])
        P_guide = tf.eye(self.state_dim, batch_shape=[self.batch_size])
        eta_aux = tf.random.normal([self.batch_size, self.state_dim])

        # This should execute successfully without invoking the BVP ODE solver
        final_particles, final_log_det = linear_filter._flow_with_det(
            eta_0, Y_t, num_flow_steps_unused=1, P_guide=P_guide, eta_aux=eta_aux
        )
        self.assertEqual(final_particles.shape, (self.batch_size, self.num_particles, self.state_dim))

    def test_full_filter_execution(self):
        """Test the complete PF-PF filtering loop over a sequence of observations."""
        time_steps = 3

        # Dummy observation sequence [Batch, Time, Obs_Dim]
        y_obs = tf.random.normal([self.batch_size, time_steps, self.obs_dim])

        # Execute the filter (num_flow_steps is overridden by continuous flow, but required by signature)
        x_filt, P_filt, all_particles, all_weights = self.optimal_filter.filter(
            observations=y_obs, num_flow_steps=1
        )

        # Check history shapes
        self.assertEqual(x_filt.shape, (self.batch_size, time_steps, self.state_dim))
        self.assertEqual(P_filt.shape, (self.batch_size, time_steps, self.state_dim, self.state_dim))
        self.assertEqual(all_particles.shape, (self.batch_size, time_steps, self.num_particles, self.state_dim))
        self.assertEqual(all_weights.shape, (self.batch_size, time_steps, self.num_particles))

        # Check weights validity (should sum to 1 across particle dimension)
        weight_sums = tf.reduce_sum(all_weights, axis=-1)
        self.assertAllClose(weight_sums, tf.ones([self.batch_size, time_steps]), atol=1e-5)


if __name__ == '__main__':
    # Run the tests
    # Use verbosity=2 to see detailed output for each test case
    tf.test.main()