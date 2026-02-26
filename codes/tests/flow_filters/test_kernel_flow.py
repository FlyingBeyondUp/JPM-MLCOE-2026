import unittest
import tensorflow as tf
import tensorflow_probability as tfp
from Filters.flow_filters import KernelPFF
from models import NLSSM

# Set standard precision as defined in your module
dtype = tf.float64
tfd = tfp.distributions


class TestKernelPFF(unittest.TestCase):
    """Professional Unit Test Suite for the Exact Daum-Huang Kernel Particle Flow Filter."""

    def setUp(self):
        """Sets up a dummy Nonlinear State-Space Model (NLSSM) and test tensors."""
        tf.random.set_seed(42)

        self.state_dim = 6
        self.obs_dim = 2
        self.batch_size = 4
        self.num_particles = 15
        self.num_flow_steps = 3

        # Initialize distributions using float64 to match KernelPFF requirements
        self.x0 = tf.zeros([self.state_dim], dtype=dtype)
        self.init_noise = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.state_dim, dtype=dtype),
            scale_diag=tf.ones(self.state_dim, dtype=dtype)
        )
        self.process_noise = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.state_dim, dtype=dtype),
            scale_diag=tf.ones(self.state_dim, dtype=dtype) * 0.1
        )
        self.observation_noise = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.obs_dim, dtype=dtype),
            scale_diag=tf.ones(self.obs_dim, dtype=dtype) * 0.5
        )

        # Dummy linear transition and observation functions for testing gradient propagation
        def transition_fn(x, noise):
            return x + noise

        def observation_fn(x, noise):
            # Extracts the first two state dimensions
            H = tf.cast(tf.constant([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]), dtype=dtype)
            return tf.matmul(x, H, transpose_b=True) + noise

        self.model = NLSSM(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            process_noise=self.process_noise,
            observation_noise=self.observation_noise,
            init_noise=self.init_noise,
            x0=self.x0
        )

        # Mock particles and observations
        self.particles = tf.random.normal(
            [self.batch_size, self.num_particles, self.state_dim], dtype=dtype
        )
        self.observations = tf.random.normal(
            [self.batch_size, self.obs_dim], dtype=dtype
        )
        self.step_sizes = tf.constant([0.05, 0.1, 0.15], dtype=dtype)

    def test_initialization_parameters(self):
        """Verifies that the KernelPFF initializes the localized distance matrices correctly."""
        pff = KernelPFF(model=self.model, num_particles=self.num_particles)

        self.assertEqual(pff.alpha, 1.0 / self.num_particles)
        self.assertEqual(pff.C_loc_mat.shape, (self.state_dim, self.state_dim))
        self.assertEqual(pff.R_inv_diag.shape, (self.obs_dim,))

        # Ensure the covariance scaling factor is strictly positive
        self.assertTrue(tf.reduce_all(pff.C_loc_mat > 0.0))

    def test_compute_gradients_log_posterior(self):
        """Verifies shape and numerical stability of the log-posterior gradients."""
        pff = KernelPFF(model=self.model, num_particles=self.num_particles)

        x_bar = tf.reduce_mean(self.particles, axis=1, keepdims=True)
        centered = self.particles - x_bar
        B_sample = tf.matmul(centered, centered, transpose_a=True) / (self.num_particles - 1.0)
        B_inv = tf.linalg.inv(B_sample + 1e-5 * tf.eye(self.state_dim, dtype=dtype))

        grads = pff._compute_gradients_log_posterior(
            self.particles, self.observations, B_inv, x_bar
        )

        # Check gradient shape matches particle tensor shape [B, N, D]
        self.assertEqual(grads.shape, (self.batch_size, self.num_particles, self.state_dim))

        # Ensure no NaNs or Infs are produced during gradient computation
        self.assertFalse(tf.reduce_any(tf.math.is_nan(grads)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(grads)))

    def test_flow_update_matrix_kernel(self):
        """Tests the particle flow update using the matrix-valued kernel."""
        pff = KernelPFF(model=self.model, num_particles=self.num_particles, kernel_type='matrix')

        updated_particles = pff._flow_update(
            observations=self.observations,
            particles=self.particles,
            num_flow_steps=self.num_flow_steps,
            step_sizes=self.step_sizes
        )

        # Shape should remain invariant
        self.assertEqual(updated_particles.shape, self.particles.shape)
        # Ensure particles actually moved
        self.assertFalse(tf.reduce_all(tf.math.equal(updated_particles, self.particles)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(updated_particles)))

    def test_flow_update_scalar_kernel(self):
        """Tests the particle flow update using the scalar-valued kernel."""
        pff = KernelPFF(model=self.model, num_particles=self.num_particles, kernel_type='scalar')

        updated_particles = pff._flow_update(
            observations=self.observations,
            particles=self.particles,
            num_flow_steps=self.num_flow_steps,
            step_sizes=self.step_sizes
        )

        self.assertEqual(updated_particles.shape, self.particles.shape)
        self.assertFalse(tf.reduce_all(tf.math.equal(updated_particles, self.particles)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(updated_particles)))


if __name__ == '__main__':
    unittest.main()