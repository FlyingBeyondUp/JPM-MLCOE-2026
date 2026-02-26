import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Assuming these are the module names in your project structure
from codes.models import LearnableSSM
from codes.Filters.differentiable_filters import SoftResamplingParticleFilter

tfd = tfp.distributions


class DummyTransition(tf.keras.layers.Layer):
    """A simple linear transition layer for testing."""

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2, kernel_initializer='identity')

    def call(self, inputs):
        return self.dense(inputs)


class DummyObservation(tf.keras.layers.Layer):
    """A simple linear observation layer for testing."""

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2, kernel_initializer='identity')

    def call(self, inputs):
        return self.dense(inputs)


def create_dummy_model():
    """Helper to create a differentiable State Space Model."""
    state_dim = 2
    obs_dim = 2

    return LearnableSSM(
        state_dim=state_dim,
        obs_dim=obs_dim,
        transition_layers=DummyTransition(),
        observation_layers=DummyObservation(),
        learn_noise=True,  # Enables trainable noise parameters
        learn_init_state=True,  # Enables trainable initial state
        init_noise=tfd.MultivariateNormalDiag(
            loc=tf.zeros(state_dim),
            scale_diag=tf.ones(state_dim)
        )
    )


class TestSoftResamplingParticleFilter(unittest.TestCase):

    def setUp(self):
        """Set up common variables before each test."""
        self.batch_size = 4
        self.num_particles = 50
        self.state_dim = 2
        self.obs_dim = 2
        self.seq_len = 5
        self.model = create_dummy_model()

    def test_initialization(self):
        """Test if the filter initializes with the correct attributes."""
        alpha_val = 0.7
        pf = SoftResamplingParticleFilter(self.model, num_particles=self.num_particles, alpha=alpha_val)

        self.assertEqual(pf.num_particles, self.num_particles)
        self.assertEqual(pf.alpha, alpha_val)
        self.assertIsInstance(pf.optimizer, tf.keras.optimizers.Optimizer)
        self.assertEqual(pf.method, 'soft')  # Base class method name

    def test_resample_shapes_and_normalization(self):
        """Test the _resample method for shape correctness and weight normalization."""
        pf = SoftResamplingParticleFilter(self.model, num_particles=self.num_particles, alpha=0.5)

        # Create dummy particles and uniform weights
        particles = tf.random.normal([self.batch_size, self.num_particles, self.state_dim])
        weights = tf.ones([self.batch_size, self.num_particles]) / self.num_particles

        resampled_particles, resampled_weights = pf._resample(particles, weights)

        # Check shapes
        self.assertEqual(resampled_particles.shape, (self.batch_size, self.num_particles, self.state_dim))
        self.assertEqual(resampled_weights.shape, (self.batch_size, self.num_particles))

        # Check if new weights sum to 1 (normalized)
        weight_sums = tf.reduce_sum(resampled_weights, axis=1)
        tf.debugging.assert_near(weight_sums, tf.ones([self.batch_size]), atol=1e-5)

    def test_resample_hard_alpha(self):
        """Test if alpha=1.0 strictly behaves like standard resampling (weights reset to uniform)."""
        pf = SoftResamplingParticleFilter(self.model, num_particles=self.num_particles, alpha=1.0)

        particles = tf.random.normal([self.batch_size, self.num_particles, self.state_dim])
        # Skewed weights
        weights = tf.random.uniform([self.batch_size, self.num_particles])
        weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

        _, resampled_weights = pf._resample(particles, weights)

        # With alpha=1.0, w_new ~ w^(1-1) = w^0 = 1, so weights should be perfectly uniform
        expected_weights = tf.ones([self.batch_size, self.num_particles]) / self.num_particles
        tf.debugging.assert_near(resampled_weights, expected_weights, atol=1e-5)

    def test_train_step_updates_variables(self):
        """Test if train_step successfully computes gradients and updates model variables."""
        pf = SoftResamplingParticleFilter(self.model, num_particles=20, alpha=0.5)
        # Dummy observations: [Batch, Time, Obs_Dim]
        observations = tf.random.normal([self.batch_size, self.seq_len, self.obs_dim])

        # warm up the filter to ensure all variables are initialized and included in the trainable_variables list
        _ = pf.filter_summarized(observations)

        # Capture pre-update variable values (now includes all Dense weights)
        pre_update_vars = [tf.identity(v) for v in self.model.trainable_variables]
        # Perform one training step
        loss = pf.train_step(observations)
        # Capture post-update variable values
        post_update_vars = self.model.trainable_variables


        self.assertIsNotNone(loss)
        self.assertFalse(tf.math.is_nan(loss))

        # Assert at least one trainable variable has been modified by the optimizer
        has_changed = any(
            not tf.reduce_all(tf.math.equal(pre, post))
            for pre, post in zip(pre_update_vars, post_update_vars)
        )
        self.assertTrue(has_changed, "Model variables were not updated during train_step.")

    def test_fit_loop(self):
        """Test if the fit method can successfully iterate over a tf.data.Dataset."""
        pf = SoftResamplingParticleFilter(self.model, num_particles=10)

        # Create a dummy dataset
        raw_data = tf.random.normal([10, self.seq_len, self.obs_dim])  # 10 total sequences
        dataset = tf.data.Dataset.from_tensor_slices(raw_data).batch(self.batch_size)

        try:
            # Run fit for 10 epochs to ensure the loop functions without crashing
            pf.fit(dataset, epochs=10)
        except Exception as e:
            self.fail(f"fit() method raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()