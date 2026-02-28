# python
# file: codes/tests/differentiable_filters/test_entropy_regularized_DPF.py

import unittest
import tensorflow as tf
import tensorflow_probability as tfp

from Filters.differentiable_filters.entropy_regularized_OT import DifferentiableParticleFilter
from models.base_models import LearnableSSM

tfd = tfp.distributions


class DummyProposal(tf.keras.layers.Layer):
    """proposal\_layers(x\_pre, y) \-\> (loc, std) with broadcasting."""
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.log_std = tf.Variable(tf.zeros([state_dim], dtype=tf.float32), trainable=True)

    def __call__(self, x_pre, y):
        # x_pre: [B, N, D], y: [B, obs_dim]
        loc = x_pre
        std = tf.nn.softplus(self.log_std) + 1e-3  # [D]
        return loc, std


class DifferentiableParticleFilterTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        tf.random.set_seed(1234)

    def _make_model(self, state_dim=2, obs_dim=2, learn_noise=True):
        transition_layers = tf.keras.layers.Dense(state_dim, use_bias=False)
        observation_layers = tf.keras.layers.Dense(obs_dim, use_bias=False)
        proposal_layers = DummyProposal(state_dim)

        init_noise = tfd.MultivariateNormalDiag(
            loc=tf.zeros([state_dim], tf.float32),
            scale_diag=tf.ones([state_dim], tf.float32) * 0.1,
        )

        model = LearnableSSM(
            state_dim=state_dim,
            obs_dim=obs_dim,
            transition_layers=transition_layers,
            observation_layers=observation_layers,
            proposal_layers=proposal_layers,
            x0_init=tf.zeros([state_dim], dtype=tf.float32),
            init_noise=init_noise,
            learn_noise=learn_noise,
            learn_init_state=True,
            init_noise_scale=0.1,
        )

        # build Keras weights so trainable_variables is populated
        _ = transition_layers(tf.zeros([1, state_dim], tf.float32))
        _ = observation_layers(tf.zeros([1, state_dim], tf.float32))
        _ = proposal_layers(tf.zeros([1, 1, state_dim], tf.float32), tf.zeros([1, obs_dim], tf.float32))

        return model

    def _make_dpf(self, num_particles=8, epsilon=0.5, sinkhorn_iter=20):
        model = self._make_model(state_dim=2, obs_dim=2, learn_noise=True)
        return DifferentiableParticleFilter(
            model=model,
            num_particles=num_particles,
            epsilon=epsilon,
            sinkhorn_iter=sinkhorn_iter,
            scaling=0.75,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        )

    def test_sinkhorn_potentials_shapes(self):
        dpf = self._make_dpf(num_particles=6, sinkhorn_iter=4)
        B, N = 3, dpf.num_particles

        a = tf.ones([B, N], tf.float32) / tf.cast(N, tf.float32)
        b = tf.ones([B, N], tf.float32) / tf.cast(N, tf.float32)
        C = tf.random.uniform([B, N, N], dtype=tf.float32)

        f, g = dpf._sinkhorn_potentials(a, b, C)

        self.assertEqual(tuple(f.shape), (B, N))
        self.assertEqual(tuple(g.shape), (B, N))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(f)).numpy()))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(g)).numpy()))

    def test_differentiable_resample_shapes_and_uniform_weights(self):
        dpf = self._make_dpf(num_particles=10, sinkhorn_iter=5)
        B, N, D = 2, dpf.num_particles, dpf.model.state_dim

        particles = tf.random.normal([B, N, D], dtype=tf.float32)
        w = tf.random.uniform([B, N], minval=0.1, maxval=1.0, dtype=tf.float32)
        w = w / tf.reduce_sum(w, axis=1, keepdims=True)

        new_particles, new_w = dpf._differentiable_resample(particles, w)

        self.assertEqual(tuple(new_particles.shape), (B, N, D))
        self.assertEqual(tuple(new_w.shape), (B, N))

        expected = tf.ones([B, N], tf.float32) / tf.cast(N, tf.float32)
        tf.debugging.assert_near(new_w, expected, atol=1e-5)

        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(new_particles)).numpy()))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(new_w)).numpy()))

    def test_resample_branch_switch(self):
        dpf = self._make_dpf(num_particles=8, sinkhorn_iter=3)
        B, N, D = 1, dpf.num_particles, dpf.model.state_dim

        particles = tf.random.normal([B, N, D], dtype=tf.float32)
        w = tf.ones([B, N], tf.float32) / tf.cast(N, tf.float32)

        dpf.use_differentiable_resample = True
        p1, w1 = dpf._resample(particles, w)
        self.assertEqual(tuple(p1.shape), (B, N, D))
        self.assertEqual(tuple(w1.shape), (B, N))

        dpf.use_differentiable_resample = False
        p2, w2 = dpf._resample(particles, w)
        self.assertEqual(tuple(p2.shape), (B, N, D))
        self.assertEqual(tuple(w2.shape), (B, N))

    def test_filter_forces_disable_differentiable_resample(self):
        dpf = self._make_dpf(num_particles=6, sinkhorn_iter=3)
        dpf.use_differentiable_resample = True

        obs = tf.random.normal([2, 4, dpf.model.obs_dim], dtype=tf.float32)
        _ = dpf.filter(obs)

        self.assertFalse(dpf.use_differentiable_resample)

    def test_train_step_updates_parameters(self):
        dpf = self._make_dpf(num_particles=32, epsilon=0.25, sinkhorn_iter=50)
        dpf.use_differentiable_resample = True

        B, T = 16, 10
        obs = tf.random.normal([B, T, dpf.model.obs_dim], dtype=tf.float32)

        vars_before = [v.numpy().copy() for v in dpf.model.trainable_variables]

        loss = dpf.train_step(obs)

        print(loss)
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(bool(tf.math.is_finite(loss).numpy()))

        vars_after = [v.numpy().copy() for v in dpf.model.trainable_variables]
        changed = any((vb != va).any() for vb, va in zip(vars_before, vars_after))
        self.assertTrue(changed, "train\\_step() should update model parameters, but no changes detected.")

    def test_sinkhorn_correctness(self):
        """
        Professional Check: Does the transport plan P satisfy marginal constraints?
        P * 1 = a (source weights) AND P.T * 1 = b (target weights)
        """
        # Setup: 2 Batches, 10 Particles
        dpf = self._make_dpf(num_particles=10, epsilon=0.25, sinkhorn_iter=100)
        B, N = 2, 10

        # Create random Cost
        C = tf.random.uniform([B, N, N], dtype=tf.float32)
        # Source weights (normalized)
        a = tf.random.uniform([B, N], dtype=tf.float32)
        a = a / tf.reduce_sum(a, axis=1, keepdims=True)
        # Target weights (uniform 1/N)
        b = tf.ones([B, N], dtype=tf.float32) / float(N)

        # Run Sinkhorn
        f, g = dpf._sinkhorn_potentials(a, b, C)

        # Reconstruct Transport Matrix P (Log domain conversion)
        # log P = log(a) + log(b) + (f + g - C) / epsilon
        log_P = tf.expand_dims(tf.math.log(a), 2) + \
                tf.expand_dims(tf.math.log(b), 1) + \
                (tf.expand_dims(f, 2) + tf.expand_dims(g, 1) - C) / dpf.epsilon
        P = tf.exp(log_P)

        # Check Marginal 1: Sum over columns should equal source weights 'a'
        marginal_a = tf.reduce_sum(P, axis=2)
        # Check Marginal 2: Sum over rows should equal target weights 'b'
        marginal_b = tf.reduce_sum(P, axis=1)

        # Allow small numerical error (Sinkhorn is approximate)
        tf.debugging.assert_near(marginal_a, a, atol=1e-3, message="Transport plan does not match source marginals")
        tf.debugging.assert_near(marginal_b, b, atol=1e-3, message="Transport plan does not match target marginals")

    def test_resampling_preserves_moments(self):
        """
        Professional Check: The weighted mean before resampling must approx. equal
        the unweighted mean after resampling.
        """
        dpf = self._make_dpf(num_particles=100, epsilon=0.1, sinkhorn_iter=50)
        B, N, D = 1, 100, 2

        # Create particles clustered far apart to make mean sensitive
        particles = tf.concat([
            tf.random.normal([B, N // 2, D], mean=-10.0),
            tf.random.normal([B, N // 2, D], mean=10.0)
        ], axis=1)

        # Skew weights heavily towards the positive cluster
        # First half (neg cluster) gets low weight, Second half (pos cluster) gets high weight
        w_logits = tf.concat([tf.zeros([B, N // 2]) - 10.0, tf.zeros([B, N // 2]) + 10.0], axis=1)
        w = tf.nn.softmax(w_logits, axis=1)

        # 1. Compute Expected Mean BEFORE resampling
        # weighted_mean = sum(w * x)
        expected_mean = tf.reduce_sum(particles * tf.expand_dims(w, -1), axis=1)

        # 2. Resample
        resampled_p, resampled_w = dpf._differentiable_resample(particles, w)

        # 3. Compute Mean AFTER resampling (weights should be uniform 1/N)
        actual_mean = tf.reduce_mean(resampled_p, axis=1)

        # The deterministic OT resampling is usually very accurate for means
        tf.debugging.assert_near(actual_mean, expected_mean, atol=0.5,
                                 message="Resampling significantly shifted the particle distribution mean.")

    def test_training_convergence_on_simple_sequence(self):
        """
        Professional Check: Can the model actually learn (reduce loss) on a dummy sequence?
        """
        dpf = self._make_dpf(num_particles=32, epsilon=0.25,sinkhorn_iter=50)
        dpf.use_differentiable_resample = True

        # Create a very simple observation sequence (e.g., constant value)
        # The model should easily learn to predict this.
        obs = tf.random.normal([16, 10, dpf.model.obs_dim], dtype=tf.float32)

        initial_loss,init_grad = dpf.train_step(obs,requires_grad=True)
        print('initial loss:', initial_loss.numpy())
        print('initial grad norm:', init_grad)

        # Train for a few steps
        for _ in range(5):
            final_loss,final_grad = dpf.train_step(obs,requires_grad=True)
            print(f"Loss: {final_loss.numpy():.4f}")
            print(f"Gradient norm: {tf.linalg.global_norm(final_grad).numpy():.4f}")

        self.assertLess(final_loss, initial_loss, "Loss did not decrease after training steps.")

    def _assert_all_finite(self, x, name: str):
        x = tf.convert_to_tensor(x)
        tf.debugging.assert_all_finite(
            x,
            f"`{name}` contains NaN/Inf; shape={x.shape}, "
            f"min={tf.reduce_min(x).numpy()}, max={tf.reduce_max(x).numpy()}",
        )

    def test_debug_noise_scales_are_finite_and_bounded(self):
        dpf = self._make_dpf(num_particles=32, epsilon=0.25, sinkhorn_iter=50)

        # Check raw parameters
        self._assert_all_finite(dpf.model.log_process_noise_scale, "log_process_noise_scale")
        self._assert_all_finite(dpf.model.log_obs_noise_scale, "log_obs_noise_scale")

        # Check actual std used by distributions (should be >= 1e-3 per base_models.py)
        proc_scale = tf.exp(dpf.model.log_process_noise_scale)
        obs_scale = tf.exp(dpf.model.log_obs_noise_scale)
        self._assert_all_finite(proc_scale, "exp(log_process_noise_scale)")
        self._assert_all_finite(obs_scale, "exp(log_obs_noise_scale)")

        self.assertTrue(bool(tf.reduce_all(proc_scale >= 0.0).numpy()))
        self.assertTrue(bool(tf.reduce_all(obs_scale >= 0.0).numpy()))

        # Check distribution stds (after min clamp inside properties)
        proc_dist = dpf.model.process_noise
        obs_dist = dpf.model.observation_noise
        self._assert_all_finite(proc_dist.scale.diag, "process_noise.scale_diag")
        self._assert_all_finite(obs_dist.scale.diag, "observation_noise.scale_diag")
        self.assertTrue(bool(tf.reduce_all(proc_dist.scale.diag >= 1e-3).numpy()))
        self.assertTrue(bool(tf.reduce_all(obs_dist.scale.diag >= 1e-3).numpy()))

    def test_debug_proposal_std_is_finite_and_positive(self):
        dpf = self._make_dpf(num_particles=16, epsilon=0.25, sinkhorn_iter=20)

        B, N, D = 4, dpf.num_particles, dpf.model.state_dim
        x_pre = tf.random.normal([B, N, D], dtype=tf.float32)
        y = tf.random.normal([B, dpf.model.obs_dim], dtype=tf.float32)

        proposal = dpf.model.get_proposal_dist(x_pre, y)
        std = proposal.scale.diag  # [D] or broadcasted
        self._assert_all_finite(std, "proposal_std")
        self.assertTrue(bool(tf.reduce_all(std > 0.0).numpy()))

    def test_debug_filter_summarized_outputs_finite(self):
        dpf = self._make_dpf(num_particles=32, epsilon=0.25, sinkhorn_iter=50)
        dpf.use_differentiable_resample = True

        B, T = 16, 10
        obs = tf.random.normal([B, T, dpf.model.obs_dim], dtype=tf.float32)

        # We do not assume exact return semantics; we only assert finiteness on ll_batch.
        out = dpf.filter_summarized(obs)
        ll_batch = out[-1]

        self._assert_all_finite(ll_batch, "log_likelihood_batch")
        # Also guard extreme values that often precede Inf/NaN later
        self.assertTrue(bool(tf.reduce_all(ll_batch > -1e20).numpy()))
        self.assertTrue(bool(tf.reduce_all(ll_batch < 1e20).numpy()))

    def test_debug_train_step_loss_and_gradients_all_finite(self):
        dpf = self._make_dpf(num_particles=32, epsilon=0.25, sinkhorn_iter=50)
        dpf.use_differentiable_resample = True

        B, T = 16, 10
        obs = tf.random.normal([B, T, dpf.model.obs_dim], dtype=tf.float32)

        loss, grads = dpf.train_step(obs, requires_grad=True)
        self._assert_all_finite(loss, "loss")

        # Identify first variable with bad grad (fails with informative message)
        for v, g in zip(dpf.model.trainable_variables, grads):
            if g is None:
                continue
            try:
                self._assert_all_finite(g, f"grad::{v.name}")
            except Exception as e:
                raise AssertionError(f"Gradient became NaN/Inf for `{v.name}`") from e

        # Global norm should also be finite
        gn = tf.linalg.global_norm([g for g in grads if g is not None])
        self._assert_all_finite(gn, "global_grad_norm")

    def test_debug_train_step_two_steps_no_nan_regression(self):
        """
        Minimal reproduction: many NaN bugs appear only after the first update.
        This test catches \`step\_2\` exploding even if \`step\_1\` is finite.
        """
        dpf = self._make_dpf(num_particles=32, epsilon=0.25, sinkhorn_iter=50)
        dpf.use_differentiable_resample = True

        B, T = 16, 10
        obs = tf.random.normal([B, T, dpf.model.obs_dim], dtype=tf.float32)

        loss1 = dpf.train_step(obs)
        self._assert_all_finite(loss1, "loss_step_1")

        loss2 = dpf.train_step(obs)
        self._assert_all_finite(loss2, "loss_step_2")


if __name__ == "__main__":
    unittest.main()
