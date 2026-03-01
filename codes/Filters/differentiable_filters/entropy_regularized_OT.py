import tensorflow as tf
import tensorflow_probability as tfp
from Filters.basic_filters import ParticleFilter
from models.base_models import LearnableSSM

tfd = tfp.distributions
dtype = tf.float32


class DifferentiableParticleFilter(ParticleFilter):
    def __init__(self, model: LearnableSSM, num_particles: int,
                 epsilon: float = 0.5,
                 sinkhorn_iter: int = 50,
                 scaling: bool = True,
                 optimizer: tf.keras.optimizers.Optimizer = None,
                 resample_method: str = 'multinomial'):
        """
        Args:
            model: Instance of modified NLSSM.
            epsilon: Entropy regularization weight (from paper).
            sinkhorn_iter: Steps for Sinkhorn loop.
            scaling: Optional scaling for cost matrix stability.
            optimizer: Keras optimizer for training.
        """
        super().__init__(model, num_particles, resample_method=resample_method)
        self.epsilon = epsilon
        self.sinkhorn_iter = sinkhorn_iter
        self.scaling = scaling
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.use_differentiable_resample = False

    def _sinkhorn_potentials(self, a, b, C):
        """Computes dual potentials f, g using Log-domain Sinkhorn iterations."""
        f = tf.zeros_like(a)
        g = tf.zeros_like(b)
        eps = self.epsilon

        def loop_body(i, f, g):
            term_f = (tf.expand_dims(g, 1) - C) / eps + tf.expand_dims(tf.math.log(b), 1)
            f_new = 0.5 * (f - eps * tf.reduce_logsumexp(term_f, axis=2))

            term_g = (tf.expand_dims(f_new, 2) - C) / eps + tf.expand_dims(tf.math.log(a), 2)
            g_new = 0.5 * (g - eps * tf.reduce_logsumexp(term_g, axis=1))

            return i + 1, f_new, g_new

        _, f, g = tf.while_loop(
            lambda i, f, g: i < self.sinkhorn_iter,
            loop_body,
            [0, f, g]
        )
        return f, g

    def _differentiable_resample(self, particles, weights):
        """Differentiable Ensemble Transform (DET) Resampling via Sinkhorn."""
        weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-16)
        a = tf.maximum(weights, 1e-6)

        N = self.num_particles

        norm_sq = tf.reduce_sum(tf.square(particles), axis=-1, keepdims=True)
        C = norm_sq + tf.transpose(norm_sq, [0, 2, 1]) - 2 * tf.matmul(particles, particles, transpose_b=True)

        if self.scaling:
            d_x = tf.cast(tf.shape(particles)[-1], dtype)
            std_x = tf.math.reduce_std(particles, axis=1)
            max_std = tf.reduce_max(std_x, axis=-1, keepdims=True)
            delta = tf.sqrt(d_x) * max_std
            delta = tf.maximum(delta, 1e-8)
            delta_expanded = tf.expand_dims(delta, axis=2)
            C = C / tf.stop_gradient(tf.square(delta_expanded))

        b = tf.ones_like(weights) / float(N)

        f, g = self._sinkhorn_potentials(a, b, C)

        log_P = tf.expand_dims(tf.math.log(a + 1e-16), 2) + \
                tf.expand_dims(tf.math.log(b + 1e-16), 1) + \
                (tf.expand_dims(f, 2) + tf.expand_dims(g, 1) - C) / self.epsilon
        P = tf.exp(log_P)

        resampled_particles = float(N) * tf.matmul(P, particles, transpose_a=True)
        resampled_weights = tf.ones_like(weights) / float(N)

        return resampled_particles, resampled_weights

    # ==========================================
    # API ALIGNMENT FIXES
    # ==========================================
    def _resample(self, particles: tf.Tensor, weights: tf.Tensor, ess: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Intercepts resampling to route to OT when training."""
        if self.use_differentiable_resample:
            return self._differentiable_resample(particles, weights)
        return super()._resample(particles, weights, ess)

    def predict(self, t: int, state: tuple) -> tuple:
        """Bypass standard predict if a proposal network is used."""
        if getattr(self.model, 'proposal_layers', None) is not None:
            return state
        return super().predict(t, state)

    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        """Handles proposal sampling and differentiable weight updates."""
        pre_particles, weights = state
        B = tf.shape(pre_particles)[0]
        N = self.num_particles

        def _compute_lik(x):
            Y_t_expanded = tf.expand_dims(observation, 1)
            log_lik = self._compute_log_prob(
                target=Y_t_expanded, source=x,
                map_fn=self.model.observation_fn, noise_dist=self.model.observation_noise,
                dist_fn=getattr(self.model, 'get_observation_dist', None)
            )
            return tf.reduce_sum(log_lik, axis=-1) if len(log_lik.shape) > 2 else log_lik

        def _standard_update():
            # Standard Bootstrap fallback (or t=0)
            log_lik = _compute_lik(pre_particles)
            return pre_particles, tf.math.log(weights + 1e-10) + log_lik

        def _proposal_update():
            # Advanced Proposal Network Update
            obs_expanded = tf.tile(tf.expand_dims(observation, 1), [1, N, 1])
            dist_q = self.model.get_proposal_dist(pre_particles, obs_expanded)
            sampled_particles = dist_q.sample()

            log_lik = _compute_lik(sampled_particles)

            log_trans = self._compute_log_prob(
                target=sampled_particles, source=pre_particles,
                map_fn=self.model.transition_fn, noise_dist=self.model.process_noise,
                dist_fn=getattr(self.model, 'get_transition_dist', None)
            )
            if len(log_trans.shape) > 2: log_trans = tf.reduce_sum(log_trans, axis=-1)

            log_prop = dist_q.log_prob(sampled_particles)
            if len(log_prop.shape) > 2: log_prop = tf.reduce_sum(log_prop, axis=-1)

            log_w = tf.math.log(weights + 1e-10) + log_lik + log_trans - log_prop
            return sampled_particles, log_w

        if getattr(self.model, 'proposal_layers', None) is not None:
            # Only use the proposal network for t > 0
            particles, log_weights = tf.cond(t > 0, _proposal_update, _standard_update)
        else:
            particles, log_weights = _standard_update()

        # Shared metrics and statistics calculation
        log_norm_const = tf.math.reduce_logsumexp(log_weights, axis=1, keepdims=True)
        new_weights = tf.math.exp(log_weights - log_norm_const)
        log_lik_inc = tf.squeeze(log_norm_const, axis=1)

        sum_sq_weights = tf.reduce_sum(tf.square(new_weights), axis=1)
        ess = 1.0 / (sum_sq_weights + 1e-10)

        w_expanded = tf.expand_dims(new_weights, axis=-1)
        x_filt = tf.reduce_sum(w_expanded * particles, axis=1)
        diff = particles - tf.expand_dims(x_filt, axis=1)
        weighted_diff = diff * w_expanded
        P_filt = tf.matmul(weighted_diff, diff, transpose_a=True)

        resampled_particles, resampled_weights = self._resample(particles, new_weights, ess)

        return (resampled_particles, resampled_weights), (log_lik_inc, x_filt, P_filt, ess)

    @tf.function
    def train_step(self, observations, requires_grad=False):
        """Standardized training step executing the global AutoGraph filter loop."""
        self.use_differentiable_resample = True
        with tf.GradientTape() as tape:
            # The base_filter will now execute seamlessly inside the tape
            res = super().filter(observations)
            log_likelihood_batch = res['log_likelihood']

            # Maximize Likelihood => Minimize Negative Likelihood
            loss = -tf.reduce_mean(log_likelihood_batch)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if requires_grad:
            return loss, gradients
        return loss

    def fit(self, dataset: tf.data.Dataset, epochs: int = 10):
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch in dataset:
                loss = self.train_step(batch)
                total_loss += float(loss)
                steps += 1
            print(f"Epoch {epoch + 1}: Loss = {total_loss / max(steps, 1):.4f}")

    def filter(self, observations):
        self.use_differentiable_resample = False
        return super().filter(observations)