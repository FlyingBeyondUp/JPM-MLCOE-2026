import tensorflow as tf
import tensorflow_probability as tfp
from Filters.basic_filters import ParticleFilter
from models import LearnableSSM

tfd = tfp.distributions

class DifferentiableParticleFilter(ParticleFilter):
    def __init__(self, model:LearnableSSM, num_particles,
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
        # Force 'det' method for clarity, though we override _resample anyway
        super().__init__(model, num_particles, resample_method=resample_method)
        self.epsilon = epsilon
        self.sinkhorn_iter = sinkhorn_iter
        self.scaling = scaling
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.use_differentiable_resample = False

    def _sinkhorn_potentials(self, a, b, C):
        """
        Computes dual potentials f, g using Log-domain Sinkhorn iterations.
        [cite_start]Algorithm 2 from the paper[cite: 138].
        """
        # a, b: [Batch, N], C: [Batch, N, N]
        f = tf.zeros_like(a)
        g = tf.zeros_like(b)
        eps = self.epsilon

        def loop_body(i, f, g):
            # Update f: f = 0.5 * (f + softmin_row(C - g - log_b))
            # Implemented in log-domain for stability
            term_f = (tf.expand_dims(g, 1) - C) / eps + tf.expand_dims(tf.math.log(b), 1)
            f_new = 0.5 * (f - eps * tf.reduce_logsumexp(term_f, axis=2))

            # Update g: g = 0.5 * (g + softmin_col(C - f - log_a))
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
        """
        Differentiable Ensemble Transform (DET) Resampling.
        Overrides the base class resampling to be differentiable.
        return: [B,N,D], [B,N] (particles, weights)
        """
        # Ensure weights are normalized (they should be from _update, but for safety)
        weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-16)

        # Stability Fix 2: Clip weights for gradient safety
        # We clamp the lower bound to 1e-6. The gradient for any weight < 1e-6 becomes 0.
        # This prevents the 1/x explosion.
        a = tf.maximum(weights, 1e-6)

        N = self.num_particles

        # Cost Matrix C
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        # Cost Matrix C
        norm_sq = tf.reduce_sum(tf.square(particles), axis=-1, keepdims=True)  # [B, N, 1]
        C = norm_sq + tf.transpose(norm_sq, [0, 2, 1]) - 2 * tf.matmul(particles, particles, transpose_b=True)

        if self.scaling:
            # Get the state dimension (d_x)
            d_x = tf.cast(tf.shape(particles)[-1], tf.float32)
            # Compute the standard deviation across particles for each dimension
            # particles shape: [Batch, N, d_x] -> std_x shape: [Batch, d_x]
            std_x = tf.math.reduce_std(particles, axis=1)
            # Find the maximum standard deviation among all dimensions
            # max_std shape: [Batch, 1]
            max_std = tf.reduce_max(std_x, axis=-1, keepdims=True)
            # Compute delta(X_t) = sqrt(d_x) * max(std)
            delta = tf.sqrt(d_x) * max_std

            # Prevent division by zero if particles collapse
            delta = tf.maximum(delta, 1e-8)
            # Expand dims to match C shape [Batch, 1, 1]
            delta_expanded = tf.expand_dims(delta, axis=2)
            # Rescale C by delta^2 (since C is squared distance)
            # Crucial: Use tf.stop_gradient so the network doesn't learn to
            # artificially inflate particle variance just to minimize the Sinkhorn cost!
            C = C / tf.stop_gradient(tf.square(delta_expanded))

        # Target distribution 'b' is uniform (1/N)
        # Source distribution 'a' is current weights
        b = tf.ones_like(weights) / float(N)

        # 3. Compute Potentials f, g via Sinkhorn
        f, g = self._sinkhorn_potentials(a, b, C)

        # 4. Compute Transport Matrix P (in log domain then exp)
        # log P_ij = log(a_i) + log(b_j) + (f_i + g_j - C_ij) / eps
        log_P = tf.expand_dims(tf.math.log(a + 1e-16), 2) + \
                tf.expand_dims(tf.math.log(b + 1e-16), 1) + \
                (tf.expand_dims(f, 2) + tf.expand_dims(g, 1) - C) / self.epsilon
        P = tf.exp(log_P)  # [Batch, N, N]

        # 5. Transport Particles (Barycentric Projection)
        # New particles are weighted combinations of old particles based on transport plan P
        # Equation 13 in paper: X_new = N * P^T * X
        # P maps Source(i) -> Target(j). We want to construct Target(j).
        # Target_j = sum_i P_ij * Source_i / b_j
        # Since b_j = 1/N, this becomes Target_j = N * sum_i P_ij * Source_i
        # This corresponds to P^T @ X
        resampled_particles = float(N) * tf.matmul(P, particles, transpose_a=True)

        # After DET, particles are uniformly weighted
        resampled_weights = tf.ones_like(weights) / float(N)

        return resampled_particles, resampled_weights

    def _resample(self, particles, weights):
        # use differentiable resampling during training, standard resampling during evaluation
        if self.use_differentiable_resample:
            return self._differentiable_resample(particles, weights)
        return super()._resample(particles, weights)

    def _transition(self, particles: tf.Tensor, observation: tf.Tensor = None) -> tf.Tensor:
        """
        Professional Transition Step:
        If a proposal network exists and an observation is provided, sample from the proposal q(x_t | x_{t-1}, y_t).
        Otherwise, fall back to the system dynamics p(x_t | x_{t-1}).
        """
        # the following check is to ensure that NLSSMs without proposal layers can still use this DPF without modification.
        if getattr(self.model, 'proposal_layers', None) is not None and observation is not None:
            # Expand observation to match particle count
            # obs: [B, Dy] -> [B, N, Dy]
            obs_expanded = tf.tile(tf.expand_dims(observation, 1), [1, self.num_particles, 1])
            # Get distribution from learnable proposal network
            # q(.): returns a distribution (e.g., MultivariateNormalDiag)
            dist_q = self.model.get_proposal_dist(particles, obs_expanded)
            # Sample new particles using the Reparameterization Trick
            # This allows gradients to flow through the sampling step.
            # shape of predicted_particles: [B, N, Dx]
            predicted_particles = dist_q.sample()
            return predicted_particles

        # Fallback to standard transition (Bootstrap)
        return super()._transition(particles)

    def _update(self, pre_particles, particles, weights, Y_t):
        # Log Likelihood: log p(y_t | x_t)
        Y_t_expanded = tf.expand_dims(Y_t, 1)
        log_lik = self._compute_log_prob(
            target=Y_t_expanded, source=particles,
            map_fn=self.model.observation_fn,
            noise_dist=self.model.observation_noise,
            dist_fn=getattr(self.model, 'get_observation_dist', None)
        )
        if len(log_lik.shape) > 2: log_lik = tf.reduce_sum(log_lik, axis=-1)

        # Transition Prior: log p(x_t | x_{t-1})
        log_trans = self._compute_log_prob(
            target=particles, source=pre_particles,
            map_fn=self.model.transition_fn,
            noise_dist=self.model.process_noise,
            dist_fn=getattr(self.model, 'get_transition_dist', None)
        )
        if len(log_trans.shape) > 2: log_trans = tf.reduce_sum(log_trans, axis=-1)

        # Proposal Posterior: log q(x_t | x_{t-1}, y_t)
        # Only compute this if we actually used a proposal network!
        if getattr(self.model, 'proposal_layers', None) is not None:
            Y_t_expanded_N = tf.tile(Y_t_expanded, [1, self.num_particles, 1])
            dist_q = self.model.get_proposal_dist(pre_particles, Y_t_expanded_N)
            log_prop = dist_q.log_prob(particles)
            if len(log_prop.shape) > 2: log_prop = tf.reduce_sum(log_prop, axis=-1)
            # Professional Weight Update
            log_weights = tf.math.log(weights + 1e-16) + log_lik + log_trans - log_prop
        else:
            # Bootstrap Fallback (terms cancel out)
            log_weights = tf.math.log(weights + 1e-16) + log_lik
            # Log-Sum-Exp trick
        log_norm_const = tf.math.reduce_logsumexp(log_weights, axis=1, keepdims=True)
        new_weights = tf.math.exp(log_weights - log_norm_const)
        log_lik_increment = tf.squeeze(log_norm_const, axis=1)
        return new_weights, log_lik_increment


    @tf.function
    def train_step(self, observations,requires_grad=False):
        """
        Performs one gradient update step using the log-likelihood returned by filter().
        observations: [Batch, T, Obs_Dim]
        """
        with tf.GradientTape() as tape:
            # 1. Run Filter
            # We assume the base class filter returns (trajectories, variances, log_likelihood_accum)
            # The gradients will flow back through:
            # log_likelihood -> _update -> _compute_log_prob -> model parameters
            # AND (Crucially for DPF):
            # log_likelihood -> _resample (DET) -> _update (prev step) -> model parameters
            _, _,_,log_likelihood_batch = super().filter_summarized(observations) # shape: [Batch] (log-likelihood per sequence in batch)

            # 2. Loss = Negative Mean Log-Likelihood
            # Maximize Likelihood => Minimize Negative Likelihood
            loss = -tf.reduce_mean(log_likelihood_batch)

        # 3. Compute Gradients
        # This will capture gradients for:
        # - NLSSM learnable noise scales
        # - NLSSM learnable x0
        # - NLSSM transition/observation layers (Dense weights or explicit matrices)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply Gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if requires_grad:
            return loss, gradients
        return loss

    def fit(self, dataset: tf.data.Dataset, epochs: int = 10):
        """
        Training loop compatible with tf.data.Dataset.
        dataset: yields [Batch, T, Obs_Dim]
        """
        self.use_differentiable_resample = True  # Enable differentiable resampling during training
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch in dataset:
                loss = self.train_step(batch)
                total_loss += float(loss)
                steps += 1

            print(f"Epoch {epoch + 1}: Loss = {total_loss / max(steps, 1):.4f}")

    def filter(self, observations):
        '''
            Override the filter method to disable differentiable resampling during evaluation.
            This allows us to compare the learned model's performance with standard particle filter.
        '''
        self.use_differentiable_resample = False
        return super().filter(observations)