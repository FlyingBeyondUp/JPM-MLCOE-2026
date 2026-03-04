import tensorflow as tf
import tensorflow_probability as tfp
from Filters.basic_filters.particle_filter import ParticleFilter

tfd = tfp.distributions
dtype = tf.float32


class SoftResamplingParticleFilter(ParticleFilter):
    def __init__(self, model, num_particles,
                 alpha: float = 0.5,
                 resample_threshold: float = 1.0,
                 optimizer: tf.keras.optimizers.Optimizer = None):
        """
        Args:
            model: Instance of modified NLSSM or LearnableSSM.
            num_particles: Number of particles.
            alpha: Softness parameter in [0, 1].
                   1.0 = Hard/Standard Resampling.
                   0.0 = Uniform/No Resampling.
            resample_threshold: ESS threshold fraction to trigger resampling.
            optimizer: Keras optimizer for training.
        """
        super().__init__(model, num_particles, resample_method='soft', resample_threshold=resample_threshold)
        self.alpha = alpha
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Add the boolean state indicators
        self.is_training = False

    def train(self):
        """Activates training mode."""
        self.is_training = True

    def eval(self):
        """Activates evaluation/inference mode."""
        self.is_training = False

    def _resample(self, particles: tf.Tensor, weights: tf.Tensor, ess: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Implements Conditional Soft Resampling.
        Signature explicitly matches the base ParticleFilter.
        """
        batch_size = tf.shape(particles)[0]
        N = self.num_particles

        # Check trigger condition: ESS < threshold * N
        resample_cond = ess < (self.resample_threshold * tf.cast(self.num_particles, dtype))

        # 1. Compute Sampling Probabilities: q ~ w^alpha
        log_weights = tf.math.log(weights + 1e-16)
        log_q = log_weights * self.alpha

        # Normalize q
        q_logits = log_q - tf.reduce_logsumexp(log_q, axis=1, keepdims=True)

        # 2. Sample Indices (Non-differentiable routing)
        categorical = tfd.Categorical(logits=q_logits)
        indices = categorical.sample(N)
        indices = tf.transpose(indices)  # [Batch, N]

        # 3. Gather Particles
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, N])
        gather_indices = tf.stack([batch_indices, indices], axis=-1)

        p_new = tf.gather_nd(particles, gather_indices)

        # 4. Importance Weight Correction (w_new ~ w_old / q ~ w^(1-alpha))
        selected_log_weights = tf.gather_nd(log_weights, gather_indices)

        # In log space: (1 - alpha) * log_w
        log_new_weights_unnorm = selected_log_weights * (1.0 - self.alpha)

        # Normalize new weights
        log_new_weights_norm = log_new_weights_unnorm - tf.reduce_logsumexp(
            log_new_weights_unnorm, axis=1, keepdims=True)
        w_new = tf.exp(log_new_weights_norm)

        # 5. Apply Conditionally
        # Independently update only the sequences in the batch that require resampling
        cond_expanded_p = tf.reshape(resample_cond, [batch_size, 1, 1])
        final_particles = tf.where(cond_expanded_p, p_new, particles)

        cond_expanded_w = tf.reshape(resample_cond, [batch_size, 1])
        final_weights = tf.where(cond_expanded_w, w_new, weights)

        return final_particles, final_weights

    @tf.function
    def train_step(self, observations, requires_grad=False, clip_norm=5.0):
        """
        Standardized training step executing the global AutoGraph filter loop
        with professional gradient stabilization techniques.
        """
        self.train()
        T = tf.cast(tf.shape(observations)[1], dtype)

        with tf.GradientTape() as tape:
            # Run filter and extract dictionary results
            res = super().filter(observations)
            log_likelihood_batch = res['log_likelihood']

            # Time-Normalized Loss
            loss = -tf.reduce_mean(log_likelihood_batch) / T

        # Compute raw gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Safely filter out variables that have no gradient (None)
        valid_grads_and_vars = [
            (g, v) for g, v in zip(gradients, trainable_vars) if g is not None
        ]
        valid_grads = [g for g, v in valid_grads_and_vars]
        valid_vars = [v for g, v in valid_grads_and_vars]

        # Defensive Programming: NaN/Inf Masking
        safe_grads = [
            tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            for g in valid_grads
        ]

        # Global Gradient Clipping
        clipped_grads, global_norm = tf.clip_by_global_norm(safe_grads, clip_norm)

        # Apply the stabilized gradients
        self.optimizer.apply_gradients(zip(clipped_grads, valid_vars))

        if requires_grad:
            return loss, clipped_grads
        return loss

    def fit(self, dataset: tf.data.Dataset, epochs: int = 10, clip_norm=5.0):
        """
        Standard training loop.
        """
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch in dataset:
                loss = self.train_step(batch, clip_norm=clip_norm)
                total_loss += float(loss)
                steps += 1

            print(f"Epoch {epoch + 1}: Loss = {total_loss / max(steps, 1):.4f}")

    def filter(self, observations):
        """Overrides filter to ensure standard state tracking."""
        return super().filter(observations)