import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as layers
from codes.Filters.basic_filters import ParticleFilter

tfd = tfp.distributions


class SoftResamplingParticleFilter(ParticleFilter):
    def __init__(self, model, num_particles,
                 alpha: float = 0.5,
                 optimizer: tf.keras.optimizers.Optimizer = None):
        """
        Args:
            model: Instance of modified NLSSM.
            alpha: Softness parameter in [0, 1].
                   1.0 = Hard/Standard Resampling.
                   0.0 = Uniform/No Resampling.
            optimizer: Keras optimizer for training.
        """
        # We assume the base ParticleFilter handles the flow
        super().__init__(model, num_particles, resample_method='soft')
        self.alpha = alpha
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=1e-3)

    def _resample(self, particles, weights):
        """
        Implements Soft Resampling.
        particles: [Batch, N, State_Dim]
        weights: [Batch, N]
        """
        batch_size = tf.shape(particles)[0]
        N = self.num_particles

        # 1. Compute Sampling Probabilities: q ~ w^alpha
        # We work in log space for stability
        log_weights = tf.math.log(weights + 1e-16)
        log_q = log_weights * self.alpha

        # Normalize q
        q_logits = log_q - tf.reduce_logsumexp(log_q, axis=1, keepdims=True)

        # 2. Sample Indices (Non-differentiable step)
        # We use Categorical distribution to sample indices
        # indices shape: [Batch, N]
        categorical = tfd.Categorical(logits=q_logits)
        indices = categorical.sample(N)
        indices = tf.transpose(indices)  # [Batch, N]

        # 3. Gather Particles
        # Create batch indices for gather_nd
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, N])
        gather_indices = tf.stack([batch_indices, indices], axis=-1)

        resampled_particles = tf.gather_nd(particles, gather_indices)

        # 4. Importance Weight Correction
        # w_new ~ w_old / q ~ w^(1-alpha)

        # Gather the original weights corresponding to the selected particles
        selected_log_weights = tf.gather_nd(log_weights, gather_indices)

        # Calculate new log weights: w_new = w^1 / w^alpha = w^(1-alpha)
        # In log space: (1 - alpha) * log_w
        log_new_weights_unnorm = selected_log_weights * (1.0 - self.alpha)

        # Normalize new weights
        log_new_weights_norm = log_new_weights_unnorm - tf.reduce_logsumexp(log_new_weights_unnorm, axis=1,
                                                                            keepdims=True)
        resampled_weights = tf.exp(log_new_weights_norm)

        return resampled_particles, resampled_weights

    @tf.function
    def train_step(self, observations):
        """
        Performs gradient update using the log-likelihood from the filter.
        Note: The gradient flow here is partial. It flows through the weights
        contributing to the likelihood, but NOT through the resampling indices.
        """
        with tf.GradientTape() as tape:
            # Run filter to get log-likelihood (accumulated log_norm_const)
            # Corrected implementation:
            _, _, _, log_likelihood_batch = self.filter_summarized(observations)

            # Loss = Negative Log Likelihood
            loss = -tf.reduce_mean(log_likelihood_batch)

        # Compute gradients
        # Gradients will flow through:
        # 1. The Likelihood term (log_norm_const) computed in _update
        # 2. The weights carried over from the PREVIOUS step's _resample (via w^(1-alpha))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def fit(self, dataset: tf.data.Dataset, epochs: int = 10):
        """
        Standard training loop.
        """
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch in dataset:
                loss = self.train_step(batch)
                total_loss += float(loss)
                steps += 1

            print(f"Epoch {epoch + 1}: Loss = {total_loss / max(steps, 1):.4f}")