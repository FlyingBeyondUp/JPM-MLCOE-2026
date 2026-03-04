import tensorflow as tf
import tensorflow_probability as tfp

from Filters.basic_filters.base_filter import BaseFilter
from models.base_models import AbstractSSM

tfd = tfp.distributions
dtype = tf.float32


class ParticleFilter(BaseFilter):
    """
    Standard Particle Filter (Sequential Importance Resampling).

    Inherits from BaseFilter. Tracks the state distribution empirically using
    a set of weighted particles. The opaque tracking state is defined as a
    2-tuple: (particles, weights).
    """

    def __init__(self, model: AbstractSSM, num_particles: int,
                 resample_method: str = 'multinomial', resample_threshold: float = 1.0):
        super().__init__(model)
        self.num_particles = num_particles
        self.method = resample_method
        self.resample_threshold = resample_threshold

    def _init_state(self, batch_size: int) -> tuple:
        """
        Initializes particles and weights at t=0.
        State is strictly defined as (particles, weights).
        """
        total_particles = batch_size * self.num_particles
        noise = self.model.init_noise.sample(total_particles)

        if len(noise.shape) == 1:
            noise = tf.expand_dims(noise, axis=-1)

        # Broadcast initial state mean across particles
        x0_expanded = tf.expand_dims(self.model.x0, 0)
        particles_flat = x0_expanded + noise

        particles = tf.reshape(particles_flat, [batch_size, self.num_particles, self.model.state_dim])
        weights = tf.ones([batch_size, self.num_particles], dtype=dtype) / tf.cast(self.num_particles, dtype)

        return (particles, weights)

    def _init_trajectory(self, time_steps: int) -> tuple:
        """Initializes TensorArrays to store the sequential Monte Carlo trajectory."""
        particles_ta = tf.TensorArray(dtype, size=time_steps)
        weights_ta = tf.TensorArray(dtype, size=time_steps)
        x_filt_ta = tf.TensorArray(dtype, size=time_steps)
        P_filt_ta = tf.TensorArray(dtype, size=time_steps)
        ess_ta = tf.TensorArray(dtype, size=time_steps)
        logl_ta = tf.TensorArray(dtype, size=time_steps)

        return (particles_ta, weights_ta, x_filt_ta, P_filt_ta, ess_ta, logl_ta)

    def predict(self, t: int, state: tuple) -> tuple:
        """
        Propagates particles through the state transition model.
        Computes p(x_t | x_{t-1}).
        """
        particles, weights = state

        B = tf.shape(particles)[0]
        N = self.num_particles
        D = self.model.state_dim

        particles_flat = tf.reshape(particles, [B * N, D])

        process_noise = self.model.process_noise.sample(B * N)
        if len(process_noise.shape) == 1:
            process_noise = tf.expand_dims(process_noise, -1)

        process_noise_flat = tf.reshape(process_noise, [B * N, D])

        predicted_flat = self.model.transition_fn(particles_flat, process_noise_flat)
        predicted_particles = tf.reshape(predicted_flat, [B, N, D])

        return (predicted_particles, weights)

    def _compute_log_prob(self, target: tf.Tensor, source: tf.Tensor, map_fn, noise_dist, dist_fn=None) -> tf.Tensor:
        """
        Generic method to calculate log p(target | source).
        Dynamically routes between explicit neural-network density functions or additive noise.
        """
        B = tf.shape(source)[0]
        N = tf.shape(source)[1]

        if dist_fn is not None:
            dist = dist_fn(source)
            log_prob = dist.log_prob(target)
        else:
            # Additive noise fallback
            # target: [B, 1, Obs_Dim], source: [B, N, State_Dim]
            # noise must match particle axis: [B, N, Obs_Dim]
            obs_dim = tf.shape(target)[-1]
            zero_noise = tf.zeros([B, N, obs_dim], dtype=source.dtype)

            pred = map_fn(source, zero_noise)  # [B, N, Obs_Dim]
            residual = target - pred  # broadcast [B,1,Obs_Dim] -> [B,N,Obs_Dim]
            log_prob = noise_dist.log_prob(residual)  # [B, N] or [B, N, ...] depending on dist
        return tf.reshape(log_prob, [B, N])

    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        """
        Incorporates an observation to correct particle weights and compute
        filtered sufficient statistics.
        Resampling is triggered here if Effective Sample Size (ESS) drops.
        """
        particles, weights = state
        B = tf.shape(particles)[0]

        # 1. Evaluate Observation Likelihood p(y_t | x_t)
        Y_t_expanded = tf.expand_dims(observation, 1)  # [B, 1, Obs_Dim]
        log_prob = self._compute_log_prob(
            target=Y_t_expanded,
            source=particles,
            map_fn=self.model.observation_fn,
            noise_dist=self.model.observation_noise,
            dist_fn=getattr(self.model, 'get_observation_dist', None)
        )
        if len(log_prob.shape) > 2:
            log_prob = tf.reduce_sum(log_prob, axis=-1)

        # 2. Update Weights
        log_weights = tf.math.log(weights + 1e-10) + log_prob
        log_norm_const = tf.math.reduce_logsumexp(log_weights, axis=1, keepdims=True)
        new_weights = tf.math.exp(log_weights - log_norm_const)
        log_lik_inc = tf.squeeze(log_norm_const, axis=1)

        # 3. Calculate ESS (Effective Sample Size)
        sum_sq_weights = tf.reduce_sum(tf.square(new_weights), axis=1)
        ess = 1.0 / (sum_sq_weights + 1e-10)

        # 4. Extract Analytical Filtered Statistics (Before resampling for exactness)
        w_expanded = tf.expand_dims(new_weights, axis=-1)
        x_filt = tf.reduce_sum(w_expanded * particles, axis=1)
        diff = particles - tf.expand_dims(x_filt, axis=1)
        weighted_diff = diff * w_expanded
        P_filt = tf.matmul(weighted_diff, diff, transpose_a=True)

        # 5. Resample if degenerate
        resampled_particles, resampled_weights = self._resample(particles, new_weights, ess)

        new_state = (resampled_particles, resampled_weights)
        metrics = (log_lik_inc, x_filt, P_filt, ess)
        return new_state, metrics

    def _get_resample_indices(self, weights: tf.Tensor, batch_size: int, num_particles: int) -> tf.Tensor:
        """Generates indices for Sequential Importance Resampling."""
        if self.method == 'multinomial':
            categorical = tfd.Categorical(probs=weights)
            indices = categorical.sample(num_particles)
            indices = tf.transpose(indices)
        elif self.method == 'systematic':
            cumulative_sum = tf.cumsum(weights, axis=-1)
            positions = (tf.range(num_particles, dtype=dtype) + tf.random.uniform([1], 0, 1)) / tf.cast(num_particles,
                                                                                                        dtype)
            positions = tf.tile(tf.expand_dims(positions, axis=0), [batch_size, 1])
            indices = tf.searchsorted(cumulative_sum, positions, side='right')
        else:
            indices = tf.tile(tf.expand_dims(tf.range(num_particles), 0), [batch_size, 1])

        return tf.clip_by_value(indices, 0, num_particles - 1)

    def _resample(self, particles: tf.Tensor, weights: tf.Tensor, ess: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Conditionally resamples particles to prevent weight degeneracy."""
        batch_size = tf.shape(particles)[0]

        # Trigger condition: ESS < threshold * N
        resample_cond = ess < (self.resample_threshold * tf.cast(self.num_particles, dtype))

        indices = self._get_resample_indices(weights, batch_size, self.num_particles)
        p_new = tf.gather(particles, indices, batch_dims=1)
        w_new = tf.fill([batch_size, self.num_particles], 1.0 / tf.cast(self.num_particles, dtype))

        # Independently update only the sequences in the batch that require resampling
        cond_expanded_p = tf.reshape(resample_cond, [batch_size, 1, 1])
        particles = tf.where(cond_expanded_p, p_new, particles)

        cond_expanded_w = tf.reshape(resample_cond, [batch_size, 1])
        weights = tf.where(cond_expanded_w, w_new, weights)

        return particles, weights

    def forecast(self, observations: tf.Tensor) -> tuple:
        """
        Generates the empirical forecast of the next observation based on the provided trajectory.
        """
        res = self.filter(observations)
        particles_last = res['particles'][:, -1, :, :]
        weights_last = res['weights'][:, -1, :]

        # 1. Propagate the empirical distribution to T+1
        state_last = (particles_last, weights_last)
        particles_pred, _ = self.predict(0, state_last)

        B = tf.shape(particles_pred)[0]
        N = self.num_particles

        # 2. Map predicted particles into observation space
        particles_pred_flat = tf.reshape(particles_pred, [B * N, self.model.state_dim])
        zero_noise = tf.zeros([B * N, self.model.obs_dim], dtype=dtype)
        y_pred_flat = self.model.observation_fn(particles_pred_flat, zero_noise)
        y_pred = tf.reshape(y_pred_flat, [B, N, self.model.obs_dim])

        # 3. Extract Analytical Statistics
        w_expanded = tf.expand_dims(weights_last, -1)
        y_pred_mean = tf.reduce_sum(w_expanded * y_pred, axis=1)

        diff = y_pred - tf.expand_dims(y_pred_mean, 1)
        weighted_diff = diff * w_expanded
        S_pred = tf.matmul(weighted_diff, diff, transpose_a=True)

        # Add additive observation noise covariance
        R = tf.expand_dims(self._get_cov(self.model.observation_noise), 0)
        S_next = S_pred + R

        return y_pred_mean, S_next

    def _write_trajectory(self, t: int, trajectory: tuple, state: tuple, metrics: tuple) -> tuple:
        """Writes current empirical state and metrics into the TensorArrays."""
        particles_ta, weights_ta, x_filt_ta, P_filt_ta, ess_ta, logl_ta = trajectory
        particles, weights = state
        log_l, x_filt, P_filt, ess = metrics

        return (
            particles_ta.write(t, particles), weights_ta.write(t, weights),
            x_filt_ta.write(t, x_filt), P_filt_ta.write(t, P_filt),
            ess_ta.write(t, ess), logl_ta.write(t, log_l)
        )

    def _format_output(self, trajectory: tuple) -> dict:
        """Stacks the TensorArrays and returns a clean empirical dictionary."""
        particles_ta, weights_ta, x_filt_ta, P_filt_ta, ess_ta, logl_ta = trajectory

        return {
            "particles": tf.transpose(particles_ta.stack(), perm=[1, 0, 2, 3]),
            "weights": tf.transpose(weights_ta.stack(), perm=[1, 0, 2]),
            "x_filt": tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),
            "P_filt": tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),
            "ess": tf.transpose(ess_ta.stack(), perm=[1, 0]),
            "log_likelihood": tf.reduce_sum(tf.transpose(logl_ta.stack(), perm=[1, 0]), axis=-1)
        }