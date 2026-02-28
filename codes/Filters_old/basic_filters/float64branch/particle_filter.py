import tensorflow as tf
import tensorflow_probability as tfp
from models import NLSSM,LearnableSSM
from typing import Union

tfd = tfp.distributions
dtype=tf.float32

class ParticleFilter:
    def __init__(self,model: Union[NLSSM,LearnableSSM], num_particles: int,resample_method: str = 'multinomial',resample_threshold: float = 1.0):
        self.model = model
        self.num_particles = num_particles
        self.method = resample_method
        self.resample_threshold = resample_threshold

    def _initialize(self,batch_size:int=1)-> tuple[tf.Tensor, tf.Tensor]:
        # Initialize particles from the initial distribution
        total_particles=batch_size*self.num_particles
        noise=self.model.init_noise.sample(total_particles)
        if len(noise.shape) == 1:
            # ensure noise has shape [num_particles*batch_size, state_dim]
            noise = tf.expand_dims(noise, axis=-1)
        particles = self.model.x0 + noise # broadcasting to [total_particles, state_dim]
        weights = tf.ones([batch_size,self.num_particles],dtype=dtype) / tf.cast(self.num_particles,dtype=dtype) # match dtype
        return tf.reshape(particles,[batch_size,self.num_particles,self.model.state_dim]), weights

    def _transition(self, particles: tf.Tensor,observation:Union[None,tf.Tensor]=None)-> tf.Tensor:
        # Propagate particles through the state transition model
        # particles: [Batch_size,num_particles, state_dim]

        Batch_size = tf.shape(particles)[0]

        particles_reshaped = tf.reshape(particles, [-1, self.model.state_dim])  # [Batch_size*num_particles, state_dim]
        process_noise = self.model.process_noise.sample(Batch_size * self.num_particles)
        process_noise = tf.reshape(process_noise, [-1, self.model.state_dim])  # [Batch_size*num_particles, state_dim]
        predicted_particles = self.model.transition_fn(particles_reshaped, process_noise)

        return tf.reshape(predicted_particles, [Batch_size, self.num_particles, self.model.state_dim])  # [Batch_size,num_particles, state_dim]

    def _compute_log_prob(self, target, source, map_fn, noise_dist, dist_fn=None):
        """
        Generic method to calculate log p(target | source).
        Can be used for:
        1. Transition: p(x_t | x_{t-1}) -> target=x_t, source=x_{t-1}, map_fn=transition_fn
        2. Observation: p(y_t | eta_1)  -> target=y_t, source=eta_1,   map_fn=observation_fn

        Args:
            target: [B, 1, D_out] tensor for observations, [B, N, D_state] for states
            source: [B, N, D_in] tensor
            map_fn: Function mapping source -> target (e.g., transition_fn)
            noise_dist: Distribution of the additive noise
            dist_fn: Optional function to get full distribution (e.g., get_transition_dist)
        """
        B = tf.shape(source)[0]
        N = tf.shape(source)[1]

        # Flatten inputs to [B*N, D] for vectorized processing
        # source_flat = tf.reshape(source, [B * N, -1])
        # target_flat = tf.reshape(target, [B * N, -1])
        source_flat = source
        target_flat = target

        if dist_fn is not None:
            dist = dist_fn(source_flat)
            log_prob = dist.log_prob(target_flat)
        else:
            # default: additive noise model
            zero_noise = tf.zeros_like(target_flat)
            pred_flat = map_fn(source_flat, zero_noise)
            residual = target_flat - pred_flat
            log_prob = noise_dist.log_prob(residual)

        return tf.reshape(log_prob, [B, N])

    def _update(self, pre_particles,particles: tf.Tensor, weights: tf.Tensor,Y_t: tf.Tensor):
        '''
        Update particle weights based on the observation likelihood

        :param pre_particles: [Batch_size,num_particles, state_dim], particles before transition,
                              used for computing transition likelihood in the differentiable version of particle filter
                              but not used in the standard particle filter, which only considers observation likelihood
        :param particles: [Batch_size,num_particles, state_dim]
        :param weights: [Batch_size,num_particles]
        :param Y_t: [Batch_size, obs_dim]
        :return: new_weights: [Batch_size,num_particles]
        '''
        # log p(y_t | x_t)
        # Expand Y_t to [B, 1, obs_dim] for likelihood computation
        #Y_t_expanded = tf.tile(tf.expand_dims(Y_t, 1), [1, self.num_particles, 1])
        # The tf.tile costs too much memory when num_particles is large
        Y_t_expanded = tf.expand_dims(Y_t, 1) # use the broadcasting mechanism of tf
        log_prob = self._compute_log_prob(
            target=Y_t_expanded,
            source=particles,
            map_fn=self.model.observation_fn,
            noise_dist=self.model.observation_noise,
            dist_fn=getattr(self.model, 'get_observation_dist', None)
        )
        if len(log_prob.shape) > 2:
            log_prob = tf.reduce_sum(log_prob, axis=-1)

        # w_t = w_{t-1} * p(y_t | x_t)
        # log(w_t) = log(w_{t-1}) + log_prob, shape: [Batch_size,num_particles]
        log_weights = tf.math.log(weights + 1e-10*tf.ones_like(weights)) + log_prob
        # Log-Sum-Exp trick for numerical stability
        log_norm_const = tf.math.reduce_logsumexp(log_weights, axis=1, keepdims=True)
        new_weights = tf.math.exp(log_weights - log_norm_const)
        # The log-likelihood increment is exactly the log_norm_const
        log_lik_increment = tf.squeeze(log_norm_const, axis=1)
        return new_weights, log_lik_increment

    def _get_resample_indices(self, weights: tf.Tensor, batch_size: int, num_particles: int) -> tf.Tensor:
        """
        Helper method to generate resampling indices.
        Can be used by child classes (e.g., InvertiblePFPF) to synchronize resampling.
        """
        if self.method == 'multinomial':
            categorical = tfd.Categorical(probs=weights)
            indices = categorical.sample(num_particles)
            indices = tf.transpose(indices)  # [Batch_size, num_particles]
        elif self.method == 'systematic':
            # Systematic resampling
            cumulative_sum = tf.cumsum(weights, axis=-1)  # [Batch_size, num_particles]
            # [num_particles]
            positions = (tf.range(num_particles, dtype=dtype) + tf.random.uniform([1], 0, 1,dtype=dtype)) / tf.cast(
                num_particles, dtype)
            # [Batch_size, num_particles]
            positions = tf.tile(tf.expand_dims(positions, axis=0), [batch_size, 1])
            indices = tf.searchsorted(cumulative_sum, positions, side='right')  # [Batch_size, num_particles]
        else:
            # Fallback or 'none' - just return range indices
            indices = tf.tile(tf.expand_dims(tf.range(num_particles), 0), [batch_size, 1])

        return tf.clip_by_value(indices, 0, num_particles - 1)

    def _resample(self, particles: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        effective_batch_size = 1.0 / (tf.reduce_sum(tf.square(weights), axis=1) + 1e-10)
        resample_cond = effective_batch_size < (self.resample_threshold * tf.cast(self.num_particles, dtype))

        batch_size = tf.shape(particles)[0]
        indices = self._get_resample_indices(weights, batch_size, self.num_particles)

        p_new = tf.gather(particles, indices, batch_dims=1)
        w_new = tf.fill([batch_size, self.num_particles], 1.0 / float(self.num_particles))
        w_new=tf.cast(w_new, dtype=dtype)

        # Independently update only the batches that need resampling
        cond_expanded_p = tf.reshape(resample_cond, [batch_size, 1, 1])
        particles = tf.where(cond_expanded_p, p_new, particles)

        cond_expanded_w = tf.reshape(resample_cond, [batch_size, 1])
        weights = tf.where(cond_expanded_w, w_new, weights)

        return particles, weights

    def _filter_update(self,pre_particles,particles,weights,Y_t):
        weights,_ = self._update(pre_particles,particles, weights, Y_t)

        x_filt = tf.reduce_sum(tf.expand_dims(weights, axis=-1) * particles, axis=1)  # [B, state_dim]
        diff = particles - tf.expand_dims(x_filt, axis=1)  # [B, num_particles, state_dim]
        # weights: [B, N] -> [B, N, 1] for broadcasting
        weighted_diff = diff * tf.expand_dims(weights, axis=-1)
        # [B, D, N] @ [B, N, D] -> [B, D, D]
        # transpose_a=True effectively permutes weighted_diff to [B, D, N]
        P_filt = tf.matmul(weighted_diff, diff, transpose_a=True)
        return weights, x_filt, P_filt

    @tf.function
    def filter(self, observations: tf.Tensor):
        # Batch-enabled particle filter
        # observations: [B,T, obs_dim]

        batch_size,T=observations.shape[0],observations.shape[1]
        particles, weights = self._initialize(batch_size=batch_size)
        Y_time_major=tf.transpose(observations,perm=[1,0,2]) # [T,B,obs_dim]

        all_particles = tf.TensorArray(dtype=dtype, size=T)
        all_weights = tf.TensorArray(dtype=dtype, size=T)
        x_filt_ta=tf.TensorArray(dtype=dtype,size=T)
        P_filt_ta=tf.TensorArray(dtype=dtype,size=T)

        for t in tf.range(T): # if use native python loop, tf.function will unroll the loop, costing too much memory
            Y_t = Y_time_major[t]  # [B, obs_dim]
            pre_particles=particles
            particles = tf.cond(
                t > 0,
                lambda: self._transition(particles),
                lambda: particles
            )

            weights,x_filt,P_filt = self._filter_update(pre_particles,particles,weights,Y_t)

            x_filt_ta = x_filt_ta.write(t, x_filt)
            P_filt_ta=P_filt_ta.write(t,P_filt)
            all_particles=all_particles.write(t,particles)
            all_weights=all_weights.write(t,weights)

            particles, weights = self._resample(particles, weights)

        return (
            tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),  # [B, T, state_dim]
            tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),  # [B, T, state_dim, state_dim]
            tf.transpose(all_particles.stack(), perm=[1, 0, 2, 3]),  # [B, T, num_particles, state_dim]
            tf.transpose(all_weights.stack(), perm=[1, 0, 2])  # [B, T, num_particles]
        )

    @tf.function
    def filter_summarized(self, observations: tf.Tensor):
        # Only return filtered mean, covariance, and effective sample size
        # while discard particles and weights
        # Batch-enabled particle filter
        # observations: [B,T, obs_dim]

        batch_size, T = tf.shape(observations)[0], tf.shape(observations)[1]

        # Initialize
        # particles: [B, N, state_dim], weights: [B, N]
        particles, weights = self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])  # [T, B, obs_dim]

        # TensorArrays for outputs
        x_filt_ta = tf.TensorArray(dtype=dtype, size=T)
        P_filt_ta = tf.TensorArray(dtype=dtype, size=T)
        ess_ta = tf.TensorArray(dtype=dtype, size=T)  # New accumulator for ESS

        log_l = tf.zeros([batch_size], dtype=dtype)

        for t in tf.range(T):
            Y_t = Y_time_major[t]
            pre_particles=particles
            # Predict
            particles = tf.cond(
                t > 0,
                lambda: self._transition(particles,Y_t),
                lambda: particles
            )
            weights, log_lik_inc = self._update(pre_particles, particles, weights, Y_t)
            # Fix 3: Accumulate safely
            log_l += log_lik_inc

            # ESS = 1 / sum(w^2)
            # weights: [B, N] -> sum_sq: [B] -> ess: [B]
            sum_sq_weights = tf.reduce_sum(tf.square(weights), axis=1)
            ess = 1.0 / (sum_sq_weights + 1e-8)  # Avoid div by zero

            w_expanded=tf.expand_dims(weights, axis=-1)
            x_filt = tf.reduce_sum(w_expanded * particles, axis=1)

            diff = particles - tf.expand_dims(x_filt, axis=1)
            weighted_diff = diff * w_expanded
            P_filt = tf.matmul(weighted_diff, diff, transpose_a=True)

            # --- Write to History ---
            x_filt_ta = x_filt_ta.write(t, x_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)
            ess_ta = ess_ta.write(t, ess)


            particles, weights = self._resample(particles, weights)

        # Stack and Transpose results
        return (
            tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),  # [B, T, state_dim]
            tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),  # [B, T, state_dim, state_dim]
            tf.transpose(ess_ta.stack(), perm=[1, 0]),  # [B, T]
            log_l
        )