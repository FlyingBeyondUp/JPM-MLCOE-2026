from nonlinearSSM import NLSSM,get1DStochasticVolModel
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

tfd = tfp.distributions

class ParticleFilter:
    def __init__(self,model: NLSSM, num_particles: int,resample_method: str = 'multinomial',resample_threshold: float = 0.5):
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
        weights = tf.ones([batch_size,self.num_particles]) / tf.cast(self.num_particles, tf.float32) # match dtype
        return tf.reshape(particles,[batch_size,self.num_particles,self.model.state_dim]), weights

    def _transition(self, particles: tf.Tensor)-> tf.Tensor:
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

    def _update(self, particles: tf.Tensor, weights: tf.Tensor,Y_t: tf.Tensor):
        '''
        Update particle weights based on the observation likelihood
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
            dist_fn=self.model.get_observation_dist
        )
        if len(log_prob.shape) > 2:
            log_prob = tf.reduce_sum(log_prob, axis=-1)

        # w_t = w_{t-1} * p(y_t | x_t)
        # log(w_t) = log(w_{t-1}) + log_prob, shape: [Batch_size,num_particles]
        log_weights = tf.math.log(weights + 1e-10*tf.ones_like(weights)) + log_prob
        log_norm_const = tf.math.reduce_logsumexp(log_weights, axis=1,keepdims=True) # average over particles
        new_weights = tf.math.exp(log_weights - log_norm_const)
        return new_weights

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
            positions = (tf.range(num_particles, dtype=tf.float32) + tf.random.uniform([1], 0, 1)) / tf.cast(
                num_particles, tf.float32)
            # [Batch_size, num_particles]
            positions = tf.tile(tf.expand_dims(positions, axis=0), [batch_size, 1])
            indices = tf.searchsorted(cumulative_sum, positions, side='right')  # [Batch_size, num_particles]
        else:
            # Fallback or 'none' - just return range indices
            indices = tf.tile(tf.expand_dims(tf.range(num_particles), 0), [batch_size, 1])

        return tf.clip_by_value(indices, 0, num_particles - 1)

    def _resample(self, particles: tf.Tensor, weights: tf.Tensor)-> tuple[tf.Tensor, tf.Tensor]:
        '''
        Resample particles based on their weights
        particles: [Batch_size,num_particles, state_dim]
        weights: [Batch_size,num_particles]
        return:
                particles: [Batch_size,num_particles, state_dim],
                weights:   [Batch_size,num_particles]
        '''
        # effective_batch_size = 1/tf.reduce_sum(tf.square(weights), axis=1)  # [Batch_size]
        # resample_cond = effective_batch_size < (self.resample_threshold * tf.cast(self.num_particles, tf.float32))  # [Batch_size]
        # if not resample_cond:
        #     return particles, weights

        if self.method == 'multinomial':
            categorical = tfd.Categorical(probs=weights)
            indices = categorical.sample(self.num_particles)
            indices=tf.transpose(indices) # [Batch_size,num_particles]
        elif self.method == 'systematic':
            # Systematic resampling
            cumulative_sum = tf.cumsum(weights, axis=-1)  # [Batch_size,num_particles]
            positions = (tf.range(self.num_particles, dtype=tf.float32) + tf.random.uniform([1], 0, 1)) / self.num_particles  # [num_particles]
            positions = tf.tile(tf.expand_dims(positions, axis=0), [tf.shape(weights)[0], 1])  # [Batch_size,num_particles]

            indices = tf.searchsorted(cumulative_sum, positions, side='right')  # [Batch_size,num_particles]
        elif self.method=='none':
            return particles, weights
        else:
            raise ValueError(f"Unknown resampling method: {self.method}")
        indices = tf.clip_by_value(indices, 0, self.num_particles - 1)
        batch_size = tf.shape(particles)[0]
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=-1), [1, self.num_particles])  # [Batch_size,num_particles]
        resampled_particles = tf.gather_nd(particles, tf.stack([batch_indices, indices], axis=-1))  # [Batch_size,num_particles, state_dim]
        resampled_weights = tf.ones_like(weights) / self.num_particles  # [Batch_size,num_particles]
        return resampled_particles, resampled_weights

    @tf.function
    def filter(self, observations: tf.Tensor):
        # Batch-enabled particle filter
        # observations: [B,T, obs_dim]

        batch_size,T=observations.shape[0],observations.shape[1]
        particles, weights = self._initialize(batch_size=batch_size)
        Y_time_major=tf.transpose(observations,perm=[1,0,2]) # [T,B,obs_dim]

        all_particles = tf.TensorArray(dtype=tf.float32, size=T)
        all_weights = tf.TensorArray(dtype=tf.float32, size=T)
        x_filt_ta=tf.TensorArray(dtype=tf.float32,size=T)
        P_filt_ta=tf.TensorArray(dtype=tf.float32,size=T)

        for t in tf.range(T): # if use native python loop, tf.function will unroll the loop, costing too much memory
            Y_t = Y_time_major[t]  # [B, obs_dim]
            particles = tf.cond(
                t > 0,
                lambda: self._transition(particles),
                lambda: particles
            )
            weights = self._update(particles, weights, Y_t)

            x_filt=tf.reduce_sum(tf.expand_dims(weights, axis=-1) * particles, axis=1)  # [B, state_dim]
            diff = particles - tf.expand_dims(x_filt, axis=1)  # [B, num_particles, state_dim]
            # weights: [B, N] -> [B, N, 1] for broadcasting
            weighted_diff = diff * tf.expand_dims(weights, axis=-1)
            # [B, D, N] @ [B, N, D] -> [B, D, D]
            # transpose_a=True effectively permutes weighted_diff to [B, D, N]
            P_filt = tf.matmul(weighted_diff, diff, transpose_a=True)

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
        particles, weights = self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])  # [T, B, obs_dim]

        # TensorArrays for outputs
        x_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        ess_ta = tf.TensorArray(dtype=tf.float32, size=T)  # New accumulator for ESS

        for t in tf.range(T):
            Y_t = Y_time_major[t]

            # Predict
            particles = tf.cond(
                t > 0,
                lambda: self._transition(particles),
                lambda: particles
            )
            weights = self._update(particles, weights, Y_t)

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

            # Resample
            particles, weights = self._resample(particles, weights)

        # Stack and Transpose results
        return (
            tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),  # [B, T, state_dim]
            tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),  # [B, T, state_dim, state_dim]
            tf.transpose(ess_ta.stack(), perm=[1, 0])  # [B, T]
        )

def run_resampling_experiment(model: NLSSM, num_particles: int, x_true:tf.Tensor,y_obs: tf.Tensor,num_runs: int = 20):
    results={'multinomial':[], 'systematic':[]}
    for method in results.keys():
        pf = ParticleFilter(model=model, num_particles=num_particles, resample_method=method)
        estimates=[]
        for run in range(num_runs):
            x_filt, _,_,_ = pf.filter(y_obs)
            estimates.append(x_filt[0, :, 0].numpy())
        results[method]=np.array(estimates)
    x_true_np=x_true.numpy()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    var_mult = np.var(results['multinomial'], axis=0)
    var_syst = np.var(results['systematic'], axis=0)
    plt.plot(var_mult, label='Multinomial Variance', color='red', alpha=0.7)
    plt.plot(var_syst, label='Systematic Variance', color='blue', alpha=0.7)
    plt.title('Variance of Estimator across Trials') #  (Lower is Better)
    plt.xlabel('Time Step')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    rmse_mult = np.sqrt(np.mean((results['multinomial'] - x_true_np) ** 2, axis=1))
    rmse_syst = np.sqrt(np.mean((results['systematic'] - x_true_np) ** 2, axis=1))
    plt.boxplot([rmse_mult, rmse_syst], labels=['Multinomial', 'Systematic'])
    plt.title(f'RMSE Distribution over {num_runs} Trials')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Resampling_Methods_Comparison.pdf",bbox_inches='tight')
    plt.show()

    print(f"Multinomial Mean Variance: {np.mean(var_mult):.6f}")
    print(f"Systematic Mean Variance:  {np.mean(var_syst):.6f}")


def experiment_weight_degeneracy(model, y_obs,method='none'):
    print("\n--- Running Weight Degeneracy Experiment (No Resampling) ---")
    # 使用 'none' 方法禁用重采样
    pf = ParticleFilter(model=model, num_particles=1000, resample_method='systematic')

    _, _, _, all_weights = pf.filter(y_obs)

    # all_weights shape: [Batch, T, Num_Particles]
    weights_np = all_weights.numpy()[0]

    # 计算 N_eff = 1 / sum(w^2)
    n_eff = 1.0 / np.sum(weights_np ** 2, axis=1) # sum over particles

    pf = ParticleFilter(model=model, num_particles=1000, resample_method='none')

    _, _, _, all_weights = pf.filter(y_obs)

    # all_weights shape: [Batch, T, Num_Particles]
    weights_np = all_weights.numpy()[0]

    # 计算 N_eff = 1 / sum(w^2)
    n_eff_no_resample = 1.0 / np.sum(weights_np ** 2, axis=1)  # sum over particles

    plt.figure(figsize=(10, 5))
    plt.plot(n_eff, label='Effective Sample Size ($N_{eff}$)')
    plt.plot(n_eff_no_resample, label='Effective Sample Size without Resampling ($N_{eff}$)', linestyle='--')
    plt.axhline(y=1000, color='r', linestyle='--', label='Total Particles (Ideal)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Weight Degeneracy: $N_{eff}$ over Time, {method}')
    plt.xlabel('Time Step')
    plt.ylabel('$N_{eff}$')
    #plt.yscale('log')  # 使用对数坐标更明显
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Weight_Degeneracy.pdf",bbox_inches='tight')
    plt.show()


def experiment_static_impoverishment_wrong_convergence(T: int = 100):
    true_state_value = 0.0

    # define a static model where the state does not change
    def transition_fn(x, noise):
        return x + noise

    def observation_fn(x, noise):
        return x + noise

    static_model = NLSSM(
        state_dim=1, obs_dim=1,
        x0=tf.ones([1, 1]) * true_state_value,
        init_noise=tfd.Normal(loc=0.0, scale=0.1),
        process_noise=tfd.Normal(loc=0.0, scale=1e-10),
        observation_noise=tfd.Normal(loc=0.0, scale=2.0),
        transition_fn=transition_fn,
        observation_fn=observation_fn
    )

    x_true, y_obs = static_model.sample(T)

    # Ensure y_obs has shape [Batch, T, obs_dim]
    if len(y_obs.shape) == 2:
        y_obs = tf.expand_dims(y_obs, axis=0)
    elif len(y_obs.shape) == 1:
        y_obs = tf.reshape(y_obs, [1, -1, 1])

    pf = ParticleFilter(model=static_model, num_particles=100, resample_method='multinomial')
    x_filt, P_filt, all_particles, _ = pf.filter(y_obs)

    particles_np = all_particles.numpy()[0, :, :, 0]  # [T, N]
    estimated_mean = x_filt.numpy()[0, :, 0]  # [T]
    estimated_var = P_filt.numpy()[0, :, 0, 0]  # [T]

    final_estimate = estimated_mean[-1]
    final_error = abs(final_estimate - true_state_value)
    final_variance = estimated_var[-1]

    print(f"True State: {true_state_value}")
    print(f"Final Estimate: {final_estimate:.4f}")
    print(f"Final Error: {final_error:.4f}")
    print(f"Final Variance (Uncertainty): {final_variance:.10f}")

    plt.figure(figsize=(12, 6))
    plt.plot(estimated_mean, color='blue', linewidth=2, label='Estimated Mean')
    plt.plot(x_true.numpy().reshape(-1), color='red', linewidth=2, label='True State')
    plt.scatter(range(T), y_obs.numpy().flatten(), color='grey', marker='o', label='Observations', s=10, alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('State Value/Observations')
    plt.title('Estimated Mean vs True State with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 6))
    for t in range(T):
        plt.scatter(np.full_like(particles_np[t], t), particles_np[t],
                    s=5, alpha=0.1, color='k', marker='.')
    plt.plot(estimated_mean, color='blue', linewidth=2, label='PF Estimate')
    estimated_std = np.sqrt(estimated_var)
    plt.fill_between(range(T),
                     estimated_mean - 2 * estimated_std,  # 2 standard deviations for 95% CI
                     estimated_mean + 2 * estimated_std,
                     color='blue', alpha=0.2, label='Uncertainty (95% CI)')
    plt.axhline(y=x_true.numpy().reshape(-1)[0], color='red', linestyle='--', linewidth=2, label='True State (0.0)')
    plt.title(f'Static Impoverishment: Converged to {final_estimate:.2f} instead of {true_state_value}')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Example usage with 1D Stochastic Volatility Model
    sv_model = get1DStochasticVolModel(alpha=0.9, beta=0.5, sigma=1)
    pf = ParticleFilter(model=sv_model, num_particles=10000, resample_method='multinomial')

    # Simulate data
    T = 200
    x_true, y_obs = sv_model.sample(T)
    if len(y_obs.shape) == 1:
        y_obs = tf.reshape(y_obs, [1,-1, 1])  # Ensure y_obs has shape [B, T, obs_dim]

    #Run particle filter
    t0=tf.timestamp()
    x_filt, P_filt, all_particles, all_weights = pf.filter(y_obs)
    t1=tf.timestamp()
    print(f"Particle filter completed in {t1 - t0:.2f} seconds.")

    # # Plot results
    # plt.figure(figsize=(12, 6))
    # plt.plot(x_true.numpy(), label='True State', color='g')
    # plt.plot(x_filt[0,:,0].numpy(), label='Filtered State', color='b')
    # plt.fill_between(range(T),
    #                  x_filt[0,:,0].numpy() - 2*tf.sqrt(P_filt[0,:,0,0]).numpy(),
    #                  x_filt[0,:,0].numpy() + 2*tf.sqrt(P_filt[0,:,0,0]).numpy(),
    #                  color='b', alpha=0.2, label='95% Confidence Interval')
    # #plt.scatter(range(T), y_obs.numpy(), label='Observations', color='r', s=10)
    # plt.legend()
    # plt.title('Particle Filter on 1D Stochastic Volatility Model')
    # plt.xlabel('Time')
    # plt.ylabel('State / Observation')
    # plt.savefig("ParticleFilter_1DSVM.pdf",bbox_inches='tight')
    # plt.show()

    #run_resampling_experiment(sv_model, num_particles=1000, x_true=x_true, y_obs=y_obs, num_runs=50)
    experiment_weight_degeneracy(sv_model, y_obs, method='none')
    #experiment_static_impoverishment_wrong_convergence(T)