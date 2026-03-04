import tensorflow as tf
import tensorflow_probability as tfp
from models import NLSSM,get1DLogSquaredSVM
from Filters.basic_filters import UnscentedKalmanFilter, ParticleFilter
from Filters.flow_filters import EDHFlow, LEDHFlow
from typing import Callable
import numpy as np

tfd = tfp.distributions

class ParticleUKF(UnscentedKalmanFilter):
    """
    Vectorized UKF for Particle Filters.
    """

    @classmethod
    def from_ukf(cls, ukf_instance: UnscentedKalmanFilter):
        """
        Factory method to create a ParticleUKF from an existing UnscentedKalmanFilter.
        Transfers all parameters and learned noise states.
        Usage:
            p_ukf = ParticleUKF.from_ukf(existing_ukf)
        """
        if ukf_instance.train_noise:
            ukf_instance.sync_model()

        alpha_val = ukf_instance.alpha
        beta_val = ukf_instance.beta
        kappa_val =ukf_instance.kappa

        return cls(
            model=ukf_instance.model,
            alpha=alpha_val,
            beta=beta_val,
            kappa=kappa_val,
            train_noise=ukf_instance.train_noise
        )

    @tf.function
    def predict_step(self, x_curr, P_curr, Wm, Wc, lam, Q, noise_proc_mean):
        """
        x_curr: [Batch, Particles, Dim]
        """
        # Check if we have particle dimension
        if x_curr.shape.ndims == 3:
            shape = tf.shape(x_curr)
            B, N, D = shape[0], shape[1], shape[2]
            #    Layout in memory: [b0p0, b0p1... b1p0, b1p1...]
            x_flat = tf.reshape(x_curr, [-1, D])

            if P_curr.shape.ndims == 3:
                # P is [B, D, D] -> [B*N, D, D]
                P_flat = tf.repeat(P_curr, repeats=N, axis=0)
            else:
                # If P is [B, N, D, D]
                P_flat = tf.reshape(P_curr, [-1, D, D])
            P_flat=0.5*(P_flat + tf.linalg.matrix_transpose(P_flat)) + 1e-4 * tf.eye(D, batch_shape=[B*N])

            Q_arg = Q
            if Q.shape.ndims == 3:
                # Repeat batch elements to match flattened layout:
                # [Q0, Q1] -> [Q0, Q0... Q1, Q1...]
                Q_arg = tf.repeat(Q, repeats=N, axis=0)

            x_pred_flat, P_pred_flat,_ = super().predict_step(
                x_flat, P_flat, Wm, Wc, lam, Q_arg, noise_proc_mean
            )

            return tf.reshape(x_pred_flat, [B, N, D]), tf.reshape(P_pred_flat, [B, N, D, D])

        else:
            # Fallback for standard [Batch, Dim] input
            return super().predict_step(x_curr, P_curr, Wm, Wc, lam, Q, noise_proc_mean)[:-1]

    @tf.function
    def update_step(self, x_pred, P_pred, y, Wm, Wc, lam, R, noise_obs_mean):
        """
        x_pred: [Batch, Particles, Dim]
        y:      [Batch, Obs] (Shared across particles in the same batch)
        """
        if x_pred.shape.ndims == 3:
            shape = tf.shape(x_pred)
            B, N, D = shape[0], shape[1], shape[2]

            x_flat = tf.reshape(x_pred, [-1, D])
            P_flat = tf.reshape(P_pred, [-1, D, D])

            # y is [Batch, Obs]. We need to repeat it for every particle in the batch.
            # [y0, y1] -> [y0, y0... (N times), y1, y1...]
            y_flat = tf.repeat(y, repeats=N, axis=0)

            R_arg = R
            if R.shape.ndims == 3:
                # Batched R [B, Obs, Obs] -> Repeat to [B*N, Obs, Obs]
                R_arg = tf.repeat(R, repeats=N, axis=0)

            x_new_flat, P_new_flat, _ = super().update_step(
                x_flat, P_flat, y_flat, Wm, Wc, lam, R_arg, noise_obs_mean
            )

            x_new = tf.reshape(x_new_flat, [B, N, D])
            P_new = tf.reshape(P_new_flat, [B, N, D, D])

            return x_new, P_new

        else:
            return super().update_step(x_pred, P_pred, y, Wm, Wc, lam, R, noise_obs_mean)[:-1]



class InvertiblePFPF(ParticleFilter):
    def __init__(self, model: NLSSM, num_particles: int,
                 ukf: UnscentedKalmanFilter,
                 flow_class: Callable = EDHFlow,
                 num_flow_steps: int = 10,
                 resample_method: str = 'systematic',
                 resample_threshold: float = 1.0):
        super().__init__(model, num_particles, resample_method=resample_method,resample_threshold=resample_threshold)

        self.flow_algo = flow_class(model, num_particles)
        self.num_flow_steps = num_flow_steps
        self.ukf = ParticleUKF.from_ukf(ukf)  # Store the UKF

    @tf.function
    def _flow_with_det(self, eta_0, Y_t, num_flow_steps, P_guide=None, eta_aux=None,step_sizes:tf.Tensor=None) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Runs the flow with External Guidance P.
        eta_aux is x_filt transformed by transition function without noise,
        which is used as auxiliary input to compute flow parameters to ensure the invertibility of the flow.
        """
        const_delta_lambda = 1.0 / float(num_flow_steps)
        current_particles = eta_0
        eta_0_aux = tf.identity(eta_aux)
        eta_0_aux=tf.expand_dims(eta_0_aux, -1) # [B, D,1] for EDH, [B, N, D, 1] for LEDH
        eta_aux=tf.expand_dims(eta_aux, -1) # [B, D,1] for EDH, [B, N, D, 1] for LEDH

        B = tf.shape(eta_0)[0]
        N = self.num_particles
        D = self.model.state_dim

        # Accumulator for log determinant [Batch, Num_Particles]
        log_det_jacobian = tf.zeros((B, N))

        lam = 0.0
        for k in tf.range(num_flow_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (log_det_jacobian, tf.TensorShape([None, N])),
                    (current_particles, tf.TensorShape([None, N, D])),
                    (eta_aux,
                     tf.TensorShape([None, D, 1]) if self.flow_algo.__class__.__name__ == "EDHFlow" else tf.TensorShape(
                         [None, N, D, 1]))
                ]
            )

            if step_sizes is None:
                dt = const_delta_lambda
            else:
                dt = step_sizes[k]
            lam+= dt

            # A shape: [B, D, D] (Shared/EDH) or [B, N, D, D] (Per-Particle/LEDH)
            # b shape: [B, D] (Shared/EDH) or [B, N, D] (Per-Particle/LEDH)
            A, b = self.flow_algo.compute_flow_parameters(
                current_particles, Y_t, lam, P_xx=P_guide, linearization_points=eta_aux, eta_0_mean=eta_0_aux
            )


            if len(A.shape) == 3 and self.flow_algo.__class__.__name__ == "EDHFlow":
                # A: [B, D, D], b: [B, D]
                # Broadcasting: [B,N,D] @ [B,D,D] -> [B,N,D]
                drift = tf.matmul(current_particles, A, transpose_b=True) + tf.expand_dims(b, 1)
                current_particles = current_particles + dt * drift

                drift_aux = tf.matmul(A, eta_aux) + tf.expand_dims(b, -1)  # [B, D, 1]
                eta_aux = eta_aux + dt * drift_aux

            elif len(A.shape) == 4 and self.flow_algo.__class__.__name__ == "LEDHFlow":
                # A: [B, N, D, D], b: [B, N, D]
                # Expand x to [B, N, D, 1] to match A's particle dimension
                x_expanded = tf.expand_dims(current_particles, -1)

                # Ax: [B, N, D, D] @ [B, N, D, 1] -> [B, N, D, 1]
                b_expanded = tf.expand_dims(b, -1)

                drift = tf.matmul(A, x_expanded) + b_expanded
                drift = tf.squeeze(drift, -1)  # [B, N, D]
                current_particles = current_particles + dt * drift

                # det(J_i) varies per particle, so it affects relative weights.
                identity = tf.eye(D, batch_shape=[B, N])
                _, step_log_det = tf.linalg.slogdet(identity + dt * A)  # [B, N]
                log_det_jacobian += step_log_det

                drift_aux = tf.matmul(A, eta_aux) + tf.expand_dims(b, -1)  # [B, D, 1] or [B, N, D, 1]
                eta_aux = eta_aux + dt * drift_aux
            else:
                raise ValueError("A must be either [B, D, D] or [B, N, D, D]")

        return current_particles, log_det_jacobian

    @tf.function
    def _update_pfpf(self, x_prev, eta_0, eta_1, weights, Y_t, log_det_jacobian, is_initial_step):
        '''
        Updates particle weights according to Eq. (18) in the paper:
        w_new = w_old * [ p(y|eta1) * p(eta1|x_prev) * |det(J)| ] / p(eta0|x_prev)

        Args:
            x_prev: Particles at t-1 (or None if t=0)
            eta_0:  Predicted particles before flow (prior samples)
            eta_1:  Particles after flow (proposal samples)
            weights: Previous weights
            Y_t:    Current observation
            log_det_jacobian: Log determinant of the flow Jacobian
            is_initial_step: Boolean tensor/flag
        '''
        N = self.num_particles

        # Likelihood p(y_t | eta_1)
        # Expand Y_t to [B, N, Obs]
        Y_t_expanded = tf.tile(tf.expand_dims(Y_t, 1), [1, N, 1])

        log_likelihood = self._compute_log_prob(
            target=Y_t_expanded,
            source=eta_1,
            map_fn=self.model.observation_fn,
            noise_dist=self.model.observation_noise,
            dist_fn=getattr(self.model, 'get_observation_dist', None),
            flatten=True
        )
        if len(log_likelihood.shape) > 2:
            log_likelihood = tf.reduce_sum(log_likelihood, axis=-1)

        # p(x) is based on init_noise (centered at x0)
        def _compute_init_log_prob(x):
            # Model defines x0 as mean, so noise = x - x0
            noise = x - self.model.x0
            lp = self.model.init_noise.log_prob(noise)
            if len(lp.shape) == 3:
                lp = tf.reduce_sum(lp, axis=-1)
            return lp

        # p(x_t | x_{t-1}) uses transition function
        def _compute_trans_log_prob(target, source):
            return self._compute_log_prob(
                target=target,
                source=source,
                map_fn=self.model.transition_fn,
                noise_dist=self.model.process_noise,
                dist_fn=getattr(self.model, 'get_transition_dist', None),
                flatten=True
            )

        log_p_eta1_prior, log_p_eta0_prior = tf.cond(
            is_initial_step,
            lambda: (_compute_init_log_prob(eta_1), _compute_init_log_prob(eta_0)),
            lambda: (_compute_trans_log_prob(eta_1, x_prev), _compute_trans_log_prob(eta_0, x_prev))
        )

        # log(w_new) = log(w_old) + log p(y|eta1) + log p(eta1) - log p(eta0) + log |J|
        log_weights_prev = tf.math.log(weights + 1e-10)
        log_update = (log_likelihood + log_p_eta1_prior - log_p_eta0_prior + log_det_jacobian)
        log_weights_new = log_weights_prev + log_update
        log_norm_const = tf.math.reduce_logsumexp(log_weights_new, axis=1, keepdims=True)
        new_weights = tf.math.exp(log_weights_new - log_norm_const)

        return new_weights

    @tf.function
    def filter(self, observations: tf.Tensor, num_flow_steps: int = 10,step_sizes:tf.Tensor=None) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size, T = observations.shape[0], observations.shape[1]

        x_ukf, P_ukf, Wm, Wc, lam_ukf, Q, R, q_mean, r_mean = self.ukf.initialize_filter(batch_size)
        x_filt = x_ukf
        if self.flow_algo.__class__.__name__ == "LEDHFlow":
            P_ukf = tf.tile(tf.expand_dims(P_ukf, 1), [1, self.num_particles, 1, 1]) ## [B, N, D, D]
            x_ukf = tf.tile(tf.expand_dims(x_ukf, 1), [1, self.num_particles, 1])  # [B, N, D]

        all_particles = tf.TensorArray(dtype=tf.float32, size=T)
        all_weights = tf.TensorArray(dtype=tf.float32, size=T)
        x_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)

        particles, weights = self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])  # [T,B,obs_dim]

        #@tf.function
        def _transition_branch(particles, x, P_ukf):
            p_pred = self._transition(particles)
            x_u, P_u= self.ukf.predict_step(x, P_ukf, Wm, Wc, lam_ukf, Q, q_mean)
            return p_pred, x_u, P_u

        for t in tf.range(T):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (particles, tf.TensorShape([None, self.num_particles, self.model.state_dim])),
                    (weights, tf.TensorShape([None, self.num_particles])),
                    (x_filt, tf.TensorShape([None, self.model.state_dim])),
                    (x_ukf, tf.TensorShape([None,
                                            self.model.state_dim]) if self.flow_algo.__class__.__name__ == "EDHFlow" else tf.TensorShape(
                        [None, self.num_particles, self.model.state_dim])),
                    (P_ukf, tf.TensorShape([None, self.model.state_dim,
                                            self.model.state_dim]) if self.flow_algo.__class__.__name__ == "EDHFlow" else tf.TensorShape(
                        [None, self.num_particles, self.model.state_dim, self.model.state_dim]))
                ]
            )

            Y_t = Y_time_major[t]
            #pre_imag=tf.cond(self.flow_algo.__class__.__name__ == "EDHFlow",lambda: x_filt,lambda:particles)
            if self.flow_algo.__class__.__name__ == "EDHFlow":
                pre_imag = x_filt
            else:
                pre_imag = particles
            if t > 0:
                eta_0, x_pred_ukf, P_pred_ukf = _transition_branch(particles, pre_imag, P_ukf)
                eta_aux=self.model.transition_fn(pre_imag,tf.zeros_like(pre_imag))
            else:
                eta_0, x_pred_ukf, P_pred_ukf = particles, x_ukf, P_ukf
                eta_aux=pre_imag

            eta_1, log_det = self._flow_with_det(
                eta_0, Y_t, num_flow_steps, P_guide=P_pred_ukf, eta_aux=eta_aux,step_sizes=step_sizes
            )
            is_initial = tf.equal(t, 0)
            weights = self._update_pfpf(particles, eta_0, eta_1, weights, Y_t, log_det, is_initial)

            x_ukf, P_ukf= self.ukf.update_step(
                x_pred_ukf, P_pred_ukf, Y_t, Wm, Wc, lam_ukf, R, r_mean
            )

            particles = eta_1
            x_filt = tf.reduce_sum(tf.expand_dims(weights, axis=-1) * particles, axis=1)  # [B, D]
            diff = particles - tf.expand_dims(x_filt, axis=1)
            weights_expanded = tf.reshape(weights, [batch_size, self.num_particles, 1, 1])
            P_filt = tf.reduce_sum(weights_expanded * tf.expand_dims(diff, axis=-1) * tf.expand_dims(diff, axis=-2),
                                   axis=1)

            x_filt_ta = x_filt_ta.write(t, x_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)
            all_particles = all_particles.write(t, particles)
            all_weights = all_weights.write(t, weights)

            if self.flow_algo.__class__.__name__ == "EDHFlow":
                particles, weights = self._resample(particles, weights)
            elif self.flow_algo.__class__.__name__ == "LEDHFlow":
                particles, weights, P_ukf = self._resample_all(particles, weights, P_ukf,batch_size)

        return (
            tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),
            tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),
            tf.transpose(all_particles.stack(), perm=[1, 0, 2, 3]),
            tf.transpose(all_weights.stack(), perm=[1, 0, 2])
        )

    def _resample_all(self, particles, w, P_u, batch_size):
        effective_batch_size = 1.0 / (tf.reduce_sum(tf.square(w), axis=1) + 1e-10)
        resample_cond = effective_batch_size < (self.resample_threshold * tf.cast(self.num_particles, tf.float32))

        indices = self._get_resample_indices(w, batch_size, self.num_particles)
        p_new = tf.gather(particles, indices, batch_dims=1)
        P_u_new = tf.gather(P_u, indices, batch_dims=1)
        w_new = tf.fill([batch_size, self.num_particles], 1.0 / float(self.num_particles))

        cond_expanded_p = tf.reshape(resample_cond, [batch_size, 1, 1])
        particles = tf.where(cond_expanded_p, p_new, particles)

        # Handle P_u based on EDH [B, D, D] vs LEDH [B, N, D, D]
        cond_expanded_P = tf.reshape(resample_cond, [batch_size, 1, 1, 1]) if len(P_u.shape) == 4 else cond_expanded_p
        P_u = tf.where(cond_expanded_P, P_u_new, P_u)

        cond_expanded_w = tf.reshape(resample_cond, [batch_size, 1])
        w = tf.where(cond_expanded_w, w_new, w)

        return particles, w, P_u


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sv_model = get1DLogSquaredSVM(alpha=0.9, beta=0.5, sigma=1)

    ukf = UnscentedKalmanFilter(model=sv_model, alpha=1e-3, beta=2.0, kappa=0.0)
    pf = EDHFlow(model=sv_model, num_particles=1000,ukf=ukf)
    lpf = LEDHFlow(model=sv_model, num_particles=1000,ukf=ukf)
    pfpf = InvertiblePFPF(model=sv_model, num_particles=1000, ukf=ukf, flow_class=EDHFlow)
    lpfpf= InvertiblePFPF(model=sv_model, num_particles=1000, ukf=ukf, flow_class=LEDHFlow)


    # Simulate data
    T = 200
    x_true, y_obs = sv_model.sample(T)
    if len(y_obs.shape) == 1:
        y_obs = tf.reshape(y_obs, [1, -1, 1])  # Ensure y_obs has shape [B, T, obs_dim]

    pf.filter_with_ukf(y_obs, num_flow_steps=5)
    t0 = tf.timestamp()
    x_filt, P_filt, all_particles = pf.filter_with_ukf(y_obs, num_flow_steps=5)
    t1 = tf.timestamp()
    print(f"EDH Flow completed in {t1 - t0:.4f} seconds.")

    lpf.filter_with_ukf(y_obs, num_flow_steps=5)
    t2 = tf.timestamp()
    x_filt_ledh, P_filt_ledh, all_particles_ledh = lpf.filter_with_ukf(y_obs, num_flow_steps=5)
    t3 = tf.timestamp()
    print(f"LEDH Flow completed in {t3 - t2:.4f} seconds.")

    pfpf.filter(y_obs, num_flow_steps=5)
    t4 = tf.timestamp()
    x_filt_pfpf, P_filt_pfpf, all_particles_pfpf, all_weights_pfpf = pfpf.filter(y_obs, num_flow_steps=5)
    t5 = tf.timestamp()
    print(f"Particle Flow Particle Filter completed in {t5 - t4:.4f} seconds.")
    print(x_filt_pfpf.shape)

    lpfpf.filter(y_obs, num_flow_steps=5)
    t6= tf.timestamp()
    x_filt_lpfpf, P_filt_lpfpf, all_particles_lpfpf, all_weights_lpfpf = lpfpf.filter(y_obs, num_flow_steps=5)
    t7 = tf.timestamp()
    print(f"LEDH Particle Flow Particle Filter completed in {t7 - t6:.4f} seconds.")

    print(f'shape of x_filt: {x_filt.shape}, shape of P_filt: {P_filt.shape}')
    print(f'shape of x_filt_ledh: {x_filt_ledh.shape}, shape of P_filt_ledh: {P_filt_ledh.shape}')
    print(f'shape of x_filt_pfpf: {x_filt_pfpf.shape}, shape of P_filt_pfpf: {P_filt_pfpf.shape}')
    print(f'shape of x_filt_lpfpf: {x_filt_lpfpf.shape}, shape of P_filt_lpfpf: {P_filt_lpfpf.shape}')
    # Plot results
    plt.figure(figsize=(12, 6))

    # Safely squeeze all tensors to 1D (T,) for Matplotlib
    x_true_flat = tf.squeeze(x_true).numpy()
    x_filt_flat = tf.squeeze(x_filt).numpy()
    x_filt_ledh_flat = tf.squeeze(x_filt_ledh).numpy()
    x_filt_pfpf_flat = tf.squeeze(x_filt_pfpf).numpy()

    # For the covariance/confidence interval, safely squeeze out the extra dims
    p_filt_pfpf_flat = tf.squeeze(P_filt_pfpf).numpy()

    plt.plot(x_true_flat, label='True State', color='g')
    plt.plot(x_filt_flat, label='EDH Filtered State', color='b')
    plt.plot(x_filt_ledh_flat, label='LEDH Filtered State', color='y', linestyle='--')
    plt.plot(x_filt_pfpf_flat, label='PF-PF (EDH) Filtered State', color='m', linestyle='--')

    plt.fill_between(range(T),
                     x_filt_pfpf_flat - np.sqrt(p_filt_pfpf_flat),
                     x_filt_pfpf_flat + np.sqrt(p_filt_pfpf_flat),
                     color='b', alpha=0.2, label='95% Confidence Interval')

    plt.scatter(range(T), tf.squeeze(y_obs).numpy(), label='Observations', color='r', s=10)
    plt.legend()
    plt.title('Particle Filter on 1D Stochastic Volatility Model')
    plt.xlabel('Time')
    plt.ylabel('State / Observation')
    plt.savefig("EDH_LEDH_PFPF_filter.pdf", bbox_inches='tight')
    plt.show()