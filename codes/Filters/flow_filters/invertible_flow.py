import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable
import numpy as np
from models.base_models import NLSSM
from models.models import get1DLogSquaredSVM
from Filters.basic_filters import UnscentedKalmanFilter
from Filters.basic_filters import ParticleFilter
from Filters.flow_filters import EDHFlow, LEDHFlow

tfd = tfp.distributions
dtype = tf.float32


class ParticleUKF(UnscentedKalmanFilter):
    """
    Vectorized UKF tailored for Particle Filters.

    Overrides the standard UKF predict and update methods to intercept 3D
    particle tensors [Batch, Particles, Dim], temporarily flatten them to 2D
    for the standard UKF execution, and reshape them back automatically.
    """

    @classmethod
    def from_ukf(cls, ukf_instance: UnscentedKalmanFilter):
        """Factory method to clone an existing UKF into a ParticleUKF."""
        if ukf_instance.train_noise:
            ukf_instance.sync_model()

        return cls(
            model=ukf_instance.model,
            alpha=float(ukf_instance.alpha.numpy()),
            beta=float(ukf_instance.beta.numpy()),
            kappa=float(ukf_instance.kappa.numpy()),
            train_noise=ukf_instance.train_noise
        )

    def predict(self, t: int, state: tuple) -> tuple:
        """Intercepts 3D states, flattens, runs standard predict, and reshapes."""
        x_filt, P_filt, x_pred, P_pred, Wm, Wc, lam = state

        if x_filt.shape.ndims == 3:
            B, N, D = tf.shape(x_filt)[0], tf.shape(x_filt)[1], tf.shape(x_filt)[2]
            x_flat = tf.reshape(x_filt, [B * N, D])

            if P_filt.shape.ndims == 3:
                P_flat = tf.repeat(P_filt, repeats=N, axis=0)
            else:
                P_flat = tf.reshape(P_filt, [B * N, D, D])

            P_flat = 0.5 * (P_flat + tf.linalg.matrix_transpose(P_flat)) + 1e-4 * tf.eye(D, batch_shape=[B * N],
                                                                                         dtype=dtype)

            flat_state = (x_flat, P_flat, x_flat, P_flat, Wm, Wc, lam)
            flat_pred_state = super().predict(t, flat_state)

            x_pred_flat, P_pred_flat, _, _, _, _, _ = flat_pred_state

            x_pred_out = tf.reshape(x_pred_flat, [B, N, D])
            P_pred_out = tf.reshape(P_pred_flat, [B, N, D, D])

            return (x_pred_out, P_pred_out, x_pred_out, P_pred_out, Wm, Wc, lam)
        else:
            return super().predict(t, state)

    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        """Intercepts 3D states, repeats the observation, and routes through standard UKF update."""
        x_filt_dummy, P_filt_dummy, x_pred, P_pred, Wm, Wc, lam = state

        if x_pred.shape.ndims == 3:
            B, N, D = tf.shape(x_pred)[0], tf.shape(x_pred)[1], tf.shape(x_pred)[2]

            x_pred_flat = tf.reshape(x_pred, [B * N, D])
            if P_pred.shape.ndims == 3:
                P_pred_flat = tf.repeat(P_pred, repeats=N, axis=0)
            else:
                P_pred_flat = tf.reshape(P_pred, [B * N, D, D])

            obs_flat = tf.repeat(observation, repeats=N, axis=0)

            flat_state = (x_pred_flat, P_pred_flat, x_pred_flat, P_pred_flat, Wm, Wc, lam)
            flat_new_state, flat_metrics = super().update(t, flat_state, obs_flat)

            x_filt_flat, P_filt_flat, _, _, _, _, _ = flat_new_state

            x_filt_out = tf.reshape(x_filt_flat, [B, N, D])
            P_filt_out = tf.reshape(P_filt_flat, [B, N, D, D])

            log_l_flat = flat_metrics[0]
            log_l = tf.reduce_mean(tf.reshape(log_l_flat, [B, N]), axis=1)

            return (x_filt_out, P_filt_out, x_pred, P_pred, Wm, Wc, lam), (log_l,)
        else:
            return super().update(t, state, observation)


class InvertiblePFPF(ParticleFilter):
    """
    Invertible Particle Flow Particle Filter (PFPF).

    Inherits from the standard ParticleFilter, replacing the standard importance
    sampling proposal with a deterministic ODE flow (EDH or LEDH) and adjusting
    the particle weights using the invertible Jacobian determinant of the flow.
    """

    def __init__(self, model: NLSSM, num_particles: int, ukf: UnscentedKalmanFilter,
                 flow_class: Callable = EDHFlow, num_flow_steps: int = 10,
                 resample_method: str = 'systematic', resample_threshold: float = 1.0, step_sizes=None):
        super().__init__(model, num_particles, resample_method=resample_method, resample_threshold=resample_threshold)

        self.flow_algo = flow_class(model, num_particles)
        self.num_flow_steps = num_flow_steps
        self.step_sizes = step_sizes
        self.ukf = ParticleUKF.from_ukf(ukf)

    def _transition(self, particles: tf.Tensor) -> tf.Tensor:
        """Helper to safely flatten and propagate particles through the transition model."""
        B = tf.shape(particles)[0]
        N = tf.shape(particles)[1]
        D = self.model.state_dim

        particles_reshaped = tf.reshape(particles, [B * N, D])

        process_noise = self.model.process_noise.sample(B * N)
        if len(process_noise.shape) == 1:
            process_noise = tf.expand_dims(process_noise, -1)
        process_noise = tf.reshape(process_noise, [B * N, D])

        predicted_particles = self.model.transition_fn(particles_reshaped, process_noise)
        return tf.reshape(predicted_particles, [B, N, D])

    def _init_state(self, batch_size: int) -> tuple:
        """Combines the Particle and UKF states into a single 11-element Opaque State."""
        particles, weights = super()._init_state(batch_size)
        x_ukf, P_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam = self.ukf._init_state(batch_size)

        if self.flow_algo.__class__.__name__ == "LEDHFlow":
            x_ukf = tf.tile(tf.expand_dims(x_ukf, 1), [1, self.num_particles, 1])
            P_ukf = tf.tile(tf.expand_dims(P_ukf, 1), [1, self.num_particles, 1, 1])
            x_pred_ukf, P_pred_ukf = x_ukf, P_ukf
            eta_aux = particles
        else:
            eta_aux = x_ukf

        x_prev = particles
        return (particles, weights, x_prev, eta_aux, x_ukf, P_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam)

    def predict(self, t: int, state: tuple) -> tuple:
        """Advances both the Monte Carlo particles and the analytical UKF guidance."""
        particles, weights, x_prev, eta_aux, x_ukf, P_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam = state

        particles_pred = self._transition(particles)

        ukf_state_prev = (x_ukf, P_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam)
        ukf_state_pred = self.ukf.predict(t, ukf_state_prev)
        _, _, x_pred_ukf_new, P_pred_ukf_new, _, _, _ = ukf_state_pred

        pre_imag = x_ukf if self.flow_algo.__class__.__name__ == "EDHFlow" else particles

        B = tf.shape(pre_imag)[0]
        if pre_imag.shape.ndims == 3:
            N, D = tf.shape(pre_imag)[1], tf.shape(pre_imag)[2]
            pre_imag_flat = tf.reshape(pre_imag, [B * N, D])
            eta_aux_pred_flat = self.model.transition_fn(pre_imag_flat, tf.zeros_like(pre_imag_flat))
            eta_aux_pred = tf.reshape(eta_aux_pred_flat, [B, N, D])
        else:
            eta_aux_pred = self.model.transition_fn(pre_imag, tf.zeros_like(pre_imag))

        new_state = (particles_pred, weights, particles, eta_aux_pred, x_ukf, P_ukf, x_pred_ukf_new, P_pred_ukf_new,
                         Wm, Wc, lam)
            # COMPILER FIX: Restore compiler metadata mapping so main loop doesn't panic
        for n, o in zip(new_state, state):
            if isinstance(n, tf.Tensor) and isinstance(o, tf.Tensor): n.set_shape(o.shape)
        return new_state

    def _flow_with_det(self, eta_0, Y_t, P_guide, eta_aux) -> tuple[tf.Tensor, tf.Tensor]:
        """Runs the deterministic ODE flow while tracking the log determinant of the Jacobian."""
        const_delta_lambda = 1.0 / float(self.num_flow_steps)
        current_particles = eta_0
        eta_0_aux = tf.expand_dims(tf.identity(eta_aux), -1)
        eta_aux_col = tf.expand_dims(eta_aux, -1)

        B, N, D = tf.shape(eta_0)[0], self.num_particles, self.model.state_dim
        log_det_jacobian = tf.zeros((B, N), dtype=dtype)

        lam = 0.0
        for k in tf.range(self.num_flow_steps):
            # SAFEGURAD: Tell AutoGraph to expect dynamically widened batch sizes (from 1 to None)
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (log_det_jacobian, tf.TensorShape([None, self.num_particles])),
                    (current_particles, tf.TensorShape([None, self.num_particles, self.model.state_dim])),
                    (eta_aux_col, tf.TensorShape([None, self.model.state_dim,
                                                  1]) if self.flow_algo.__class__.__name__ == "EDHFlow" else tf.TensorShape(
                        [None, self.num_particles, self.model.state_dim, 1]))
                ]
            )

            dt = const_delta_lambda if self.step_sizes is None else self.step_sizes[k]
            lam += dt

            A, b = self.flow_algo.compute_flow_parameters(
                current_particles, Y_t, lam, P_xx=P_guide, linearization_points=eta_aux_col, eta_0_mean=eta_0_aux
            )

            if self.flow_algo.__class__.__name__ == "EDHFlow":
                drift = tf.matmul(current_particles, A, transpose_b=True) + tf.expand_dims(b, 1)
                current_particles = current_particles + dt * drift

                drift_aux = tf.matmul(A, eta_aux_col) + tf.expand_dims(b, -1)
                eta_aux_col = eta_aux_col + dt * drift_aux

                # COMPILER FIX: Satisfy AutoGraph's requirement that all shape_invariants are modified
                log_det_jacobian = log_det_jacobian + 0.0

            else:  # LEDHFlow
                x_expanded = tf.expand_dims(current_particles, -1)
                b_expanded = tf.expand_dims(b, -1)

                drift = tf.matmul(A, x_expanded) + b_expanded
                current_particles = current_particles + dt * tf.squeeze(drift, -1)

                identity = tf.eye(D, batch_shape=[B, N], dtype=dtype)
                _, step_log_det = tf.linalg.slogdet(identity + dt * A)
                log_det_jacobian += step_log_det

                drift_aux = tf.matmul(A, eta_aux_col) + b_expanded
                eta_aux_col = eta_aux_col + dt * drift_aux

        return current_particles, log_det_jacobian

    def _update_pfpf_weights(self, x_prev, eta_0, eta_1, weights, Y_t, log_det_jacobian, is_initial_step):
        """Eq. (18): w_new = w_old * [ p(y|eta1) * p(eta1|x_prev) * |det(J)| ] / p(eta0|x_prev)"""

        def _safe_log_prob(target, source, map_fn, noise_dist, dist_fn):
            B, N = tf.shape(source)[0], tf.shape(source)[1]
            D_in, D_out = tf.shape(source)[-1], tf.shape(target)[-1]

            if tf.shape(target)[1] == 1:
                target = tf.tile(target, [1, N, 1])

            source_flat = tf.reshape(source, [B * N, D_in])
            target_flat = tf.reshape(target, [B * N, D_out])

            if dist_fn is not None:
                dist = dist_fn(source_flat)
                lp = dist.log_prob(target_flat)
            else:
                zero_n = tf.zeros_like(target_flat)
                pred = map_fn(source_flat, zero_n)
                lp = noise_dist.log_prob(target_flat - pred)

            return tf.reshape(lp, [B, N])

        Y_t_expanded = tf.expand_dims(Y_t, 1)

        log_likelihood = _safe_log_prob(
            target=Y_t_expanded, source=eta_1,
            map_fn=self.model.observation_fn, noise_dist=self.model.observation_noise,
            dist_fn=getattr(self.model, 'get_observation_dist', None)
        )
        if len(log_likelihood.shape) > 2: log_likelihood = tf.reduce_sum(log_likelihood, axis=-1)

        def _compute_prior(source, target, map_fn, noise_dist, dist_fn):
            lp = _safe_log_prob(target, source, map_fn, noise_dist, dist_fn)
            return tf.reduce_sum(lp, axis=-1) if len(lp.shape) > 2 else lp

        x0_tiled = tf.tile(tf.reshape(self.model.x0, [1, 1, -1]), [tf.shape(eta_1)[0], self.num_particles, 1])

        safe_x_prev = x_prev if x_prev is not None else eta_0

        log_p_eta1_prior, log_p_eta0_prior = tf.cond(
            is_initial_step,
            lambda: (
                _compute_prior(x0_tiled, eta_1, lambda x, n: x + n, self.model.init_noise, None),
                _compute_prior(x0_tiled, eta_0, lambda x, n: x + n, self.model.init_noise, None)
            ),
            lambda: (
                _compute_prior(safe_x_prev, eta_1, self.model.transition_fn, self.model.process_noise,
                               getattr(self.model, 'get_transition_dist', None)),
                _compute_prior(safe_x_prev, eta_0, self.model.transition_fn, self.model.process_noise,
                               getattr(self.model, 'get_transition_dist', None))
            )
        )

        log_weights_prev = tf.math.log(weights + 1e-10)
        log_update = (log_likelihood + log_p_eta1_prior - log_p_eta0_prior + log_det_jacobian)
        log_weights_new = log_weights_prev + log_update

        log_norm_const = tf.math.reduce_logsumexp(log_weights_new, axis=1, keepdims=True)
        new_weights = tf.math.exp(log_weights_new - log_norm_const)
        log_lik_inc = tf.squeeze(log_norm_const, axis=1)

        return new_weights, log_lik_inc

    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        """Executes the invertible flow and normalizes the resulting particle weights."""
        eta_0, weights, x_prev, eta_aux, x_ukf, P_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam = state

        eta_1, log_det = self._flow_with_det(eta_0, observation, P_guide=P_pred_ukf, eta_aux=eta_aux)

        weights_new, log_l_inc = self._update_pfpf_weights(
            x_prev, eta_0, eta_1, weights, observation, log_det, tf.equal(t, 0)
        )

        ukf_state_pred = (x_ukf, P_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam)
        ukf_state_filt, _ = self.ukf.update(t, ukf_state_pred, observation)
        x_ukf_filt, P_ukf_filt, _, _, _, _, _ = ukf_state_filt

        sum_sq_weights = tf.reduce_sum(tf.square(weights_new), axis=1)
        ess = 1.0 / (sum_sq_weights + 1e-10)

        w_expanded = tf.expand_dims(weights_new, axis=-1)
        x_filt = tf.reduce_sum(w_expanded * eta_1, axis=1)
        diff = eta_1 - tf.expand_dims(x_filt, axis=1)
        weighted_diff = diff * w_expanded
        P_filt = tf.matmul(weighted_diff, diff, transpose_a=True)

        res_particles, res_weights, x_ukf_res, P_ukf_res = self._resample_all(
            eta_1, weights_new, x_ukf_filt, P_ukf_filt, ess
        )

        new_state = (res_particles, res_weights, x_prev, eta_aux, x_ukf_res, P_ukf_res, x_pred_ukf, P_pred_ukf, Wm, Wc,
                     lam)
        metrics = (log_l_inc, x_filt, P_filt, ess)
        # COMPILER FIX: Force the dynamic tensor back to the static shape of the incoming state
        for n, o in zip(new_state, state):
            if isinstance(n, tf.Tensor) and isinstance(o, tf.Tensor):
                n.set_shape(o.shape)

        return new_state, metrics

    def _resample_all(self, particles, w, x_u, P_u, ess):
        batch_size = tf.shape(particles)[0]
        resample_cond = ess < (self.resample_threshold * tf.cast(self.num_particles, dtype))

        indices = self._get_resample_indices(w, batch_size, self.num_particles)

        p_new = tf.gather(particles, indices, batch_dims=1)
        w_new = tf.fill([batch_size, self.num_particles], 1.0 / float(self.num_particles))

        cond_p = tf.reshape(resample_cond, [batch_size, 1, 1])
        particles = tf.where(cond_p, p_new, particles)

        cond_w = tf.reshape(resample_cond, [batch_size, 1])
        w = tf.where(cond_w, w_new, w)

        if self.flow_algo.__class__.__name__ == "LEDHFlow":
            x_u_new = tf.gather(x_u, indices, batch_dims=1)
            x_u = tf.where(cond_p, x_u_new, x_u)

            P_u_new = tf.gather(P_u, indices, batch_dims=1)
            cond_P = tf.reshape(resample_cond, [batch_size, 1, 1, 1])
            P_u = tf.where(cond_P, P_u_new, P_u)

        return particles, w, x_u, P_u

    def _write_trajectory(self, t: int, trajectory: tuple, state: tuple, metrics: tuple) -> tuple:
        particles_ta, weights_ta, x_filt_ta, P_filt_ta, ess_ta, logl_ta = trajectory
        particles, weights = state[0], state[1]
        log_l, x_filt, P_filt, ess = metrics

        return (
            particles_ta.write(t, particles), weights_ta.write(t, weights),
            x_filt_ta.write(t, x_filt), P_filt_ta.write(t, P_filt),
            ess_ta.write(t, ess), logl_ta.write(t, log_l)
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sv_model = get1DLogSquaredSVM(alpha=0.9, beta=0.5, sigma=1)

    flow_steps = 10

    ukf = UnscentedKalmanFilter(model=sv_model, alpha=1e-3, beta=2.0, kappa=0.0)

    pf = EDHFlow(model=sv_model, num_particles=500, ukf=ukf, num_flow_steps=flow_steps)
    lpf = LEDHFlow(model=sv_model, num_particles=500, ukf=ukf, num_flow_steps=flow_steps)
    pfpf = InvertiblePFPF(model=sv_model, num_particles=500, ukf=ukf, flow_class=EDHFlow, num_flow_steps=flow_steps)
    lpfpf = InvertiblePFPF(model=sv_model, num_particles=500, ukf=ukf, flow_class=LEDHFlow, num_flow_steps=flow_steps)

    T = 200
    x_true, y_obs = sv_model.sample(batch_size=1, T=T)
    if len(y_obs.shape) == 1:
        y_obs = tf.reshape(y_obs, [1, -1, 1])

    # --- EDH Flow ---
    print("Compiling EDHFlow graph...")
    _ = pf.filter(y_obs)
    t0 = tf.timestamp()
    res_edh = pf.filter(y_obs)
    t1 = tf.timestamp()
    print(f"EDH Flow completed in {t1 - t0:.4f} seconds.\n")
    x_filt, P_filt = res_edh['x_filt'], res_edh['P_filt']

    # --- LEDH Flow ---
    print("Compiling LEDHFlow graph...")
    _ = lpf.filter(y_obs)
    t2 = tf.timestamp()
    res_ledh = lpf.filter(y_obs)
    t3 = tf.timestamp()
    print(f"LEDH Flow completed in {t3 - t2:.4f} seconds.\n")
    x_filt_ledh, P_filt_ledh = res_ledh['x_filt'], res_ledh['P_filt']

    # --- Invertible PFPF (EDH) ---
    print("Compiling PFPF (EDH) graph...")
    _ = pfpf.filter(y_obs)
    t4 = tf.timestamp()
    res_pfpf = pfpf.filter(y_obs)
    t5 = tf.timestamp()
    print(f"Particle Flow Particle Filter completed in {t5 - t4:.4f} seconds.\n")
    x_filt_pfpf, P_filt_pfpf = res_pfpf['x_filt'], res_pfpf['P_filt']

    # --- Invertible PFPF (LEDH) ---
    print("Compiling PFPF (LEDH) graph...")
    _ = lpfpf.filter(y_obs)
    t6 = tf.timestamp()
    res_lpfpf = lpfpf.filter(y_obs)
    t7 = tf.timestamp()
    print(f"LEDH Particle Flow Particle Filter completed in {t7 - t6:.4f} seconds.\n")
    x_filt_lpfpf, P_filt_lpfpf = res_lpfpf['x_filt'], res_lpfpf['P_filt']

    # Print diagnostics
    print(f'shape of x_filt: {x_filt.shape}, shape of P_filt: {P_filt.shape}')
    print(f'shape of x_filt_ledh: {x_filt_ledh.shape}, shape of P_filt_ledh: {P_filt_ledh.shape}')
    print(f'shape of x_filt_pfpf: {x_filt_pfpf.shape}, shape of P_filt_pfpf: {P_filt_pfpf.shape}')
    print(f'shape of x_filt_lpfpf: {x_filt_lpfpf.shape}, shape of P_filt_lpfpf: {P_filt_lpfpf.shape}')

    # Plot results
    plt.figure(figsize=(12, 6))

    x_true_flat = tf.squeeze(x_true).numpy()
    x_filt_flat = tf.squeeze(x_filt).numpy()
    x_filt_ledh_flat = tf.squeeze(x_filt_ledh).numpy()
    x_filt_pfpf_flat = tf.squeeze(x_filt_pfpf).numpy()
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
    plt.title('Particle Flow Filters on 1D Stochastic Volatility Model')
    plt.xlabel('Time')
    plt.ylabel('State / Observation')
    plt.savefig("EDH_LEDH_PFPF_filter.pdf", bbox_inches='tight')
    plt.show()