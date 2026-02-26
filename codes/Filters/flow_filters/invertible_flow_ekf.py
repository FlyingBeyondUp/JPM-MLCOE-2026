import tensorflow as tf
import tensorflow_probability as tfp
from models import NLSSM, get1DLogSquaredSVM
from codes.Filters.basic_filters import UnscentedKalmanFilter, ExtendedKalmanFilter, ParticleFilter
from codes.Filters.flow_filters import EDHFlow, LEDHFlow
from typing import Callable, Union
import numpy as np

tfd = tfp.distributions


class ParticleEKF:
    """
    Vectorized EKF for guiding Particle Flow Particle Filters.
    Uses automatic differentiation for Jacobians.
    Mirrors the ParticleUKF interface (predict_step / update_step / initialize_filter)
    so it can be used as a drop-in replacement.
    """

    def __init__(self, model: NLSSM):
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim

    # ------------------------------------------------------------------ #
    #  Covariance helpers                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_cov(dist):
        try:
            cov = dist.covariance()
        except (AttributeError, NotImplementedError):
            var = dist.variance()
            if len(var.shape) == 0:
                var = tf.reshape(var, [1])
            cov = tf.linalg.diag(var)
        if len(cov.shape) == 0:
            cov = tf.reshape(cov, [1, 1])
        elif len(cov.shape) == 1:
            cov = tf.linalg.diag(cov)
        return tf.cast(cov, tf.float32)

    # ------------------------------------------------------------------ #
    #  Batch Jacobian via AD                                               #
    # ------------------------------------------------------------------ #
    def _batch_jacobian(self, fn, x, noise_dim):
        """
        Compute dh/dx at a batch of points using AD.
        x:  [M, D]
        Returns: f_val [M, out_dim], J [M, out_dim, D]
        """
        zero_noise = tf.zeros([tf.shape(x)[0], noise_dim], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            val = fn(x, zero_noise)
        J = tape.batch_jacobian(val, x)  # [M, out_dim, D]
        return val, J

    # ------------------------------------------------------------------ #
    #  initialize_filter  – same signature as UKF version                  #
    # ------------------------------------------------------------------ #
    def initialize_filter(self, batch_size: int):
        Q = self._get_cov(self.model.process_noise)
        R = self._get_cov(self.model.observation_noise)
        P0 = self._get_cov(self.model.init_noise)

        x_init = tf.reshape(self.model.x0, [1, self.state_dim])
        x_init = tf.tile(x_init, [batch_size, 1])

        P_init = tf.expand_dims(P0, 0)
        P_init = tf.tile(P_init, [batch_size, 1, 1])
        return x_init, P_init, Q, R

    # ------------------------------------------------------------------ #
    #  predict_step                                                        #
    # ------------------------------------------------------------------ #
    @tf.function
    def predict_step(self, x_curr, P_curr, Q):
        """
        EKF predict.
        Supports both:
            x_curr: [B, D]        P_curr: [B, D, D]         → shared
            x_curr: [B, N, D]     P_curr: [B, N, D, D]      → per-particle
        Returns same rank as input.
        """
        per_particle = (x_curr.shape.ndims == 3)

        if per_particle:
            shape = tf.shape(x_curr)
            B, N, D = shape[0], shape[1], shape[2]
            x_flat = tf.reshape(x_curr, [-1, D])          # [B*N, D]
            P_flat = tf.reshape(P_curr, [-1, D, D])        # [B*N, D, D]
        else:
            x_flat = x_curr    # [B, D]
            P_flat = P_curr    # [B, D, D]

        # Jacobian of transition at x_flat
        f_val, F = self._batch_jacobian(
            self.model.transition_fn, x_flat, self.state_dim
        )  # f_val [M, D], F [M, D, D]

        # P_pred = F P F^T + Q
        P_pred = tf.matmul(F, tf.matmul(P_flat, F, transpose_b=True)) + Q
        P_pred = 0.5 * (P_pred + tf.linalg.matrix_transpose(P_pred))

        if per_particle:
            return (tf.reshape(f_val, [B, N, D]),
                    tf.reshape(P_pred, [B, N, D, D]))
        return f_val, P_pred

    # ------------------------------------------------------------------ #
    #  update_step                                                         #
    # ------------------------------------------------------------------ #
    @tf.function
    def update_step(self, x_pred, P_pred, y, R):
        """
        EKF update (Joseph form).
        Supports both:
            x_pred: [B, D]        P_pred: [B, D, D]        y: [B, obs]
            x_pred: [B, N, D]     P_pred: [B, N, D, D]     y: [B, obs]
        For per-particle mode, y is broadcast to every particle.
        Returns same rank as input.
        """
        per_particle = (x_pred.shape.ndims == 3)

        if per_particle:
            shape = tf.shape(x_pred)
            B, N, D = shape[0], shape[1], shape[2]
            x_flat = tf.reshape(x_pred, [-1, D])
            P_flat = tf.reshape(P_pred, [-1, D, D])
            y_flat = tf.repeat(y, repeats=N, axis=0)  # [B*N, obs]
        else:
            x_flat = x_pred
            P_flat = P_pred
            y_flat = y

        M = tf.shape(x_flat)[0]

        # Jacobian of observation at x_flat
        h_val, H = self._batch_jacobian(
            self.model.observation_fn, x_flat, self.obs_dim
        )  # h_val [M, obs], H [M, obs, D]

        # Innovation
        innov = y_flat - h_val  # [M, obs]

        # S = H P H^T + R
        PHt = tf.matmul(P_flat, H, transpose_b=True)  # [M, D, obs]
        S = tf.matmul(H, PHt) + R                       # [M, obs, obs]
        S = 0.5 * (S + tf.linalg.matrix_transpose(S)) \
            + 1e-6 * tf.eye(self.obs_dim, batch_shape=[M])

        # Kalman gain  K = P H^T S^{-1}
        S_chol = tf.linalg.cholesky(S)
        # solve S K^T = (H P)  →  K^T = S^{-1} H P
        Kt = tf.linalg.cholesky_solve(S_chol, tf.linalg.matrix_transpose(PHt))
        K = tf.linalg.matrix_transpose(Kt)  # [M, D, obs]

        # State update
        x_new = x_flat + tf.squeeze(
            tf.matmul(K, tf.expand_dims(innov, -1)), -1
        )  # [M, D]

        # Covariance update – Joseph form for numerical stability
        I_KC = tf.eye(self.state_dim, batch_shape=[M]) - tf.matmul(K, H)
        P_new = tf.matmul(I_KC, tf.matmul(P_flat, I_KC, transpose_b=True)) \
                + tf.matmul(K, tf.matmul(R, K, transpose_b=True))
        P_new = 0.5 * (P_new + tf.linalg.matrix_transpose(P_new))

        if per_particle:
            return (tf.reshape(x_new, [B, N, D]),
                    tf.reshape(P_new, [B, N, D, D]))
        return x_new, P_new


class ParticleUKF(UnscentedKalmanFilter):
    """
    Vectorized UKF for Particle Filters.
    """

    @classmethod
    def from_ukf(cls, ukf_instance: UnscentedKalmanFilter):
        if ukf_instance.train_noise:
            ukf_instance.sync_model()
        return cls(
            model=ukf_instance.model,
            alpha=ukf_instance.alpha,
            beta=ukf_instance.beta,
            kappa=ukf_instance.kappa,
            train_noise=ukf_instance.train_noise
        )

    @tf.function
    def predict_step(self, x_curr, P_curr, Wm, Wc, lam, Q, noise_proc_mean):
        if x_curr.shape.ndims == 3:
            shape = tf.shape(x_curr)
            B, N, D = shape[0], shape[1], shape[2]
            x_flat = tf.reshape(x_curr, [-1, D])
            if P_curr.shape.ndims == 3:
                P_flat = tf.repeat(P_curr, repeats=N, axis=0)
            else:
                P_flat = tf.reshape(P_curr, [-1, D, D])
            P_flat = 0.5 * (P_flat + tf.linalg.matrix_transpose(P_flat)) \
                     + 1e-4 * tf.eye(D, batch_shape=[B * N])
            Q_arg = Q
            if Q.shape.ndims == 3:
                Q_arg = tf.repeat(Q, repeats=N, axis=0)
            x_pred_flat, P_pred_flat, _ = super().predict_step(
                x_flat, P_flat, Wm, Wc, lam, Q_arg, noise_proc_mean)
            return (tf.reshape(x_pred_flat, [B, N, D]),
                    tf.reshape(P_pred_flat, [B, N, D, D]))
        else:
            return super().predict_step(x_curr, P_curr, Wm, Wc, lam, Q, noise_proc_mean)[:-1]

    @tf.function
    def update_step(self, x_pred, P_pred, y, Wm, Wc, lam, R, noise_obs_mean):
        if x_pred.shape.ndims == 3:
            shape = tf.shape(x_pred)
            B, N, D = shape[0], shape[1], shape[2]
            x_flat = tf.reshape(x_pred, [-1, D])
            P_flat = tf.reshape(P_pred, [-1, D, D])
            y_flat = tf.repeat(y, repeats=N, axis=0)
            R_arg = R
            if R.shape.ndims == 3:
                R_arg = tf.repeat(R, repeats=N, axis=0)
            x_new_flat, P_new_flat, _ = super().update_step(
                x_flat, P_flat, y_flat, Wm, Wc, lam, R_arg, noise_obs_mean)
            return (tf.reshape(x_new_flat, [B, N, D]),
                    tf.reshape(P_new_flat, [B, N, D, D]))
        else:
            return super().update_step(x_pred, P_pred, y, Wm, Wc, lam, R, noise_obs_mean)[:-1]


class InvertiblePFPF(ParticleFilter):
    def __init__(self, model: NLSSM, num_particles: int,
                 ukf: UnscentedKalmanFilter = None,
                 flow_class: Callable = EDHFlow,
                 num_flow_steps: int = 10,
                 resample_method: str = 'systematic',
                 resample_threshold: float = 1.0):
        super().__init__(model, num_particles,
                         resample_method=resample_method,
                         resample_threshold=resample_threshold)

        self.flow_algo = flow_class(model, num_particles)
        self.num_flow_steps = num_flow_steps

        # Guide filters – create both, use whichever is requested at filter time
        if ukf is not None:
            self.ukf = ParticleUKF.from_ukf(ukf)
        else:
            self.ukf = None
        self.ekf = ParticleEKF(model)

    # ------------------------------------------------------------------ #
    #  Flow with Jacobian determinant accumulation                         #
    # ------------------------------------------------------------------ #
    @tf.function
    def _flow_with_det(self, eta_0, Y_t, num_flow_steps,
                       P_guide=None, eta_aux=None,
                       step_sizes: tf.Tensor = None) -> tuple[tf.Tensor, tf.Tensor]:
        const_delta_lambda = 1.0 / float(num_flow_steps)
        current_particles = eta_0
        eta_0_aux = tf.identity(eta_aux)
        eta_0_aux = tf.expand_dims(eta_0_aux, -1)
        eta_aux = tf.expand_dims(eta_aux, -1)

        B = tf.shape(eta_0)[0]
        N = self.num_particles
        D = self.model.state_dim

        log_det_jacobian = tf.zeros((B, N))

        lam = 0.0
        for k in tf.range(num_flow_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (log_det_jacobian, tf.TensorShape([None, N])),
                    (current_particles, tf.TensorShape([None, N, D])),
                    (eta_aux,
                     tf.TensorShape([None, D, 1])
                     if self.flow_algo.__class__.__name__ == "EDHFlow"
                     else tf.TensorShape([None, N, D, 1]))
                ]
            )

            if step_sizes is None:
                dt = const_delta_lambda
            else:
                dt = step_sizes[k]
            lam += dt

            A, b = self.flow_algo.compute_flow_parameters(
                current_particles, Y_t, lam,
                P_xx=P_guide, linearization_points=eta_aux,
                eta_0_mean=eta_0_aux
            )

            if len(A.shape) == 3 and self.flow_algo.__class__.__name__ == "EDHFlow":
                drift = tf.matmul(current_particles, A, transpose_b=True) \
                        + tf.expand_dims(b, 1)
                current_particles = current_particles + dt * drift

                drift_aux = tf.matmul(A, eta_aux) + tf.expand_dims(b, -1)
                eta_aux = eta_aux + dt * drift_aux

            elif len(A.shape) == 4 and self.flow_algo.__class__.__name__ == "LEDHFlow":
                x_expanded = tf.expand_dims(current_particles, -1)
                b_expanded = tf.expand_dims(b, -1)
                drift = tf.matmul(A, x_expanded) + b_expanded
                drift = tf.squeeze(drift, -1)
                current_particles = current_particles + dt * drift

                identity = tf.eye(D, batch_shape=[B, N])
                _, step_log_det = tf.linalg.slogdet(identity + dt * A)
                log_det_jacobian += step_log_det

                drift_aux = tf.matmul(A, eta_aux) + tf.expand_dims(b, -1)
                eta_aux = eta_aux + dt * drift_aux
            else:
                raise ValueError("A must be either [B, D, D] or [B, N, D, D]")

        return current_particles, log_det_jacobian

    # ------------------------------------------------------------------ #
    #  Weight update                                                       #
    # ------------------------------------------------------------------ #
    @tf.function
    def _update_pfpf(self, x_prev, eta_0, eta_1, weights, Y_t,
                     log_det_jacobian, is_initial_step):
        N = self.num_particles

        Y_t_expanded = tf.tile(tf.expand_dims(Y_t, 1), [1, N, 1])
        log_likelihood = self._compute_log_prob(
            target=Y_t_expanded, source=eta_1,
            map_fn=self.model.observation_fn,
            noise_dist=self.model.observation_noise,
            dist_fn=getattr(self.model, 'get_observation_dist', None)
        )
        if len(log_likelihood.shape) > 2:
            log_likelihood = tf.reduce_sum(log_likelihood, axis=-1)

        def _compute_init_log_prob(x):
            noise = x - self.model.x0
            lp = self.model.init_noise.log_prob(noise)
            if len(lp.shape) == 3:
                lp = tf.reduce_sum(lp, axis=-1)
            return lp

        def _compute_trans_log_prob(target, source):
            return self._compute_log_prob(
                target=target, source=source,
                map_fn=self.model.transition_fn,
                noise_dist=self.model.process_noise,
                dist_fn=getattr(self.model, 'get_transition_dist', None)
            )

        log_p_eta1_prior, log_p_eta0_prior = tf.cond(
            is_initial_step,
            lambda: (_compute_init_log_prob(eta_1),
                     _compute_init_log_prob(eta_0)),
            lambda: (_compute_trans_log_prob(eta_1, x_prev),
                     _compute_trans_log_prob(eta_0, x_prev))
        )

        log_weights_prev = tf.math.log(weights + 1e-10)
        log_update = (log_likelihood
                      + log_p_eta1_prior
                      - log_p_eta0_prior
                      + log_det_jacobian)
        log_weights_new = log_weights_prev + log_update
        log_norm_const = tf.math.reduce_logsumexp(
            log_weights_new, axis=1, keepdims=True)
        new_weights = tf.math.exp(log_weights_new - log_norm_const)
        return new_weights

    # ------------------------------------------------------------------ #
    #  Resampling helpers                                                  #
    # ------------------------------------------------------------------ #
    def _resample_all(self, particles, w, P_u, batch_size):
        effective_batch_size = 1.0 / (
            tf.reduce_sum(tf.square(w), axis=1) + 1e-10)
        resample_cond = effective_batch_size < (
            self.resample_threshold * tf.cast(self.num_particles, tf.float32))

        indices = self._get_resample_indices(w, batch_size, self.num_particles)
        p_new = tf.gather(particles, indices, batch_dims=1)
        P_u_new = tf.gather(P_u, indices, batch_dims=1)
        w_new = tf.fill([batch_size, self.num_particles],
                        1.0 / float(self.num_particles))

        cond_p = tf.reshape(resample_cond, [batch_size, 1, 1])
        particles = tf.where(cond_p, p_new, particles)

        if len(P_u.shape) == 4:
            cond_P = tf.reshape(resample_cond, [batch_size, 1, 1, 1])
        else:
            cond_P = cond_p
        P_u = tf.where(cond_P, P_u_new, P_u)

        cond_w = tf.reshape(resample_cond, [batch_size, 1])
        w = tf.where(cond_w, w_new, w)
        return particles, w, P_u


    @tf.function
    def filter(self, observations: tf.Tensor,
               num_flow_steps: int = 10,
               step_sizes: tf.Tensor = None
               ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size, T = observations.shape[0], observations.shape[1]

        (x_ukf, P_ukf, Wm, Wc, lam_ukf,
         Q, R, q_mean, r_mean) = self.ukf.initialize_filter(batch_size)
        x_filt = x_ukf

        if self.flow_algo.__class__.__name__ == "LEDHFlow":
            P_ukf = tf.tile(tf.expand_dims(P_ukf, 1),
                            [1, self.num_particles, 1, 1])
            x_ukf = tf.tile(tf.expand_dims(x_ukf, 1),
                            [1, self.num_particles, 1])

        all_particles = tf.TensorArray(dtype=tf.float32, size=T)
        all_weights   = tf.TensorArray(dtype=tf.float32, size=T)
        x_filt_ta     = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta     = tf.TensorArray(dtype=tf.float32, size=T)

        particles, weights = self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])

        def _transition_branch(particles, x, P_ukf):
            p_pred = self._transition(particles)
            x_u, P_u = self.ukf.predict_step(
                x, P_ukf, Wm, Wc, lam_ukf, Q, q_mean)
            return p_pred, x_u, P_u

        for t in tf.range(T):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (particles, tf.TensorShape(
                        [None, self.num_particles, self.model.state_dim])),
                    (weights, tf.TensorShape(
                        [None, self.num_particles])),
                    (x_filt, tf.TensorShape(
                        [None, self.model.state_dim])),
                    (x_ukf, tf.TensorShape(
                        [None, self.model.state_dim])
                     if self.flow_algo.__class__.__name__ == "EDHFlow"
                     else tf.TensorShape(
                        [None, self.num_particles, self.model.state_dim])),
                    (P_ukf, tf.TensorShape(
                        [None, self.model.state_dim, self.model.state_dim])
                     if self.flow_algo.__class__.__name__ == "EDHFlow"
                     else tf.TensorShape(
                        [None, self.num_particles,
                         self.model.state_dim, self.model.state_dim]))
                ]
            )

            Y_t = Y_time_major[t]

            if self.flow_algo.__class__.__name__ == "EDHFlow":
                pre_imag = x_filt
            else:
                pre_imag = particles

            if t > 0:
                eta_0, x_pred_ukf, P_pred_ukf = _transition_branch(
                    particles, pre_imag, P_ukf)
                eta_aux = self.model.transition_fn(
                    pre_imag, tf.zeros_like(pre_imag))
            else:
                eta_0 = particles
                x_pred_ukf, P_pred_ukf = x_ukf, P_ukf
                eta_aux = pre_imag

            eta_1, log_det = self._flow_with_det(
                eta_0, Y_t, num_flow_steps,
                P_guide=P_pred_ukf, eta_aux=eta_aux,
                step_sizes=step_sizes)

            is_initial = tf.equal(t, 0)
            weights = self._update_pfpf(
                particles, eta_0, eta_1, weights,
                Y_t, log_det, is_initial)

            x_ukf, P_ukf = self.ukf.update_step(
                x_pred_ukf, P_pred_ukf, Y_t,
                Wm, Wc, lam_ukf, R, r_mean)

            particles = eta_1
            x_filt = tf.reduce_sum(
                tf.expand_dims(weights, -1) * particles, axis=1)
            diff = particles - tf.expand_dims(x_filt, 1)
            w_exp = tf.reshape(
                weights, [batch_size, self.num_particles, 1, 1])
            P_filt = tf.reduce_sum(
                w_exp * tf.expand_dims(diff, -1)
                      * tf.expand_dims(diff, -2), axis=1)

            x_filt_ta     = x_filt_ta.write(t, x_filt)
            P_filt_ta     = P_filt_ta.write(t, P_filt)
            all_particles = all_particles.write(t, particles)
            all_weights   = all_weights.write(t, weights)

            if self.flow_algo.__class__.__name__ == "EDHFlow":
                particles, weights = self._resample(particles, weights)
            elif self.flow_algo.__class__.__name__ == "LEDHFlow":
                particles, weights, P_ukf = self._resample_all(
                    particles, weights, P_ukf, batch_size)

        return (
            tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),
            tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),
            tf.transpose(all_particles.stack(), perm=[1, 0, 2, 3]),
            tf.transpose(all_weights.stack(), perm=[1, 0, 2]),
        )


    #  filter_with_ekf() — EKF-guided                                      #
    #  1.Per-particle EKF predict / update for LEDH                      #
    #  2.Single EKF predict / update for EDH                             #
    @tf.function
    def filter_with_ekf(
        self,
        observations: tf.Tensor,
        num_flow_steps: int = 10,
        step_sizes: tf.Tensor = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        PF-PF filter using an EKF covariance guide.

        Args:
            observations: [Batch, T, obs_dim]
            num_flow_steps: number of λ integration steps per time step
            step_sizes: optional [num_flow_steps] tensor of Δλ values

        Returns:
            x_filt:        [B, T, D]
            P_filt:        [B, T, D, D]
            all_particles: [B, T, N, D]
            all_weights:   [B, T, N]
        """
        batch_size, T = observations.shape[0], observations.shape[1]
        D = self.model.state_dim
        N = self.num_particles
        is_ledh = (self.flow_algo.__class__.__name__ == "LEDHFlow")

        # ----- Initialise EKF guide -----------------------------------
        x_ekf_init, P_ekf_init, Q, R = self.ekf.initialize_filter(batch_size)
        # x_ekf_init: [B, D],  P_ekf_init: [B, D, D]

        if is_ledh:
            # Per-particle covariance: [B, N, D, D]
            P_ekf = tf.tile(
                tf.expand_dims(P_ekf_init, 1),
                [1, N, 1, 1])
            # Per-particle "EKF state" (used only for predict linearisation
            # point; the actual particles carry the state values)
            x_ekf = tf.tile(
                tf.expand_dims(x_ekf_init, 1),
                [1, N, 1])     # [B, N, D]
        else:
            # Shared (EDH): single covariance per batch
            P_ekf = P_ekf_init   # [B, D, D]
            x_ekf = x_ekf_init   # [B, D]

        x_filt = x_ekf_init  # weighted-mean estimate [B, D]

        particles, weights = self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])  # [T,B,obs]

        all_particles = tf.TensorArray(dtype=tf.float32, size=T)
        all_weights   = tf.TensorArray(dtype=tf.float32, size=T)
        x_filt_ta     = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta     = tf.TensorArray(dtype=tf.float32, size=T)

        for t in range(T):
            # Shape invariants for AutoGraph
            if is_ledh:
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[
                        (particles, tf.TensorShape([None, N, D])),
                        (weights,   tf.TensorShape([None, N])),
                        (x_filt,    tf.TensorShape([None, D])),
                        (x_ekf,     tf.TensorShape([None, N, D])),
                        (P_ekf,     tf.TensorShape([None, N, D, D])),
                    ])
            else:
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[
                        (particles, tf.TensorShape([None, N, D])),
                        (weights,   tf.TensorShape([None, N])),
                        (x_filt,    tf.TensorShape([None, D])),
                        (x_ekf,     tf.TensorShape([None, D])),
                        (P_ekf,     tf.TensorShape([None, D, D])),
                    ])

            Y_t = Y_time_major[t]  # [B, obs]

            # ---- Determine the "pre-image" used for auxiliary flow ----
            # EDH: single mean;  LEDH: per-particle
            if is_ledh:
                pre_imag = particles       # [B, N, D]
            else:
                pre_imag = x_filt          # [B, D]

            # ---- Predict step ----------------------------------------
            if t > 0:
                # Propagate particles (with noise) → η₀
                eta_0 = self._transition(particles)

                # EKF predict for covariance guide
                x_ekf_pred, P_ekf_pred = self.ekf.predict_step(
                    x_ekf, P_ekf, Q)

                # Deterministic auxiliary for flow linearisation
                eta_aux = self.model.transition_fn(
                    pre_imag, tf.zeros_like(pre_imag))
            else:
                eta_0 = particles
                x_ekf_pred = x_ekf
                P_ekf_pred = P_ekf
                eta_aux = pre_imag

            # ---- Particle flow ---------------------------------------
            eta_1, log_det = self._flow_with_det(
                eta_0, Y_t, num_flow_steps,
                P_guide=P_ekf_pred,
                eta_aux=eta_aux,
                step_sizes=step_sizes)

            # ---- Weight update ---------------------------------------
            is_initial = tf.equal(t, 0)
            weights = self._update_pfpf(
                particles, eta_0, eta_1, weights,
                Y_t, log_det, is_initial)

            # ---- EKF update for guide covariance ---------------------
            # For LEDH the update runs per-particle (flattened inside);
            # for EDH it runs once on the shared [B, D] state.
            x_ekf, P_ekf = self.ekf.update_step(
                x_ekf_pred, P_ekf_pred, Y_t, R)

            # ---- Form filtered estimate ------------------------------
            particles = eta_1
            x_filt = tf.reduce_sum(
                tf.expand_dims(weights, -1) * particles, axis=1)  # [B, D]
            diff = particles - tf.expand_dims(x_filt, 1)
            w_exp = tf.reshape(weights, [batch_size, N, 1, 1])
            P_filt = tf.reduce_sum(
                w_exp * tf.expand_dims(diff, -1)
                      * tf.expand_dims(diff, -2), axis=1)

            x_filt_ta     = x_filt_ta.write(t, x_filt)
            P_filt_ta     = P_filt_ta.write(t, P_filt)
            all_particles = all_particles.write(t, particles)
            all_weights   = all_weights.write(t, weights)

            # ---- Resample -------------------------------------------
            if is_ledh:
                particles, weights, P_ekf = self._resample_all(
                    particles, weights, P_ekf, batch_size)
            else:
                particles, weights = self._resample(particles, weights)

        # ----- Stack & return -----------------------------------------
        return (
            tf.transpose(x_filt_ta.stack(),     perm=[1, 0, 2]),
            tf.transpose(P_filt_ta.stack(),     perm=[1, 0, 2, 3]),
            tf.transpose(all_particles.stack(), perm=[1, 0, 2, 3]),
            tf.transpose(all_weights.stack(),   perm=[1, 0, 2]),
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sv_model = get1DLogSquaredSVM(alpha=0.9, beta=0.5, sigma=1)

    ukf = UnscentedKalmanFilter(model=sv_model, alpha=1e-3, beta=2.0, kappa=0.0)
    pf = EDHFlow(model=sv_model, num_particles=1000, ukf=ukf)
    lpf = LEDHFlow(model=sv_model, num_particles=1000, ukf=ukf)
    pfpf = InvertiblePFPF(model=sv_model, num_particles=1000, ukf=ukf,
                          flow_class=EDHFlow)
    lpfpf = InvertiblePFPF(model=sv_model, num_particles=1000, ukf=ukf,
                           flow_class=LEDHFlow)

    T = 200
    x_true, y_obs = sv_model.sample(T)
    if len(y_obs.shape) == 1:
        y_obs = tf.reshape(y_obs, [1, -1, 1])

    # --- UKF-guided ---
    pfpf.filter(y_obs, num_flow_steps=5)
    t0 = tf.timestamp()
    x_filt_pfpf, P_filt_pfpf, _, w_pfpf = pfpf.filter(y_obs, num_flow_steps=5)
    t1 = tf.timestamp()
    print(f"PF-PF (EDH, UKF guide): {t1 - t0:.4f}s")

    # --- EKF-guided ---
    pfpf.filter_with_ekf(y_obs, num_flow_steps=5)
    t2 = tf.timestamp()
    x_filt_ekf, P_filt_ekf, _, w_ekf = pfpf.filter_with_ekf(
        y_obs, num_flow_steps=5)
    t3 = tf.timestamp()
    print(f"PF-PF (EDH, EKF guide): {t3 - t2:.4f}s")

    # --- Compare ESS ---
    ess_ukf = 1.0 / tf.reduce_sum(tf.square(w_pfpf), axis=-1)
    ess_ekf = 1.0 / tf.reduce_sum(tf.square(w_ekf), axis=-1)
    print(f"Mean ESS (UKF guide): {tf.reduce_mean(ess_ukf):.1f}")
    print(f"Mean ESS (EKF guide): {tf.reduce_mean(ess_ekf):.1f}")