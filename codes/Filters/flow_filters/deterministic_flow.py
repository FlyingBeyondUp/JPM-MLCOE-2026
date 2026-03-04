import tensorflow as tf
import tensorflow_probability as tfp

from Filters.basic_filters.base_filter import BaseFilter
from models.base_models import NLSSM
from Filters.basic_filters import UnscentedKalmanFilter

tfd = tfp.distributions
dtype = tf.float32


class EDHFlow(BaseFilter):
    """
    Implements the Exact Daum-Huang (EDH) particle flow for non-linear state-space models.

    Inherits from BaseFilter. Uses a deterministic ODE to migrate particles from the
    prior to the posterior, avoiding the weight degeneracy issues of standard resampling.

    Can optionally embed an UnscentedKalmanFilter (UKF) inside its opaque state to
    guide the prior covariance and resampling structural stability.
    """

    def __init__(self, model: NLSSM, num_particles: int, num_flow_steps: int = 10,
                 step_sizes: tf.Tensor = None, ukf: UnscentedKalmanFilter = None,
                 resample_from_ukf: bool = False):
        super().__init__(model)
        self.num_particles = num_particles
        self.num_flow_steps = num_flow_steps
        self.step_sizes = step_sizes
        self.ukf = ukf
        self.resample_from_ukf = resample_from_ukf

    def _init_state(self, batch_size: int) -> tuple:
        """
        Initializes particles and optionally packs the UKF internal state.
        State Tuple: (particles, x_ukf, P_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam)
        """
        total_particles = batch_size * self.num_particles
        noise = self.model.init_noise.sample(total_particles)

        if len(noise.shape) == 1:
            noise = tf.expand_dims(noise, axis=-1)

        x0_expanded = tf.expand_dims(self.model.x0, 0)
        particles_flat = x0_expanded + noise
        particles = tf.reshape(particles_flat, [batch_size, self.num_particles, self.model.state_dim])

        # Pack the UKF state if provided, otherwise fill with dummy tensors
        if self.ukf is not None:
            ukf_state = self.ukf._init_state(batch_size)
        else:
            D = self.model.state_dim
            x_dummy = tf.zeros([batch_size, D], dtype=dtype)
            P_dummy = tf.zeros([batch_size, D, D], dtype=dtype)
            w_dummy = tf.zeros([1], dtype=dtype)
            ukf_state = (x_dummy, P_dummy, x_dummy, P_dummy, w_dummy, w_dummy, tf.zeros([], dtype=dtype))

        return (particles, *ukf_state)

    def _init_trajectory(self, time_steps: int) -> tuple:
        """Initializes TensorArrays to store the flow trajectory."""
        particles_ta = tf.TensorArray(dtype, size=time_steps)
        x_filt_ta = tf.TensorArray(dtype, size=time_steps)
        P_filt_ta = tf.TensorArray(dtype, size=time_steps)
        logl_ta = tf.TensorArray(dtype, size=time_steps)

        return (particles_ta, x_filt_ta, P_filt_ta, logl_ta)

    def _transition(self, particles: tf.Tensor) -> tf.Tensor:
        """Propagates particles through the state transition model."""
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

    def predict(self, t: int, state: tuple) -> tuple:
        """Executes the predict step for both particles and the embedded UKF."""
        particles, x_filt_ukf, P_filt_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam = state

        # 1. Transition particles
        particles_pred = self._transition(particles)

        # 2. Transition UKF (if enabled)
        if self.ukf is not None:
            ukf_state_prev = (x_filt_ukf, P_filt_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam)
            ukf_state_pred = self.ukf.predict(t, ukf_state_prev)
        else:
            ukf_state_pred = (x_filt_ukf, P_filt_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam)

        return (particles_pred, *ukf_state_pred)

    def compute_flow_parameters(self, particles: tf.Tensor, observation: tf.Tensor, lam: float,
                                P_xx=None, linearization_points=None, eta_0_mean=None):
        """
        Computes flow parameters A(λ) and b(λ) using Statistical Linearization.
        """
        shape = tf.shape(particles)
        B, N, state_dim = shape[0], shape[1], particles.shape[-1]
        obs_dim = observation.shape[-1]

        # Dynamically fetch observation noise for learning capability
        R = tf.expand_dims(self._get_cov(self.model.observation_noise), axis=0)

        if linearization_points is not None:
            mean_x_flat = linearization_points
            if len(mean_x_flat.shape) > 2:
                mean_x_flat = tf.squeeze(mean_x_flat, -1)
        else:
            mean_x_flat = tf.reduce_mean(particles, axis=1)

        zero_noise = tf.zeros((B, self.model.obs_dim), dtype=dtype)
        with tf.GradientTape() as tape:
            tape.watch(mean_x_flat)
            mean_y_flat = self.model.observation_fn(mean_x_flat, zero_noise)

        H = tape.batch_jacobian(mean_y_flat, mean_x_flat)

        P_Ht = tf.matmul(P_xx, H, transpose_b=True)
        HP_Ht = tf.matmul(H, P_Ht)

        S = R + lam * HP_Ht
        S = 0.5 * (S + tf.linalg.matrix_transpose(S)) + 1e-4 * tf.eye(obs_dim, batch_shape=[B], dtype=dtype)

        obs_expanded = tf.expand_dims(observation, -1)
        mean_y_expanded = tf.expand_dims(mean_y_flat, -1)
        innovation = obs_expanded - mean_y_expanded

        rhs = tf.concat([H, innovation], axis=-1)
        solution = tf.linalg.solve(S, rhs)
        S_inv_H, S_inv_innov = tf.split(solution, [state_dim, 1], axis=-1)

        A = -0.5 * tf.matmul(P_Ht, S_inv_H)

        HP = tf.transpose(P_Ht, perm=[0, 2, 1])
        R_batch = tf.tile(R, [B, 1, 1])
        Kt = tf.linalg.solve(R_batch, HP)
        K = tf.transpose(Kt, perm=[0, 2, 1])

        mean_x_col = tf.expand_dims(mean_x_flat, -1)
        z_minus_e = innovation + tf.matmul(H, mean_x_col)

        K_ze = tf.matmul(K, z_minus_e)

        eye_d = tf.eye(state_dim, batch_shape=[B], dtype=dtype)
        I_lamA = eye_d + lam * A
        term_left = tf.matmul(I_lamA, K_ze)

        if eta_0_mean is None:
            eta_0_mean = tf.reduce_mean(particles, axis=1)
        if len(eta_0_mean.shape) == 2:
            eta_0_mean = tf.expand_dims(eta_0_mean, -1)

        term_right = tf.matmul(A, eta_0_mean)
        bracket = term_left + term_right

        I_2lamA = eye_d + 2.0 * lam * A
        b = tf.matmul(I_2lamA, bracket)
        b = tf.squeeze(b, -1)

        return A, b

    def _flow_update(self, observation: tf.Tensor, particles: tf.Tensor, P_xx=None) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor]:
        """Perform the EDH particle flow given an observation by integrating dx/dλ = Ax + b."""
        const_delta_lambda = 1.0 / float(self.num_flow_steps)
        current_particles = particles
        eta_0_mean = tf.reduce_mean(particles, axis=1)

        if P_xx is None:
            # Fallback to Sample Covariance
            mean_x = tf.reduce_mean(particles, axis=1, keepdims=True)
            dx = particles - mean_x
            N_float = tf.cast(tf.shape(particles)[1], dtype)
            P_xx = tf.matmul(dx, dx, transpose_a=True) / (N_float - 1.0)

        lam = 0.0
        for k in tf.range(self.num_flow_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (current_particles, tf.TensorShape([None, self.num_particles, self.model.state_dim]))
                ]
            )
            delta_lambda = self.step_sizes[k] if self.step_sizes is not None else const_delta_lambda
            lam += delta_lambda

            A, b = self.compute_flow_parameters(current_particles, observation, lam, P_xx=P_xx, eta_0_mean=eta_0_mean)
            drift = tf.matmul(current_particles, A, transpose_b=True) + tf.expand_dims(b, axis=1)
            current_particles = current_particles + delta_lambda * drift

        x_filt = tf.reduce_mean(current_particles, axis=1)
        P_filt = tfp.stats.covariance(current_particles, sample_axis=1)

        return current_particles, x_filt, P_filt

    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        """Executes the ODE flow update and coordinates the UKF resampling path."""
        particles_pred, x_filt_ukf, P_filt_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam = state

        P_xx = P_pred_ukf if self.ukf is not None else None

        particles_filt, x_filt_flow, P_filt_flow = self._flow_update(observation, particles_pred, P_xx=P_xx)

        if self.ukf is not None:
            ukf_state_pred = (x_filt_ukf, P_filt_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam)
            ukf_state_filt, ukf_metrics = self.ukf.update(t, ukf_state_pred, observation)
            log_l = ukf_metrics[0]

            if self.resample_from_ukf:
                # Use x_filt for accuracy, but P_ukf for structural stability to prevent collapse
                P_u = ukf_state_filt[1]
                P_sym = 0.5 * (P_u + tf.linalg.matrix_transpose(P_u)) + 1e-4 * tf.eye(self.model.state_dim,
                                                                                      batch_shape=[
                                                                                          tf.shape(particles_filt)[0]])
                mvn = tfd.MultivariateNormalFullCovariance(loc=x_filt_flow, covariance_matrix=P_sym)
                samples = mvn.sample(self.num_particles)
                particles_filt = tf.transpose(samples, perm=[1, 0, 2])
        else:
            ukf_state_filt = (x_filt_ukf, P_filt_ukf, x_pred_ukf, P_pred_ukf, Wm, Wc, lam)
            log_l = tf.zeros([tf.shape(observation)[0]], dtype=dtype)

        new_state = (particles_filt, *ukf_state_filt)
        metrics = (log_l, x_filt_flow, P_filt_flow)

        # COMPILER FIX: Force the dynamic tensor back to the static shape of the incoming state
        for n, o in zip(new_state, state):
            if isinstance(n, tf.Tensor) and isinstance(o, tf.Tensor):
                n.set_shape(o.shape)

        return new_state, metrics

    def forecast(self, observations: tf.Tensor) -> tuple:
        res = self.filter(observations)
        particles_last = res['particles'][:, -1, :, :]

        particles_pred = self._transition(particles_last)

        B = tf.shape(particles_pred)[0]
        N = self.num_particles

        particles_pred_flat = tf.reshape(particles_pred, [B * N, self.model.state_dim])
        zero_noise = tf.zeros([B * N, self.model.obs_dim], dtype=dtype)
        y_pred_flat = self.model.observation_fn(particles_pred_flat, zero_noise)
        y_pred = tf.reshape(y_pred_flat, [B, N, self.model.obs_dim])

        y_pred_mean = tf.reduce_mean(y_pred, axis=1)
        diff = y_pred - tf.expand_dims(y_pred_mean, 1)
        S_pred = tf.matmul(diff, diff, transpose_a=True) / (tf.cast(N, dtype) - 1.0)

        R = tf.expand_dims(self._get_cov(self.model.observation_noise), 0)
        S_next = S_pred + R

        return y_pred_mean, S_next

    def _write_trajectory(self, t: int, trajectory: tuple, state: tuple, metrics: tuple) -> tuple:
        particles_ta, x_filt_ta, P_filt_ta, logl_ta = trajectory
        particles = state[0]
        log_l, x_filt, P_filt = metrics

        return (
            particles_ta.write(t, particles),
            x_filt_ta.write(t, x_filt),
            P_filt_ta.write(t, P_filt),
            logl_ta.write(t, log_l)
        )

    def _format_output(self, trajectory: tuple) -> dict:
        particles_ta, x_filt_ta, P_filt_ta, logl_ta = trajectory
        return {
            "particles": tf.transpose(particles_ta.stack(), perm=[1, 0, 2, 3]),
            "x_filt": tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),
            "P_filt": tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),
            "log_likelihood": tf.reduce_sum(tf.transpose(logl_ta.stack(), perm=[1, 0]), axis=-1)
        }


class LEDHFlow(EDHFlow):
    """
    Implements the Local Exact Daum-Huang (LEDH) particle flow.

    Inherits from EDHFlow. Computes distinct flow parameters A_i and b_i for each particle
    using individual Jacobian linearizations while sharing the global covariance P.
    """

    def __init__(self, model: NLSSM, num_particles: int, num_flow_steps: int = 10,
                 step_sizes: tf.Tensor = None, ukf: UnscentedKalmanFilter = None,
                 resample_from_ukf: bool = False):
        super().__init__(model, num_particles, num_flow_steps, step_sizes, ukf, resample_from_ukf)

    def compute_flow_parameters(self, particles: tf.Tensor, observation: tf.Tensor, lam: float,
                                P_xx=None, linearization_points=None, eta_0_mean=None) -> tuple[tf.Tensor, tf.Tensor]:
        """Computes flow parameters per particle using LEDH equations."""
        B, N, D = tf.shape(particles)[0], tf.shape(particles)[1], tf.shape(particles)[2]
        obs_dim = observation.shape[-1]

        if P_xx.shape.ndims == 3:
            P_expanded = tf.expand_dims(P_xx, 1)
        else:
            P_expanded = P_xx

        if linearization_points is None:
            linearization_points = tf.expand_dims(particles, -1)

        flat_particles = tf.reshape(linearization_points, [B * N, D])
        zero_noise = tf.zeros((B * N, self.model.obs_dim), dtype=dtype)

        with tf.GradientTape() as tape:
            tape.watch(flat_particles)
            pred_obs_flat = self.model.observation_fn(flat_particles, zero_noise)

        H_flat = tape.batch_jacobian(pred_obs_flat, flat_particles)
        H = tf.reshape(H_flat, [B, N, obs_dim, D])

        pred_obs = tf.reshape(pred_obs_flat, [B, N, obs_dim])

        PHt = tf.matmul(P_expanded, H, transpose_b=True)
        HPHt = tf.matmul(H, PHt)

        # Dynamically fetch R
        R = self._get_cov(self.model.observation_noise)
        R_expanded = tf.reshape(R, [1, 1, obs_dim, obs_dim])
        R_tiled = tf.tile(R_expanded, [B, N, 1, 1])

        S = R_tiled + lam * HPHt
        S = 0.5 * (S + tf.linalg.matrix_transpose(S)) + 1e-5 * tf.eye(obs_dim, batch_shape=[B, N], dtype=dtype)

        S_inv_H = tf.linalg.solve(S, H)
        A = -0.5 * tf.matmul(PHt, S_inv_H)

        HP = tf.matmul(H, P_expanded)
        Kt = tf.linalg.solve(R_tiled, HP)
        K = tf.transpose(Kt, perm=[0, 1, 3, 2])

        y_true_expanded = tf.expand_dims(observation, 1)
        innov = y_true_expanded - pred_obs
        innov = tf.expand_dims(innov, -1)

        Hx = tf.matmul(H, linearization_points)
        z_minus_e = innov + Hx

        K_ze = tf.matmul(K, z_minus_e)

        if eta_0_mean is None:
            eta_0_mean = tf.reduce_mean(particles, axis=1)
        if len(eta_0_mean.shape) == 2:
            eta_0_mean_expanded = tf.reshape(eta_0_mean, [B, 1, D, 1])
        else:
            eta_0_mean_expanded = eta_0_mean

        I_lamA_Kze = K_ze + lam * tf.matmul(A, K_ze)
        bracket = I_lamA_Kze + tf.matmul(A, eta_0_mean_expanded)

        b_expanded = bracket + 2.0 * lam * tf.matmul(A, bracket)
        b = tf.squeeze(b_expanded, -1)

        return A, b

    def _flow_update(self, observation: tf.Tensor, particles: tf.Tensor, P_xx=None, linearization_points=None) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor]:
        """Overridden flow update for LEDH. Handles dimensions where A is [B, N, D, D] (per-particle)."""
        const_delta_lambda = 1.0 / float(self.num_flow_steps)
        current_particles = particles
        eta_0_mean = tf.expand_dims(particles, -1)

        if P_xx is None:
            mean_x = tf.reduce_mean(particles, axis=1, keepdims=True)
            dx = particles - mean_x
            N_float = tf.cast(tf.shape(particles)[1], dtype)
            P_xx = tf.matmul(dx, dx, transpose_a=True) / (N_float - 1.0)

        lam = 0.0
        for k in tf.range(self.num_flow_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (current_particles, tf.TensorShape([None, self.num_particles, self.model.state_dim]))
                ]
            )
            delta_lambda = self.step_sizes[k] if self.step_sizes is not None else const_delta_lambda
            lam += delta_lambda

            A, b = self.compute_flow_parameters(current_particles, observation, lam, P_xx=P_xx, eta_0_mean=eta_0_mean)

            x_expanded = tf.expand_dims(current_particles, -1)
            b_expanded = tf.expand_dims(b, -1)
            drift = tf.matmul(A, x_expanded) + b_expanded
            drift = tf.squeeze(drift, -1)
            current_particles = current_particles + delta_lambda * drift

        x_filt = tf.reduce_mean(current_particles, axis=1)
        P_filt = tfp.stats.covariance(current_particles, sample_axis=1)

        return current_particles, x_filt, P_filt