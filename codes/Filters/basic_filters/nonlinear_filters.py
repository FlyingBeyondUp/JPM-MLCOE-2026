import tensorflow as tf
import tensorflow_probability as tfp
import math
from Filters.basic_filters.base_filter import BaseFilter
from models.base_models import NLSSM

tfd = tfp.distributions
dtype = tf.float32


class ExtendedKalmanFilter(BaseFilter):
    """
    Extended Kalman Filter (EKF) for Non-Linear State-Space Models.

    Assumes strictly additive process and observation noise:
    x_t = f(x_{t-1}) + q_t
    y_t = h(x_t) + r_t
    """

    def __init__(self, model: NLSSM, requires_stabilization: bool = True):
        super().__init__(model)
        self.requires_stabilization = requires_stabilization

    def _batch_linearize(self, fn, x):
        """
        Batch linearization.
        Only computes the Jacobian with respect to the state (A or C).
        """
        # Pass zero noise to isolate the deterministic function
        zero_noise = tf.zeros_like(x) if fn.__name__ == 'transition_fn' else tf.zeros(
            [tf.shape(x)[0], self.model.obs_dim])

        with tf.GradientTape() as tape:
            tape.watch(x)
            val = fn(x, zero_noise)

        J_x = tape.batch_jacobian(val, x)
        return val, J_x

    def _init_state(self, batch_size: int) -> tuple:
        state_dim = self.model.state_dim

        x0_reshaped = tf.reshape(self.model.x0, [state_dim])
        P0 = self._get_cov(self.model.init_noise)

        x_init = tf.tile(tf.expand_dims(x0_reshaped, 0), [batch_size, 1])
        P_init = tf.tile(tf.expand_dims(P0, 0), [batch_size, 1, 1])
        A_init = tf.tile(tf.expand_dims(tf.eye(state_dim, dtype=dtype), 0), [batch_size, 1, 1])

        return (x_init, P_init, x_init, P_init, A_init)

    def _init_trajectory(self, time_steps: int) -> tuple:
        x_filt_ta = tf.TensorArray(dtype, size=time_steps)
        P_filt_ta = tf.TensorArray(dtype, size=time_steps)
        x_pred_ta = tf.TensorArray(dtype, size=time_steps)
        P_pred_ta = tf.TensorArray(dtype, size=time_steps)
        A_ta = tf.TensorArray(dtype, size=time_steps)
        logl_ta = tf.TensorArray(dtype, size=time_steps)

        return (x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, A_ta, logl_ta)

    def predict(self, t: int, state: tuple) -> tuple:
        x_filt_prev, P_filt_prev, _, _, _ = state

        Q = tf.expand_dims(self._get_cov(self.model.process_noise), 0)

        # 1. Linearize just the state
        x_pred, A = self._batch_linearize(self.model.transition_fn, x_filt_prev)

        # 2. Additive noise update (No W matrix needed!)
        P_pred = tf.matmul(A, tf.matmul(P_filt_prev, A, transpose_b=True)) + Q
        P_pred = 0.5 * (P_pred + tf.linalg.matrix_transpose(P_pred))

        return (x_pred, P_pred, x_pred, P_pred, A)

    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        _, _, x_pred, P_pred, A_t = state

        obs_dim = self.model.obs_dim
        state_dim = self.model.state_dim

        R = tf.expand_dims(self._get_cov(self.model.observation_noise), 0)

        # 1. Linearize just the state
        h_val, C = self._batch_linearize(self.model.observation_fn, x_pred)

        innov = tf.expand_dims(observation - h_val, -1)

        # 2. Additive noise update (No V matrix needed, R_eff is just R)
        S_t = tf.matmul(C, tf.matmul(P_pred, C, transpose_b=True)) + R + 1e-6 * tf.eye(obs_dim, dtype=dtype)
        S_t = 0.5 * (S_t + tf.linalg.matrix_transpose(S_t))
        S_chol = tf.linalg.cholesky(S_t)

        Kt_transposed = tf.linalg.cholesky_solve(S_chol, tf.matmul(C, P_pred))
        K_t = tf.linalg.matrix_transpose(Kt_transposed)

        x_filt = x_pred + tf.squeeze(tf.matmul(K_t, innov), -1)

        I_KC = tf.eye(state_dim, dtype=dtype) - tf.matmul(K_t, C)
        if self.requires_stabilization:
            term1 = tf.matmul(I_KC, tf.matmul(P_pred, I_KC, transpose_b=True))
            term2 = tf.matmul(K_t, tf.matmul(R, K_t, transpose_b=True))
            P_filt = term1 + term2
        else:
            P_filt = tf.matmul(I_KC, P_pred)

        P_filt = 0.5 * (P_filt + tf.linalg.matrix_transpose(P_filt))

        const_term = -0.5 * float(obs_dim) * tf.math.log(2 * math.pi)
        diag_S = tf.maximum(tf.linalg.diag_part(S_chol), 1e-6)
        log_det_S = 2 * tf.reduce_sum(tf.math.log(diag_S), axis=-1)

        quad_term = tf.squeeze(tf.matmul(tf.linalg.matrix_transpose(innov), tf.linalg.cholesky_solve(S_chol, innov)),
                               [1, 2])
        log_l = const_term - 0.5 * log_det_S - 0.5 * quad_term

        new_state = (x_filt, P_filt, x_pred, P_pred, A_t)
        return new_state, (log_l,)

    def forecast(self, observations: tf.Tensor) -> tuple:
        res = self.filter(observations)
        x_filt_last = res['x_filt'][:, -1, :]
        P_filt_last = res['P_filt'][:, -1, :, :]

        state_last = (x_filt_last, P_filt_last, None, None, None)
        x_pred_next, P_pred_next, _, _, _ = self.predict(0, state_last)

        y_pred_next, C = self._batch_linearize(self.model.observation_fn, x_pred_next)

        R = tf.expand_dims(self._get_cov(self.model.observation_noise), 0)
        S_next = tf.matmul(C, tf.matmul(P_pred_next, C, transpose_b=True)) + R

        return y_pred_next, S_next

    def _write_trajectory(self, t: int, trajectory: tuple, state: tuple, metrics: tuple) -> tuple:
        x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, A_ta, logl_ta = trajectory
        x_filt, P_filt, x_pred, P_pred, A_t = state
        log_l = metrics[0]

        return (
            x_filt_ta.write(t, x_filt), P_filt_ta.write(t, P_filt),
            x_pred_ta.write(t, x_pred), P_pred_ta.write(t, P_pred),
            A_ta.write(t, A_t), logl_ta.write(t, log_l)
        )

    def _format_output(self, trajectory: tuple) -> dict:
        x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, A_ta, logl_ta = trajectory
        return {
            "x_filt": tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),
            "P_filt": tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),
            "x_pred": tf.transpose(x_pred_ta.stack(), perm=[1, 0, 2]),
            "P_pred": tf.transpose(P_pred_ta.stack(), perm=[1, 0, 2, 3]),
            "A_t": tf.transpose(A_ta.stack(), perm=[1, 0, 2, 3]),
            "log_likelihood": tf.reduce_sum(tf.transpose(logl_ta.stack(), perm=[1, 0]), axis=-1)
        }

    @tf.function
    def smooth(self, Y: tf.Tensor):
        """Batched RTS Smoother (Remains unchanged, as smoothing only requires A)."""
        res = self.filter(Y)
        x_filt, P_filt = res['x_filt'], res['P_filt']
        x_pred, P_pred = res['x_pred'], res['P_pred']
        A_arr = res['A_t']
        log_l = res['log_likelihood']

        batch_size = tf.shape(x_filt)[0]
        T = tf.shape(x_filt)[1]
        state_dim = self.model.state_dim

        x_smooth_ta = tf.TensorArray(dtype=dtype, size=T, clear_after_read=False)
        P_smooth_ta = tf.TensorArray(dtype=dtype, size=T, clear_after_read=False)
        P_cross_ta = tf.TensorArray(dtype=dtype, size=tf.maximum(1, T - 1))

        x_smooth_ta = x_smooth_ta.write(T - 1, tf.expand_dims(x_filt[:, T - 1, :], -1))
        P_smooth_ta = P_smooth_ta.write(T - 1, P_filt[:, T - 1, :, :])

        for t in tf.range(T - 2, -1, -1):
            P_filt_t = P_filt[:, t, :, :]
            P_pred_next = P_pred[:, t + 1, :, :]
            A_t = A_arr[:, t + 1, :, :]

            P_pred_next_chol = tf.linalg.cholesky(P_pred_next + 1e-6 * tf.eye(state_dim, dtype=dtype))

            J_transposed = tf.linalg.cholesky_solve(P_pred_next_chol, tf.matmul(A_t, P_filt_t))
            J_t = tf.linalg.matrix_transpose(J_transposed)

            x_next_smooth = x_smooth_ta.read(t + 1)
            P_next_smooth = P_smooth_ta.read(t + 1)

            dx = x_next_smooth - tf.expand_dims(x_pred[:, t + 1, :], -1)
            x_curr = tf.expand_dims(x_filt[:, t, :], -1) + tf.matmul(J_t, dx)

            dP = P_next_smooth - P_pred_next
            P_curr = P_filt_t + tf.matmul(J_t, tf.matmul(dP, J_t, transpose_b=True))
            P_curr = 0.5 * (P_curr + tf.linalg.matrix_transpose(P_curr))

            P_cross_val = tf.matmul(P_next_smooth, J_t, transpose_b=True)

            x_smooth_ta = x_smooth_ta.write(t, x_curr)
            P_smooth_ta = P_smooth_ta.write(t, P_curr)
            P_cross_ta = P_cross_ta.write(t, P_cross_val)

        x_smooth = tf.squeeze(tf.transpose(x_smooth_ta.stack(), perm=[1, 0, 2, 3]), axis=-1)
        P_smooth = tf.transpose(P_smooth_ta.stack(), perm=[1, 0, 2, 3])
        P_cross = tf.transpose(P_cross_ta.stack(), perm=[1, 0, 2, 3])

        return x_smooth, P_smooth, P_cross, log_l

    def fit(self, Y: tf.Tensor, n_iter: int = 10, **kwargs):
        """EM Solver for Additive Noise EKF."""
        if len(Y.shape) == 2 and self.model.obs_dim == 1:
            Y = tf.expand_dims(Y, axis=-1)

        batch_size = tf.shape(Y)[0]
        T = tf.shape(Y)[1]
        state_dim = self.model.state_dim
        obs_dim = self.model.obs_dim

        total_samples = tf.cast(batch_size * T, dtype)
        total_transitions = tf.cast(batch_size * (T - 1), dtype)

        init_Q_std = self._get_mean(self.model.process_noise.stddev(), state_dim)
        init_R_std = self._get_mean(self.model.observation_noise.stddev(), obs_dim)

        Q_var = tf.Variable(init_Q_std, dtype=dtype)
        R_var = tf.Variable(init_R_std, dtype=dtype)

        self.model.process_noise = tfd.Normal(loc=tf.zeros(state_dim), scale=Q_var)
        self.model.observation_noise = tfd.Normal(loc=tf.zeros(obs_dim), scale=R_var)

        log_likelihoods = []
        for i in range(n_iter):
            x_smooth, P_smooth, P_cross, log_L = self.smooth(Y)
            mean_log_L = float(tf.reduce_mean(log_L).numpy())
            log_likelihoods.append(mean_log_L)

            if i >= 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-3:
                print(f'EM converged at iteration {i}.')
                break

            y_flat = tf.reshape(Y, [-1, obs_dim])
            x_flat = tf.reshape(x_smooth, [-1, state_dim])
            P_flat = tf.reshape(P_smooth, [-1, state_dim, state_dim])

            # Update R
            h_val, H = self._batch_linearize(self.model.observation_fn, x_flat)
            res_y = tf.expand_dims(y_flat - h_val, -1)

            # term_R calculation naturally assumes additive noise!
            term_R = tf.matmul(res_y, res_y, transpose_b=True) + tf.matmul(H, tf.matmul(P_flat, H, transpose_b=True))
            new_R = tf.reduce_sum(term_R, axis=0) / total_samples

            # Update Q
            x_curr = x_smooth[:, :-1, :]
            x_next = x_smooth[:, 1:, :]
            P_curr = P_smooth[:, :-1, :, :]
            P_next = P_smooth[:, 1:, :, :]
            P_cross_flat = tf.reshape(P_cross, [-1, state_dim, state_dim])

            x_curr_flat = tf.reshape(x_curr, [-1, state_dim])
            x_next_flat = tf.reshape(x_next, [-1, state_dim])
            P_curr_flat = tf.reshape(P_curr, [-1, state_dim, state_dim])
            P_next_flat = tf.reshape(P_next, [-1, state_dim, state_dim])

            f_val, A = self._batch_linearize(self.model.transition_fn, x_curr_flat)
            res_x = tf.expand_dims(x_next_flat - f_val, -1)

            # term_Q calculation naturally assumes additive noise!
            term_Q = (tf.matmul(res_x, res_x, transpose_b=True) +
                      P_next_flat +
                      tf.matmul(A, tf.matmul(P_curr_flat, A, transpose_b=True)) -
                      tf.matmul(P_cross_flat, A, transpose_b=True) -
                      tf.matmul(A, P_cross_flat, transpose_b=True))
            new_Q = tf.reduce_sum(term_Q, axis=0) / total_transitions

            Q_diag = tf.maximum(tf.linalg.diag_part(new_Q), 1e-6)
            R_diag = tf.maximum(tf.linalg.diag_part(new_R), 1e-6)

            Q_var.assign(tf.sqrt(Q_diag))
            R_var.assign(tf.sqrt(R_diag))

            if i % 10 == 0:
                print(f"Iter {i}: Log-Likelihood={mean_log_L:.4f}")

        return log_likelihoods


class UnscentedKalmanFilter(BaseFilter, tf.Module):
    """
    Unscented Kalman Filter (UKF) for Non-Linear State-Space Models.

    Inherits from BaseFilter. Uses the Unscented Transform (UT) to propagate
    mean and covariance through non-linear functions.
    """

    def __init__(self, model: NLSSM, alpha=1e-3, beta=2.0, kappa=0.0, train_noise=False):
        BaseFilter.__init__(self, model)
        tf.Module.__init__(self)

        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim

        # Core UKF Hyperparameters (Learnable)
        self.alpha = tf.Variable(alpha, dtype=dtype, name='ukf_alpha')
        self.beta = tf.Variable(beta, dtype=dtype, name='ukf_beta')
        self.kappa = tf.Variable(kappa, dtype=dtype, name='ukf_kappa')

        # Noise Parameters (Learnable)
        self.train_noise = train_noise
        self.proc_log_scale = self._create_log_scale(model.process_noise, self.state_dim, 'proc')
        self.obs_log_scale = self._create_log_scale(model.observation_noise, self.obs_dim, 'obs')

        # NOTE: We DO NOT compute self.Wm, self.Wc, self.lam here anymore!
        # Computing them here hides them from tf.GradientTape during the .fit() loop.

    def _create_log_scale(self, dist, dim, name):
        """Helper to safely create trainable log-scale variables, bypassing LinearOperator errors."""
        if not self.train_noise: return None

        try:
            val = dist.stddev()
        except (AttributeError, NotImplementedError):
            try:
                val = tf.sqrt(tf.linalg.diag_part(dist.covariance()))
            except (AttributeError, NotImplementedError):
                val = tf.ones([dim], dtype=dtype)

        val = tf.convert_to_tensor(val, dtype=dtype)
        if len(val.shape) == 0:
            val = tf.fill([dim], val)

        return tf.Variable(tf.math.log(val + 1e-6), name=f'{name}_log_scale')

    def _compute_weights(self):
        """
        Dynamically computes UT weights.
        Because this is called inside the forward pass (via _init_state),
        tf.GradientTape can track the exact mathematical relationship
        between the trainable parameters (alpha, beta, kappa) and the outputs.
        """
        n_float = tf.cast(self.state_dim, dtype)
        lam = (self.alpha ** 2) * (n_float + self.kappa) - n_float

        Wm_0 = lam / (n_float + lam)
        Wm_rest = 0.5 / (n_float + lam)
        # Multiplying by tf.ones ensures the gradient flows cleanly through the vector creation
        Wm_rest_vec = tf.ones([2 * self.state_dim], dtype=dtype) * Wm_rest
        Wm = tf.concat([[Wm_0], Wm_rest_vec], axis=0)

        Wc_0 = Wm_0 + (1.0 - self.alpha ** 2 + self.beta)
        Wc = tf.concat([[Wc_0], Wm_rest_vec], axis=0)
        return Wm, Wc, lam

    def _init_state(self, batch_size: int) -> tuple:
        """
        Initializes the tracking state at t=0.

        DESIGN CHOICE: The "Opaque State" Tuple
        To avoid recomputing the static UT weights at every single time step,
        we compute them EXACTLY ONCE here at the beginning of the sequence.
        We then pack them into the state tuple. Because BaseFilter treats the
        state tuple as opaque, it simply passes these weights along the time
        loop to `predict` and `update` without any extra overhead.
        """
        x_init = tf.tile(tf.reshape(self.model.x0, [1, self.state_dim]), [batch_size, 1])
        P_init = tf.tile(tf.expand_dims(self._get_cov(self.model.init_noise), 0), [batch_size, 1, 1])

        # 1. Compute weights dynamically ONCE per sequence run
        Wm, Wc, lam = self._compute_weights()

        # 2. Pack them into the 7-tuple state
        return (x_init, P_init, x_init, P_init, Wm, Wc, lam)

    def _init_trajectory(self, time_steps: int) -> tuple:
        # BaseFilter only needs history for the actual tracking states, not the static weights
        x_filt_ta = tf.TensorArray(dtype, size=time_steps)
        P_filt_ta = tf.TensorArray(dtype, size=time_steps)
        x_pred_ta = tf.TensorArray(dtype, size=time_steps)
        P_pred_ta = tf.TensorArray(dtype, size=time_steps)
        logl_ta = tf.TensorArray(dtype, size=time_steps)
        return (x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, logl_ta)

    def _generate_sigma_points(self, x, P, lam):
        n = self.state_dim
        scale = tf.sqrt(tf.cast(n, dtype) + lam)
        P_sym = 0.5 * (P + tf.linalg.matrix_transpose(P)) + 1e-6 * tf.eye(n, dtype=dtype)
        L = tf.linalg.cholesky(P_sym)
        scaled_L = scale * L

        x_expanded = tf.expand_dims(x, -1)
        right = x_expanded + scaled_L
        left = x_expanded - scaled_L
        sigmas_concat = tf.concat([x_expanded, right, left], axis=2)
        return tf.transpose(sigmas_concat, perm=[0, 2, 1])

    def _compute_stats(self, sigma_points, Wm, Wc, noise_cov):
        x_mean = tf.tensordot(sigma_points, Wm, axes=[[1], [0]])
        residuals = sigma_points - tf.expand_dims(x_mean, 1)
        Wc_b = tf.reshape(Wc, [1, -1, 1])
        weighted_res = residuals * Wc_b
        P = tf.matmul(tf.transpose(weighted_res, perm=[0, 2, 1]), residuals) + noise_cov
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))
        return x_mean, P, residuals

    def predict(self, t: int, state: tuple) -> tuple:
        # Unpack the dynamically computed weights passed from the previous step
        x_filt_prev, P_filt_prev, _, _, Wm, Wc, lam = state

        Q = tf.expand_dims(self._get_cov(self.model.process_noise, self.proc_log_scale), 0)
        q_mean = self._get_mean(self.model.process_noise, self.state_dim)

        # Use the weights normally
        sig_pts = self._generate_sigma_points(x_filt_prev, P_filt_prev, lam)
        batch_size = tf.shape(sig_pts)[0]
        num_sig = tf.shape(sig_pts)[1]

        sig_pts_flat = tf.reshape(sig_pts, [batch_size * num_sig, self.state_dim])
        sig_pts_prop_flat = self.model.transition_fn(sig_pts_flat, q_mean)
        sig_pts_prop = tf.reshape(sig_pts_prop_flat, [batch_size, num_sig, self.state_dim])

        x_pred, P_pred, _ = self._compute_stats(sig_pts_prop, Wm, Wc, Q)
        P_pred = 0.5 * (P_pred + tf.linalg.matrix_transpose(P_pred))

        # Re-pack the weights into the 7-tuple so `update` can use them
        return (x_pred, P_pred, x_pred, P_pred, Wm, Wc, lam)

    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        # Unpack the dynamically computed weights passed from the predict step
        _, _, x_pred, P_pred, Wm, Wc, lam = state

        R = tf.expand_dims(self._get_cov(self.model.observation_noise, self.obs_log_scale), 0)
        r_mean = self._get_mean(self.model.observation_noise, self.obs_dim)

        # Use the weights normally
        sig_pts_pred = self._generate_sigma_points(x_pred, P_pred, lam)
        batch_size = tf.shape(sig_pts_pred)[0]
        num_sig = tf.shape(sig_pts_pred)[1]

        sig_pts_pred_flat = tf.reshape(sig_pts_pred, [batch_size * num_sig, self.state_dim])
        sig_pts_obs_flat = self.model.observation_fn(sig_pts_pred_flat, r_mean)
        sig_pts_obs = tf.reshape(sig_pts_obs_flat, [batch_size, num_sig, self.obs_dim])

        y_pred_mean, S, y_residuals = self._compute_stats(sig_pts_obs, Wm, Wc, R)
        x_residuals = sig_pts_pred - tf.expand_dims(x_pred, 1)
        Wc_b = tf.reshape(Wc, [1, -1, 1])
        weighted_x_res = x_residuals * Wc_b

        P_xy = tf.matmul(tf.transpose(weighted_x_res, perm=[0, 2, 1]), y_residuals)
        S_chol = tf.linalg.cholesky(S + 1e-6 * tf.eye(self.obs_dim, dtype=dtype))

        Kt_transposed = tf.linalg.cholesky_solve(S_chol, tf.linalg.matrix_transpose(P_xy))
        K = tf.linalg.matrix_transpose(Kt_transposed)

        innovation = observation - y_pred_mean
        innovation_expanded = tf.expand_dims(innovation, -1)

        x_new = x_pred + tf.squeeze(tf.matmul(K, innovation_expanded), -1)
        KSKt = tf.matmul(K, tf.matmul(S, K, transpose_b=True))
        P_new = P_pred - KSKt
        P_new = 0.5 * (P_new + tf.linalg.matrix_transpose(P_new))

        const_term = -0.5 * float(self.obs_dim) * tf.math.log(2 * math.pi)
        diag_S = tf.maximum(tf.linalg.diag_part(S_chol), 1e-6)
        log_det_S = 2 * tf.reduce_sum(tf.math.log(diag_S), axis=1)
        sol = tf.linalg.cholesky_solve(S_chol, innovation_expanded)
        quad_term = tf.squeeze(tf.matmul(tf.transpose(innovation_expanded, perm=[0, 2, 1]), sol), [1, 2])
        log_l = const_term - 0.5 * log_det_S - 0.5 * quad_term

        # Re-pack the weights into the 7-tuple for the next time step t+1
        return (x_new, P_new, x_pred, P_pred, Wm, Wc, lam), (log_l,)

    def forecast(self, observations: tf.Tensor) -> tuple:
        res = self.filter(observations)
        x_filt_last = res['x_filt'][:, -1, :]
        P_filt_last = res['P_filt'][:, -1, :, :]

        # Need weights locally to push the final state into T+1
        Wm, Wc, lam = self._compute_weights()

        state_last = (x_filt_last, P_filt_last, None, None, Wm, Wc, lam)
        x_pred_next, P_pred_next, _, _, _, _, _ = self.predict(0, state_last)

        R = tf.expand_dims(self._get_cov(self.model.observation_noise, self.obs_log_scale), 0)
        r_mean = self._get_mean(self.model.observation_noise, self.obs_dim)

        sig_pts_pred = self._generate_sigma_points(x_pred_next, P_pred_next, lam)
        batch_size, num_sig = tf.shape(sig_pts_pred)[0], tf.shape(sig_pts_pred)[1]

        sig_pts_pred_flat = tf.reshape(sig_pts_pred, [batch_size * num_sig, self.state_dim])
        sig_pts_obs_flat = self.model.observation_fn(sig_pts_pred_flat, r_mean)
        sig_pts_obs = tf.reshape(sig_pts_obs_flat, [batch_size, num_sig, self.obs_dim])

        y_pred_next, S_next, _ = self._compute_stats(sig_pts_obs, Wm, Wc, R)

        return y_pred_next, S_next

    def _write_trajectory(self, t: int, trajectory: tuple, state: tuple, metrics: tuple) -> tuple:
        x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, logl_ta = trajectory

        # We unpack the 7-tuple but ignore the UT weights (Wm, Wc, lam).
        # We do not need to save the static weights to the time-series history arrays.
        x_filt, P_filt, x_pred, P_pred, _, _, _ = state
        log_l = metrics[0]

        return (
            x_filt_ta.write(t, x_filt), P_filt_ta.write(t, P_filt),
            x_pred_ta.write(t, x_pred), P_pred_ta.write(t, P_pred),
            logl_ta.write(t, log_l)
        )

    def _format_output(self, trajectory: tuple) -> dict:
        x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, logl_ta = trajectory

        return {
            "x_filt": tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),
            "P_filt": tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),
            "x_pred": tf.transpose(x_pred_ta.stack(), perm=[1, 0, 2]),
            "P_pred": tf.transpose(P_pred_ta.stack(), perm=[1, 0, 2, 3]),
            "log_likelihood": tf.reduce_sum(tf.transpose(logl_ta.stack(), perm=[1, 0]), axis=-1)
        }

    def fit(self, Y: tf.Tensor, n_iter=100, learning_rate=0.01):
        """Backpropagation through time (BPTT) using BaseFilter loop."""
        Y = tf.convert_to_tensor(Y, dtype=dtype)
        if len(Y.shape) == 2: Y = tf.expand_dims(Y, -1)

        trainable_vars = [self.alpha, self.beta, self.kappa]
        if self.proc_log_scale is not None: trainable_vars.append(self.proc_log_scale)
        if self.obs_log_scale is not None: trainable_vars.append(self.obs_log_scale)

        optimizer = tf.optimizers.Adam(learning_rate)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                results = self.filter(Y)
                loss = -tf.reduce_mean(results['log_likelihood'])
            grads = tape.gradient(loss, trainable_vars)

            # SAFEGUARD: Gracefully ignore parameters that return a None gradient.
            # This prevents tf.clip_by_norm from crashing if a parameter path is detached.
            valid_grads = []
            valid_vars = []
            for g, v in zip(grads, trainable_vars):
                if g is not None:
                    valid_grads.append(tf.clip_by_norm(g, 1.0))
                    valid_vars.append(v)

            optimizer.apply_gradients(zip(valid_grads, valid_vars))
            return loss

        losses = []
        for i in range(n_iter):
            loss_tensor = train_step()
            scalar_loss = float(loss_tensor.numpy().item() if loss_tensor.ndim == 0 else loss_tensor.numpy().mean())
            losses.append(scalar_loss)
            if i % 10 == 0: print(f"Iter {i}: Loss={scalar_loss:.4f}")

        self.sync_model()
        return losses

    def sync_model(self):
        """Updates the internal NLSSM model object with the learned noise parameters."""
        if self.proc_log_scale is not None:
            learned_std = tf.exp(self.proc_log_scale)
            self.model.process_noise = tfd.Normal(loc=tf.zeros_like(learned_std), scale=learned_std)
        if self.obs_log_scale is not None:
            learned_std = tf.exp(self.obs_log_scale)
            self.model.observation_noise = tfd.Normal(loc=tf.zeros_like(learned_std), scale=learned_std)