import tensorflow as tf
import tensorflow_probability as tfp
import math
from Filters.basic_filters.base_filter import BaseFilter
from models.base_models import LGSSM

tfd = tfp.distributions
dtype = tf.float32

class KalmanFilter(BaseFilter):
    """
    Standard Kalman Filter for Linear Gaussian State-Space Models (LGSSM).

    Inherits from BaseFilter to utilize the generic time-series tracking loop.
    The tracking state is defined as a 4-tuple to retain both prior and posterior:
    (x_filtered, P_filtered, x_predicted, P_predicted).
    """

    def __init__(self, model: LGSSM, requires_stabilization: bool = True):
        super().__init__(model)
        self.requires_stabilization = requires_stabilization

    def _init_state(self, batch_size: int) -> tuple:
        """Initializes the tracking state at t=0."""
        state_dim = self.model.state_dim

        # Ensure x0 and P0 have correct rank before expanding and tiling
        x0_reshaped = tf.reshape(self.model.x0, [state_dim, 1])
        P0_reshaped = tf.reshape(self.model.P0, [state_dim, state_dim])

        x_init = tf.tile(tf.expand_dims(x0_reshaped, 0), [batch_size, 1, 1])
        P_init = tf.tile(tf.expand_dims(P0_reshaped, 0), [batch_size, 1, 1])

        # At t=0 before any transitions, prior == posterior
        return (x_init, P_init, x_init, P_init)

    def _init_trajectory(self, time_steps: int) -> tuple:
        """Initializes TensorArrays to store the trajectory."""
        x_filt_ta = tf.TensorArray(dtype, size=time_steps)
        P_filt_ta = tf.TensorArray(dtype, size=time_steps)
        x_pred_ta = tf.TensorArray(dtype, size=time_steps)
        P_pred_ta = tf.TensorArray(dtype, size=time_steps)
        logl_ta = tf.TensorArray(dtype, size=time_steps)

        return (x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, logl_ta)

    def predict(self, t: int, state: tuple) -> tuple:
        """
        Propagates the state forward in time.
        x_t_pred = A * x_{t-1}_filt
        P_t_pred = A * P_{t-1}_filt * A^T + Q
        """
        x_filt, P_filt, _, _ = state

        A = tf.expand_dims(self.model.A, 0)
        Q = tf.expand_dims(self.model.Q, 0)

        x_pred = tf.matmul(A, x_filt)
        P_pred = tf.matmul(A, tf.matmul(P_filt, A, transpose_b=True)) + Q

        # Return a 4-tuple to satisfy tf.cond shape requirements.
        return (x_pred, P_pred, x_pred, P_pred)

    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        """
        Incorporates the observation to correct the state.
        Returns the new 4-tuple state and the log-likelihood metric.
        """
        _, _, x_pred, P_pred = state

        obs_dim = self.model.obs_dim
        state_dim = self.model.state_dim

        # Format observation
        y_t = tf.expand_dims(observation, -1)  # [B, O, 1]

        C = tf.expand_dims(self.model.C, 0)
        R = tf.expand_dims(self.model.R, 0)
        I = tf.eye(state_dim, dtype=dtype)

        # Innovation
        innov = y_t - tf.matmul(C, x_pred)
        PCt = tf.matmul(P_pred, C, transpose_b=True)
        S_t = tf.matmul(C, PCt) + R + 1e-6 * tf.eye(obs_dim, dtype=dtype)

        S_chol = tf.linalg.cholesky(S_t)

        # Kalman Gain: K_t = P_t_pred * C^T * S_t^{-1}
        Kt_transposed = tf.linalg.cholesky_solve(S_chol, tf.transpose(PCt, perm=[0, 2, 1]))
        K_t = tf.transpose(Kt_transposed, perm=[0, 2, 1])

        # Update Mean
        x_filt = x_pred + tf.matmul(K_t, innov)

        # Update Covariance
        I_KC = I - tf.matmul(K_t, C)
        if self.requires_stabilization:
            # Joseph form: P_t_filt = (I - K_t*C)*P_t_pred*(I - K_t*C)^T + K_t*R*K_t^T
            term1 = tf.matmul(I_KC, tf.matmul(P_pred, I_KC, transpose_b=True))
            term2 = tf.matmul(K_t, tf.matmul(R, K_t, transpose_b=True))
            P_filt = term1 + term2
        else:
            # Standard form: P_t_filt = (I - K_t*C)*P_t_pred
            P_filt = tf.matmul(I_KC, P_pred)

        # Log Likelihood Calculation
        log_det_S = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(S_chol)), axis=-1)
        quad_term = tf.squeeze(
            tf.matmul(tf.transpose(innov, perm=[0, 2, 1]), tf.linalg.cholesky_solve(S_chol, innov)),
            axis=[-2, -1]
        )
        const_term = -0.5 * obs_dim * tf.math.log(2 * math.pi)
        log_l = const_term - 0.5 * log_det_S - 0.5 * quad_term

        # Re-pack the 4-tuple state to pass to the next iteration
        new_state = (x_filt, P_filt, x_pred, P_pred)
        return new_state, (log_l,)

    def _write_trajectory(self, t: int, trajectory: tuple, state: tuple, metrics: tuple) -> tuple:
        """Writes the current state and metrics into the TensorArrays."""
        x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, logl_ta = trajectory
        x_filt, P_filt, x_pred, P_pred = state
        log_l = metrics[0]

        return (
            x_filt_ta.write(t, x_filt), P_filt_ta.write(t, P_filt),
            x_pred_ta.write(t, x_pred), P_pred_ta.write(t, P_pred),
            logl_ta.write(t, log_l)
        )

    def _format_output(self, trajectory: tuple) -> dict:
        """Stacks the TensorArrays and returns a clean dictionary."""
        x_filt_ta, P_filt_ta, x_pred_ta, P_pred_ta, logl_ta = trajectory

        # Stack and permute back to [Batch, Time, ...]
        x_filt = tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2, 3])
        P_filt = tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3])
        x_pred = tf.transpose(x_pred_ta.stack(), perm=[1, 0, 2, 3])
        P_pred = tf.transpose(P_pred_ta.stack(), perm=[1, 0, 2, 3])
        log_l = tf.transpose(logl_ta.stack(), perm=[1, 0])

        # Squeeze the trailing 1-dimension off the state vectors [B, T, S]
        x_filt = tf.squeeze(x_filt, axis=-1)
        x_pred = tf.squeeze(x_pred, axis=-1)

        # Accumulate log-likelihood across time [B]
        log_l_sum = tf.reduce_sum(log_l, axis=-1)

        return {
            "x_filt": x_filt,
            "P_filt": P_filt,
            "x_pred": x_pred,
            "P_pred": P_pred,
            "log_likelihood": log_l_sum
        }

    def forecast(self, observations: tf.Tensor) -> tuple:
        """
        Generates the forecast of the next observation based on the provided trajectory.
        """
        res = self.filter(observations)
        x_filt_last = tf.expand_dims(res['x_filt'][:, -1, :], -1)  # [B, Dx, 1]
        P_filt_last = res['P_filt'][:, -1, :, :]

        A = tf.expand_dims(self.model.A, 0)
        C = tf.expand_dims(self.model.C, 0)
        Q = tf.expand_dims(self.model.Q, 0)
        R = tf.expand_dims(self.model.R, 0)

        # Propagate to T+1
        x_pred_next = tf.matmul(A, x_filt_last)
        P_pred_next = tf.matmul(A, tf.matmul(P_filt_last, A, transpose_b=True)) + Q

        # Project into observation space
        y_pred_next = tf.squeeze(tf.matmul(C, x_pred_next), -1)
        S_next = tf.matmul(C, tf.matmul(P_pred_next, C, transpose_b=True)) + R

        return y_pred_next, S_next

    @tf.function
    def smooth(self, Y: tf.Tensor):
        """
        Batched RTS Smoother.
        Renamed from 'smooth_filter' to 'smooth' to match EKF naming conventions.
        """
        res = self.filter(Y)
        x_filt_in, P_filt_in = res['x_filt'], res['P_filt']
        x_pred_in, P_pred_in = res['x_pred'], res['P_pred']
        log_l = res['log_likelihood']

        batch_size = tf.shape(x_filt_in)[0]
        T = tf.shape(x_filt_in)[1]
        state_dim = self.model.state_dim
        A = tf.expand_dims(self.model.A, 0)

        x_smooth_ta = tf.TensorArray(dtype, size=T, clear_after_read=False)
        P_smooth_ta = tf.TensorArray(dtype, size=T, clear_after_read=False)
        J_ta = tf.TensorArray(dtype, size=tf.maximum(1, T - 1))

        # Initialize at T-1
        x_last = tf.expand_dims(x_filt_in[:, T - 1, :], -1)
        P_last = P_filt_in[:, T - 1, :, :]

        x_smooth_ta = x_smooth_ta.write(T - 1, x_last)
        P_smooth_ta = P_smooth_ta.write(T - 1, P_last)

        x_smooth_next = x_last
        P_smooth_next = P_last

        # Backward RTS Loop
        for t in tf.range(T - 2, -1, -1):
            P_filt_t = P_filt_in[:, t, :, :]
            P_pred_next = P_pred_in[:, t + 1, :, :]
            x_pred_next = tf.expand_dims(x_pred_in[:, t + 1, :], -1)
            x_filt_t = tf.expand_dims(x_filt_in[:, t, :], -1)

            P_pred_next_chol = tf.linalg.cholesky(P_pred_next + 1e-6 * tf.eye(state_dim))
            rhs = tf.matmul(A, P_filt_t)

            J_t_T = tf.linalg.cholesky_solve(P_pred_next_chol, rhs)
            J_t = tf.transpose(J_t_T, perm=[0, 2, 1])

            dx = x_smooth_next - x_pred_next
            x_smooth_t = x_filt_t + tf.matmul(J_t, dx)

            dP = P_smooth_next - P_pred_next
            P_smooth_t = P_filt_t + tf.matmul(J_t, tf.matmul(dP, J_t, transpose_b=True))

            x_smooth_next = x_smooth_t
            P_smooth_next = P_smooth_t

            x_smooth_ta = x_smooth_ta.write(t, x_smooth_t)
            P_smooth_ta = P_smooth_ta.write(t, P_smooth_t)
            J_ta = J_ta.write(t, J_t)

        x_smooth = tf.squeeze(tf.transpose(x_smooth_ta.stack(), perm=[1, 0, 2, 3]), axis=-1)
        P_smooth = tf.transpose(P_smooth_ta.stack(), perm=[1, 0, 2, 3])
        J_ts = tf.transpose(J_ta.stack(), perm=[1, 0, 2, 3])

        return x_smooth, P_smooth, J_ts, log_l

    def fit(self, Y: tf.Tensor, n_iter: int = 10, tol: float = 1e-3, **kwargs):
        """
        Unified API for Expectation-Maximization (EM) solver.
        Estimates the parameters (A, C, Q, R, x0, P0) of the LGSSM from data.
        """
        T = int(tf.shape(Y)[1])
        list_log_l = []

        for i in range(n_iter):
            # E-step: run Kalman smoother to get expected sufficient statistics
            X_s, P_s, J_s, log_L = self.smooth(Y)

            # Aggregate batch log-likelihoods
            total_log_L = tf.reduce_mean(log_L)
            scalar_log_L = float(total_log_L.numpy().item() if total_log_L.ndim == 0 else total_log_L.numpy().mean())
            print(f'EM Iteration {i}, Log Likelihood: {scalar_log_L:.4f}')
            list_log_l.append(scalar_log_L)
            if len(list_log_l) > 1 and abs(list_log_l[-1] - list_log_l[-2]) < tol:
                self.model.update_cholesky()
                print(f'EM converged at iteration {i}')
                break

            Exx = P_s + tf.matmul(tf.expand_dims(X_s, -1), tf.expand_dims(X_s, -2))
            Exx1 = tf.matmul(P_s[:, 1:, :, :], J_s, transpose_b=True) + \
                   tf.matmul(tf.expand_dims(X_s[:, 1:, :], -1), tf.expand_dims(X_s[:, :-1, :], -2))

            Sigma_xx = tf.reduce_sum(Exx[:, :-1, :, :], axis=[0, 1])
            Gamma_xx = Sigma_xx + tf.reduce_sum(Exx[:, -1, :, :], axis=0)
            Sigma_x1x1 = Gamma_xx - tf.reduce_sum(Exx[:, 0, :, :], axis=0)
            Sigma_xx1 = tf.reduce_sum(Exx1, axis=[0, 1])
            Gamma_yy = tf.reduce_sum(tf.matmul(tf.expand_dims(Y, -1), tf.expand_dims(Y, -2)), axis=[0, 1])
            Gamma_yx = tf.reduce_sum(tf.matmul(tf.expand_dims(Y, -1), tf.expand_dims(X_s, -2)), axis=[0, 1])

            # M-step: update model parameters using the expected sufficient statistics
            Sigma_xx_chol = tf.linalg.cholesky(Sigma_xx + 1e-6 * tf.eye(self.model.state_dim, dtype=Sigma_xx.dtype))
            Sigma_x1x1_chol = tf.linalg.cholesky(Sigma_x1x1 + 1e-6 * tf.eye(self.model.state_dim, dtype=Sigma_x1x1.dtype))

            A_new = tf.transpose(tf.linalg.cholesky_solve(Sigma_x1x1_chol, tf.transpose(Sigma_xx1)))
            C_new = tf.transpose(tf.linalg.cholesky_solve(Sigma_xx_chol, tf.transpose(Gamma_yx)))

            R_term = Gamma_yy - C_new @ tf.transpose(Gamma_yx) - Gamma_yx @ tf.transpose(C_new) + C_new @ Gamma_xx @ tf.transpose(C_new)
            R_new = R_term / (tf.cast(tf.shape(Y)[0] * T, dtype))
            R_new = 0.5 * (R_new + tf.transpose(R_new)) + 1e-6 * tf.eye(self.model.obs_dim)

            Q_term = Sigma_x1x1 - A_new @ tf.transpose(Sigma_xx1) - Sigma_xx1 @ tf.transpose(A_new) + A_new @ Sigma_xx @ tf.transpose(A_new)
            Q_new = Q_term / (tf.cast(tf.shape(Y)[0] * (T - 1), dtype))
            Q_new = 0.5 * (Q_new + tf.transpose(Q_new)) + 1e-6 * tf.eye(self.model.state_dim)

            x0_new = tf.reduce_mean(X_s[:, 0, :], axis=0, keepdims=True)
            centered_x0 = X_s[:, 0, :] - x0_new
            cov_means = tf.matmul(tf.expand_dims(centered_x0, -1), tf.expand_dims(centered_x0, -2))

            P0_new = tf.reduce_mean(P_s[:, 0, :, :] + cov_means, axis=0)
            P0_new = 0.5 * (P0_new + tf.transpose(P0_new)) + 1e-6 * tf.eye(self.model.state_dim)

            self.model.set_params([A_new, C_new, Q_new, R_new, tf.transpose(x0_new), P0_new])

        self.model.update_cholesky()
        return list_log_l


def EM_initializer(Y, state_dim):
    """
    Initializes a Kalman Filter using PCA on the observations.
    """
    Y_flat = tf.reshape(Y, [-1, Y.shape[-1]])
    S, U, V = tf.linalg.svd(Y_flat - tf.reduce_mean(Y_flat, axis=0, keepdims=True), full_matrices=False)

    top_s = tf.linalg.diag(tf.sqrt(S[:state_dim]))
    top_v = V[:, :state_dim]
    C_init = tf.matmul(top_v, top_s)

    residuals = Y_flat @ (tf.eye(top_v.shape[0], dtype=Y.dtype) - tf.matmul(top_v, tf.transpose(top_v)))
    residual_variance = tf.math.reduce_variance(residuals, axis=0)
    R_init = tf.linalg.diag(residual_variance + 1e-6)

    x0_init = tf.reshape(tf.reduce_mean(Y, axis=[0, 1], keepdims=True), (1, -1)) @ top_v
    P0_init = tf.eye(state_dim, dtype=Y.dtype) * 1.0
    A_init = tf.eye(state_dim, dtype=Y.dtype) * 0.9
    Q_init = tf.eye(state_dim, dtype=Y.dtype) * 0.1

    model = LGSSM(state_dim, Y.shape[-1], params=[A_init, C_init, Q_init, R_init, tf.transpose(x0_init), P0_init])
    kf = KalmanFilter(model)
    return kf