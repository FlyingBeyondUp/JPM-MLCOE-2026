import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable
from codes.Filters.flow_filters.invertible_flow import InvertiblePFPF
from models import NLSSM
from codes.Filters.basic_filters import UnscentedKalmanFilter
from codes.Filters.flow_filters import EDHFlow, LEDHFlow


class OptimalInvertiblePFPF(InvertiblePFPF):
    """
    Integrates the stiffness mitigation optimal log-homotopy (Dai & Daum)
    into the Invertible Particle Flow Particle Filter (Li & Coates).
    """

    def __init__(self, model: NLSSM, num_particles: int,
                 ukf: UnscentedKalmanFilter,
                 flow_class: Callable = EDHFlow,
                 resample_method: str = 'systematic',
                 resample_threshold: float = 1.0,
                 stiffness_weight: float = 0.01,
                 bvp_grid_size: int = 100,
                 cfl_margin: float = 0.5):

        super().__init__(model, num_particles, ukf, flow_class,
                         1, resample_method, resample_threshold)
        self.mu = stiffness_weight
        self.bvp_grid_size = bvp_grid_size
        self.cfl_margin = cfl_margin

    def _get_obs_hessian(self, x: tf.Tensor):
        B = tf.shape(x)[0]
        with tf.GradientTape() as tape:
            tape.watch(x)
            zero_noise = tf.zeros((B, self.model.obs_dim))
            mean_y = self.model.observation_fn(x, zero_noise)

        H = tape.batch_jacobian(mean_y, x)

        try:
            R_cov = self.model.observation_noise.covariance()
        except (NotImplementedError, AttributeError):
            var = self.model.observation_noise.variance()
            if len(var.shape) == 0:
                R_cov = tf.reshape(var, [1, 1])
            else:
                R_cov = tf.linalg.diag(var)

        if len(R_cov.shape) == 1:
            R_cov = tf.linalg.diag(R_cov)

        R_inv = tf.linalg.pinv(R_cov)
        Hh = tf.linalg.matrix_transpose(H) @ R_inv[tf.newaxis, :, :] @ H
        return -Hh

    def _solve_optimal_path(self, P_xx: tf.Tensor, Hh: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        B = tf.shape(P_xx)[0]
        state_dim = tf.shape(P_xx)[-1]

        P_xx = 0.5 * (P_xx + tf.linalg.matrix_transpose(P_xx))
        M0 = tf.linalg.pinv(P_xx)
        Mh = -Hh

        def ode_fn(t, state):
            beta = state['beta']
            beta_dot = state['beta_dot']

            beta_clamped = tf.clip_by_value(beta, 0.0, 1.5)
            M = M0 + tf.reshape(beta_clamped, [-1, 1, 1]) * Mh
            M_inv = tf.linalg.pinv(M)

            tr_M = tf.linalg.trace(M)
            tr_Minv = tf.linalg.trace(M_inv)
            tr_Mh = tf.linalg.trace(Mh)

            M_inv_Mh = tf.matmul(M_inv, Mh)
            tr_Minv_Mh_Minv = tf.linalg.trace(tf.matmul(M_inv_Mh, M_inv))

            accel = self.mu * (tr_Mh * tr_Minv - tr_M * tr_Minv_Mh_Minv)
            return {'beta': beta_dot, 'beta_dot': accel}

        solver = tfp.math.ode.DormandPrince(rtol=1e-1, atol=1e-2)
        solution_times = tf.linspace(0.0, 1.0, self.bvp_grid_size)

        def integrate(u0):
            initial_state = {'beta': tf.zeros([B]), 'beta_dot': u0}
            results = solver.solve(ode_fn, initial_time=0.0, initial_state=initial_state,
                                   solution_times=solution_times)
            return results.states['beta'][-1], results.states['beta'], results.states['beta_dot']

        lower = tf.zeros([B])
        upper = tf.ones([B]) * 2.0

        for _ in tf.range(10):
            end_beta, _, _ = integrate(upper)
            if tf.reduce_all(end_beta > 1.0):
                break
            upper = upper * 2.0

        for _ in tf.range(20):
            mid = (lower + upper) / 2.0
            end_beta, _, _ = integrate(mid)
            mask = end_beta < 1.0
            lower = tf.where(mask, mid, lower)
            upper = tf.where(mask, upper, mid)

        final_u0 = (lower + upper) / 2.0
        _, betas, beta_dots = integrate(final_u0)

        # FIX 2: Prevent interpolation of negative betas from solver overshoot
        betas = tf.maximum(betas, 0.0)

        return tf.transpose(betas), tf.transpose(beta_dots)

    @tf.function
    def _flow_with_det(self, eta_0, Y_t, num_flow_steps_unused, P_guide=None, eta_aux=None, step_sizes=None) -> tuple[
        tf.Tensor, tf.Tensor]:
        B = tf.shape(eta_0)[0]
        N = self.num_particles
        D = self.model.state_dim
        is_edh = self.flow_algo.__class__.__name__ == "EDHFlow"

        has_nans = tf.reduce_any(tf.math.is_nan(P_guide)) | tf.reduce_any(tf.math.is_nan(eta_aux))

        # Branch 1: The Fallback if Upstream UKF failed
        def _fallback():
            return eta_0, tf.zeros([B, N], dtype=tf.float32)

        # Branch 2: The Healthy Optimal Flow
        def _compute_flow():
            # FIX 3: MatrixSolve Underflow Protection (Jitter)
            P_guide_safe = P_guide + 1e-4 * tf.eye(D, batch_shape=tf.shape(P_guide)[:-2])

            eta_aux_mean = tf.reduce_mean(eta_aux, axis=1) if not is_edh else eta_aux
            Hh = self._get_obs_hessian(eta_aux_mean)
            P_bvp = tf.reduce_mean(P_guide_safe, axis=1) if len(P_guide_safe.shape) == 4 else P_guide_safe

            if self.mu > 0.0:
                grid_betas, grid_beta_dots = self._solve_optimal_path(P_bvp, Hh)
            else:
                linear_betas = tf.linspace(0.0, 1.0, self.bvp_grid_size)
                grid_betas = tf.tile(tf.expand_dims(linear_betas, 0), [B, 1])
                grid_beta_dots = tf.ones_like(grid_betas)

            lam = tf.constant(0.0, dtype=tf.float32)
            beta = tf.zeros([B], dtype=tf.float32)
            current_particles = eta_0
            current_aux = tf.expand_dims(tf.identity(eta_aux), -1)
            eta_0_aux = tf.expand_dims(tf.identity(eta_aux), -1)
            log_det_jacobian = tf.zeros([B, N], dtype=tf.float32)

            base_dl = tf.constant(0.05, dtype=tf.float32)
            cfl = tf.constant(self.cfl_margin, dtype=tf.float32)

            def cond(lam_c, beta_c, parts, aux, log_det):
                return lam_c < 1.0

            def body(lam_c, beta_c, parts, aux, log_det):
                u_lam = tfp.math.interp_regular_1d_grid(
                    lam_c, x_ref_min=0.0, x_ref_max=1.0, y_ref=grid_beta_dots
                )

                # FIX 2: Strict enforcement of positive-definiteness for MatrixSolve
                beta_c_clamped = tf.maximum(beta_c, 0.0)

                if is_edh:
                    beta_bcast = tf.reshape(beta_c_clamped, [B, 1, 1])
                else:
                    beta_bcast = tf.reshape(beta_c_clamped, [B, 1, 1, 1])

                A, b = self.flow_algo.compute_flow_parameters(
                    parts, Y_t, beta_bcast, P_xx=P_guide_safe,
                    linearization_points=aux, eta_0_mean=eta_0_aux
                )

                abs_A = tf.abs(A)
                row_sums = tf.reduce_sum(abs_A, axis=-1)

                if is_edh:
                    norm_A = tf.reduce_max(row_sums, axis=-1)
                else:
                    norm_A = tf.reduce_max(row_sums, axis=[-2, -1])

                # FIX 1: Absolute Velocity Limit to prevent backwards time travel
                abs_u_lam = tf.abs(u_lam)
                max_dt_batch = cfl / (norm_A * abs_u_lam + 1e-6)

                safe_dt = tf.reduce_min(max_dt_batch)
                dt = tf.minimum(base_dl, safe_dt)
                dt = tf.minimum(dt, 1.0 - lam_c)

                if is_edh:
                    u_part = tf.reshape(u_lam, [B, 1, 1])
                    u_A = tf.reshape(u_lam, [B, 1, 1])
                    u_aux = tf.reshape(u_lam, [B, 1, 1])

                    drift_part = (tf.matmul(parts, A, transpose_b=True) + tf.expand_dims(b, 1)) * u_part
                    new_parts = parts + dt * drift_part

                    drift_aux = (tf.matmul(A, aux) + tf.expand_dims(b, -1)) * u_aux
                    new_aux = aux + dt * drift_aux

                    I = tf.eye(D, batch_shape=[B])
                    A_step = dt * u_A * A
                    _, step_log_det = tf.linalg.slogdet(I + A_step)
                    new_log_det = log_det + tf.expand_dims(step_log_det, 1)

                else:
                    u_part = tf.reshape(u_lam, [B, 1, 1])
                    u_A = tf.reshape(u_lam, [B, 1, 1, 1])
                    u_aux = tf.reshape(u_lam, [B, 1, 1, 1])

                    x_expanded = tf.expand_dims(parts, -1)
                    b_expanded = tf.expand_dims(b, -1)

                    drift_part = tf.squeeze(tf.matmul(A, x_expanded) + b_expanded, -1) * u_part
                    new_parts = parts + dt * drift_part

                    drift_aux = (tf.matmul(A, aux) + b_expanded) * u_aux
                    new_aux = aux + dt * drift_aux

                    I = tf.eye(D, batch_shape=[B, N])
                    A_step = dt * u_A * A
                    _, step_log_det = tf.linalg.slogdet(I + A_step)
                    new_log_det = log_det + step_log_det

                new_lam = lam_c + dt
                new_beta = tfp.math.interp_regular_1d_grid(
                    new_lam, x_ref_min=0.0, x_ref_max=1.0, y_ref=grid_betas
                )

                return new_lam, new_beta, new_parts, new_aux, new_log_det

            final_lam, final_beta, final_particles, final_aux, final_log_det = tf.while_loop(
                cond, body,
                loop_vars=(lam, beta, current_particles, current_aux, log_det_jacobian),
                maximum_iterations=200,
                shape_invariants=(
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, N, D]),
                    tf.TensorShape([None, D, 1]) if is_edh else tf.TensorShape([None, N, D, 1]),
                    tf.TensorShape([None, N])
                )
            )
            return final_particles, final_log_det

        # Use tf.cond to evaluate branches cleanly, bypassing AutoGraph variable tracking
        return tf.cond(has_nans, _fallback, _compute_flow)