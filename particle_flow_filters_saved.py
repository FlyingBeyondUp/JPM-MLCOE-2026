import tensorflow as tf
import tensorflow_probability as tfp
from nonlinearSSM import NLSSM, get1DLogSquaredSVM, UnscentedKalmanFilter
from typing import Callable
from particle_filter import ParticleFilter

tfd = tfp.distributions


class EDHFlow:
    '''Implements the Exact Daum-Huang (EDH) particle flow for nonlinear state-space models.'''

    def __init__(self, model: NLSSM, num_particles: int,ukf: UnscentedKalmanFilter = None):
        self.model = model
        self.num_particles = num_particles
        self.R = self._get_cov(self.model.observation_noise)
        self.ukf=ukf

    def _initialize(self, batch_size: int = 1) -> tf.Tensor:
        # Initialize particles from the initial distribution
        total_particles = batch_size * self.num_particles
        noise = self.model.init_noise.sample(total_particles)
        if len(noise.shape) == 1:
            # ensure noise has shape [num_particles*batch_size, state_dim]
            noise = tf.expand_dims(noise, axis=-1)
        particles = self.model.x0 + noise  # broadcasting to [total_particles, state_dim]
        return tf.reshape(particles, [batch_size, self.num_particles, self.model.state_dim])

    def _get_cov(self, dist):
        """Extracts covariance, ensuring 2D shape."""
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

        return tf.cast(cov, dtype=tf.float32)

    def compute_flow_parameters(self, particles: tf.Tensor, observation: tf.Tensor, lam: float,
                                P_xx=None, eta_mean=None, eta_0_mean=None) -> tuple[tf.Tensor, tf.Tensor]:
        '''
        Computes flow parameters A(λ) and b(λ) using Statistical Linearization.
        Equations derived from the Exact Daum-Huang Log-Homotopy.
        '''
        B, N, state_dim = particles.shape
        obs_dim = observation.shape[-1]
        R = tf.expand_dims(self.R, axis=0)  # [1, obs_dim, obs_dim]

        if eta_mean is not None:
            mean_x_flat = eta_mean  # [B, D]
            # Ensure Rank 2
            if len(mean_x_flat.shape) > 2:
                mean_x_flat = tf.squeeze(mean_x_flat, -1)
        else:
            mean_x_flat = tf.reduce_mean(particles, axis=1)  # [B, D]

        zero_noise = tf.zeros((B, self.model.obs_dim))
        with tf.GradientTape() as tape:
            tape.watch(mean_x_flat)
            # We pass zero noise as we are linearizing the deterministic part of h(x)
            mean_y_flat = self.model.observation_fn(mean_x_flat, zero_noise)

        # H = dh/dx | mean_x  [B, obs_dim, state_dim]
        # uses batch_jacobian to compute the Jacobian at mean_x
        H = tape.batch_jacobian(mean_y_flat, mean_x_flat)

        # approximate P_xy ~ P_xx * H^T
        P_Ht = tf.matmul(P_xx, H, transpose_b=True)  # [B, state_dim, obs_dim]
        # S = R + λ * H * P_xx * H^T= R + λ * H * P_Ht
        HP_Ht = tf.matmul(H, P_Ht)
        S = R + lam * HP_Ht
        S = 0.5 * (S + tf.linalg.matrix_transpose(S)) + 1e-4 * tf.eye(obs_dim, batch_shape=[B])

        obs_expanded = tf.expand_dims(observation, -1)  # [B, obs_dim, 1]
        mean_y_expanded = tf.expand_dims(mean_y_flat, -1)  # [B, obs_dim, 1]
        innovation = obs_expanded - mean_y_expanded

        rhs = tf.concat([H, innovation], axis=-1)  # [B, obs_dim, D + 1]
        solution = tf.linalg.solve(S, rhs)  # [B, obs_dim, D + 1]
        # S_inv_H: [B, obs_dim, D], S_inv_innov: [B, obs_dim, 1]
        S_inv_H, S_inv_innov = tf.split(solution, [state_dim, 1], axis=-1)

        A = -0.5 * tf.matmul(P_Ht, S_inv_H)

        # K = P H^T R^-1
        HP = tf.transpose(P_Ht, perm=[0, 2, 1])
        R_batch = tf.tile(R, [B, 1, 1])  # Broadcasting R to batch size
        Kt = tf.linalg.solve(R_batch, HP)
        K = tf.transpose(Kt, perm=[0, 2, 1])  # [B, D, obs_dim]

        # z-e = (z - h) + Hx_bar
        # innovation = z - h(x_bar)

        # H * x_bar
        mean_x_col = tf.expand_dims(mean_x_flat, -1)  # [B, D, 1]
        # z_minus_e = (z - h) + Hx
        z_minus_e = innovation + tf.matmul(H, mean_x_col)  # [B, obs_dim, 1]

        # K * (z - e)
        K_ze = tf.matmul(K, z_minus_e)  # [B, D, 1]

        # (I + λA) * K_ze
        eye_d = tf.eye(state_dim, batch_shape=[B])
        I_lamA = eye_d + lam * A
        term_left = tf.matmul(I_lamA, K_ze)  # [B, D, 1]

        # A * eta_0_mean
        # Ensure eta_0_mean is [B, D, 1]
        if eta_0_mean is None:
            # should be avoided, since by definition eta_0_mean is the mean of the states before the flow
            eta_0_mean = tf.reduce_mean(particles, axis=1)
        if len(eta_0_mean.shape) == 2:
            eta_0_mean = tf.expand_dims(eta_0_mean, -1)
        term_right = tf.matmul(A, eta_0_mean)  # [B, D, 1]
        bracket = term_left + term_right

        # b = (I + 2λA) * bracket
        I_2lamA = eye_d + 2.0 * lam * A
        b = tf.matmul(I_2lamA, bracket)  # [B, D, 1]

        return A, tf.squeeze(b, -1)  # A: [B, D, D], b: [B, D]


    @tf.function
    def _flow_update(self, observation: tf.Tensor, particles: tf.Tensor, num_flow_steps,P_xx=None,step_sizes=None) -> \
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        Perform the EDH particle flow given an observation by integrating dx/dλ = Ax + b
        '''
        # Step size for numerical integration (λ goes from 0 to 1)
        const_delta_lambda = 1.0 / float(num_flow_steps)
        current_particles = particles  # [Batch, particles, state_dim]
        eta_0_mean = tf.reduce_mean(particles, axis=1)  # [B, D]
        N=particles.shape[1]

        if P_xx is None:
            # Fallback to Sample Covariance (Noisy in high-dimensional state-space, leads to low ESS)
            mean_x = tf.reduce_mean(particles, axis=1, keepdims=True)
            dx = particles - mean_x
            P_xx = tf.matmul(dx, dx, transpose_a=True) / (tf.cast(N, tf.float32) - 1)

        lam=0.0
        for k in tf.range(num_flow_steps):
            if step_sizes is None:
                delta_lambda = const_delta_lambda
            else:
                delta_lambda = step_sizes[k]
            lam += delta_lambda
            # Compute flow parameters based on current particle distribution
            # A: [Batch, state_dim, state_dim], b: [Batch, state_dim]
            A, b = self.compute_flow_parameters(current_particles, observation, lam, P_xx=P_xx,eta_0_mean=eta_0_mean)

            # v = Ax + b for a single particle
            # batch version: X@A^T + b
            drift = tf.matmul(current_particles, A, transpose_b=True) + tf.expand_dims(b, 1)
            current_particles = current_particles + delta_lambda * drift

        x_filt = tf.reduce_mean(current_particles, axis=1)
        P_filt = tfp.stats.covariance(current_particles, sample_axis=1)

        return current_particles, x_filt, P_filt

    @tf.function
    def _transition(self, particles: tf.Tensor) -> tf.Tensor:
        # Propagate particles through the state transition model
        # particles: [Batch_size,num_particles, state_dim]
        Batch_size = tf.shape(particles)[0]

        particles_reshaped = tf.reshape(particles, [-1, self.model.state_dim])  # [Batch_size*num_particles, state_dim]
        process_noise = self.model.process_noise.sample(Batch_size * self.num_particles)
        process_noise = tf.reshape(process_noise, [-1, self.model.state_dim])  # [Batch_size*num_particles, state_dim]
        predicted_particles = self.model.transition_fn(particles_reshaped, process_noise)

        return tf.reshape(predicted_particles, [Batch_size, self.num_particles,
                                                self.model.state_dim])  # [Batch_size,num_particles, state_dim]

    @tf.function
    def filter(self, observations: tf.Tensor, num_flow_steps: int = 10,step_sizes:tf.Tensor=None) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Batch-enabled particle flow particle filter
        # observations: [B,T, obs_dim]

        batch_size, T = observations.shape[0], observations.shape[1]
        particles = self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])  # [T,B,obs_dim]

        all_particles = tf.TensorArray(dtype=tf.float32, size=T)
        x_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            Y_t = Y_time_major[t]  # [B, obs_dim]
            particles = tf.cond(
                t > 0,
                lambda: self._transition(particles),
                lambda: particles
            )
            particles, x_filt, P_filt = self._flow_update(Y_t, particles, num_flow_steps=num_flow_steps,step_sizes=step_sizes)

            x_filt_ta = x_filt_ta.write(t, x_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)
            all_particles = all_particles.write(t, particles)

        return (
            tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),  # [B, T, state_dim]
            tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),  # [B, T, state_dim, state_dim]
            tf.transpose(all_particles.stack(), perm=[1, 0, 2, 3]),  # [B, T, num_particles, state_dim]
        )


    def filter_with_ukf(self, observations: tf.Tensor, num_flow_steps: int = 10,step_sizes:tf.Tensor=None,resample:bool=True) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Batch-enabled particle flow particle filter
        # observations: [B,T, obs_dim]
        if self.ukf is None:
            raise ValueError("UKF instance is not provided for filter_with_ukf method.")
        batch_size, T = observations.shape[0], observations.shape[1]

        x_ukf, P_ukf, Wm, Wc, lam_ukf, Q, R, q_mean, r_mean = self.ukf.initialize_filter(batch_size)

        all_particles = tf.TensorArray(dtype=tf.float32, size=T)
        x_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)

        particles= self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])  # [T,B,obs_dim]

        for t in tf.range(T):
            Y_t = Y_time_major[t]
            eta_0, x_pred_ukf, P_pred_ukf = tf.cond(
                t > 0,
                lambda: (
                    self._transition(particles),
                    *self.ukf.predict_step(x_ukf, P_ukf, Wm, Wc, lam_ukf, Q, q_mean)
                ),
                lambda: (particles, x_ukf, P_ukf)
            )

            particles,x_filt,P_filt = self._flow_update(
                Y_t, eta_0, num_flow_steps=num_flow_steps,P_xx=P_pred_ukf,step_sizes=step_sizes
            )

            x_ukf, P_ukf, _ = self.ukf.update_step(
                x_pred_ukf, P_pred_ukf, Y_t, Wm, Wc, lam_ukf, R, r_mean
            )

            x_filt_ta = x_filt_ta.write(t, x_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)
            all_particles = all_particles.write(t, particles)

            if resample:
                mvn = tfd.MultivariateNormalFullCovariance(loc=x_filt, covariance_matrix=P_filt)
                samples = mvn.sample(self.num_particles) # [num_particles, batch_size, state_dim]
                particles = tf.transpose(samples, perm=[1, 0, 2]) # [batch_size, num_particles, state_dim]

        return (
            tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),
            tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),
            tf.transpose(all_particles.stack(), perm=[1, 0, 2, 3]),
        )


class LEDHFlow(EDHFlow):
    '''
    Implements the Local Exact Daum-Huang (LEDH) particle flow.
    Computes distinct flow parameters A_i and b_i for each particle using
    individual Jacobian linearizations while sharing the global covariance P.
    '''

    def __init__(self, model, num_particles: int):
        super().__init__(model, num_particles)

    def compute_flow_parameters(self, particles: tf.Tensor, observation: tf.Tensor, lam: float,
                                P_xx=None, eta_mean=None, eta_0_mean=None) -> tuple[tf.Tensor, tf.Tensor]:
        '''
        [cite_start]Computes flow parameters per particle using LEDH equations (13) and (14) from the paper[cite: 177, 182].
        A^i = -0.5 * P * H^i.T * (R + lam * H^i * P * H^i.T)^-1 * H^i
        b^i = (I + 2*lam*A^i) * [ (I + lam*A^i) * K^i * (z - e^i) + A^i * eta_0_mean ]
        '''
        B, N, D = tf.shape(particles)[0], tf.shape(particles)[1], tf.shape(particles)[2]
        obs_dim = observation.shape[-1]

        # P [B, D, D]
        if P_xx is None:
            # Fallback to Sample Covariance
            mean_x = tf.reduce_mean(particles, axis=1, keepdims=True)
            dx = particles - mean_x
            P_xx = tf.matmul(dx, dx, transpose_a=True) / (tf.cast(N, tf.float32) - 1)

        # Expand P for broadcasting: [B, 1, D, D]
        P_expanded = tf.expand_dims(P_xx, 1)

        # Compute Jacobian H_i per particle [B, N, Obs, D]
        flat_particles = tf.reshape(particles, [B * N, D])
        zero_noise = tf.zeros((B * N, self.model.obs_dim))

        with tf.GradientTape() as tape:
            tape.watch(flat_particles)
            pred_obs_flat = self.model.observation_fn(flat_particles, zero_noise)

        # Jacobian H: [B*N, Obs, D] -> Reshape to [B, N, Obs, D]
        H_flat = tape.batch_jacobian(pred_obs_flat, flat_particles)
        H = tf.reshape(H_flat, [B, N, obs_dim, D])

        # Prediction y_i: [B, N, Obs]
        pred_obs = tf.reshape(pred_obs_flat, [B, N, obs_dim])

        # Compute A_i [B, N, D, D]
        # P * H^T: [B, 1, D, D] @ [B, N, D, Obs] -> [B, N, D, Obs]
        PHt = tf.matmul(P_expanded, H, transpose_b=True)

        # S = R + lambda * H * P * H^T
        # HPHt: [B, N, Obs, D] @ [B, N, D, Obs] -> [B, N, Obs, Obs]
        HPHt = tf.matmul(H, PHt)

        # This prevents the ValueError in tf.linalg.solve
        R_expanded = tf.reshape(self.R, [1, 1, obs_dim, obs_dim])
        R_tiled = tf.tile(R_expanded, [B, N, 1, 1])
        S = R_tiled + lam * HPHt

        # S_inv_H = S^-1 * H
        # [B, N, Obs, Obs] \ [B, N, Obs, D] -> [B, N, Obs, D]
        S_inv_H = tf.linalg.solve(S, H)

        # A = -0.5 * P * H^T * S^-1 * H = -0.5 * PHt * S_inv_H
        # [B, N, D, Obs] @ [B, N, Obs, D] -> [B, N, D, D]
        A = -0.5 * tf.matmul(PHt, S_inv_H)

        # b^i = (I + 2*lam*A) * [ (I + lam*A) * K * (z - e) + A * eta_0_mean ]
        # Calculate K = P * H^T * R^-1
        # We compute (R^-1 * H * P)^T = (solve(R, HP))^T
        HP = tf.matmul(H, P_expanded)  # [B, N, Obs, D]

        # Kt = R^-1 * HP
        # solve requires explicit broadcasting of R to match tensor shapes
        Kt = tf.linalg.solve(R_tiled, HP)
        K = tf.transpose(Kt, perm=[0, 1, 3, 2])  # [B, N, D, Obs]

        # e^i = h(x) - H*x
        # z - e^i = z - h(x) + H*x = (z - h(x)) + H*x
        y_true_expanded = tf.expand_dims(observation, 1)  # [B, 1, Obs]
        innov = y_true_expanded - pred_obs  # [B, N, Obs]
        innov = tf.expand_dims(innov, -1)  # [B, N, Obs, 1]

        # H*x: [B, N, Obs, D] @ [B, N, D, 1] -> [B, N, Obs, 1]
        x_expanded = tf.expand_dims(particles, -1)
        Hx = tf.matmul(H, x_expanded)

        z_minus_e = innov + Hx  # [B, N, Obs, 1]

        # Term: K * (z - e)
        # [B, N, D, Obs] @ [B, N, Obs, 1] -> [B, N, D, 1]
        K_ze = tf.matmul(K, z_minus_e)

        # Term: A * eta_0_mean
        if eta_0_mean is None:
            eta_0_mean = tf.reduce_mean(particles, axis=1)  # [B, D]

        # reshape eta_0_mean to [B, 1, D, 1] for broadcasting
        if len(eta_0_mean.shape) == 2:
            eta_0_mean_expanded = tf.reshape(eta_0_mean, [B, 1, D, 1])
        else:
            eta_0_mean_expanded = eta_0_mean

        A_eta0 = tf.matmul(A, eta_0_mean_expanded)  # [B, N, D, 1]
        # Bracket term: (I + lam*A)*K_ze + A_eta0
        # (I + lam*A) * K_ze = K_ze + lam * A * K_ze
        I_lamA_Kze = K_ze + lam * tf.matmul(A, K_ze)
        bracket = I_lamA_Kze + A_eta0  # [B, N, D, 1]
        # Final b: (I + 2*lam*A) * bracket
        # = bracket + 2*lam * A * bracket
        b_expanded = bracket + 2.0 * lam * tf.matmul(A, bracket)
        b = tf.squeeze(b_expanded, -1)  # [B, N, D]

        return A, b

    @tf.function
    def _flow_update(self, observation: tf.Tensor, particles: tf.Tensor, num_flow_steps: int = 10,
                     step_sizes: tf.Tensor = None, P_xx=None) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        Overridden flow update for LEDH.
        Handles dimensions where A is [B, N, D, D] (per-particle) vs EDH's [B, D, D] (shared).
        '''
        const_delta_lambda = 1.0 / float(num_flow_steps)
        current_particles = particles  # [B, N, D]

        # Calculate eta_0_mean (prior mean) ONCE at the start
        eta_0_mean = tf.reduce_mean(particles, axis=1)  # [B, D]

        lam = 0.0
        for k in tf.range(num_flow_steps):
            if step_sizes is None:
                delta_lambda = const_delta_lambda
            else:
                delta_lambda = step_sizes[k]
            lam += delta_lambda

            # A: [B, N, D, D], b: [B, N, D]
            A, b = self.compute_flow_parameters(
                current_particles, observation, lam, P_xx=P_xx, eta_0_mean=eta_0_mean
            )

            # Drift: v = A_i * x_i + b_i
            x_expanded = tf.expand_dims(current_particles, -1)  # [B, N, D, 1]

            # Ax: [B, N, D, D] @ [B, N, D, 1] -> [B, N, D, 1]
            Ax = tf.matmul(A, x_expanded)

            # b: [B, N, D] -> [B, N, D, 1]
            b_expanded = tf.expand_dims(b, -1)

            drift = Ax + b_expanded
            drift = tf.squeeze(drift, -1)  # [B, N, D]

            current_particles = current_particles + delta_lambda * drift

        x_filt = tf.reduce_mean(current_particles, axis=1)
        P_filt = tfp.stats.covariance(current_particles, sample_axis=1)

        return current_particles, x_filt, P_filt


class InvertiblePFPF(ParticleFilter):
    def __init__(self, model: NLSSM, num_particles: int,
                 ukf: UnscentedKalmanFilter,
                 flow_class: Callable = EDHFlow,
                 num_flow_steps: int = 10, resample_method: str = 'systematic'):
        super().__init__(model, num_particles, resample_method=resample_method)

        self.flow_algo = flow_class(model, num_particles)
        self.num_flow_steps = num_flow_steps
        self.ukf = ukf  # Store the UKF

    @tf.function
    def _flow_with_det(self, eta_0, Y_t, num_flow_steps, P_guide=None, eta_mean=None,step_sizes:tf.Tensor=None) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Runs the flow with External Guidance P.
        eta_mean is x_filt transformed by transition function without noise.
        """
        const_delta_lambda = 1.0 / float(num_flow_steps)
        current_particles = eta_0
        eta_0_mean = tf.identity(eta_mean)

        B = tf.shape(eta_0)[0]
        N = self.num_particles
        D = self.model.state_dim

        # Accumulator for log determinant [Batch, Num_Particles]
        log_det_jacobian = tf.zeros((B, N))

        lam=0.0
        for k in tf.range(num_flow_steps):
            if step_sizes is None:
                dt = const_delta_lambda
            else:
                dt = step_sizes[k]
            lam+= dt

            # A shape: [B, D, D] (Shared/EDH) or [B, N, D, D] (Per-Particle/LEDH)
            A, b = self.flow_algo.compute_flow_parameters(
                current_particles, Y_t, lam, P_xx=P_guide, eta_mean=eta_mean, eta_0_mean=eta_0_mean
            )

            if len(A.shape) == 3:
                # A: [B, D, D], b: [B, D]
                # Broadcasting: [B,N,D] @ [B,D,D] -> [B,N,D]
                drift = tf.matmul(current_particles, A, transpose_b=True) + tf.expand_dims(b, 1)
                current_particles = current_particles + dt * drift

                # Correct update for eta_mean (keeping it [B, D])
                # Ensure correct dimensionality for matmul: [B, D, D] @ [B, D, 1]
                eta_mean_expanded = tf.expand_dims(eta_mean, -1)  # [B, D, 1]
                drift_mean = tf.matmul(A, eta_mean_expanded)  # [B, D, 1]
                drift_mean = tf.squeeze(drift_mean, -1) + b  # [B, D]
                eta_mean = eta_mean + dt * drift_mean

            else:
                # A: [B, N, D, D], b: [B, N, D]
                # Expand x to [B, N, D, 1] to match A's particle dimension
                x_expanded = tf.expand_dims(current_particles, -1)

                # Ax: [B, N, D, D] @ [B, N, D, 1] -> [B, N, D, 1]
                Ax = tf.matmul(A, x_expanded)
                b_expanded = tf.expand_dims(b, -1)

                drift = Ax + b_expanded
                drift = tf.squeeze(drift, -1)  # [B, N, D]
                current_particles = current_particles + dt * drift

                # det(J_i) varies per particle, so it affects relative weights.
                identity = tf.eye(D, batch_shape=[B, N])
                J = identity + dt * A
                _, step_log_det = tf.linalg.slogdet(J)  # [B, N]
                log_det_jacobian += step_log_det

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
            dist_fn=self.model.get_observation_dist
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
                dist_fn=self.model.get_transition_dist
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

    # @tf.function
    def filter(self, observations: tf.Tensor, num_flow_steps: int = 10,step_sizes:tf.Tensor=None) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size, T = observations.shape[0], observations.shape[1]

        x_ukf, P_ukf, Wm, Wc, lam_ukf, Q, R, q_mean, r_mean = self.ukf.initialize_filter(batch_size)

        all_particles = tf.TensorArray(dtype=tf.float32, size=T)
        all_weights = tf.TensorArray(dtype=tf.float32, size=T)
        x_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_ta = tf.TensorArray(dtype=tf.float32, size=T)

        particles, weights = self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])  # [T,B,obs_dim]

        @tf.function
        def _transition_branch(particles, x_ukf, P_ukf):
            p_pred = self._transition(particles)
            x_u, P_u = self.ukf.predict_step(x_ukf, P_ukf, Wm, Wc, lam_ukf, Q, q_mean)
            return p_pred, x_u, P_u

        for t in tf.range(T):
            Y_t = Y_time_major[t]
            if t > 0:
                eta_0, x_pred_ukf, P_pred_ukf = _transition_branch(particles, x_ukf, P_ukf)
                eta_mean = self.model.transition_fn(x_filt, tf.zeros_like(x_filt))
            else:
                eta_0, x_pred_ukf, P_pred_ukf = particles, x_ukf, P_ukf
                eta_mean = tf.identity(x_ukf)

            eta_1, log_det = self._flow_with_det(
                eta_0, Y_t, num_flow_steps, P_guide=P_pred_ukf, eta_mean=eta_mean,step_sizes=step_sizes
            )
            is_initial = tf.equal(t, 0)
            weights = self._update_pfpf(particles, eta_0, eta_1, weights, Y_t, log_det, is_initial)

            x_ukf, P_ukf, _ = self.ukf.update_step(
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

            particles, weights = self._resample(particles, weights)

        return (
            tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2]),
            tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3]),
            tf.transpose(all_particles.stack(), perm=[1, 0, 2, 3]),
            tf.transpose(all_weights.stack(), perm=[1, 0, 2])
        )
