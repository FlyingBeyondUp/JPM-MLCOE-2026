import tensorflow as tf
import tensorflow_probability as tfp

from models import NLSSM
from Filters.basic_filters import UnscentedKalmanFilter, ParticleFilter


tfd = tfp.distributions
dtype=tf.float32

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

        return tf.cast(cov, dtype=dtype)

    def compute_flow_parameters(self, particles: tf.Tensor, observation: tf.Tensor, lam: float,
                                P_xx=None, linearization_points=None, eta_0_mean=None):
        '''
        Computes flow parameters A(λ) and b(λ) using Statistical Linearization.
        Equations derived from the Exact Daum-Huang Log-Homotopy.
        '''
        shape = tf.shape(particles)
        B = shape[0]
        N = shape[1]
        state_dim = particles.shape[-1]
        obs_dim = observation.shape[-1]
        R = tf.expand_dims(self.R, axis=0)  # [1, obs_dim, obs_dim]

        if linearization_points is not None:
            mean_x_flat = linearization_points  # [B, D]
            # Ensure Rank 2
            if len(mean_x_flat.shape) > 2:
                mean_x_flat = tf.squeeze(mean_x_flat, -1)
        else:
            mean_x_flat = tf.reduce_mean(particles, axis=1)  # [B, D]

        zero_noise = tf.zeros((B, self.model.obs_dim),dtype=dtype)
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
        S = 0.5 * (S + tf.linalg.matrix_transpose(S)) + 1e-4 * tf.eye(obs_dim, batch_shape=[B],dtype=dtype)

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
        eye_d = tf.eye(state_dim, batch_shape=[B],dtype=dtype)
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
        b=tf.squeeze(b, -1) # b: [B, D]

        return A,b


    @tf.function
    def _flow_update(self, observations: tf.Tensor, particles: tf.Tensor,
                     num_flow_steps:int,P_xx=None,step_sizes=None) -> \
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        Perform the EDH particle flow given an observation by integrating dx/dλ = Ax + b
        '''
        # Step size for numerical integration (λ goes from 0 to 1)
        const_delta_lambda = 1.0 / float(num_flow_steps)
        current_particles = particles  # [Batch, particles, state_dim]
        eta_0_mean = tf.reduce_mean(particles, axis=1)  # [B, D]  # [B, D,1]
        N=particles.shape[1]

        if P_xx is None:
            # Fallback to Sample Covariance (Noisy in high-dimensional state-space, leads to low ESS)
            mean_x = tf.reduce_mean(particles, axis=1, keepdims=True)
            dx = particles - mean_x
            P_xx = tf.matmul(dx, dx, transpose_a=True) / (tf.cast(N, dtype) - 1)

        lam=tf.cast(0.0,dtype)
        for k in tf.range(num_flow_steps):
            if step_sizes is None:
                delta_lambda = const_delta_lambda
            else:
                delta_lambda = step_sizes[k]
            lam += delta_lambda
            # Compute flow parameters based on current particle distribution
            # A: [Batch, state_dim, state_dim], b: [Batch, state_dim]
            A,b = self.compute_flow_parameters(current_particles, observations, lam,
                                                P_xx=P_xx,eta_0_mean=eta_0_mean)
            drift=tf.matmul(current_particles, A, transpose_b=True) + tf.expand_dims(b, axis=1)  # [Batch, particles, state_dim]
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

        all_particles = tf.TensorArray(dtype=dtype, size=T)
        x_filt_ta = tf.TensorArray(dtype=dtype, size=T)
        P_filt_ta = tf.TensorArray(dtype=dtype, size=T)

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

    @tf.function
    def filter_with_ukf(self, observations: tf.Tensor, num_flow_steps: int = 10,step_sizes:tf.Tensor=None,resample:bool=False) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Batch-enabled particle flow particle filter
        # observations: [B,T, obs_dim]
        if self.ukf is None:
            raise ValueError("UKF instance is not provided for filter_with_ukf method.")
        obs_shape=tf.shape(observations)
        batch_size, T = obs_shape[0], obs_shape[1]

        x_ukf, P_ukf, Wm, Wc, lam_ukf, Q, R, q_mean, r_mean = self.ukf.initialize_filter(batch_size)

        all_particles = tf.TensorArray(dtype=dtype, size=T)
        x_filt_ta = tf.TensorArray(dtype=dtype, size=T)
        P_filt_ta = tf.TensorArray(dtype=dtype, size=T)

        particles= self._initialize(batch_size=batch_size)
        Y_time_major = tf.transpose(observations, perm=[1, 0, 2])  # [T,B,obs_dim]
        x_filt,P_filt=x_ukf, P_ukf

        def _transition_branch(particles,x_filt,P_ukf):
            particles = self._transition(particles)
            # Safely unpack exactly 3 values from the standard UKF
            x_pred_ukf, P_pred_ukf, _ = self.ukf.predict_step(x_filt, P_ukf, Wm, Wc, lam_ukf, Q, q_mean)
            return particles, x_pred_ukf, P_pred_ukf

        for t in tf.range(T):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (particles, tf.TensorShape([None, self.num_particles, self.model.state_dim])),
                    (x_filt, tf.TensorShape([None, self.model.state_dim])),
                    (x_ukf, tf.TensorShape([None, self.model.state_dim])),
                    (P_filt, tf.TensorShape([None, self.model.state_dim, self.model.state_dim])),
                    (P_ukf, tf.TensorShape([None, self.model.state_dim, self.model.state_dim]))
                ]
            )

            if resample:
                # Use x_filt for accuracy, but P_ukf for structural stability to prevent collapse
                P_ukf_sym = 0.5 * (P_ukf + tf.linalg.matrix_transpose(P_ukf)) + 1e-4 * tf.eye(self.model.state_dim,batch_shape=[batch_size],dtype=dtype)
                mvn = tfd.MultivariateNormalFullCovariance(loc=x_filt, covariance_matrix=P_ukf_sym)
                samples = mvn.sample(self.num_particles)
                particles = tf.transpose(samples, perm=[1, 0, 2])

            Y_t = Y_time_major[t]
            if t > 0:
                particles, x_pred_ukf, P_pred_ukf = _transition_branch(particles, x_filt, P_ukf)
            else:
                particles, x_pred_ukf, P_pred_ukf = particles, x_ukf, P_ukf

            particles,x_filt,P_filt = self._flow_update(
                Y_t, particles, num_flow_steps=num_flow_steps,P_xx=P_pred_ukf,step_sizes=step_sizes
            )

            x_ukf, P_ukf, _ = self.ukf.update_step(
                x_pred_ukf, P_pred_ukf, Y_t, Wm, Wc, lam_ukf, R, r_mean
            )

            particles = tf.ensure_shape(particles, [None, self.num_particles, self.model.state_dim])
            x_filt = tf.ensure_shape(x_filt, [None, self.model.state_dim])
            P_filt = tf.ensure_shape(P_filt, [None, self.model.state_dim, self.model.state_dim])

            x_filt_ta = x_filt_ta.write(t, x_filt)
            P_filt_ta = P_filt_ta.write(t, P_filt)
            all_particles = all_particles.write(t, particles)

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

    def __init__(self, model, num_particles: int,ukf: UnscentedKalmanFilter = None):
        super().__init__(model, num_particles,ukf)

    def compute_flow_parameters(self, particles: tf.Tensor, observation: tf.Tensor, lam: float,
                                P_xx=None, linearization_points=None, eta_0_mean=None) -> tuple[tf.Tensor, tf.Tensor]:
        '''
        [cite_start]Computes flow parameters per particle using LEDH equations (13) and (14) from the paper[cite: 177, 182].
        A^i = -0.5 * P * H^i.T * (R + lam * H^i * P * H^i.T)^-1 * H^i
        b^i = (I + 2*lam*A^i) * [ (I + lam*A^i) * K^i * (z - e^i) + A^i * eta_0_mean ]
        '''
        B, N, D = tf.shape(particles)[0], tf.shape(particles)[1], tf.shape(particles)[2]
        obs_dim = observation.shape[-1]
        # Expand P for broadcasting: [B, 1, D, D]
        if P_xx.shape.ndims == 3:
            # [B, D, D] Shared covariance, expand to [B, 1, D, D] for broadcasting against N
            P_expanded = tf.expand_dims(P_xx, 1)
        else:
            # [B, N, D, D] already Per-particle covariance
            P_expanded = P_xx

        # Compute Jacobian H_i per particle [B, N, Obs, D]
        if linearization_points is None:
            linearization_points = tf.expand_dims(particles, -1)
        flat_particles = tf.reshape(linearization_points, [B * N, D])
        zero_noise = tf.zeros((B * N, self.model.obs_dim),dtype=dtype)

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
        R_expanded = tf.reshape(self.R, [1, 1, obs_dim, obs_dim])
        R_tiled = tf.tile(R_expanded, [B, N, 1, 1])
        S = R_tiled + lam * HPHt
        S = 0.5 * (S + tf.linalg.matrix_transpose(S)) + 1e-5 * tf.eye(obs_dim, batch_shape=[B, N],dtype=dtype)
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
        Kt = tf.linalg.solve(R_tiled, HP)
        K = tf.transpose(Kt, perm=[0, 1, 3, 2])  # [B, N, D, Obs]

        # e^i = h(x) - H*x
        # z - e^i = z - h(x) + H*x = (z - h(x)) + H*x
        # should estimated at the linearization points
        y_true_expanded = tf.expand_dims(observation, 1)  # [B, 1, Obs]
        innov = y_true_expanded - pred_obs  # [B, N, Obs]
        innov = tf.expand_dims(innov, -1)  # [B, N, Obs, 1]

        # H*x: [B, N, Obs, D] @ [B, N, D, 1] -> [B, N, Obs, 1]
        Hx = tf.matmul(H, linearization_points)

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
            eta_0_mean_expanded = eta_0_mean # shape already [B, N, D, 1]

        # Bracket term: (I + lam*A)*K_ze + A_eta0
        # (I + lam*A) * K_ze = K_ze + lam * A * K_ze
        I_lamA_Kze = K_ze + lam * tf.matmul(A, K_ze)
        bracket = I_lamA_Kze + tf.matmul(A, eta_0_mean_expanded)  # [B, N, D, 1]
        # Final b: (I + 2*lam*A) * bracket
        # = bracket + 2*lam * A * bracket
        b_expanded = bracket + 2.0 * lam * tf.matmul(A, bracket)
        b = tf.squeeze(b_expanded, -1)  # [B, N, D]

        # Ax: [B, N, D, D] @ [B, N, D, 1] -> [B, N, D, 1]
        # b: [B, N, D] -> [B, N, D, 1]
        return A,b

    @tf.function
    def _flow_update(self, observation: tf.Tensor, particles: tf.Tensor, linearization_points:tf.Tensor=None,
                     num_flow_steps: int = 10,step_sizes: tf.Tensor = None, P_xx=None) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        Overridden flow update for LEDH.
        Handles dimensions where A is [B, N, D, D] (per-particle) vs EDH's [B, D, D] (shared).
        '''
        const_delta_lambda = 1.0 / float(num_flow_steps)
        current_particles = particles  # [B, N, D]

        # Calculate eta_0_mean (prior mean) ONCE at the start
        # eta_0_mean = tf.reduce_mean(particles, axis=1,keepdims=True)  # [B,1, D]
        # eta_0_mean = tf.expand_dims(eta_0_mean, -1)  # [B, 1, D, 1]
        eta_0_mean = tf.expand_dims(particles, -1)
        if P_xx is None:
            # Fallback to Sample Covariance
            mean_x = tf.reduce_mean(particles, axis=1, keepdims=True)
            dx = particles - mean_x
            P_xx = tf.matmul(dx, dx, transpose_a=True) / (tf.cast(tf.shape(particles)[1], dtype) - 1)

        lam = tf.cast(0.0, dtype)
        for k in tf.range(num_flow_steps):
            if step_sizes is None:
                delta_lambda = const_delta_lambda
            else:
                delta_lambda = step_sizes[k]
            lam += delta_lambda

            # A: [B, N, D, D], b: [B, N, D]
            A,b= self.compute_flow_parameters(
                current_particles, observation, lam, P_xx=P_xx,eta_0_mean=eta_0_mean
            )
            x_expanded = tf.expand_dims(current_particles, -1)
            # Ax: [B, N, D, D] @ [B, N, D, 1] -> [B, N, D, 1]
            b_expanded = tf.expand_dims(b, -1)
            drift = tf.matmul(A, x_expanded) + b_expanded
            drift = tf.squeeze(drift, -1)
            current_particles = current_particles + delta_lambda * drift

        x_filt = tf.reduce_mean(current_particles, axis=1)
        P_filt = tfp.stats.covariance(current_particles, sample_axis=1)

        return current_particles, x_filt, P_filt