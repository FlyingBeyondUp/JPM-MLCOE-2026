import tensorflow as tf
import tensorflow_probability as tfp
from models import NLSSM
from .deterministic_flow import EDHFlow

tfd = tfp.distributions

class StochasticFlow(EDHFlow):
    def __init__(self, model: NLSSM, num_particles, stiffness_weight=0.01,
                 diffusion_q=None,diffusion_dim=None,getQ=None,requires_stiffness=False,rng_seed: int = None):
        """
        :param stiffness_weight: Parameter 'mu' from the paper (Eq 25). Controls tradeoff between
                                 energy and stiffness mitigation.
                                 zero means no stiffness mitigation, higher values prioritize better conditioning of the flow.
                                 For mu=0, use linear homotopy schedule (beta = lambda) without solving the optimal control problem.
        :param diffusion_cov:    diffusion matrix Q.
                                 If non-zero, implements Stochastic Flow.
                                 If 0, implements Stiffness-Mitigated Deterministic Flow.

        The Q is the diffusion matrix in the SDE, not the process noise covariance in the model.
        It can be set to a function of beta or P_xx for more complex flows.
        """
        super().__init__(model, num_particles)
        self.model = model
        self.num_particles = num_particles
        self.R = self._get_cov(self.model.observation_noise)
        self.R_inv=tf.linalg.inv(self.R+1e-6*tf.eye(self.model.obs_dim))
        self.mu = stiffness_weight
        self.q = diffusion_q[tf.newaxis,:,:] if diffusion_q is not None else None # shape [1,1,D,m]
        self.m=diffusion_dim
        self.rng_seed = rng_seed

        self.betas,self.beta_dots=None,None

        if self.q is None:
            self.getQ=getQ
        else:
            self.Q=self.q@tf.linalg.matrix_transpose(self.q) # shape [1,D,D]

        if self.rng_seed is not None:
            tf.random.set_seed(int(self.rng_seed))
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass

        self.requires_stiffness = requires_stiffness
        if requires_stiffness:
            self.stiffness_records=[]

    def getObsHessian(self, x: tf.Tensor) :
        '''
        Computes the Hessian of the negative log-likelihood (observation density) at mean_x.
        For Gaussian likelihood, this is simply H^T R^-1 H where H is the Jacobian of h(x) at mean_x.

        :param mean_x: [B, D] The point at which to compute the Hessian
        :return: Hessian matrix [B, D, D]
        '''
        B = tf.shape(x)[0]
        with tf.GradientTape() as tape:
            tape.watch(x)
            zero_noise = tf.zeros((B, self.model.obs_dim))
            mean_y = self.model.observation_fn(x, zero_noise)

        H = tape.batch_jacobian(mean_y, x)  # [B, obs_dim, state_dim]
        Hh=tf.linalg.matrix_transpose(H)@self.R_inv[tf.newaxis,:,:]@H  # [B, state_dim, state_dim]
        return H,-Hh

    def _get_optimal_path(self, P_xx: tf.Tensor, Hh: tf.Tensor, num_flow_steps: int):
        B = tf.shape(P_xx)[0]
        state_dim = tf.shape(P_xx)[-1]

        M0 = tf.linalg.inv(P_xx + 1e-6 * tf.eye(state_dim))
        Mh = -Hh

        def ode_fn(t,state):
            beta = state['beta']  # [B]
            beta_dot = state['beta_dot']  # [B]
            beta_clamped = tf.clip_by_value(beta, -0.05, 1.5)
            M = M0 + tf.reshape(beta_clamped, [-1, 1, 1]) * Mh
            M_inv = tf.linalg.inv(M + 1e-6 * tf.eye(state_dim))

            tr_M = tf.linalg.trace(M)
            tr_Minv = tf.linalg.trace(M_inv)
            tr_Mh = tf.linalg.trace(Mh)
            M_inv_Mh = tf.matmul(M_inv, Mh)
            tr_Minv_Mh_Minv = tf.linalg.trace(tf.matmul(M_inv_Mh, M_inv))

            accel = self.mu * (tr_Mh * tr_Minv - tr_M * tr_Minv_Mh_Minv)
            return {'beta': beta_dot, 'beta_dot': accel}

        solver = tfp.math.ode.DormandPrince(rtol=1e-3, atol=1e-3)
        solution_times = tf.linspace(0.0, 1.0, num_flow_steps)

        def integrate(u0):
            initial_state = {'beta': tf.zeros([B]), 'beta_dot': u0}
            results = solver.solve(ode_fn, initial_time=0.0, initial_state=initial_state,
                                   solution_times=solution_times)
            betas = results.states['beta']  # [num_flow_steps, B]
            beta_dots = results.states['beta_dot']
            return betas[-1], betas, beta_dots

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

        self.betas = tf.transpose(betas)
        self.beta_dots = tf.transpose(beta_dots)


    def _get_beta(self,lam,B=1):
        # Return shape: [B,1,1,1] for the shape [B,N,D,D]
        # meaning that the path is determined by the mean and covariance of the prior
        # and does not depend on particles' x

        beta_index = tf.cast(lam * tf.cast(tf.shape(self.betas)[-1] - 1, tf.float32), tf.int32)
        beta_index = tf.clip_by_value(beta_index, 0, tf.shape(self.betas)[-1] - 1)

        if len(self.betas.shape) == 1:
            # Linear schedule: betas shape [num_flow_steps]
            beta = tf.reshape(self.betas[beta_index], [1, 1, 1])  # scalar
            beta_dot = tf.reshape(self.beta_dots[beta_index],[1,1,1])  # scalar

            # Broadcast to [B, 1, 1]
            beta = tf.tile(beta, [B, 1, 1])
            beta_dot = tf.tile(beta_dot, [B, 1, 1])
        else:
            # Optimal schedule: betas shape [B, num_flow_steps]
            beta = tf.reshape(self.betas[:, beta_index],[B,1,1])
            beta_dot = tf.reshape(self.beta_dots[:, beta_index],[B,1,1])
        return beta, beta_dot

    @tf.function(reduce_retracing=True)
    def compute_flow(self, particles: tf.Tensor, observation: tf.Tensor, lam: float,
                                P_xx=None, linearization_points=None, eta_0_mean=None):
        """
        Computes the drift for the stochastic flow.

        Ref: Eq 22 in paper for Jacobian F.
        Drift = beta_dot * f_EDH(beta) + 0.5 * Q * grad(log p)

        :param particles: [B, N, D]
        :param observation: [B, Obs]
        :param P_xx: [B, D, D]
        :param eta_0_mean: [B, 1, D]
        :param beta: The current value of the homotopy parameter (lambda in paper eq 22)
        :param beta_dot: The rate of change of beta w.r.t integration steps.

        :return : drift: The drift term for the SDE, shape [B, N, D]
                  q: The diffusion matrix for the SDE, shape [B, D, m]
        """
        B, N, D = particles.shape

        mean_particles = tf.reduce_mean(particles, axis=1)  # [B, D]
        Hg = -tf.linalg.inv(P_xx + 1e-6 * tf.eye(D)) # [B,D,D]
        grad_log_g=tf.expand_dims(Hg,axis=1)@tf.expand_dims(particles - eta_0_mean,axis=-1) # [B,N,D,1]
        # The Hessian matrices of the likelihood at the mean particle and the prior w.r.t. state
        H,Hh=self.getObsHessian(mean_particles)

        H=tf.expand_dims(H,1) # shape: [B,1,Obs, D]
        residue=tf.expand_dims(observation,axis=1) - self.model.observation_fn(particles, tf.zeros_like(particles))
        # Wrap to [-pi, pi]
        residue = (residue + 3.14159) % (2 * 3.14159) - 3.14159
        # H: [B, 1, Obs, D], R_inv -> [1,1,Obs, Obs], residue: [B,N,Obs,1]
        # R_inv@residue: [B, N, Obs, 1]
        # grad log h shape: [B, N, D,1]
        grad_log_h=tf.linalg.matrix_transpose(H)@self.R_inv[tf.newaxis,tf.newaxis,:,:]@residue[...,tf.newaxis]

        beta,beta_dot = self._get_beta(lam,B) # [B,1,1]
        grad_log_p = grad_log_g + tf.expand_dims(beta,axis=1)*grad_log_h # [B,N,D,1]

        Hp=beta*Hh+Hg  # Hessian of log p at the current pseudo-time, shape [B, D, D]
        Hp_inv=tf.linalg.inv(Hp-tf.expand_dims(1e-6*tf.eye(D),axis=0)) # [B,D,D], -1e-6 since Hp is negative definite
        if self.q is None:
            q=self.getQ(Hh,Hp_inv)
            Q=q@tf.linalg.matrix_transpose(q)
        else:
            q,Q=self.q,self.Q  # q shape: [1,D,m], Q shape: [1,D,D]

        # F=1/2*(Q@Hp-beta_dot*Hp_inv@Hh) # shape [B, D, D]
        # if self.requires_stiffness:
        #     eigenvals=tf.abs(tf.math.real(tf.linalg.eigvals(F)))
        #     stiffness=tf.math.log(tf.reduce_max(eigenvals,axis=-1)/tf.maximum(tf.reduce_min(eigenvals,axis=-1),1e-10))
        #     self.stiffness_records.append(stiffness.numpy())

        K=1/2*(Q+beta_dot*Hp_inv@Hh@Hp_inv) # [B,D,D]
        K=tf.expand_dims(K,1) # [B,1,D,D]
        Hp_inv=tf.expand_dims(Hp_inv,1)  # [B,1,D,D]
        f=K@grad_log_p-tf.expand_dims(beta_dot,axis=1)*Hp_inv@grad_log_h # [B,N,D,1]
        f=tf.reshape(f,[B,N,D])
        return f,q

    def update_path(self,num_flow_steps:int,P_xx=None,Hh=None):
        if self.mu == 0:
            # Linear schedule
            self.betas = tf.linspace(0.0, 1.0, num_flow_steps)
            self.beta_dots = tf.ones(num_flow_steps)
        else:
            # Optimal schedule
            self._get_optimal_path(P_xx, Hh, num_flow_steps)

    #@tf.function
    def _flow_update(self, observations: tf.Tensor, particles: tf.Tensor,
                     num_flow_steps:int,P_xx=None,step_sizes=None) -> \
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        Perform the EDH particle flow given an observation by integrating dx/dλ = Ax + b
        '''
        # Step size for numerical integration (λ goes from 0 to 1)
        B, N, D = particles.shape
        const_delta_lambda = 1.0 / float(num_flow_steps)
        current_particles = particles  # [Batch, particles, state_dim]
        eta_0_mean = tf.reduce_mean(particles, axis=1,keepdims=True)  # [B,1, D]


        # Use stateless normal if a seed is set on the instance; otherwise use stateful normal
        # diffusion_noise shape: [num_flow_steps, B, N, m] where m is the diffusion dimension (number of independent noise sources)
        if self.rng_seed is not None:
            seed = tf.constant([int(self.rng_seed), int(num_flow_steps)], dtype=tf.int32)
            diffusion_noise = tf.random.stateless_normal(
                shape=(num_flow_steps, B, N, self.m),
                seed=seed,
                dtype=tf.float32
            )*tf.sqrt(const_delta_lambda)
        else:
            diffusion_noise = tf.random.normal(
                shape=(num_flow_steps,B, N, self.m))*tf.sqrt(const_delta_lambda)

        if P_xx is None:
            mean_x = tf.reduce_mean(particles, axis=1, keepdims=True)
            dx = particles - mean_x
            P_xx = tf.matmul(dx, dx, transpose_a=True) / (tf.cast(N, tf.float32) - 1)

        _,Hh=self.getObsHessian(tf.reshape(eta_0_mean,(B,D)))
        self.update_path(num_flow_steps,P_xx,Hh)

        lam=0.0
        for k in tf.range(num_flow_steps):
            if step_sizes is None:
                delta_lambda = const_delta_lambda
            else:
                delta_lambda = step_sizes[k]
            # Compute flow parameters based on current particle distribution
            # drift shape: [B, N, D], q shape: [1, D, m]
            drift,q = self.compute_flow(current_particles, observations, lam,
                                                P_xx=P_xx,eta_0_mean=eta_0_mean)

            current_particles = current_particles + delta_lambda * drift+diffusion_noise[k]@tf.linalg.matrix_transpose(q)
            lam += delta_lambda

        x_filt = tf.reduce_mean(current_particles, axis=1)
        P_filt = tfp.stats.covariance(current_particles, sample_axis=1)

        return current_particles, x_filt, P_filt

