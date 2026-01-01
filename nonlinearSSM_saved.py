import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable,Union
import matplotlib.pyplot as plt
import math
from linearSSM import KalmanFilter, LGSSM
import time

tfd=tfp.distributions

# non-linear state-space model
class NLSSM:
    def __init__(self,state_dim:int,obs_dim:int,
                 transition_fn:Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 observation_fn:Callable[[tf.Tensor,tf.Tensor],tf.Tensor],
                 process_noise:tfd.Distribution,
                 observation_noise:tfd.Distribution,
                 init_noise:tfd.Distribution,
                 x0: tf.Tensor,observation_dist:Union[None,tfd.Distribution]=None):
        '''
        The transition and observation functions should support batch inputs
        '''
        self.transition_fn=transition_fn
        self.observation_fn=observation_fn

        self.process_noise=process_noise
        self.observation_noise=observation_noise
        self.observation_dist = observation_dist

        self.init_noise=init_noise
        self.x0=x0

        self.state_dim=state_dim
        self.obs_dim=obs_dim

    def sample(self,T:int,control_input=None)->tuple[tf.Tensor,tf.Tensor]:
        '''
        Generate a sample sequence of length T
        :param T: The length of the sequence
        :return: sequence of states x and observations y
                 with shape [T,state_dim] and [T,obs_dim]
        '''
        x = tf.TensorArray(dtype=tf.float32, size=T)
        y = tf.TensorArray(dtype=tf.float32, size=T)

        x_t = self.x0 + self.init_noise.sample()

        for t in range(T):
            if t>0:
                x_t = self.transition_fn(x_t, self.process_noise.sample())
            y_t = self.observation_fn(x_t, self.observation_noise.sample())
            x = x.write(t, x_t)
            y = y.write(t, y_t)
        x,y=x.stack(),y.stack()
        return tf.squeeze(x,axis=-1),tf.squeeze(y,axis=-1)

    def batch_sample(self,T:int,batch_size:int)->tuple[tf.Tensor,tf.Tensor]:
        '''
        :param T: The length of the sequences
        :return: x: [batch_size,T,state_dim], y: [batch_size,T,obs_dim]
        '''
        return tf.vectorized_map(lambda _: self.sample(T), tf.range(batch_size))

def get1DStochasticVolModel(alpha:float,sigma:float,beta:float,heavy_tail=False)->NLSSM:
    '''
    Stochastic Volatility Model in 1D
    x_{t+1} = alpha * x_t + process_noise, process_noise ~ N(0,sigma^2)
    y_t = beta * exp(x_t / 2) * observation_noise, observation_noise ~ N(0,1)
    :param alpha: state transition coefficient
    :param sigma: standard deviation of process noise
    :param beta: scaling factor for observation noise
    :return: NLSSM instance
    '''
    state_dim=1
    obs_dim=1

    def transition_fn(x:tf.Tensor,process_noise:tf.Tensor)->tf.Tensor:
        return alpha * x + process_noise

    def observation_fn(x:tf.Tensor,observation_noise:tf.Tensor)->tf.Tensor:
        return beta * tf.exp(x / 2) * observation_noise

    if not heavy_tail:
        process_noise_dist = tfd.Normal(loc=0.0, scale=sigma)
    else:
        process_noise_dist = tfd.StudentT(df=3.0, loc=0.0, scale=sigma)
    observation_noise_dist = tfd.Normal(loc=0.0, scale=1.0)
    init_noise_dist = tfd.Normal(loc=0.0, scale=sigma/(1-alpha**2)**0.5)

    x0 = tf.zeros((state_dim,))

    return NLSSM(state_dim, obs_dim,
                 transition_fn,
                 observation_fn,
                 process_noise_dist,
                 observation_noise_dist,
                 init_noise_dist,
                 x0)

def get1DLogSquaredSVM(alpha:float,sigma:float,beta:float,heavy_tail=False)->NLSSM:
    '''
    Log-Squared Stochastic Volatility Model in 1D
    x_{t+1} = alpha * x_t + process_noise, process_noise ~ N(0,sigma^2)
    y_t = beta * exp(x_t / 2) * observation_noise, observation_noise ~ N(0,1)
    log(y_t^2) = x_t + log(beta^2) + log(observation_noise^2)
    Mean of log(observation_noise^2) is digamma(0.5) - log(0.5) ~ -1.27, added to the bias term
    Variance of log(observation_noise^2) is pi^2 / 2 ~ 4.93

    :param alpha: state transition coefficient
    :param sigma: standard deviation of process noise
    :param beta: scaling factor for observation noise
    :return: NLSSM instance
    '''
    state_dim=1
    obs_dim=1
    bias=tf.math.log(beta**2)+tf.math.digamma(0.5)-tf.math.log(0.5)

    def transition_fn(x:tf.Tensor,process_noise:tf.Tensor)->tf.Tensor:
        return alpha * x + process_noise

    def observation_fn(x:tf.Tensor,observation_noise:tf.Tensor)->tf.Tensor:
        return x + bias + observation_noise

    if heavy_tail:
        process_noise_dist = tfd.StudentT(df=3.0, loc=0.0, scale=sigma)
    else:
        process_noise_dist = tfd.Normal(loc=0.0, scale=sigma)
    observation_noise_dist = tfd.Normal(loc=0.0, scale=(math.pi**2 / 2)**0.5)
    init_noise_dist = tfd.Normal(loc=0.0, scale=sigma/(1-alpha**2)**0.5)

    x0 = tf.zeros((state_dim,))

    return NLSSM(state_dim, obs_dim,
                 transition_fn,
                 observation_fn,
                 process_noise_dist,
                 observation_noise_dist,
                 init_noise_dist,
                 x0)


def get_LogSVM_LGSSM(alpha: float, sigma: float, beta: float) -> tuple[LGSSM, tf.Tensor]:
    """
    Constructs a Linear Gaussian State-Space Model (LGSSM) approximation
    for the Log-Squared Stochastic Volatility Model.

    Model:
      x_{t+1} = alpha * x_t + eta_t,      eta_t ~ N(0, sigma^2)
      z_t     = x_t + bias + v_t,         v_t   ~ N(0, pi^2/2)

    Where z_t = log(y_t^2).

    Returns:
        model: An LGSSM instance configured with:
               A=alpha, C=1, Q=sigma^2, R=pi^2/2.
        bias:  The scalar bias term [log(beta^2) + E[log(epsilon^2)]].
               This must be subtracted from log(y^2) before filtering.
    """
    state_dim = 1
    obs_dim = 1

    # 1. Transition Matrix A = [[alpha]]
    A = tf.constant([[alpha]], dtype=tf.float32)

    # 2. Observation Matrix C = [[1.0]]
    C = tf.constant([[1.0]], dtype=tf.float32)

    # 3. Process Noise Covariance Q = [[sigma^2]]
    Q = tf.constant([[sigma**2]], dtype=tf.float32)

    # 4. Observation Noise Covariance R = [[pi^2 / 2]]
    # The variance of log(chi^2_1) is exactly pi^2 / 2 (~4.93)
    R = tf.constant([[math.pi**2 / 2]], dtype=tf.float32)

    # 5. Initial State Mean x0 = [[0.0]]
    x0 = tf.zeros([state_dim, 1], dtype=tf.float32)

    # 6. Initial State Covariance P0 = [[sigma^2 / (1 - alpha^2)]]
    # This is the stationary variance of the AR(1) process
    P0 = tf.constant([[sigma**2 / (1 - alpha**2)]], dtype=tf.float32)

    # 7. Bias Calculation
    # E[log(epsilon^2)] = digamma(0.5) - log(0.5) approx -1.27
    expected_log_chi2 = tf.math.digamma(0.5) - tf.math.log(0.5)
    bias = tf.math.log(beta**2) + expected_log_chi2

    # Create the LGSSM instance
    params = [A, C, Q, R, x0, P0]
    model = LGSSM(state_dim, obs_dim, params=params)

    return model, bias


def getVasicekBondPriceModel(kappa: float, theta: float, sigma: float, tau: float, dt: float,x0:tf.Tensor=tf.zeros((1,))) -> NLSSM:
    """
    Vasicek Short Rate Model with Zero-Coupon Bond Observation.

    State (Hidden): Centered Short Rate x_t = r_t - theta
    Dynamics: dx_t = -kappa * x_t * dt + sigma * dW_t
    Observation: y_t = Price(r_t, tau) + noise

    Price P(r, tau) = A(tau) * exp(-B(tau) * r)

    :param kappa: Mean reversion speed
    :param theta: Long-term mean rate
    :param sigma: Volatility of the rate
    :param tau: Time to maturity of the bond (in years)
    :param dt: Time step size for the simulation
    """
    state_dim = 1
    obs_dim = 1

    # Vasicek Bond Pricing Coefficients
    # B(tau) = (1 - exp(-kappa*tau)) / kappa
    B = (1.0 - math.exp(-kappa * tau)) / kappa

    # A(tau) term (standard affine term structure formula)
    term1 = (theta - (sigma ** 2) / (2 * kappa ** 2)) * (B - tau)
    term2 = (sigma ** 2) / (4 * kappa) * (B ** 2)
    A_coeff = math.exp(term1 - term2)

    # Discrete time parameters
    # x_{t+1} = (1 - kappa*dt) * x_t + noise
    proc_noise_std = sigma * math.sqrt(dt)

    # Observation noise (market noise in bond prices)
    obs_noise_std = 0.01  # e.g., 2 cents on a $1 par value

    ar_coef_tf = tf.constant(1.0 - kappa * dt, dtype=tf.float32)
    A_coeff_tf = tf.constant(A_coeff, dtype=tf.float32)
    B_tf = tf.constant(B, dtype=tf.float32)
    theta_tf = tf.constant(theta, dtype=tf.float32)

    def transition_fn(x, process_noise):
        return ar_coef_tf * x + process_noise

    def observation_fn(x, observation_noise):
        r_t = x + theta_tf
        price = A_coeff_tf * tf.exp(-B_tf * r_t)
        return price + observation_noise

    process_noise_dist = tfd.Normal(loc=0.0, scale=proc_noise_std)
    observation_noise_dist = tfd.Normal(loc=0.0, scale=obs_noise_std)

    # Initial state: Start at the mean (x=0 implies r=theta)
    stationary_variance = (sigma ** 2) / (2 * kappa)
    init_noise_dist = tfd.Normal(loc=0.0, scale=math.sqrt(stationary_variance))
    x0 = tf.reshape(x0, [state_dim, 1])

    return NLSSM(state_dim, obs_dim,
                 transition_fn,
                 observation_fn,
                 process_noise_dist,
                 observation_noise_dist,
                 init_noise_dist,
                 x0)


def getVasicekLGSSM(kappa: float, theta: float, sigma: float, tau: float, dt: float,x0:tf.Tensor=tf.zeros((1,))) -> tuple[LGSSM, float]:
    """
    Constructs a Static Linear Approximation (LGSSM) of the Vasicek Bond Model.
    Linearizes the bond price function around the long-term mean (r = theta).

    Returns:
        model: LGSSM
        bias: The price at the mean P(theta), to be subtracted from observations.
    """
    state_dim = 1
    obs_dim = 1

    # 1. Coefficients (Same as above)
    B = (1.0 - math.exp(-kappa * tau)) / kappa
    term1 = (theta - (sigma ** 2) / (2 * kappa ** 2)) * (B - tau)
    term2 = (sigma ** 2) / (4 * kappa) * (B ** 2)
    A_coeff = math.exp(term1 - term2)

    # 2. Linearization Point (r = theta, so x = 0)
    price_at_mean = A_coeff * math.exp(-B * theta)

    # Jacobian at mean: dP/dr = -B * P(r)
    # Since x = r - theta, dP/dx = dP/dr
    C_val = -B * price_at_mean

    # 3. LGSSM Matrices
    A_mat = tf.constant([[1.0 - kappa * dt]], dtype=tf.float32)
    C_mat = tf.constant([[C_val]], dtype=tf.float32)
    Q_mat = tf.constant([[(sigma * math.sqrt(dt)) ** 2]], dtype=tf.float32)
    R_mat = tf.constant([[0.02 ** 2]], dtype=tf.float32)  # Same obs noise as NLSSM

    x0 = x0
    stationary_variance = (sigma ** 2) / (2 * kappa)
    P0 = tf.constant([[stationary_variance]], dtype=tf.float32)

    params = [A_mat, C_mat, Q_mat, R_mat, x0, P0]
    model = LGSSM(state_dim, obs_dim, params=params)

    return model, price_at_mean


class ExtendedKalmanFilter:
    def __init__(self, model):
        # model: state_dim, obs_dim, transition_fn, observation_fn
        #        process_noise, observation_noise, init_noise, x0
        self.model = model

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

    @tf.function
    def _linearize(self, fn, x, noise):
        """Single sample linearization."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(noise)
            val = fn(x, noise)

        J_x = tape.jacobian(val, x, experimental_use_pfor=False)
        J_noise = tape.jacobian(val, noise, experimental_use_pfor=False)
        del tape
        return val, J_x, J_noise

    @tf.function
    def _batch_linearize(self, fn, x, noise):
        """
        Batch linearization using batch_jacobian for efficiency.
        x: [Batch, Dim]
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(noise)
            val = fn(x, noise)

        # batch_jacobian returns [Batch, Dim_out, Dim_in]
        J_x = tape.batch_jacobian(val, x, experimental_use_pfor=True)
        J_noise = tape.batch_jacobian(val, noise, experimental_use_pfor=True)
        del tape
        return val, J_x, J_noise

    @tf.function
    def filter(self, y: tf.Tensor, T: int, requires_stabilization=True, return_jacobians=False):
        state_dim = self.model.state_dim
        obs_dim = self.model.obs_dim
        Q = self._get_cov(self.model.process_noise)
        R = self._get_cov(self.model.observation_noise)
        P0 = self._get_cov(self.model.init_noise)

        # Initialization
        x_curr = tf.reshape(self.model.x0, [state_dim, 1])
        P_curr = P0

        # Pre-allocate zero noise for linearization
        zero_proc = tf.zeros([state_dim], dtype=tf.float32)
        zero_obs = tf.zeros([obs_dim], dtype=tf.float32)

        # Storage
        x_pred_arr = tf.TensorArray(dtype=tf.float32, size=T)
        P_pred_arr = tf.TensorArray(dtype=tf.float32, size=T)
        x_filt_arr = tf.TensorArray(dtype=tf.float32, size=T)
        P_filt_arr = tf.TensorArray(dtype=tf.float32, size=T)

        # A_arr stores Jacobian F_t for transition t -> t+1
        # Size T-1 is sufficient for smoothing, but T is safer for indexing
        A_arr = tf.TensorArray(dtype=tf.float32, size=T)

        log_l = 0.0
        const_term = -0.5 * obs_dim * tf.math.log(2 * math.pi)
        for t in range(T):
            # --- Prediction Step ---
            if t == 0:
                x_pred = x_curr
                P_pred = P_curr
            else:
                # 1. Retrieve Jacobian A_{t-1} computed in previous step
                # In this loop structure, we actually compute prediction using the *previous* state
                # So we must compute linearization at x_{t-1|t-1}
                # Note: x_curr holds x_{t-1|t-1} here
                f_val, A, W = self._linearize(self.model.transition_fn, x_curr, zero_proc)
                A = tf.reshape(A, [state_dim, state_dim])
                W = tf.reshape(W, [state_dim, state_dim])
                # Store Jacobian A_{t-1} which drives (t-1) -> t
                A_arr = A_arr.write(t - 1, A)

                x_pred = f_val
                P_pred = A @ P_curr @ tf.transpose(A) + W @ Q @ tf.transpose(W)
                # Symmetrize Prediction
                P_pred = 0.5 * (P_pred + tf.transpose(P_pred))

            # --- Update Step ---
            h_val, C, V = self._linearize(self.model.observation_fn, x_pred, zero_obs)
            C = tf.reshape(C, [obs_dim, state_dim])
            V = tf.reshape(V, [obs_dim, obs_dim])

            innov = tf.reshape(y[t] - h_val, [obs_dim, 1])
            R_eff = V @ R @ tf.transpose(V)
            S_t = C @ P_pred @ tf.transpose(C) + R_eff + 1e-6 * tf.eye(obs_dim)
            # Stable Inversion
            S_chol = tf.linalg.cholesky(S_t)
            # K = P C^T S^{-1}
            # We solve S K^T = C P  => K^T = S^{-1} C P
            Kt_transposed = tf.linalg.cholesky_solve(S_chol, C @ P_pred)
            K_t = tf.transpose(Kt_transposed)

            x_curr = x_pred + K_t @ innov
            I_KC = tf.eye(state_dim) - K_t @ C
            if requires_stabilization:
                # Joseph Form: (I-KC)P(I-KC)' + KRK'
                P_curr = I_KC @ P_pred @ tf.transpose(I_KC) + K_t @ R_eff @ tf.transpose(K_t)
            else:
                P_curr = I_KC @ P_pred

            # Enforce Symmetry
            P_curr = 0.5 * (P_curr + tf.transpose(P_curr))
            # Log Likelihood Calculation
            log_det_S = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(S_chol)))
            quad_term = tf.squeeze(tf.transpose(innov) @ tf.linalg.cholesky_solve(S_chol, innov))
            log_l += const_term - 0.5 * log_det_S - 0.5 * quad_term

            x_pred_arr = x_pred_arr.write(t, x_pred)
            P_pred_arr = P_pred_arr.write(t, P_pred)
            x_filt_arr = x_filt_arr.write(t, x_curr)
            P_filt_arr = P_filt_arr.write(t, P_curr)

        # For the very last step, we don't strictly need A_T, but let's pad it
        # Or, simpler: The smoother loop only goes up to T-2 accessing A_{T-2}

        results = (x_filt_arr.stack(), P_filt_arr.stack(), x_pred_arr.stack(), P_pred_arr.stack(), log_l)
        if return_jacobians:
            return results + (A_arr.stack(),)
        return results

    @tf.function
    def smooth(self, y: tf.Tensor, T: int):
        """
        RTS Smoother.
        Fixed Indexing: A_arr[t] now corresponds to transition t -> t+1.
        """
        x_filt, P_filt, x_pred, P_pred, log_l, A_arr = self.filter(y, T, return_jacobians=True)

        x_smooth = tf.TensorArray(dtype=tf.float32, size=T, clear_after_read=False)
        P_smooth = tf.TensorArray(dtype=tf.float32, size=T, clear_after_read=False)
        P_cross = tf.TensorArray(dtype=tf.float32, size=T - 1)

        # Initialize with last filtered state
        x_smooth = x_smooth.write(T - 1, x_filt[T - 1])
        P_smooth = P_smooth.write(T - 1, P_filt[T - 1])

        # range(T-2, -1, -1) means we iterate t from T-2 down to 0
        for t in tf.range(T - 2, -1, -1):
            P_filt_t = P_filt[t]
            P_pred_next = P_pred[t + 1]

            # Jacobian for transition t -> t+1
            # In the filter loop above, we wrote A_{t} at index t.
            A_t = A_arr[t]
            # Smoother Gain: J = P_{t|t} * A^T * P_{t+1|t}^{-1}
            # Solve P_{t+1|t} * J^T = A * P_{t|t}
            P_pred_next_chol = tf.linalg.cholesky(P_pred_next + 1e-6 * tf.eye(self.model.state_dim))

            # J = (P_filt_t @ A_t.T) @ inv(P_pred_next)
            # transpose(J) = inv(P_pred_next) @ (A_t @ P_filt_t)
            J_transposed = tf.linalg.cholesky_solve(P_pred_next_chol, A_t @ P_filt_t)
            J_t = tf.transpose(J_transposed)

            x_next_smooth = x_smooth.read(t + 1)
            P_next_smooth = P_smooth.read(t + 1)

            x_curr = x_filt[t] + J_t @ (x_next_smooth - x_pred[t + 1])
            P_curr = P_filt_t + J_t @ (P_next_smooth - P_pred_next) @ tf.transpose(J_t)
            P_curr = 0.5 * (P_curr + tf.transpose(P_curr))  # Symmetrize
            # Cross covariance P_{t+1, t | T} for EM
            # Standard approx: P_{t+1, t | T} approx P_{t+1|T} J_t^T
            P_cross_val = P_next_smooth @ tf.transpose(J_t)

            x_smooth = x_smooth.write(t, x_curr)
            P_smooth = P_smooth.write(t, P_curr)
            P_cross = P_cross.write(t, P_cross_val)

        return x_smooth.stack(), P_smooth.stack(), P_cross.stack(), log_l

    @tf.function
    def batch_smooth(self, Y):
        T = int(tf.shape(Y)[1])
        return tf.vectorized_map(lambda seq: self.smooth(seq, T), Y)

    def fit_EM(self, Y: tf.Tensor, n_iter: int = 10):
        """
        Infering model parameters (noise covariance matrices)
        using EM on a batch of sequences.
        """
        # Ensure Y has 3 dims: [Batch, T, Obs]
        if len(Y.shape) == 2 and self.model.obs_dim == 1:
            Y = tf.expand_dims(Y, axis=-1)

        batch_size = tf.shape(Y)[0]
        T = tf.shape(Y)[1]
        state_dim = self.model.state_dim
        obs_dim = self.model.obs_dim

        total_samples = tf.cast(batch_size * T, tf.float32)
        total_transitions = tf.cast(batch_size * (T - 1), tf.float32)
        # Get current values and ensure they are vectors [dim]
        init_Q_std = self.model.process_noise.stddev()
        if len(init_Q_std.shape) == 0:
            init_Q_std = tf.fill([state_dim], init_Q_std)
        else:
            init_Q_std = tf.broadcast_to(init_Q_std, [state_dim])

        init_R_std = self.model.observation_noise.stddev()
        if len(init_R_std.shape) == 0:
            init_R_std = tf.fill([obs_dim], init_R_std)
        else:
            init_R_std = tf.broadcast_to(init_R_std, [obs_dim])

        Q_var = tf.Variable(init_Q_std, dtype=tf.float32)
        R_var = tf.Variable(init_R_std, dtype=tf.float32)
        # Replace model distributions ONCE with variable-backed distributions
        self.model.process_noise = tfd.Normal(loc=tf.zeros(state_dim), scale=Q_var)
        self.model.observation_noise = tfd.Normal(loc=tf.zeros(obs_dim), scale=R_var)

        log_likelihoods = []
        for i in range(n_iter):
            # E-Step: Run Smoothing on Batch
            # Use vectorized_map or a custom batch_smooth function if defined
            # Here we wrap the smooth function
            x_smooth, P_smooth, P_cross, log_L = self.batch_smooth(Y)
            mean_log_L = tf.reduce_mean(log_L)
            log_likelihoods.append(mean_log_L)
            if i>=1 and tf.abs(log_likelihoods[-1]-log_likelihoods[-2])<1e-3:
                print(f'EM converged at iteration {i}.')
                break

            # M-Step
            y_flat = tf.reshape(Y, [-1, obs_dim])
            x_flat = tf.reshape(x_smooth, [-1, state_dim])
            P_flat = tf.reshape(P_smooth, [-1, state_dim, state_dim])
            # update R by R = 1/N * Sum (res*res.T + H*P*H.T)
            zero_obs = tf.zeros([tf.shape(x_flat)[0], obs_dim])
            h_val, H, _ = self._batch_linearize(self.model.observation_fn, x_flat, zero_obs)

            res_y = tf.expand_dims(y_flat - h_val, -1)  # [N, Obs, 1]
            term_R = res_y @ tf.transpose(res_y, perm=[0, 2, 1]) + \
                     H @ P_flat @ tf.transpose(H, perm=[0, 2, 1])
            new_R = tf.reduce_sum(term_R, axis=0) / total_samples
            # Update Process Noise Q
            # P_cross is [Batch, T-1, State, State]
            x_curr = x_smooth[:, :-1, :]
            x_next = x_smooth[:, 1:, :]
            P_curr = P_smooth[:, :-1, :, :]
            P_next = P_smooth[:, 1:, :, :]

            x_curr_flat = tf.reshape(x_curr, [-1, state_dim])
            x_next_flat = tf.reshape(x_next, [-1, state_dim])
            P_curr_flat = tf.reshape(P_curr, [-1, state_dim, state_dim])
            P_next_flat = tf.reshape(P_next, [-1, state_dim, state_dim])
            P_cross_flat = tf.reshape(P_cross, [-1, state_dim, state_dim])

            zero_proc = tf.zeros([tf.shape(x_curr_flat)[0], state_dim])
            f_val, A, _ = self._batch_linearize(self.model.transition_fn, x_curr_flat, zero_proc)
            res_x = tf.expand_dims(x_next_flat - f_val, -1)

            term_Q = (res_x @ tf.transpose(res_x, perm=[0, 2, 1]) +
                      P_next_flat +
                      A @ P_curr_flat @ tf.transpose(A, perm=[0, 2, 1]) -
                      P_cross_flat @ tf.transpose(A, perm=[0, 2, 1]) -
                      A @ tf.transpose(P_cross_flat, perm=[0, 2, 1]))
            new_Q = tf.reduce_sum(term_Q, axis=0) / total_transitions
            new_R_std = tf.sqrt(tf.linalg.diag_part(new_R))
            new_Q_std = tf.sqrt(tf.linalg.diag_part(new_Q))

            Q_var.assign(new_Q_std)
            R_var.assign(new_R_std)
        return log_likelihoods



class UnscentedKalmanFilter(tf.Module):
    def __init__(self, model, alpha=1e-3, beta=2.0, kappa=0.0,train_noise=False):
        super(UnscentedKalmanFilter, self).__init__()
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        # UKF Hyperparameters
        self.alpha = tf.Variable(alpha, dtype=tf.float32, name='ukf_alpha')
        self.beta = tf.Variable(beta, dtype=tf.float32, name='ukf_beta')
        self.kappa = tf.Variable(kappa, dtype=tf.float32, name='ukf_kappa')

        # Noise Parameters (Trainable)
        # defined as part of the model
        self.train_noise = train_noise
        self.proc_log_scale = self._create_log_scale(model.process_noise, self.state_dim, 'proc')
        self.obs_log_scale = self._create_log_scale(model.observation_noise, self.obs_dim, 'obs')

    def _create_log_scale(self, dist, dim, name):
        """Helper to create trainable log-scale variables from existing distributions."""
        if not self.train_noise:
            return None

        # Get initial value from model
        if hasattr(dist, 'scale'):
            val = dist.scale
        elif hasattr(dist, 'stddev'):
            val = dist.stddev()
        else:
            val = 1.0

        val = tf.convert_to_tensor(val, dtype=tf.float32)
        if len(val.shape) == 0: val = tf.fill([dim], val)

        # Initialize with log(val) for numerical stability
        return tf.Variable(tf.math.log(val + 1e-6), name=f'{name}_log_scale')

    def _compute_weights(self):
        n = self.state_dim
        lam = self.alpha ** 2 * (n + self.kappa) - n
        Wm_0 = lam / (n + lam)
        Wm_rest = 0.5 / (n + lam)
        # Weights shape: [2*n + 1]
        Wm = tf.concat([[Wm_0], tf.fill([2 * n], Wm_rest)], axis=0)
        Wc_0 = Wm_0 + (1 - self.alpha ** 2 + self.beta)
        Wc = tf.concat([[Wc_0], tf.fill([2 * n], Wm_rest)], axis=0)
        return Wm, Wc, lam

    @staticmethod
    def _get_mean(dist, dim):
        """
        Safely extracts the mean from a distribution.
        """
        try:
            # Most TFP distributions implement .mean()
            mean = dist.mean()
        except (AttributeError, NotImplementedError):
            # Fallback: if mean is not defined, assume 0 (e.g. simple centered noise)
            mean = tf.zeros([dim])

        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        # If mean is scalar (e.g. 0.0), broadcast to [Dim]
        if len(mean.shape) == 0:
            mean = tf.fill([dim], mean)
        return mean

    @staticmethod
    def _get_cov(dist, log_scale_var=None):
        """
        Computes covariance. Prioritizes the trainable variable if it exists.
        """
        if log_scale_var is not None:
            # Use the trainable variable: Variance = exp(log_scale)^2
            scale = tf.exp(log_scale_var)
            return tf.linalg.diag(tf.square(scale))

        # Fallback to the fixed distribution object
        try:
            cov = dist.covariance()
        except (AttributeError, NotImplementedError):
            var = dist.variance()
            if len(var.shape) == 0: var = tf.reshape(var, [1])
            cov = tf.linalg.diag(var)

        if len(cov.shape) == 0:
            cov = tf.reshape(cov, [1, 1])
        elif len(cov.shape) == 1:
            cov = tf.linalg.diag(cov)
        return tf.cast(cov, dtype=tf.float32)

    def _generate_sigma_points(self, x, P, lam):
        # x: [Batch, Dim]
        # P: [Batch, Dim, Dim]
        n = self.state_dim
        scale = tf.sqrt(tf.cast(n + lam, tf.float32))
        P_sym = 0.5 * (P + tf.linalg.matrix_transpose(P)) + 1e-6 * tf.eye(n)
        L = tf.linalg.cholesky(P_sym)  # [Batch, n, n] Lower triangular
        scaled_L = scale * L  # [Batch, n, n]

        # 2. Broadcasting trick to avoid loops
        # x shape: [Batch, n] -> [Batch, n, 1]
        x_expanded = tf.expand_dims(x, -1)

        # [Batch, n, 1] + [Batch, n, n] -> [Batch, n, n] (Broadcasts x across columns)
        right = x_expanded + scaled_L
        left = x_expanded - scaled_L
        sigmas_concat = tf.concat([x_expanded, right, left], axis=2)
        # [Batch, 2n+1, Dim]
        return tf.transpose(sigmas_concat, perm=[0, 2, 1])

    def _compute_stats(self, sigma_points, Wm, Wc, noise_cov):
        # sigma_points: [Batch, 2n+1, Dim]
        # Wm, Wc: [2n+1]

        # Weighted Mean: Sum over sigma point dim (axis 1)
        # Wm is 1D, we broadcast it.
        x_mean = tf.tensordot(sigma_points, Wm, axes=[[1], [0]])  # [Batch, Dim]
        residuals = sigma_points - tf.expand_dims(x_mean, 1)  # [Batch, 2n+1, Dim]

        Wc_b = tf.reshape(Wc, [1, -1, 1])
        weighted_res = residuals * Wc_b  # [Batch, 2n+1, Dim]

        # Batch matmul: (B, Dim, 2n+1) @ (B, 2n+1, Dim) -> (B, Dim, Dim)
        P = tf.matmul(tf.transpose(weighted_res, perm=[0, 2, 1]), residuals)+noise_cov
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))
        return x_mean, P, residuals

    def _predict_step(self, x_curr, P_curr, Wm, Wc, lam, Q, noise_proc_mean):
        sig_pts = self._generate_sigma_points(x_curr, P_curr, lam)  # [Batch, 2n+1, n]

        # Flatten batch and sigma dims to ensure compatibility with NLSSM model
        batch_size = tf.shape(sig_pts)[0]
        num_sig = tf.shape(sig_pts)[1]
        sig_pts_flat = tf.reshape(sig_pts, [batch_size * num_sig, self.state_dim])

        # [Batch_Total, Dim]
        sig_pts_prop_flat = self.model.transition_fn(sig_pts_flat, noise_proc_mean)
        sig_pts_prop = tf.reshape(sig_pts_prop_flat, [batch_size, num_sig, self.state_dim])

        x_pred, P_pred, _ = self._compute_stats(sig_pts_prop, Wm, Wc, Q)
        P_pred = 0.5 * (P_pred + tf.linalg.matrix_transpose(P_pred))
        return x_pred, P_pred, sig_pts_prop

    def _update_step(self, x_pred, P_pred, y, Wm, Wc, lam, R, noise_obs_mean):
        # y: [Batch, Obs]

        sig_pts_pred = self._generate_sigma_points(x_pred, P_pred, lam)
        batch_size = tf.shape(sig_pts_pred)[0]
        num_sig = tf.shape(sig_pts_pred)[1]
        sig_pts_pred_flat = tf.reshape(sig_pts_pred, [batch_size * num_sig, self.state_dim])

        sig_pts_obs_flat = self.model.observation_fn(sig_pts_pred_flat, noise_obs_mean)
        sig_pts_obs = tf.reshape(sig_pts_obs_flat, [batch_size, num_sig, self.obs_dim])

        y_pred_mean, S, y_residuals = self._compute_stats(sig_pts_obs, Wm, Wc, R)
        x_residuals = sig_pts_pred - tf.expand_dims(x_pred, 1)
        Wc_b = tf.reshape(Wc, [1, -1, 1])
        weighted_x_res = x_residuals * Wc_b
        # [Batch, State, Obs]
        P_xy = tf.matmul(tf.transpose(weighted_x_res, perm=[0, 2, 1]), y_residuals)

        # S_chol: [Batch, Obs, Obs]
        S_chol = tf.linalg.cholesky(S + 1e-6 * tf.eye(self.obs_dim))

        # Solve S * K^T = P_xy^T  => K^T = S^-1 P_xy^T
        # Input  [Batch, Obs, State]
        Kt_transposed = tf.linalg.cholesky_solve(S_chol, tf.linalg.matrix_transpose(P_xy))
        K = tf.linalg.matrix_transpose(Kt_transposed)  # [Batch, State, Obs]
        innovation = y - y_pred_mean  # [Batch, Obs]
        innovation_expanded = tf.expand_dims(innovation, -1)  # [Batch, Obs, 1]
        x_update = tf.matmul(K, innovation_expanded)
        x_new = x_pred + tf.squeeze(x_update, -1)
        # P_new = P_pred - K S K^T
        # KSKt = K @ S @ K.T
        KSKt = tf.matmul(K, tf.matmul(S, K, transpose_b=True))
        P_new = P_pred - KSKt
        P_new = 0.5 * (P_new + tf.linalg.matrix_transpose(P_new))

        const_term = -0.5 * self.obs_dim * tf.math.log(2 * math.pi)
        diag_S=tf.maximum(tf.linalg.diag_part(S_chol),1e-6)
        log_det_S = 2 * tf.reduce_sum(tf.math.log(diag_S), axis=1)
        sol = tf.linalg.cholesky_solve(S_chol, innovation_expanded)
        quad_term = tf.squeeze(tf.matmul(tf.transpose(innovation_expanded, perm=[0, 2, 1]), sol), [1, 2])
        log_l = const_term - 0.5 * log_det_S - 0.5 * quad_term  # [Batch]
        # print('\n')
        # print('const: ',const_term)
        # print('log_det_S: ',log_det_S)
        # print('quad_term: ',quad_term)
        # print('min diag S_chol: ',tf.reduce_min(tf.linalg.diag_part(S_chol)))

        return x_new, P_new, log_l

    def filter(self, y: tf.Tensor, py_loop=False):
        """
        Batch-enabled UKF Filter.
        :param y: Observations [Batch, T, Obs]
                  Input MUST be standardized to Rank 3 before calling this method.
        """
        # Standardize input to [Batch, T, Obs]
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        batch_size = tf.shape(y)[0]

        Q = self._get_cov(self.model.process_noise, self.proc_log_scale)
        R = self._get_cov(self.model.observation_noise, self.obs_log_scale)
        P0 = self._get_cov(self.model.init_noise)
        Wm, Wc, lam = self._compute_weights()

        q_mean = self._get_mean(self.model.process_noise, self.state_dim)
        r_mean = self._get_mean(self.model.observation_noise, self.obs_dim)

        # Initialize State [Batch, State]
        x_init = tf.reshape(self.model.x0, [1, self.state_dim])
        x_init = tf.tile(x_init, [batch_size, 1])

        # Initialize Covariance [Batch, State, State]
        P_init = tf.expand_dims(P0, 0)
        P_init = tf.tile(P_init, [batch_size, 1, 1])

        # Transpose y to [T, Batch, Obs] for scanning over time
        y_time_major = tf.transpose(y, perm=[1, 0, 2])
        x_0, P_0, log_l_0 = self._update_step(x_init, P_init, y_time_major[0], Wm, Wc, lam, R, r_mean)

        # Carry: (x_curr, P_curr, accum_log_l)
        @tf.function
        def scan_step(carry, y_t):
            x_prev, P_prev, _ = carry
            x_pred, P_pred, _ = self._predict_step(x_prev, P_prev, Wm, Wc, lam, Q, q_mean)
            x_filt, P_filt, log_l_t = self._update_step(x_pred, P_pred, y_t, Wm, Wc, lam, R, r_mean)
            return x_filt, P_filt, log_l_t

        if not py_loop:
            y_rest = y_time_major[1:]
            # Output of scan is stacked over time: [T-1, Batch, ...]
            x_rest, P_rest, log_l_rest = tf.scan(
                scan_step,
                y_rest,
                initializer=(x_0, P_0, log_l_0)
            )

            x_hist = tf.concat([tf.expand_dims(x_0, 0), x_rest], axis=0)  # [T, Batch, State]
            P_hist = tf.concat([tf.expand_dims(P_0, 0), P_rest], axis=0)  # [T, Batch, State, State]
            log_l_hist = tf.concat([tf.expand_dims(log_l_0, 0), log_l_rest], axis=0)  # [T, Batch]

            x_hist = tf.transpose(x_hist, perm=[1, 0, 2]) # [Batch, T, State]
            P_hist = tf.transpose(P_hist, perm=[1, 0, 2, 3])
            log_l_hist = tf.transpose(log_l_hist, perm=[1, 0]) # [Batch, T]
            return x_hist, P_hist,tf.reduce_sum(log_l_hist,axis=-1) # [Batch]
        else:
            T=tf.shape(y_time_major)[0]
            log_l_ta=tf.TensorArray(dtype=tf.float32, size=T)
            log_l_ta=log_l_ta.write(0,log_l_0)
            x,P,log_l=x_0,P_0,log_l_0
            x_ta=tf.TensorArray(dtype=tf.float32, size=T)
            P_ta=tf.TensorArray(dtype=tf.float32, size=T)
            x_ta=x_ta.write(0,x)
            P_ta=P_ta.write(0,P)
            for t in range(1,T):
                x, P, log_l = scan_step((x, P, log_l), y_time_major[t])
                log_l_ta=log_l_ta.write(t,log_l)
                x_ta=x_ta.write(t,x)
                P_ta=P_ta.write(t,P)
            x_hist=x_ta.stack() # [T, Batch, State]
            x_hist = tf.transpose(x_hist, perm=[1, 0, 2]) # [Batch, T, State]
            P_hist=P_ta.stack() # [T, Batch, State, State]
            P_hist = tf.transpose(P_hist, perm=[1, 0, 2, 3]) # [Batch, T, State, State]
            log_l_hist=log_l_ta.stack() # [T, Batch]
            log_l_hist = tf.transpose(log_l_hist, perm=[1, 0]) # [Batch, T]
            return x_hist, P_hist, tf.reduce_sum(log_l_hist,axis=-1) # [Batch]

    def fit(self, y_batch: tf.Tensor, n_iter=100, learning_rate=0.01,py_loop=False):
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
        if len(y_batch.shape) == 2:
            y_batch = tf.expand_dims(y_batch, -1)
        elif len(y_batch.shape) == 1:
            y_batch = tf.reshape(y_batch, [1, -1, 1])

        trainable_vars = [self.alpha, self.beta, self.kappa]
        if self.proc_log_scale is not None: trainable_vars.append(self.proc_log_scale)
        if self.obs_log_scale is not None: trainable_vars.append(self.obs_log_scale)

        optimizer = tf.optimizers.Adam(learning_rate)

        def train_step():
            with tf.GradientTape() as tape:
                # Run filter on the WHOLE batch at once
                _, _, log_likelihoods = self.filter(y_batch, py_loop)
                loss = -tf.reduce_mean(log_likelihoods)

            grads = tape.gradient(loss, trainable_vars)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            optimizer.apply_gradients(zip(grads, trainable_vars))
            return loss

        losses=[]
        for i in range(n_iter):
            loss_tensor = train_step()
            loss_numpy = loss_tensor.numpy()
            # SAFE LOGGING: Handle both Scalar and Vector cases to prevent crash
            if loss_numpy.ndim == 0:
                # It is a scalar (ideal case)
                scalar_loss = float(loss_numpy.item())
            else:
                # It is a vector (fallback) - take mean
                scalar_loss = float(loss_numpy.mean())
            losses.append(scalar_loss)

            if i % 10 == 0:
                print(f"Iter {i}: Loss={scalar_loss:.4f}")
        self.sync_model()
        return losses

    def sync_model(self):
        """
        Updates the internal NLSSM model object with the learned noise parameters.
        """
        if self.proc_log_scale is not None:
            learned_std = tf.exp(self.proc_log_scale)
            # Create a new distribution with the learned scale
            # Assuming Normal distribution for simplicity, adapt if using StudentT
            self.model.process_noise = tfd.Normal(
                loc=tf.zeros_like(learned_std),
                scale=learned_std
            )
        if self.obs_log_scale is not None:
            learned_std = tf.exp(self.obs_log_scale)
            self.model.observation_noise = tfd.Normal(
                loc=tf.zeros_like(learned_std),
                scale=learned_std
            )


def run_example_SVM(alpha,sigma,beta,T=500):
    model = get1DStochasticVolModel(alpha=alpha, sigma=sigma, beta=beta)
    model_st = get1DStochasticVolModel(alpha=alpha, sigma=sigma, beta=beta, heavy_tail=True)
    x, y = model.sample(T)
    x_st, y_st = model_st.sample(T)
    print('max range of y_t:', tf.reduce_max(y) - tf.reduce_min(y))
    print('max range of y_t (Heavy-Tailed):', tf.reduce_max(y_st) - tf.reduce_min(y_st))

    y_log_sq = tf.math.log(y ** 2 + 1e-8)
    ekf = ExtendedKalmanFilter(get1DLogSquaredSVM(alpha=alpha, sigma=sigma, beta=beta))
    x_filt_ekf, P_filt_ekf, _, _, log_l_ekf = ekf.filter(y_log_sq, T, requires_stabilization=True)
    x_filt_ekf = tf.squeeze(x_filt_ekf, axis=-1)

    lgssm_svm, bias = get_LogSVM_LGSSM(alpha=alpha, sigma=sigma, beta=beta)
    kf = KalmanFilter(lgssm_svm)
    y_log_sq_bias_corrected = y_log_sq - bias
    x_filt_kf, P_filt_kf, _, _, log_l_kf = kf.filter(tf.reshape(y_log_sq_bias_corrected, [-1, 1]), T,
                                                     requires_stabilization=True)
    x_filt_kf = tf.squeeze(x_filt_kf, axis=-1)

    if len(y_log_sq.shape)==1:
        y_log_sq = tf.reshape(y_log_sq, [1, -1, 1])  # Batch size 1
    ukf= UnscentedKalmanFilter(get1DLogSquaredSVM(alpha=alpha, sigma=sigma, beta=beta), alpha=1, beta=2.0, kappa=0.0) # For 1D, alpha=1 is better than 1e-3?
    t0=time.time()
    x_filt_ukf, P_filt_ukf,_ = ukf.filter(y_log_sq)
    t1=time.time()
    x_filt_ukf, P_filt_ukf, _ = ukf.filter(y_log_sq, py_loop=True)
    t2=time.time()
    print(f'UKF time (tf.scan): {t1 - t0:.4f} seconds')
    print(f'UKF time (py loop): {t2 - t1:.4f} seconds')

    print('\n')
    print('EKF max error:', tf.reduce_max(tf.abs(x - x_filt_ekf)).numpy())
    print('LGSSM max error:', tf.reduce_max(tf.abs(x - x_filt_kf)).numpy())
    print('UKF max error:', tf.reduce_max(tf.abs(x - x_filt_ukf)).numpy())
    print('EKF mean error:', tf.reduce_mean(tf.abs(x - x_filt_ekf)).numpy())
    print('LGSSM mean error:', tf.reduce_mean(tf.abs(x - x_filt_kf)).numpy())
    print('UKF mean error:', tf.reduce_mean(tf.abs(x - x_filt_ukf)).numpy())

    plt.figure(figsize=(12, 6))
    plt.plot(x.numpy(), label='Hidden States x_t', c='black', linestyle='--')
    plt.plot(x_filt_ekf.numpy(), label='Hidden States from Extended Kalman Filter', c='green')
    plt.plot(x_filt_kf.numpy(), label='Hidden States from LGSSM Approximation', c='blue')
    plt.plot(x_filt_ukf.numpy().flatten(), label='Hidden States from Unscented Kalman Filter', c='orange')
    plt.title('Stochastic Volatility Model - Hidden States Estimation')
    plt.scatter(range(T), y.numpy(), c='red', s=10, label='Observations y_t')
    plt.title('Stochastic Volatility Model - Hidden States')
    plt.xlabel('Time Step')
    plt.ylabel('x_t')
    plt.legend()

    plt.figure(figsize=(12, 6))
    plt.plot(x_st.numpy(), label='Hidden State x_t (Heavy-Tailed)')
    plt.scatter(range(T), y_st.numpy(), c='red', s=10, label='Observations y_t (Heavy-Tailed)')
    plt.title('Stochastic Volatility Model with Heavy-Tailed Process Noise - Hidden States')
    plt.xlabel('Time Step')
    plt.ylabel('x_t')
    plt.legend()

    plt.show()


def run_experiment_Vasicek(kappa, theta, sigma, tau, dt, T=200,x0=tf.zeros((1,))):
    model_vasicek = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt,x0=x0)
    x, y = model_vasicek.sample(T)
    print('max range of y_t:', tf.reduce_max(y) - tf.reduce_min(y))
    print('y_t shape:', y.shape)

    # --- EKF ---
    ekf = ExtendedKalmanFilter(model_vasicek)
    x_filt_ekf_raw, P_filt_ekf, _, _, log_l_ekf = ekf.filter(y, T, requires_stabilization=True)
    x_filt_ekf = tf.squeeze(x_filt_ekf_raw, axis=-1)

    # --- LGSSM ---
    lgssm_vasicek, bias = getVasicekLGSSM(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt,x0=x0)
    kf = KalmanFilter(lgssm_vasicek)
    y_bias_corrected = y - bias
    x_filt_kf, P_filt_kf, _, _, log_l_kf = kf.filter(tf.reshape(y_bias_corrected, [-1, 1]), T,
                                                     requires_stabilization=True)
    x_filt_kf = tf.squeeze(x_filt_kf, axis=-1)

    ukf= UnscentedKalmanFilter(model_vasicek,alpha=1.0, beta=0.0, kappa=0.0) # For 1D, alpha=1 is better than 1e-3
    y=tf.reshape(y, [1, -1, 1])  # Batch size 1
    x_filt_ukf, P_filt_ukf,_ = ukf.filter(y)
    x_filt_ukf = x_filt_ukf.numpy().flatten()

    print('\n')
    print('EKF max error:', tf.reduce_max(tf.abs(x - x_filt_ekf)).numpy())
    print('UKF max error:', tf.reduce_max(tf.abs(x - x_filt_ukf)).numpy())
    print('LGSSM max error:', tf.reduce_max(tf.abs(x - x_filt_kf)).numpy())
    print('\n')
    print('EKF mean error:', tf.reduce_mean(tf.abs(x - x_filt_ekf)).numpy())
    print('UKF mean error:', tf.reduce_mean(tf.abs(x - x_filt_ukf)).numpy())
    print('LGSSM mean error:', tf.reduce_mean(tf.abs(x - x_filt_kf)).numpy())

    # y_pred = h(x_filt)
    zero_noise = tf.zeros([1])
    y_pred_ekf = []
    y_pred_kf=[]
    for t in range(T):
        state_val = tf.reshape(x_filt_ekf[t], [1])
        pred = model_vasicek.observation_fn(state_val, zero_noise)
        y_pred_ekf.append(pred)
        y_pred_kf.append(lgssm_vasicek.C @ tf.reshape(x_filt_kf[t],[-1,1]) + bias)
    y_pred_ekf = tf.stack(y_pred_ekf)
    y_pred_kf = tf.reshape(tf.stack(y_pred_kf), [-1])

    # --- Plotting ---
    fig, ax1= plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    ax1.plot(x.numpy(), label='True Hidden States (Rate Deviation)', c='black', linestyle='--')
    ax1.plot(x_filt_ekf.numpy(), label='EKF Estimate', c='green')
    ax1.plot(x_filt_kf.numpy(), label='LGSSM Estimate', c='blue')
    ax1.plot(x_filt_ukf, label='UKF Estimate', c='orange')
    ax1.set_title('Hidden States: Interest Rate Deviation (x_t)')
    ax1.set_ylabel('Rate Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ax2.scatter(range(T), y.numpy(), c='red', s=15, label='Observed Bond Prices (y_t)', alpha=0.6)
    # ax2.plot(y_pred_ekf.numpy(), label='EKF Implied Price', c='green', linewidth=2)
    # ax2.plot(y_pred_kf.numpy(), label='LGSSM Implied Price', c='blue', linewidth=2)
    # ax2.set_title('Observations: Bond Prices (y_t)')
    # ax2.set_xlabel('Time Step')
    # ax2.set_ylabel('Price')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    plt.savefig('Figures\Vasicek_hidden_state_estimation.pdf', bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def run_experiment_Vasicek_EM(kappa, theta, sigma, tau, dt, T=50, batch_size=20, n_em_iter=20):
    model_vasicek = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt)
    x_batch, y_batch = model_vasicek.batch_sample(T, batch_size)
    print('max range of y_t in batch:', tf.reduce_max(y_batch) - tf.reduce_min(y_batch))
    if len(y_batch.shape) == 2:
        y_batch = tf.expand_dims(y_batch, -1)
    print('y_batch shape:', y_batch.shape)

    # initiate EKF noise parameters to default values
    model_vasicek.process_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.state_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.state_dim,), dtype=tf.float32) * 0.01
    )
    model_vasicek.observation_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.obs_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.obs_dim,), dtype=tf.float32) * 0.02
    )
    ekf = ExtendedKalmanFilter(model_vasicek)
    log_likelihoods_ekf = ekf.fit_EM(y_batch, n_iter=n_em_iter)

    model_vasicek_ukf = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt)
    model_vasicek_ukf.process_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.state_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.state_dim,), dtype=tf.float32) * 0.01
    )
    model_vasicek_ukf.observation_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.obs_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.obs_dim,), dtype=tf.float32) * 0.02
    )
    ukf=UnscentedKalmanFilter(model_vasicek_ukf,alpha=1, beta=2.0, kappa=0.0,train_noise=True)
    t0=time.time()
    log_likelihoods_ukf = ukf.fit(y_batch, n_iter=10*n_em_iter, learning_rate=0.05)
    t1=time.time()

    model_vasicek_ukf.process_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.state_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.state_dim,), dtype=tf.float32) * 0.01
    )
    model_vasicek_ukf.observation_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.obs_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.obs_dim,), dtype=tf.float32) * 0.02
    )
    ukf = UnscentedKalmanFilter(model_vasicek_ukf, alpha=1, beta=2.0, kappa=0.0, train_noise=True)
    t2=time.time()
    log_likelihoods_ukf = ukf.fit(y_batch, n_iter=10*n_em_iter, learning_rate=0.05,py_loop=True)
    t3=time.time()
    print(f'UKF fit time (tf.scan): {t1 - t0:.4f} seconds')
    print(f'UKF fit time (py loop): {t3 - t2:.4f} seconds')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(log_likelihoods_ekf) + 1), log_likelihoods_ekf, marker='o')
    plt.title('EM Log-Likelihood Progression')
    plt.xlabel('EM Iteration')
    plt.ylabel('Average Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.savefig('EKF_EM_vasicek_loglikelihood.pdf',bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(log_likelihoods_ukf) + 1), log_likelihoods_ukf, marker='o', color='orange')
    plt.title('UKF Loss Progression')
    plt.xlabel('Training Iteration')
    plt.ylabel('Average Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.savefig('UKF_EM_vasicek_loglikelihood.pdf', bbox_inches='tight')
    plt.show()

    est_proc_std = ekf.model.process_noise.stddev().numpy().flatten()[0]
    est_obs_std = ekf.model.observation_noise.stddev().numpy().flatten()[0]
    print('\n')
    print(f'True Process Noise Std: {sigma * (dt ** 0.5):.4f}, Estimated: {est_proc_std:.4f}')
    print(f'True Observation Noise Std: {0.01:.4f}, Estimated: {est_obs_std:.4f}')

    ukf_proc_std = ukf.model.process_noise.stddev().numpy().flatten()[0]
    ukf_obs_std = ukf.model.observation_noise.stddev().numpy().flatten()[0]
    print('\n')
    print(f'UKF Estimated Process Noise Std: {ukf_proc_std:.4f}')
    print(f'UKF Estimated Observation Noise Std: {ukf_obs_std:.4f}')


if __name__ == '__main__':
    #run_example_SVM(alpha=0.9, sigma=0.3, beta=0.5)
    run_experiment_Vasicek(kappa=0.1, theta=0.05, sigma=0.02, tau=20.0, dt=1/252,x0=tf.constant((0.05,)))
    #run_experiment_Vasicek_EM(kappa=0.2, theta=0.05, sigma=0.02, tau=15.0, dt=1/252, T=100, batch_size=50, n_em_iter=10)


