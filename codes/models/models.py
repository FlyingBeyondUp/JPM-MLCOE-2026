import tensorflow as tf
import tensorflow_probability as tfp
from .base_models import LGSSM,NLSSM
import time
import matplotlib.pyplot as plt
import math

tfd = tfp.distributions

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


def getLorenz96Model(
        state_dim: int,
        obs_dim: int=2,
        F: float = 8.0,
        dt: float = 0.05,
        process_noise_std: float = 0.0,
        observation_noise_std: float = 1.0,
        observe_every_nth: int = 1
) -> NLSSM:
    """
    Lorenz96 model:
    x_{t+1} = x_t + dt * L96_dynamics(x_t) + process_noise
    y_t = H @ x_t + observation_noise

    :param state_dim: dim of the state space
    :param obs_dim: dim of the observation space, usually <= state_dim
    :param F: Lorenz96 forcing term
    :param dt: time step for integration
    :param process_noise_std: process noise standard deviation
    :param observation_noise_std: observation noise standard deviation
    :param observe_every_nth: observation operator H observes every nth state variable
    :return: NLSSM instance representing the Lorenz96 model
    """
    obs_dim = state_dim // observe_every_nth
    # Lorenz96
    def lorenz96_dynamics(x):
        # x: [Batch, Dim] or [Dim]
        x_roll_left = tf.roll(x, shift=-1, axis=-1)  # x_{i+1}
        x_roll_right = tf.roll(x, shift=1, axis=-1)  # x_{i-1}
        x_roll_right2 = tf.roll(x, shift=2, axis=-1)  # x_{i-2}

        # dx/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        dxdt = (x_roll_left - x_roll_right2) * x_roll_right - x + F
        return dxdt

    # RK4 Transition Function
    def transition_fn(x: tf.Tensor, process_noise: tf.Tensor) -> tf.Tensor:
        """
        Fourth order Runge-Kutta integration for Lorenz96
        x: [Batch, Dim]
        """
        k1 = lorenz96_dynamics(x)
        k2 = lorenz96_dynamics(x + 0.5 * dt * k1)
        k3 = lorenz96_dynamics(x + 0.5 * dt * k2)
        k4 = lorenz96_dynamics(x + dt * k3)

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next + process_noise

    # observation matrix H: selects every nth variable
    def observation_fn(x: tf.Tensor, observation_noise: tf.Tensor) -> tf.Tensor:
        """从状态中提取观测"""
        # 选择每 observe_every_nth 个状态分量
        H_indices = list(range(observe_every_nth-1, state_dim, observe_every_nth))[:obs_dim]
        if len(x.shape) == 1:
            # 单样本情况
            observed = tf.gather(x, H_indices)
        else:
            # 批量情况 [Batch, State_Dim]
            observed = tf.gather(x, H_indices, axis=-1)
        return observed + observation_noise

    process_noise_dist = tfd.Normal(
        loc=tf.zeros(state_dim, dtype=tf.float32),
        scale=tf.ones(state_dim, dtype=tf.float32) * process_noise_std
    )

    observation_noise_dist = tfd.Normal(
        loc=tf.zeros(obs_dim, dtype=tf.float32),
        scale=tf.ones(obs_dim, dtype=tf.float32) * observation_noise_std
    )

    init_noise_dist = tfd.Normal(
        loc=tf.zeros(state_dim, dtype=tf.float32),
        scale=tf.ones(state_dim, dtype=tf.float32) * 0.5
    )

    x0 = tf.ones(state_dim, dtype=tf.float32) * F
    x0 = tf.tensor_scatter_nd_update(x0, [[0]], [F + 0.01])  # 在第一个分量加小扰动

    return NLSSM(
        state_dim=state_dim,
        obs_dim=obs_dim,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        process_noise=process_noise_dist,
        observation_noise=observation_noise_dist,
        init_noise=init_noise_dist,
        x0=x0
    )















