import abc
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Union

tfd = tfp.distributions
dtype = tf.float32


class AbstractSSM(tf.keras.Model, abc.ABC):
    """
    Abstract interface for all State-Space Models (SSMs).

    This class enforces a standard contract for tracking algorithms. Whether
    the model is linear, non-linear, or driven by neural networks,
    filters will interact with it using these standard methods.

    Mathematical notation used in comments:
    - B: Batch size
    - T: Number of time steps for each sequence
    - Dx: Dimension of the hidden state (state_dim)
    - Dy: Dimension of the observation (obs_dim)
    """

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

    # ==========================================
    # 1. Properties (To be defined by subclasses)
    # ==========================================
    @property
    @abc.abstractmethod
    def x0(self) -> tf.Tensor:
        """Returns the initial state mean. Shape: [Dx] or [Dx, 1]."""
        pass

    @property
    @abc.abstractmethod
    def init_noise(self) -> tfd.Distribution:
        """Returns the distribution for the initial state uncertainty p(x_0)."""
        pass

    @property
    @abc.abstractmethod
    def process_noise(self) -> tfd.Distribution:
        """Returns the distribution for the system process noise q_t."""
        pass

    @property
    @abc.abstractmethod
    def observation_noise(self) -> tfd.Distribution:
        """Returns the distribution for the measurement noise r_t."""
        pass

    # ==========================================
    # 2. Core Transitions (To be defined by subclasses)
    # ==========================================
    @abc.abstractmethod
    def transition_fn(self, state: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """
        Propagates the state forward in time: x_t = f(x_{t-1}) + q_t.

        Args:
            state: Hidden state tensor of shape [B, Dx].
            noise: Optional process noise tensor of shape [B, Dx].
                   If None, computes the deterministic mean transition.

        Returns:
            Predicted next state of shape [B, Dx].
        """
        pass

    @abc.abstractmethod
    def observation_fn(self, state: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """
        Maps the hidden state to the observation space: y_t = h(x_t) + r_t.

        Args:
            state: Hidden state tensor of shape [B, Dx].
            noise: Optional measurement noise tensor of shape [B, Dy].
                   If None, computes the deterministic mean observation.

        Returns:
            Observation tensor of shape [B, Dy].
        """
        pass

    # ==========================================
    # 3. Universal Sequence Generation
    # ==========================================
    @tf.function
    def sample(self, batch_size: int, T: int) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Generates a batch of simulated state and observation sequences.
        It initializes a batch of states and propagates them simultaneously.

        Args:
            batch_size (int): Number of independent sequences to simulate (B).
            T (int): Number of time steps per sequence (T).

        Returns:
            x_seq: True hidden states tensor of shape [B, T, Dx].
            y_seq: Simulated observations tensor of shape [B, T, Dy].
        """
        # Standardize initial state to a 1D vector [Dx], then tile to [B, Dx]
        x_init = tf.reshape(self.x0, [self.state_dim])
        x_curr = tf.tile(tf.expand_dims(x_init, 0), [batch_size, 1])

        # Add initial state uncertainty (x_0 ~ N(mean, P0))
        init_n = self.init_noise.sample(batch_size)
        init_n = tf.reshape(init_n, [batch_size, self.state_dim])
        x_curr = x_curr + init_n

        # Setup TensorArrays to hold the time sequence.
        x_ta = tf.TensorArray(dtype=dtype, size=T)
        y_ta = tf.TensorArray(dtype=dtype, size=T)

        for t in tf.range(T):
            # Advance state for t > 0
            if t > 0:
                q_t = self.process_noise.sample(batch_size)
                q_t = tf.reshape(q_t, [batch_size, self.state_dim])
                x_curr = self.transition_fn(x_curr, q_t)

            # Generate observation from current state
            r_t = self.observation_noise.sample(batch_size)
            r_t = tf.reshape(r_t, [batch_size, self.obs_dim])
            y_curr = self.observation_fn(x_curr, r_t)

            # Write batch to history arrays
            x_ta = x_ta.write(t, x_curr)
            y_ta = y_ta.write(t, y_curr)

        # 4. Stack and transpose to yield [B, T, D] format
        # stack() creates [T, B, D]. Transposing axes 0 and 1 gives [B, T, D].
        x_seq = tf.transpose(x_ta.stack(), perm=[1, 0, 2])
        y_seq = tf.transpose(y_ta.stack(), perm=[1, 0, 2])

        return x_seq, y_seq


class LGSSM(AbstractSSM):
    """
    Linear Gaussian State-Space Model.

    Maintains explicit transition matrices (A, C, Q, R) which are required by
    closed-form Kalman filtering algorithms. It wraps these matrices into the
    AbstractSSM functional interface so that non-linear filters (like Particle
    Filters) can also process this model natively.
    """

    def __init__(self, state_dim: int, obs_dim: int, params=None):
        super().__init__(state_dim, obs_dim)

        if params is None:
            # Initialize with default stable parameters
            A0 = tf.eye(state_dim) * 0.95
            A0 += tf.random.normal([state_dim, state_dim], stddev=0.1, dtype=dtype)
            s = tf.linalg.svd(A0, compute_uv=False)
            scale = tf.minimum(1.0, 1.0 / (s[0] + 1e-9))
            self.A = tf.Variable(A0 * tf.cast(scale, dtype))

            self.C = tf.Variable(tf.random.normal([obs_dim, state_dim], stddev=1.0, dtype=dtype))
            self.Q = tf.Variable(tf.eye(state_dim) * 0.2, dtype=dtype)
            self.R = tf.Variable(tf.eye(obs_dim) * 0.1, dtype=dtype)
            self._x0 = tf.Variable(tf.zeros([state_dim, 1], dtype=dtype))
            self.P0 = tf.Variable(tf.eye(state_dim) * 1.0, dtype=dtype)
        else:
            # Unpack user-provided parameters
            self.A, self.C, self.Q, self.R, self._x0, self.P0 = [
                p if isinstance(p, tf.Variable) else tf.Variable(p, dtype=dtype) for p in params
            ]

        self.params = [self.A, self.C, self.Q, self.R, self._x0, self.P0]
        self.update_cholesky()

    def update_cholesky(self):
        """
        Pre-computes Cholesky decompositions for noise covariance matrices.
        This must be called whenever Q, R, or P0 are updated (e.g., during EM).
        """
        self.LQ = tf.linalg.cholesky(self.Q)
        self.LR = tf.linalg.cholesky(self.R)
        self.LP0 = tf.linalg.cholesky(self.P0)

    # --- Implement Abstract Properties ---
    @property
    def x0(self) -> tf.Tensor:
        return self._x0

    @property
    def init_noise(self) -> tfd.Distribution:
        # We use scale_tril to directly utilize the stable Cholesky factor
        return tfd.MultivariateNormalTriL(loc=tf.zeros(self.state_dim), scale_tril=self.LP0)

    @property
    def process_noise(self) -> tfd.Distribution:
        return tfd.MultivariateNormalTriL(loc=tf.zeros(self.state_dim), scale_tril=self.LQ)

    @property
    def observation_noise(self) -> tfd.Distribution:
        return tfd.MultivariateNormalTriL(loc=tf.zeros(self.obs_dim), scale_tril=self.LR)

    # --- Implement Abstract Methods ---
    def transition_fn(self, state: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """
        Calculates: x_t = (A * x_{t-1}^T)^T + noise
        We use transpose_b=True so that batched row vectors [B, Dx] map correctly
        to standard column-vector linear algebra.
        """
        next_state = tf.matmul(state, self.A, transpose_b=True)
        if noise is not None:
            next_state += noise
        return next_state

    def observation_fn(self, state: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """Calculates: y_t = (C * x_t^T)^T + noise"""
        obs = tf.matmul(state, self.C, transpose_b=True)
        if noise is not None:
            obs += noise
        return obs

    def set_params(self, params, update_cholesky=True):
        """Updates internal matrices with new parameters."""
        new_A, new_C, new_Q, new_R, new_x0, new_P0 = params
        self.A.assign(new_A)
        self.C.assign(new_C)
        self.Q.assign(new_Q)
        self.R.assign(new_R)
        self._x0.assign(new_x0)
        self.P0.assign(new_P0)
        if update_cholesky:
            self.update_cholesky()


class NLSSM(AbstractSSM):
    """
    Non-Linear State-Space Model.

    A flexible wrapper that constructs an SSM directly from arbitrary, user-defined
    transition and observation callables, alongside predefined noise distributions.
    """

    def __init__(self, state_dim: int, obs_dim: int,
                 transition_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 observation_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 process_noise: tfd.Distribution,
                 observation_noise: tfd.Distribution,
                 init_noise: tfd.Distribution,
                 x0: tf.Tensor,
                 observation_dist: Union[None, tfd.Distribution] = None,
                 transition_dist: Union[None, tfd.Distribution] = None):
        super().__init__(state_dim, obs_dim)

        self._transition_fn = transition_fn
        self._observation_fn = observation_fn

        self._process_noise = process_noise
        self._observation_noise = observation_noise
        self._init_noise = init_noise
        self._x0 = x0

        # Optional distributions providing exact likelihoods if noise is not additive
        self.observation_dist = observation_dist
        self.transition_dist = transition_dist

    # --- Expose internal properties ---
    @property
    def x0(self) -> tf.Tensor: return self._x0

    @property
    def init_noise(self) -> tfd.Distribution: return self._init_noise

    @property
    def process_noise(self) -> tfd.Distribution: return self._process_noise

    @process_noise.setter
    def process_noise(self, new_dist: tfd.Distribution):
        self._process_noise = new_dist

    @property
    def observation_noise(self) -> tfd.Distribution: return self._observation_noise

    @observation_noise.setter
    def observation_noise(self, new_dist: tfd.Distribution):
        self._observation_noise = new_dist

    # --- Route to provided Callables ---
    def transition_fn(self, state: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        if noise is None:
            return self._transition_fn(state, tf.zeros_like(state,dtype=dtype))
        return self._transition_fn(state, noise)

    def observation_fn(self, state: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        batch_size = tf.shape(state)[0]
        if noise is None:
            return self._observation_fn(state, tf.zeros([batch_size, self.obs_dim], dtype=dtype))
        return self._observation_fn(state, noise)


class LearnableSSM(AbstractSSM):
    """
    Learnable State-Space Model.

    Supports Neural Networks (Black Box) or parametrized explicit layers (Grey Box)
    via Keras Layers. Weights and noise scaling parameters are maintained as
    `tf.Variable` so they are automatically captured during backpropagation.
    """

    def __init__(self, state_dim: int, obs_dim: int,
                 transition_layers: tf.keras.layers.Layer,
                 observation_layers: tf.keras.layers.Layer,
                 proposal_layers: Union[None, tf.keras.layers.Layer, tfd.Distribution] = None,
                 x0_init: tf.Tensor = None,
                 init_noise: Union[None, tfd.Distribution] = None,
                 learn_noise: bool = False,
                 learn_init_state: bool = False,
                 init_process_noise_scale: float = 0.1,
                 init_obs_noise_scale: float = 0.01):
        super().__init__(state_dim, obs_dim)

        self.transition_layers = transition_layers
        self.observation_layers = observation_layers
        self.proposal_layers = proposal_layers

        # Set up learnable initial state
        if x0_init is None:
            x0_init = tf.zeros(state_dim)
        self._x0 = tf.Variable(x0_init, dtype=dtype, trainable=learn_init_state, name="x0")

        # We store the *log* of the scale to ensure the variance remains strictly positive
        # during unconstrained gradient descent optimization.
        self.log_process_noise_scale = tf.Variable(
            tf.fill([state_dim],  tf.math.log(init_process_noise_scale if init_process_noise_scale else 0.1)), trainable=learn_noise, name="log_Q")
        self.log_obs_noise_scale = tf.Variable(
            tf.fill([obs_dim],  tf.math.log(init_obs_noise_scale if init_obs_noise_scale else 0.1)), trainable=learn_noise, name="log_R")

        self._init_noise = init_noise

    @property
    def x0(self) -> tf.Tensor:
        return self._x0

    @property
    def init_noise(self) -> tfd.Distribution:
        return self._init_noise

    @property
    def process_noise(self) -> tfd.Distribution:
        # Add a floor bound to prevent scale -> 0, which would crash log_prob
        min_scale = tf.constant(1e-3, dtype=dtype)
        scale = tf.maximum(tf.exp(self.log_process_noise_scale), min_scale)
        return tfd.MultivariateNormalDiag(loc=tf.zeros(self.state_dim), scale_diag=scale)

    @property
    def observation_noise(self) -> tfd.Distribution:
        min_scale = tf.constant(1e-3, dtype=dtype)
        scale = tf.maximum(tf.exp(self.log_obs_noise_scale), min_scale)
        return tfd.MultivariateNormalDiag(loc=tf.zeros(self.obs_dim), scale_diag=scale)

    def transition_fn(self, state: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """Executes the forward pass of the Keras transition layers."""
        pred_state = self.transition_layers(state)
        if noise is not None:
            pred_state += noise
        return pred_state

    def observation_fn(self, state: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """Executes the forward pass of the Keras observation layers."""
        obs = self.observation_layers(state)
        if noise is not None:
            obs += noise
        return obs

    # --- Auxiliary Distributions (used by advanced filters) ---
    def get_observation_dist(self, particles: tf.Tensor) -> tfd.Distribution:
        """Returns the full conditional distribution p(y_t | x_t)."""
        loc = self.observation_layers(particles)
        min_scale = tf.constant(1e-3, dtype=dtype)
        scale = tf.maximum(tf.exp(self.log_obs_noise_scale), min_scale)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def get_transition_dist(self, particles: tf.Tensor) -> tfd.Distribution:
        """Returns the full conditional distribution p(x_t | x_{t-1})."""
        loc = self.transition_layers(particles)
        min_scale = tf.constant(1e-3, dtype=dtype)
        scale = tf.maximum(tf.exp(self.log_process_noise_scale), min_scale)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def get_proposal_dist(self, x_pre: tf.Tensor, y: tf.Tensor) -> tfd.Distribution:
        """Returns a custom proposal distribution for advanced importance sampling."""
        loc, std = self.proposal_layers(x_pre, y)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=std)