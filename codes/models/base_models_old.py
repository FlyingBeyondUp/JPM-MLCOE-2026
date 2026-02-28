import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Union

tfd=tfp.distributions
dtype=tf.float32

class LGSSM:
    def __init__(self,state_dim,obs_dim,params=None):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        if params is None:
            A0=tf.eye(state_dim)*0.95  # State transition matrix
            A0+=tf.random.normal([state_dim,state_dim],stddev=0.1,dtype=dtype)
            s=tf.linalg.svd(A0,compute_uv=False)
            scale=tf.minimum(1,1/(s[0]+1e-9)) # Ensure stability
            self.A=tf.Variable(A0*tf.cast(scale,dtype))
            self.C=tf.Variable(tf.random.normal([obs_dim,state_dim],stddev=1.0,dtype=dtype))  # Observation matrix
            self.Q=tf.Variable(tf.eye(state_dim)*0.2,dtype=dtype)  # Process noise covariance
            self.R=tf.Variable(tf.eye(obs_dim)*0.1,dtype=dtype)    # Observation noise covariance
            self.x0=tf.Variable(tf.zeros([state_dim,1],dtype=dtype))  # Initial state mean
            self.P0=tf.Variable(tf.eye(state_dim)*1.0,dtype=dtype)  # Initial state covariance
        else:
            self.A, self.C, self.Q, self.R, self.x0, self.P0 = [
                p if isinstance(p, tf.Variable) else tf.Variable(p,dtype=dtype) for p in params
            ]
        self.params = [self.A, self.C, self.Q, self.R, self.x0, self.P0]
        self.LQ = tf.linalg.cholesky(self.Q)
        self.LR = tf.linalg.cholesky(self.R)
        self.LP0 = tf.linalg.cholesky(self.P0)

    def sample(self,T):
        '''
        Generate a sample sequence of length T
        :param T: The length of the sequence
        :return: sequence of states x and observations y
                 with shape [T,state_dim] and [T,obs_dim]
        '''
        x = tf.TensorArray(dtype=dtype, size=T)
        y = tf.TensorArray(dtype=dtype, size=T)

        x_t = self.x0 + self.LP0 @ tf.random.normal([self.state_dim,1],dtype=dtype)

        for t in range(T):
            process_noise =  self.LQ@ tf.random.normal([self.state_dim,1],dtype=dtype)
            x_t = self.A @ x_t + process_noise

            observation_noise = self.LR @ tf.random.normal([self.obs_dim,1],dtype=dtype)
            y_t = self.C @ x_t + observation_noise

            x = x.write(t, x_t)
            y = y.write(t, y_t)
        x,y=x.stack(),y.stack()
        return tf.squeeze(x,axis=-1),tf.squeeze(y,axis=-1)

    def batch_sample(self,T,batch_size):
        '''
        :param T: The length of the sequences
        :return: x: [batch_size,T,state_dim], y: [batch_size,T,obs_dim]
        '''
        return tf.vectorized_map(lambda _: self.sample(T), tf.range(batch_size))

    def get_params(self):
        return self.params

    def update_cholesky(self):
        self.LQ = tf.linalg.cholesky(self.Q)
        self.LR = tf.linalg.cholesky(self.R)
        self.LP0 = tf.linalg.cholesky(self.P0)

    def set_params(self, params, update_cholesky=True):
        new_A, new_C, new_Q, new_R, new_x0, new_P0 = params

        self.A.assign(new_A)
        self.C.assign(new_C)
        self.Q.assign(new_Q)
        self.R.assign(new_R)
        self.x0.assign(new_x0)
        self.P0.assign(new_P0)

        if update_cholesky:
            self.update_cholesky()


class NLSSM(tf.keras.Model):
    def __init__(self,state_dim:int,obs_dim:int,
                 transition_fn:Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 observation_fn:Callable[[tf.Tensor,tf.Tensor],tf.Tensor],
                 process_noise:tfd.Distribution,
                 observation_noise:tfd.Distribution,
                 init_noise:tfd.Distribution,
                 x0: tf.Tensor,
                 observation_dist:Union[None,tfd.Distribution]=None,
                 transition_dist:Union[None,tfd.Distribution]=None):
        '''
        The transition and observation functions should support batch inputs
        '''
        super().__init__()
        self.transition_fn=transition_fn
        self.observation_fn=observation_fn

        self.process_noise=process_noise
        self.observation_noise=observation_noise

        self.observation_dist = observation_dist
        self.transition_dist = transition_dist

        self.init_noise=init_noise
        self.x0=x0

        self.state_dim=state_dim
        self.obs_dim=obs_dim

    def sample(self,T:int)->tuple[tf.Tensor,tf.Tensor]:
        '''
        Generate a sample sequence of length T
        :param T: The length of the sequence
        :return: sequence of states x and observations y
                 with shape [T,state_dim] and [T,obs_dim]
        '''
        x = tf.TensorArray(dtype=dtype, size=T)
        y = tf.TensorArray(dtype=dtype, size=T)

        x_t = self.x0 + self.init_noise.sample()

        for t in range(T):
            if t>0:
                x_t = self.transition_fn(x_t, self.process_noise.sample())
            y_t = self.observation_fn(x_t, self.observation_noise.sample())
            x = x.write(t, x_t)
            y = y.write(t, y_t)
        x,y=x.stack(),y.stack()

        if x.shape[-1] == 1:
            x = tf.squeeze(x, axis=-1)
        if y.shape[-1] == 1:
            y = tf.squeeze(y, axis=-1)

        return x, y

    def batch_sample(self,T:int,batch_size:int)->tuple[tf.Tensor,tf.Tensor]:
        '''
        :param T: The length of the sequences
        :return: x: [batch_size,T,state_dim], y: [batch_size,T,obs_dim]
        '''
        return tf.vectorized_map(lambda _: self.sample(T), tf.range(batch_size))



class LearnableSSM(tf.keras.Model):
    """
    Unified State Space Model supporting both:
    1. Grey Box: Explicit math with learnable parameters (tf.Variables).
    2. Black Box: Neural Networks (tf.keras.layers).
    """

    def __init__(self, state_dim: int, obs_dim: int,
                 transition_layers: tf.keras.layers.Layer,
                 observation_layers: tf.keras.layers.Layer,
                 proposal_layers: Union[None,tf.keras.layers.Layer, tfd.Distribution] = None,
                 x0_init: tf.Tensor = None,
                 init_noise: Union[None,tfd.Distribution] = None,
                 learn_noise: bool = False,
                 learn_init_state: bool = False,
                 init_noise_scale: float = None):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # 1. Transition & Observation Functions
        # Can be simple functions, Keras Layers, or Keras Models.
        # If they are Layers/Models, self.trainable_variables will automatically track their weights.
        self.transition_layers = transition_layers
        self.observation_layers = observation_layers
        self.proposal_layers = proposal_layers

        # 2. Learnable Initial State
        if x0_init is None:
            x0_init = tf.zeros(state_dim)
        self.x0 = tf.Variable(x0_init, dtype=dtype, trainable=learn_init_state,name="x0")

        # 3. Learnable Noise Parameters
        # We store log-scale to ensure positivity (exp(log_scale) > 0)
        # Using tf.Variable makes them learnable by default.
        if init_noise_scale is not None:
            init_log_scale = tf.math.log(init_noise_scale)
        else:
            init_log_scale = tf.math.log(0.1)  # Initialize with std=0.1
        self.log_process_noise_scale = tf.Variable(tf.fill([state_dim], init_log_scale),
                                                   trainable=learn_noise, name="log_Q")
        self.log_obs_noise_scale = tf.Variable(tf.fill([obs_dim], init_log_scale),
                                               trainable=learn_noise, name="log_R")

        # Fixed initial noise (usually not learned, but can be)
        self.init_noise=init_noise

    @property
    def process_noise(self):
        # 数值稳定：给 std 加下界，避免 scale -> 0 导致 log_prob 爆炸/inf
        min_scale = tf.constant(1e-3, dtype=dtype)
        scale = tf.maximum(tf.exp(self.log_process_noise_scale), min_scale)
        return tfd.MultivariateNormalDiag(scale_diag=scale)

    @property
    def observation_noise(self):
        min_scale = tf.constant(1e-3, dtype=dtype)
        scale = tf.maximum(tf.exp(self.log_obs_noise_scale), min_scale)
        return tfd.MultivariateNormalDiag(scale_diag=scale)

    def transition_fn(self, particles,noise=None):
        """
        Applies transition function + process noise.
        """
        # Noise must be additive or handled inside transition_fn if using reparameterization
        if noise is None:
            noise = self.process_noise.sample(tf.shape(particles)[:-1])

        # Call the user-provided function/layer
        # If it's a Neural Network, this triggers the forward pass
        pred_state = self.transition_layers(particles)

        return pred_state + noise

    def observation_fn(self, particles,noise=None):
        """
        Applies observation function.
        """
        if noise is None:
            noise = self.observation_noise.sample(tf.shape(particles)[:-1])
        return self.observation_layers(particles)+noise

    def get_observation_dist(self, particles):
        loc = self.observation_layers(particles)
        min_scale = tf.constant(1e-3, dtype=dtype)
        scale = tf.maximum(tf.exp(self.log_obs_noise_scale), min_scale)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def get_transition_dist(self, particles):
        loc = self.transition_layers(particles)
        min_scale = tf.constant(1e-3, dtype=dtype)
        scale = tf.maximum(tf.exp(self.log_process_noise_scale), min_scale)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def get_proposal_dist(self, x_pre,y):
        loc,std = self.proposal_layers(x_pre,y)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=std)


