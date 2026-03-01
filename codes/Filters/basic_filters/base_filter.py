import abc
import tensorflow as tf


class BaseFilter(abc.ABC):
    """
    Abstract base class for all sequential filters.

    This class handles the repetitive methods:
    1. setting up the time-series loop,
    2. managing TensorArrays for trajectory, and returning the stacked results.
    It treats the tracking "state" as a generic tuple, allowing subclasses to
    define whether that state contains (mean, covariance) or (particles, weights).
    """

    def __init__(self, model):
        self.model = model

    # ==========================================
    # Shared Statistical Utilities
    # ==========================================
    def _get_cov(self, dist, log_scale_var=None) -> tf.Tensor:
        """
        Safely extracts a 2D covariance matrix from a TFP distribution.
        Shared across all child filters (EKF, UKF, Particle Filters).
        """
        if log_scale_var is not None:
            scale = tf.exp(log_scale_var)
            return tf.linalg.diag(tf.square(scale))

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

    def _get_mean(self, dist, dim: int) -> tf.Tensor:
        """
        Safely extracts the mean from a TFP distribution, ensuring correct shape.
        """
        try:
            mean = dist.mean()
        except (AttributeError, NotImplementedError):
            mean = tf.zeros([dim])

        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        if len(mean.shape) == 0:
            mean = tf.fill([dim], mean)

        return mean

    # ==========================================
    # Abstract Filter Interface
    # ==========================================
    @abc.abstractmethod
    def _init_state(self, batch_size: int) -> tuple:
        """Initializes the tracking state at t=0."""
        pass

    @abc.abstractmethod
    def _init_trajectory(self, time_steps: int) -> tuple:
        """Initializes TensorArrays to store the trajectory."""
        pass

    @abc.abstractmethod
    def predict(self, t: int, state: tuple) -> tuple:
        """Propagates the state forward in time."""
        pass

    @abc.abstractmethod
    def update(self, t: int, state: tuple, observation: tf.Tensor) -> tuple:
        """Incorporates the observation to correct the state. Returns (new_state, metrics)."""
        pass

    @abc.abstractmethod
    def forecast(self, observations: tf.Tensor) -> tuple:
        """Generates the forecast of the next observation based on the provided trajectory."""
        pass

    @abc.abstractmethod
    def _write_trajectory(self, t: int, trajectory: tuple, state: tuple, metrics: tuple) -> tuple:
        """
        Writes the current state and metrics into the TensorArrays.

        IMPORTANT: This method MUST return the updated `trajectory` tuple.
        In TensorFlow Graph Mode (@tf.function), `TensorArray.write()` does not
        modify the array in-place. Instead, it returns a new immutable handle to the
        updated array. Returning this tuple allows the main `filter` loop to chain
        these handles together, explicitly defining the execution dependency order
        for the computational graph.
        """
        pass

    @abc.abstractmethod
    def _format_output(self, trajectory: tuple) -> dict:
        """Stacks the TensorArrays and returns a clean dictionary."""
        pass

    @tf.function
    def filter(self, observations: tf.Tensor) -> dict:
        """
        Executes the predict-update loop over a sequence of observations
        to produce the filtered trajectory.
        """
        batch_size = tf.shape(observations)[0]
        time_steps = tf.shape(observations)[1]

        # [Time, Batch, Obs_Dim] for easier iteration
        y_time_major = tf.transpose(observations, perm=[1, 0, 2])

        state = self._init_state(batch_size)
        trajectory = self._init_trajectory(time_steps)

        # 1. Helper to dynamically relax the batch dimension (axis 0) to None
        def _get_invariant(x):
            if not isinstance(x, tf.Tensor):
                return None
            if len(x.shape) == 0:  # Handle scalars (like 'lam')
                return tf.TensorShape([])
            return tf.TensorShape([None] + x.shape[1:])

        # 2. Map the helper across whatever complex tuple 'state' happens to be
        state_invariants = tf.nest.map_structure(_get_invariant, state)

        for t in tf.range(time_steps):
            # 3. COMPILER FIX: Enforce dynamic batching for the entire filter lifecycle
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(state, state_invariants)]
            )

            y_t = y_time_major[t]

            state = tf.cond(
                t > 0,
                lambda: self.predict(t, state),
                lambda: state
            )

            state, metrics = self.update(t, state, y_t)

            trajectory = self._write_trajectory(t, trajectory, state, metrics)

        return self._format_output(trajectory)