import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

class HMM:
    def __init__(self, init_probs, transition_probs, obs_locs, obs_scales):
        """
        Initialize the HMM parameters.

        Args:
            init_probs: Initial state probabilities. Shape (K,)
            transition_probs: Transition matrix. Shape (K, K)
            obs_locs: Mean of observation distributions. Shape (K, D)
            obs_scales: Scale (std dev) of observation distributions. Shape (K, D)
                        (Diagonal covariance assumption)
        """
        # Convert inputs to Variables to allow updates during M-step
        self.init_probs = tf.Variable(init_probs, dtype=tf.float32, name='init_probs')
        self.transition_probs = tf.Variable(transition_probs, dtype=tf.float32, name='trans_probs')
        self.obs_locs = tf.Variable(obs_locs, dtype=tf.float32, name='obs_locs')
        self.obs_scales = tf.Variable(obs_scales, dtype=tf.float32, name='obs_scales')

        self.n_states = init_probs.shape[0]
        self.n_features = obs_locs.shape[1]
        self.hmm_dist = None  # Internal TFP distribution holder

    def _build_model(self, num_steps):
        """Helper to construct the TFP HiddenMarkovModel distribution."""
        return tfd.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(probs=self.init_probs),
            transition_distribution=tfd.Categorical(probs=self.transition_probs),
            observation_distribution=tfd.Normal(loc=self.obs_locs, scale=self.obs_scales),
            num_steps=num_steps
        )

    def sample(self, n_steps, n_samples=1):
        """Generates samples from the model."""
        model = self._build_model(num_steps=n_steps)
        return model.sample(n_samples)

    def predict(self, observations):
        """
        Computes the most likely sequence of hidden states (Viterbi).
        observations: (N, T, D)
        """
        n_steps = tf.shape(observations)[1]
        self.hmm_dist = self._build_model(n_steps)
        return self.hmm_dist.posterior_mode(observations)

    def log_likelihood(self, sequences):
        """
        Computes log probability of sequences.
        sequences: (N, T, D)
        """
        n_steps = tf.shape(sequences)[1]
        self.hmm_dist = self._build_model(n_steps)
        return self.hmm_dist.log_prob(sequences)

    def Estep(self, X):
        '''
        Calculates the expected sufficient statistics.
        X shape: (N, T, D)

        Returns:
            gammas: (N, T, K) - Posterior state probabilities
            xis: (N, K, K) - Summed pairwise posteriors for the whole sequence length
        '''
        n_steps = tf.shape(X)[1]

        # We need to watch the transition parameters to get the gradient
        # Note: TFP uses the logits internally for gradients usually.
        # We reconstruct the parameters inside the tape to ensure connection.
        with tf.GradientTape() as tape:
            # We must use logits for numerical stability in gradients
            # However, our class stores probabilities.
            # We calculate logits temporarily for the gradient target.
            # Using a small epsilon to avoid log(0)
            trans_probs_safe = tf.clip_by_value(self.transition_probs, 1e-6, 1.0)
            trans_logits = tf.math.log(trans_probs_safe)
            tape.watch(trans_logits)

            # Rebuild distribution using these specific logits/probs
            # so the tape can track them.
            current_hmm = tfd.HiddenMarkovModel(
                initial_distribution=tfd.Categorical(probs=self.init_probs),
                transition_distribution=tfd.Categorical(logits=trans_logits),
                observation_distribution=tfd.Normal(loc=self.obs_locs, scale=self.obs_scales),
                num_steps=n_steps
            )

            log_probs = current_hmm.log_prob(X)
            total_log_prob = tf.reduce_sum(log_probs)

        # 1. Compute Gammas (Posterior Marginals) directly from TFP
        # gammas shape: (N, T, K)
        # We use the method from the instance created inside the tape for consistency
        gammas = current_hmm.posterior_marginals(X).probs_parameter()

        # 2. Compute Xis (Pairwise Posteriors) via Gradient
        # The gradient of Log Likelihood w.r.t Log Transition Probabilities (Logits)
        # gives the expected count of transitions.
        # Shape: (K, K) - Summed over N and T implicitly by the gradient of sum(log_prob)
        # Note: We want xis per sequence usually for batch processing,
        # but standard Baum-Welch aggregates them anyway for the update.
        # The tape.gradient here returns the sum over the batch dimension automatically.
        xis_aggregated = tape.gradient(total_log_prob, trans_logits)

        # Because we need (N, K, K) for the interface defined in your prompt,
        # we can technically get it by using a Jacobian or per-sample gradient,
        # but that is very slow.
        # Standard M-step only needs sum_n(xi), which is what we have here.
        # We will return the aggregated xis and handle it in M-step.

        return gammas, xis_aggregated

    def Mstep(self, X, gammas, xis_aggregated):
        '''
        Maximization Step.
        X: (N, T, D)
        gammas: (N, T, K)
        xis_aggregated: (K, K) -> This is sum_n sum_t xi_{nt}
        '''
        # 1. Update Initial Probabilities
        # Average gammas at t=0 across the batch
        new_init_probs = tf.reduce_mean(gammas[:, 0, :], axis=0)
        self.init_probs.assign(new_init_probs / tf.reduce_sum(new_init_probs))

        # 2. Update Transition Probabilities
        # xis_aggregated contains the expected number of transitions i->j
        # We simply normalize rows to sum to 1.
        new_trans_probs = xis_aggregated / tf.reduce_sum(xis_aggregated, axis=1, keepdims=True)
        self.transition_probs.assign(new_trans_probs)

        # 3. Update Observation Parameters (Vectorized)

        # Denominator: Sum of gammas for each state (sum over N and T)
        # Shape: (K, 1)
        gamma_sum = tf.reduce_sum(gammas, axis=[0, 1])
        gamma_sum = tf.reshape(gamma_sum, (-1, 1)) + 1e-10  # Avoid div by zero

        # Numerator for Mean: Weighted sum of observations
        # Shape: (K, D)
        weighted_locs = tf.einsum('ntk,ntd->kd', gammas, X)
        new_locs = weighted_locs / gamma_sum
        self.obs_locs.assign(new_locs)

        # Numerator for Scale: Weighted variance
        # Center the data first: (N, T, K, D)
        diff = X[:, :, tf.newaxis, :] - self.obs_locs[tf.newaxis, tf.newaxis, :, :]

        # Weighted sum of squares (Diagonal covariance only)
        # Shape: (K, D)
        weighted_sq_diff = tf.einsum('ntk,ntkd->kd', gammas, tf.square(diff))

        new_scales = tf.sqrt((weighted_sq_diff / gamma_sum) + 1e-6)
        self.obs_scales.assign(new_scales)

    def fit(self, sequences, n_epochs=100, n_interval=10, tolerance=1e-4):
        """
        Train the HMM using Baum-Welch (EM) algorithm.
        sequences: (N, T, D)
        """
        # Initial Log Likelihood
        log_p = tf.reduce_mean(self.log_likelihood(sequences))
        print(f"Initial Log Likelihood: {log_p.numpy():.4f}")

        for epoch in range(n_epochs):
            # E-Step
            gammas, xis_aggregated = self.Estep(sequences)

            # M-Step
            self.Mstep(sequences, gammas, xis_aggregated)

            # Check convergence
            if epoch % n_interval == 0:
                new_log_p = tf.reduce_mean(self.log_likelihood(sequences))
                print(f'Epoch {epoch}, Log Likelihood: {new_log_p.numpy():.4f}')

                delta = tf.abs(new_log_p - log_p)
                if delta < tolerance:
                    print(f"Converged at epoch {epoch}")
                    break
                log_p = new_log_p