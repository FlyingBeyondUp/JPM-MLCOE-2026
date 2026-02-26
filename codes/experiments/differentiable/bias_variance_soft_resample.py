import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Importing from your provided modules
from models import LearnableSSM, LGSSM
from Filters.differentiable_filters import SoftResamplingParticleFilter

tfd = tfp.distributions


# ==========================================
# 1. Custom Filter for Metric Extraction
# ==========================================
class ExperimentFilter(SoftResamplingParticleFilter):
    """
    Subclasses SoftResamplingParticleFilter to extract gradients and ESS
    during the training step for experimental logging.
    """

    @tf.function
    def train_step_with_metrics(self, observations):
        with tf.GradientTape() as tape:
            # Run filter to get log-likelihood and ESS
            # filter_summarized returns: x_filt, P_filt, ess, log_likelihood
            _, _, ess_batch, log_likelihood_batch = self.filter_summarized(observations)

            # Loss is the negative log-likelihood
            loss = -tf.reduce_mean(log_likelihood_batch)

        # Compute gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Calculate metrics
        # Global norm of the gradients (proxy for gradient scale/variance)
        grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])
        # Mean Effective Sample Size across time and batch
        mean_ess = tf.reduce_mean(ess_batch)

        return loss, grad_norm, mean_ess


# ==========================================
# 2. Model & Realistic Data Generation
# ==========================================
def generate_ground_truth_data(num_sequences: int, seq_len: int, state_dim: int, obs_dim: int) -> tf.data.Dataset:
    """
    Generates dynamically consistent data using the true LGSSM.
    """
    # Initialize the true physical/statistical model
    # The LGSSM automatically creates stable transition matrices via SVD scaling
    true_model = LGSSM(state_dim=state_dim, obs_dim=obs_dim)

    # batch_sample returns sequences of (states, observations)
    _, observations = true_model.batch_sample(T=seq_len, batch_size=num_sequences)

    # Convert to a tf.data.Dataset for standard batching
    return tf.data.Dataset.from_tensor_slices(observations).batch(16)


def create_differentiable_ssm(state_dim: int, obs_dim: int) -> LearnableSSM:
    """Creates a fresh, uninitialized LearnableSSM (Black Box) to learn the data."""
    # A small neural network to map state -> next state
    transition_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(state_dim)
    ])

    # A small neural network to map state -> observation
    observation_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(obs_dim)
    ])

    return LearnableSSM(
        state_dim=state_dim,
        obs_dim=obs_dim,
        transition_layers=transition_layer,
        observation_layers=observation_layer,
        learn_noise=True,
        learn_init_state=True,
        init_noise_scale=0.5
    )


# ==========================================
# 3. Main Experiment Loop
# ==========================================
def run_experiment():
    # Setup hyperparameters
    state_dim = 3
    obs_dim = 3
    num_particles = 40
    epochs = 20
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("Generating Ground Truth Data from LGSSM...")
    dataset = generate_ground_truth_data(num_sequences=128, seq_len=25, state_dim=state_dim, obs_dim=obs_dim)

    # Dictionaries to store results
    results = {
        'alpha': alphas,
        'final_loss': [],
        'grad_variance': [],
        'mean_ess': []
    }

    print("Starting Alpha Tuning Experiment...\n" + "=" * 45)

    for alpha in alphas:
        print(f"Training Neural SSM with Soft-Resampling alpha = {alpha}...")

        # 1. Ensure reproducibility and fresh network weights for each run
        tf.random.set_seed(101)
        model = create_differentiable_ssm(state_dim, obs_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        pf = ExperimentFilter(model, num_particles=num_particles, alpha=alpha, optimizer=optimizer)

        epoch_losses = []
        epoch_grad_norms = []
        epoch_esses = []

        # 2. Training Loop
        for epoch in range(epochs):
            batch_losses = []
            batch_grad_norms = []
            batch_esses = []

            for batch in dataset:
                loss, grad_norm, mean_ess = pf.train_step_with_metrics(batch)
                batch_losses.append(loss.numpy())
                batch_grad_norms.append(grad_norm.numpy())
                batch_esses.append(mean_ess.numpy())

            epoch_losses.append(np.mean(batch_losses))
            epoch_grad_norms.extend(batch_grad_norms)
            epoch_esses.append(np.mean(batch_esses))

        # 3. Record metrics for this alpha
        # We take the variance of the gradient norms over the final epochs to measure stability
        grad_variance = np.var(epoch_grad_norms[-60:])
        final_loss = np.mean(epoch_losses[-3:])  # Average of last 3 epochs to smooth
        final_ess = np.mean(epoch_esses[-3:])

        results['final_loss'].append(final_loss)
        results['grad_variance'].append(grad_variance)
        results['mean_ess'].append(final_ess)

        print(
            f"  -> Final Loss: {final_loss:.4f} | Grad Variance: {grad_variance:.4f} | Avg ESS: {final_ess:.2f}/{num_particles}\n")

    # ==========================================
    # 4. Plotting the Results
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Bias-Variance Trade-off in Soft-Resampling Particle Filters (LGSSM Data)", fontsize=16)

    # Plot 1: Final Training Loss (Proxy for Bias / Convergence Quality)
    axes[0].plot(results['alpha'], results['final_loss'], marker='o', color='red', linewidth=2)
    axes[0].set_title("Final Training Loss\n(Lower is Better)")
    axes[0].set_xlabel("Alpha (Softness)")
    axes[0].set_ylabel("Negative Log-Likelihood")
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Gradient Variance (Proxy for Variance)
    axes[1].plot(results['alpha'], results['grad_variance'], marker='s', color='blue', linewidth=2)
    axes[1].set_title("Gradient Norm Variance\n(Lower is Better)")
    axes[1].set_xlabel("Alpha (Softness)")
    axes[1].set_yscale('log')  # Log scale is critical to visualize variance spikes
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Average Effective Sample Size (ESS)
    axes[2].plot(results['alpha'], results['mean_ess'], marker='^', color='green', linewidth=2)
    axes[2].axhline(y=num_particles, color='black', linestyle=':', label='Max Particles')
    axes[2].set_title("Average Effective Sample Size (ESS)\n(Higher is Better)")
    axes[2].set_xlabel("Alpha (Softness)")
    axes[2].set_ylabel("ESS")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()