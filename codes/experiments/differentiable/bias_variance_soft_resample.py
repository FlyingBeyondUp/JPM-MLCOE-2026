import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Enforce float32 globally to prevent Keras/TFP dtype mismatches
tf.keras.backend.set_floatx('float32')

# Importing from your provided modules
from models import LearnableSSM, LGSSM
from Filters.basic_filters import ParticleFilter
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
            _, _, ess_batch, log_likelihood_batch = self.filter_summarized(observations)
            loss = -tf.reduce_mean(log_likelihood_batch)

        # Compute gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Calculate metrics
        grad_norm = tf.linalg.global_norm([g for g in grads if g is not None])
        mean_ess = tf.reduce_mean(ess_batch)

        return loss, grad_norm, mean_ess


# ==========================================
# 2. Model & Realistic Data Generation
# ==========================================
def generate_ground_truth_data(num_train: int, num_test: int, seq_len: int, state_dim: int, obs_dim: int):
    """Generates dynamically consistent train and test data using the true LGSSM."""
    true_model = LGSSM(state_dim=state_dim, obs_dim=obs_dim)

    # Generate Training Data
    _, train_obs = true_model.batch_sample(T=seq_len, batch_size=num_train)
    train_ds = tf.data.Dataset.from_tensor_slices(train_obs).batch(16)

    # Generate Testing Data (for Open-Loop Forecasting)
    _, test_obs = true_model.batch_sample(T=seq_len, batch_size=num_test)
    test_ds = tf.data.Dataset.from_tensor_slices(test_obs).batch(16)

    return train_ds, test_ds


def create_differentiable_ssm(state_dim: int, obs_dim: int) -> LearnableSSM:
    """Creates a fresh, uninitialized LearnableSSM (Black Box) to learn the data."""
    transition_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', dtype='float32'),
        tf.keras.layers.Dense(state_dim, dtype='float32')
    ])

    observation_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', dtype='float32'),
        tf.keras.layers.Dense(obs_dim, dtype='float32')
    ])

    # FIX: Wider initial distribution to prevent peaky likelihoods and rescue ESS
    init_noise_scale = 1.0
    init_noise_dist = tfd.MultivariateNormalDiag(
        loc=tf.zeros(state_dim, dtype=tf.float32),
        scale_diag=tf.fill([state_dim], tf.cast(2.0, tf.float32))  # Start with wider spread
    )

    return LearnableSSM(
        state_dim=state_dim,
        obs_dim=obs_dim,
        transition_layers=transition_layer,
        observation_layers=observation_layer,
        init_noise=init_noise_dist,
        learn_noise=True,
        learn_init_state=True,
        init_noise_scale=init_noise_scale
    )


# ==========================================
# 3. Open-Loop Forecasting (The Bias Test)
# ==========================================
def compute_open_loop_mse(model, test_dataset, warmup_steps=5):
    """
    Tests the True Bias of the learned model.
    Warms up the filter for a few steps to find the state, then forecasts
    the future completely blind (open-loop) and compares to ground truth.
    """
    mse_list = []

    # We use a standard hard-resampling PF just for the warmup evaluation
    # to give the model the best possible starting state estimate.
    eval_filter = SoftResamplingParticleFilter(model, num_particles=100, alpha=1.0)

    for batch in test_dataset:
        batch = tf.cast(batch, tf.float32)
        batch_size = tf.shape(batch)[0]

        # Split into warmup and forecasting horizons
        warmup_obs = batch[:, :warmup_steps, :]
        future_obs = batch[:, warmup_steps:, :]
        forecast_steps = tf.shape(future_obs)[1]

        # 1. Warmup: Get the filtered state at the end of the warmup period
        x_filt, _, _, _ = eval_filter.filter_summarized(warmup_obs)
        current_state = x_filt[:, -1, :]  # Shape: [Batch, State_Dim] (2D)

        # 2. Forecast: Roll the model forward blind (no observations)
        preds = []
        zero_process_noise = tf.zeros([batch_size, model.state_dim], dtype=tf.float32)
        zero_obs_noise_3d = tf.zeros([batch_size, 1, model.obs_dim], dtype=tf.float32)

        for _ in range(forecast_steps):
            # Predict next state (transition_fn accepts 2D)
            current_state = model.transition_fn(current_state, noise=zero_process_noise)

            # Expand to 3D for observation_fn: [Batch, 1, State_Dim]
            current_state_3d = tf.expand_dims(current_state, axis=1)

            # Predict observation (observation_fn accepts 3D)
            pred_obs_3d = model.observation_fn(current_state_3d, noise=zero_obs_noise_3d)

            # Squeeze back to 2D: [Batch, Obs_Dim] and append
            pred_obs = tf.squeeze(pred_obs_3d, axis=1)
            preds.append(pred_obs)

        # Stack predictions and compute MSE against actual future observations
        preds = tf.stack(preds, axis=1)  # [Batch, Forecast_Steps, Obs_Dim]
        mse = tf.reduce_mean(tf.square(preds - future_obs))
        mse_list.append(mse.numpy())

    return np.mean(mse_list)


# ==========================================
# 4. Main Experiment Loop
# ==========================================
def run_experiment():
    state_dim = 3
    obs_dim = 3
    num_particles = 40
    epochs = 100  # FIX: Increased epochs for proper convergence
    seq_len = 50
    # FIX: More granular alphas to capture the U-shape trend cleanly
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print("Generating Ground Truth Data from LGSSM...")
    train_ds, test_ds = generate_ground_truth_data(num_train=512, num_test=256, seq_len=seq_len, state_dim=state_dim,
                                                   obs_dim=obs_dim)

    results = {
        'alpha': alphas,
        'forecast_mse': [],  # Tracks True Bias
        'grad_variance': [],  # Tracks Variance
        'mean_ess': []  # Tracks Filter Stability
    }

    print("Starting Alpha Tuning Experiment...\n" + "=" * 55)

    for alpha in alphas:
        print(f"Training Neural SSM with Soft-Resampling alpha = {alpha}...")

        tf.random.set_seed(101)
        model = create_differentiable_ssm(state_dim, obs_dim)

        # FIX: Added global gradient clipping to survive variance spikes
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, global_clipnorm=5.0)

        pf = ExperimentFilter(model, num_particles=num_particles, alpha=alpha, optimizer=optimizer)

        epoch_grad_norms = []
        epoch_esses = []

        # Training Loop
        for epoch in range(epochs):
            batch_grad_norms = []
            batch_esses = []

            for batch in train_ds:
                batch = tf.cast(batch, tf.float32)
                _, grad_norm, mean_ess = pf.train_step_with_metrics(batch)
                batch_grad_norms.append(grad_norm.numpy())
                batch_esses.append(mean_ess.numpy())

            epoch_grad_norms.extend(batch_grad_norms)
            epoch_esses.append(np.mean(batch_esses))

        # Measure True Bias (Open-Loop Forecasting MSE on Test Data)
        forecast_mse = compute_open_loop_mse(model, test_ds, warmup_steps=5)

        # Measure Variance (Variance of gradient norms over the final converged epochs)
        grad_variance = np.var(epoch_grad_norms[-50:])
        final_ess = np.mean(epoch_esses[-10:])

        results['forecast_mse'].append(forecast_mse)
        results['grad_variance'].append(grad_variance)
        results['mean_ess'].append(final_ess)

        print(
            f"  -> Forecast MSE (Bias): {forecast_mse:.4f} | Grad Var: {grad_variance:.4f} | Avg ESS: {final_ess:.2f}/{num_particles}\n")

    # ==========================================
    # 5. Plotting the Results
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle("The True Bias-Variance Trade-off in Soft-Resampling DPFs", fontsize=16)

    # # Plot 1: Forecasting MSE (The True Bias)
    # axes[0].plot(results['alpha'], results['forecast_mse'], marker='o', color='red', linewidth=2)
    # axes[0].set_title("Open-Loop Forecast MSE (True Bias)\n(U-Shape Expected, Lower is Better)")
    # axes[0].set_xlabel("Alpha (Softness)")
    # axes[0].set_ylabel("Mean Squared Error on Test Set")
    # axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Gradient Variance (The Variance)
    axes[0].plot(results['alpha'], results['grad_variance'], marker='s', color='blue', linewidth=2)
    axes[0].set_title("Gradient Norm Variance\n(Lower is Better)")
    axes[0].set_xlabel("Alpha (Softness)")
    axes[0].set_yscale('log')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Average Effective Sample Size (ESS)
    axes[1].plot(results['alpha'], results['mean_ess'], marker='^', color='green', linewidth=1)
    axes[1].axhline(y=num_particles, color='black', linestyle=':', label='Max Particles')
    axes[1].set_title("Average Effective Sample Size (ESS)\n(Higher is Better)")
    axes[1].set_xlabel("Alpha (Softness)")
    axes[1].set_ylabel("ESS")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("bias_variance_tradeoff_soft_resampling.pdf",bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_experiment()