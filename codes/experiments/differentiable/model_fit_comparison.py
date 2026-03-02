import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

# Assuming these are available in your project structure
from models.base_models import NLSSM, LearnableSSM
from Filters.differentiable_filters import SoftResamplingParticleFilter
from Filters.differentiable_filters.entropy_regularized_OT import DifferentiableParticleFilter

tfd = tfp.distributions


# ==========================================
# 1. Custom Nonlinear Layers
# ==========================================
class LearnableNonlinearTransition(tf.keras.layers.Layer):
    """ Implements x_t = theta * sin(x_{t-1}) """

    def __init__(self, init_theta):
        super().__init__()
        self.init_theta = init_theta

    def build(self, input_shape):
        # Professional Keras variable registration
        self.theta = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(self.init_theta),
            trainable=True,
            name="theta"
        )

    def call(self, state):
        return self.theta * tf.sin(state)


class IdentityObservation(tf.keras.layers.Layer):
    """ Implements y_t = x_t """

    def call(self, state):
        return state


# ==========================================
# 2. Shared Setup Utilities
# ==========================================
def setup_experiment_data(state_dim=1, obs_dim=1, T=60, batch_size=4, true_theta=0.85):
    """Generates ground truth data using the exact NLSSM architecture."""

    def true_transition_fn(x, noise):
        return true_theta * tf.sin(x) + noise

    def true_observation_fn(x, noise):
        return x + noise

    process_noise = tfd.MultivariateNormalDiag(scale_diag=tf.ones(state_dim) * 0.5)
    observation_noise = tfd.MultivariateNormalDiag(scale_diag=tf.ones(obs_dim) * 0.2)
    init_noise = tfd.MultivariateNormalDiag(scale_diag=tf.ones(state_dim) * 1.0)
    x0_true = tf.zeros([state_dim])

    nlssm_true = NLSSM(state_dim, obs_dim, true_transition_fn, true_observation_fn,
                       process_noise, observation_noise, init_noise, x0_true)

    _, batch_y = nlssm_true.sample(batch_size=batch_size, T=T)  #
    return batch_y


def create_learnable_model(init_guess, state_dim=1, obs_dim=1):
    """Instantiates a fresh learnable SSM with the dummy-tensor build fix."""
    transition_net = LearnableNonlinearTransition(init_theta=init_guess)
    observation_net = IdentityObservation()

    # THE FIX: Explicitly build Keras layers to register 'theta'
    _ = transition_net(tf.zeros([1, state_dim], dtype=tf.float32))
    _ = observation_net(tf.zeros([1, state_dim], dtype=tf.float32))

    model = LearnableSSM(
        state_dim=state_dim, obs_dim=obs_dim,
        transition_layers=transition_net,
        observation_layers=observation_net,
        init_noise=tfd.MultivariateNormalDiag(scale_diag=tf.ones(state_dim)),
        learn_noise=False, learn_init_state=False,
        init_process_noise_scale=0.5, init_obs_noise_scale=0.2
    )
    return model, transition_net


# ==========================================
# 3. Experiment A: Statistical Comparison
# ==========================================
def run_statistical_comparison():
    batch_size = 8
    num_particles = 15  # Starved particle count to expose variance
    T = 60
    true_theta = 0.85
    init_guess = 0.20
    num_runs = 10
    epochs = 25

    print(f"\n--- EXPERIMENT A: Statistical Comparison ({num_runs} Runs) ---")
    print(f"Target Theta: {true_theta} | Initial Guess: {init_guess} | Particles: {num_particles}")

    batch_y = setup_experiment_data(T=T, batch_size=batch_size, true_theta=true_theta)

    soft_thetas, dpf_thetas = [], []
    soft_times, dpf_times = [], []

    for run in range(num_runs):
        tf.random.set_seed(run)  # Isolate noise realizations

        # 1. Setup Models
        model_soft, trans_soft = create_learnable_model(init_guess)
        model_dpf, trans_dpf = create_learnable_model(init_guess)

        srf = SoftResamplingParticleFilter(
            model=model_soft, num_particles=num_particles, alpha=0.5,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.03)
        )
        dpf = DifferentiableParticleFilter(
            model=model_dpf, num_particles=num_particles, epsilon=0.5, sinkhorn_iter=20,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.03)
        )

        # 2. Train Soft Resampling
        start_time = time.perf_counter()
        for _ in range(epochs):
            srf.train_step(batch_y, clip_norm=2.0)  #
        soft_times.append(time.perf_counter() - start_time)
        soft_thetas.append(trans_soft.theta.numpy())

        # 3. Train DPF
        start_time = time.perf_counter()
        for _ in range(epochs):
            dpf.train_step(batch_y, clip_norm=2.0)  #
        dpf_times.append(time.perf_counter() - start_time)
        dpf_thetas.append(trans_dpf.theta.numpy())

        print(f"Completed Run {run + 1}/{num_runs}")

    print("\n--- Final Method Comparison ---")
    print(f"{'Metric':<25} | {'Soft Resampling':<18} | {'DPF (Sinkhorn)':<18}")
    print("-" * 65)
    print(f"{'Mean Final Theta':<25} | {np.mean(soft_thetas):<18.4f} | {np.mean(dpf_thetas):<18.4f}")
    print(
        f"{'Absolute Bias':<25} | {np.mean(np.abs(np.array(soft_thetas) - true_theta)):<18.4f} | {np.mean(np.abs(np.array(dpf_thetas) - true_theta)):<18.4f}")
    print(f"{'Theta Variance':<25} | {np.var(soft_thetas):<18.6f} | {np.var(dpf_thetas):<18.6f}")
    print(f"{'Avg Time per Run (s)':<25} | {np.mean(soft_times):<18.2f} | {np.mean(dpf_times):<18.2f}")


# ==========================================
# 4. Experiment B: Bias-Variance Sweep (FIXED INDENTATION)
# ==========================================
def run_hyperparameter_sweep():
    print("\n--- EXPERIMENT B: Hyperparameter Bias-Variance Sweep ---")

    batch_y = setup_experiment_data(batch_size=8, T=40, true_theta=0.85)

    # 1. Sweep Alpha for Soft Resampling
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    soft_biases, soft_variances = [], []

    print("Sweeping Alpha (Soft Resampling)...")
    for alpha in alphas:
        thetas = []
        for trial in range(5):
            tf.random.set_seed(trial * 100)

            model, trans = create_learnable_model(0.20)
            srf = SoftResamplingParticleFilter(model, num_particles=20, alpha=alpha,
                                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.04))
            for _ in range(25):
                srf.train_step(batch_y, clip_norm=2.0)
            thetas.append(trans.theta.numpy())

        # VERY IMPORTANT: These must be indented under the `for alpha in alphas:` loop!
        soft_biases.append(np.mean(np.abs(np.array(thetas) - 0.85)))
        soft_variances.append(np.var(thetas))

    # 2. Sweep Epsilon for DPF
    epsilons = [0.1, 0.5, 1.0, 2.5, 5.0]
    dpf_biases, dpf_variances = [], []

    print("Sweeping Epsilon (Sinkhorn DPF)...")
    for eps in epsilons:
        thetas = []
        for trial in range(5):
            tf.random.set_seed(trial * 100 + 42)

            model, trans = create_learnable_model(0.20)
            dpf = DifferentiableParticleFilter(model, num_particles=20, epsilon=eps, sinkhorn_iter=20,
                                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.04))
            for _ in range(100):
                dpf.train_step(batch_y, clip_norm=2.0)
            thetas.append(trans.theta.numpy())

        # VERY IMPORTANT: These must be indented under the `for eps in epsilons:` loop!
        dpf_biases.append(np.mean(np.abs(np.array(thetas) - 0.85)))
        dpf_variances.append(np.var(thetas))

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Soft Resampling Plot
    ax1.set_title('Soft Resampling (Varying $\\alpha$)')
    ax1.set_xlabel('$\\alpha$ (0.0=Uniform, 1.0=Hard)')
    ax1.plot(alphas, soft_biases, 'b-o', label='Bias (Error)')
    ax1.set_ylabel('Absolute Bias', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax1_var = ax1.twinx()
    ax1_var.plot(alphas, soft_variances, 'r-s', label='Variance')
    ax1_var.set_ylabel('Variance', color='r')
    ax1_var.tick_params(axis='y', labelcolor='r')

    # DPF Plot
    ax2.set_title('DPF Sinkhorn (Varying $\\epsilon$)')
    ax2.set_xlabel('$\\epsilon$ (Entropy Regularization)')
    ax2.plot(epsilons, dpf_biases, 'b-o')
    ax2.set_ylabel('Absolute Bias', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    ax2_var = ax2.twinx()
    ax2_var.plot(epsilons, dpf_variances, 'r-s')
    ax2_var.set_ylabel('Variance', color='r')
    ax2_var.tick_params(axis='y', labelcolor='r')

    plt.tight_layout()
    plt.savefig('bias_variance_comparison.pdf',bbox_inches='tight')
    print("\nPlot saved successfully to 'bias_variance_comparison.png'!")
    plt.show()


if __name__ == "__main__":
    tf.config.optimizer.set_jit(True)
    #run_statistical_comparison()
    run_hyperparameter_sweep()