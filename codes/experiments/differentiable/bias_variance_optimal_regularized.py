import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from models.base_models import NLSSM
from Filters.basic_filters import ParticleFilter
from Filters.differentiable_filters.entropy_regularized_OT import DifferentiableParticleFilter

tfd = tfp.distributions

tf.config.run_functions_eagerly(True)


def create_stable_nlssm():
    """
    Creates a stable non-linear state-space model.
    The transition uses a sine function, making x=0 a stable equilibrium.
    """
    state_dim = 1
    obs_dim = 1

    def transition_fn(x, noise):
        return 0.9 * tf.sin(x) + noise

    def observation_fn(x, noise):
        return x + noise

    process_noise = tfd.MultivariateNormalDiag(scale_diag=tf.ones(state_dim) * tf.sqrt(0.5))
    observation_noise = tfd.MultivariateNormalDiag(scale_diag=tf.ones(obs_dim) * tf.sqrt(0.5))
    init_noise = tfd.MultivariateNormalDiag(scale_diag=tf.ones(state_dim) * tf.sqrt(0.1))
    x0_nlssm = tf.zeros([state_dim])

    nlssm = NLSSM(state_dim, obs_dim, transition_fn, observation_fn,
                  process_noise, observation_noise, init_noise, x0_nlssm)
    return nlssm


def run_state_mse_tradeoff_sweep():
    epsilons = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

    T = 100
    N_eval = 100
    num_realizations = 100

    nlssm = create_stable_nlssm()

    print(f"Simulating ground truth sequence (T={T})...")

    # Use proper batch initialization
    single_x, single_y = nlssm.sample(batch_size=1, T=T)

    batch_y = tf.tile(single_y, [num_realizations, 1, 1])
    batch_x_true = tf.tile(single_x, [num_realizations, 1, 1])

    biases = []
    variances = []
    intra_variances = []
    esses = []

    print("\nStarting Epsilon Sweep (Evaluating State Estimates)...")
    print(f"{'Epsilon':>8} | {'Bias (State Error)':>20} | {'Variance (State Fluctuation)':>28}")
    print("-" * 65)

    dpf = DifferentiableParticleFilter(nlssm, sinkhorn_iter=400, num_particles=N_eval)
    dpf.scaling = False

    for eps in epsilons:
        dpf.epsilon = eps

        # Switch to eval, but keep OT resampling active to measure its bias/variance
        dpf.eval(force_differentiable=True)
        res = dpf.filter(batch_y)

        x_filt = res['x_filt']
        P_filt = res['P_filt']
        ess = res['ess']

        ess = tf.reduce_mean(ess)
        particle_variances = tf.reduce_mean(tf.linalg.diag_part(P_filt)).numpy()
        intra_variances.append(particle_variances)

        mean_prediction = tf.reduce_mean(x_filt, axis=0)
        bias = tf.reduce_mean(tf.square(mean_prediction - batch_x_true[0])).numpy()
        variance = tf.reduce_mean(tf.square(x_filt - tf.expand_dims(mean_prediction, 0))).numpy()

        biases.append(bias)
        variances.append(variance)
        esses.append(ess)

        print(f"{eps:>8.2f} | {bias:>20.4f} | {variance:>28.4f} | {ess:>20.2f} | {particle_variances:>20.4f}")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Entropy Regularization (ε) [Log Scale]')
    ax1.set_xscale('log')
    ax1.set_ylabel('Variance of State Estimate', color=color)
    ax1.plot(epsilons, variances, marker='o', color=color, linewidth=2, label='Variance')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Bias (MSE vs Ground Truth)', color=color)
    ax2.plot(epsilons, biases, marker='s', color=color, linewidth=2, label='Bias')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('DPF State Estimation: Bias-Variance Trade-off')
    fig.tight_layout()
    #plt.savefig('dpf_state_tradeoff.png', dpi=300)
    print("\nPlot saved successfully!")


if __name__ == "__main__":
    run_state_mse_tradeoff_sweep()