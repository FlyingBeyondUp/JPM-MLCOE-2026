import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Filters.flow_filters import KernelPFF
from models import getLorenz96Model

dtype = tf.float64
tf.random.set_seed(42)


def replicate_figure_3():
    N = 20
    D = 1000
    F = 8.0

    print(f"Initializing Lorenz 96 (D={D})...")
    # 25% of the system is observed (every 4th variable)
    l96 = getLorenz96Model(state_dim=D, obs_dim=250, F=F)

    # 1. Exact Initialization as per paper
    indices = tf.range(1, D + 1)
    init_cond = tf.where(indices % 5 == 0, F + 1.0, F)
    x_true = tf.cast(tf.reshape(init_cond, (1, D)), dtype)

    # Spin up for 1000 steps to generate chaotic behavior
    print("Spinning up truth for 1000 steps...")
    for _ in range(1000):
        x_true = l96.transition_fn(x_true, tf.zeros_like(x_true))

    # 2. Generate initial ensemble and truth
    print("Generating ensemble and integrating for 20 steps...")
    # N(0, 2I) perturbation -> stddev is sqrt(2)
    prior_t0 = x_true + tf.random.normal((N, D), stddev=np.sqrt(2.0), dtype=dtype)

    # Integrate truth and ensemble for 20 steps before first DA update
    for _ in range(20):
        x_true = l96.transition_fn(x_true, tf.zeros_like(x_true))
        prior_t0 = l96.transition_fn(prior_t0, tf.zeros_like(prior_t0))

    # Generate observations at t=20
    y_full = l96.observation_fn(x_true, tf.zeros_like(x_true))
    y_obs = y_full + tf.random.normal(y_full.shape, stddev=0.5, dtype=dtype)

    # Extract specific observation value for plotting (index 4 in obs maps to x20)
    obs_val = y_obs[0, 4]

    prior = tf.expand_dims(prior_t0, 0)
    y_obs_batch = tf.expand_dims(y_obs, 0)

    # 3. Run Filters
    num_flow_steps = 500  # Used for all experiments in the paper

    print("Running Matrix Kernel PFF...")
    pff_mat = KernelPFF(l96, N, kernel_type='matrix')
    post_mat = pff_mat._flow_update(y_obs_batch, prior, num_flow_steps)

    print("Running Scalar Kernel PFF...")
    pff_scal = KernelPFF(l96, N, kernel_type='scalar')
    post_scal = pff_scal._flow_update(y_obs_batch, prior, num_flow_steps)

    # Drop batch dim for plotting
    prior_plot = prior[0]
    post_mat_plot = post_mat[0]
    post_scal_plot = post_scal[0]

    # 4. Plotting
    idx_u = 18  # Unobserved x19 (0-indexed)
    idx_o = 19  # Observed x20 (0-indexed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def plot_res(ax, post, title, is_collapsed):
        ax.scatter(prior_plot[:, idx_u], prior_plot[:, idx_o],
                   facecolors='none', edgecolors='k', label='Prior')

        # Highlight posterior particles
        c = 'r' if not is_collapsed else 'r'
        ax.scatter(post[:, idx_u], post[:, idx_o],
                   facecolors='none', edgecolors=c, label=f'Posterior')

        ax.set_title(f"{title} Kernel")
        ax.set_xlabel("x19 (Unobserved)")
        if ax == axes[0]:
            ax.set_ylabel("x20 (Observed)")
        ax.grid(True, alpha=0.3)

    plot_res(axes[0], post_mat_plot, "Matrix-Valued", False)
    plot_res(axes[1], post_scal_plot, "Scalar", True)

    #plt.savefig("posterior_kernelPFF.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    replicate_figure_3()