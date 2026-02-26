import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Filters.flow_filters import KernelPFF
from models import getLorenz96Model
from scipy.stats import multivariate_normal

dtype = tf.float64
tf.random.set_seed(42)
np.random.seed(42)


def replicate_figure_3():
    N = 20
    D = 1000
    F = 8.0

    print(f"Initializing Lorenz 96 (D={D})...")
    l96 = getLorenz96Model(state_dim=D, obs_dim=250, F=F, observation_noise_std=0.5, observe_every_nth=4)

    # 1. Spin up Truth
    x_true = tf.ones((1, D), dtype=dtype) * F
    x_true = tf.tensor_scatter_nd_update(x_true, [[0, 0]], [F + 0.01])
    for _ in range(1000):
        x_true = l96.transition_fn(x_true, tf.zeros_like(x_true))

    # Search for a 20-step window where x20 lands in a visible positive range (e.g. ~6.0)
    print("Finding suitable snapshot...")
    best_x_true = None
    for _ in range(500):
        x_test = x_true
        for _ in range(20):
            x_test = l96.transition_fn(x_test, tf.zeros_like(x_test))
        if 5.5 < x_test[0, 19] < 7.0:  # x20 is at index 19
            best_x_true = x_true
            break
        x_true = l96.transition_fn(x_true, tf.zeros_like(x_true))

    if best_x_true is not None:
        x_true = best_x_true

    # 2. Generate Prior Ensemble at t=1000 and integrate forward 20 steps
    prior_t1000 = x_true + tf.random.normal((N, D), stddev=np.sqrt(2.0), dtype=dtype)
    for _ in range(20):
        x_true = l96.transition_fn(x_true, tf.zeros_like(x_true))
        prior_t1000 = l96.transition_fn(prior_t1000, tf.zeros_like(prior_t1000))

    prior = tf.expand_dims(prior_t1000, 0)

    y_true = l96.observation_fn(x_true, tf.zeros((1,250),dtype=dtype))
    y_obs = y_true + tf.random.normal(y_true.shape, stddev=0.5, dtype=dtype)

    # 3. Run Filters
    print("Running Matrix Kernel PFF...")
    num_flow_steps = 200
    step_sizes = tf.ones(num_flow_steps, dtype=dtype) * 0.05

    pff_mat = KernelPFF(l96, N, kernel_type='matrix')
    post_mat = pff_mat._flow_update(y_obs, prior, num_flow_steps, step_sizes=step_sizes)

    print("Running Scalar Kernel PFF...")
    pff_scal = KernelPFF(l96, N, kernel_type='scalar')
    post_scal = pff_scal._flow_update(y_obs, prior, num_flow_steps, step_sizes=step_sizes)

    # 4. Compute Analytical EnKF Covariance for contours
    idx_u = 18  # Unobserved x19
    idx_o = 19  # Observed x20

    x_bar = tf.reduce_mean(prior, axis=1)
    centered = prior - tf.expand_dims(x_bar, 1)
    B_sample = tf.matmul(centered, centered, transpose_a=True) / (N - 1.0)

    # Localize B
    idx = tf.range(D)
    diff_idx = tf.abs(tf.expand_dims(idx, 0) - tf.expand_dims(idx, 1))
    dist_mat = tf.minimum(diff_idx, D - diff_idx)
    C_loc = tf.exp(-(tf.cast(dist_mat, dtype) / 4.0) ** 2)
    B_loc = B_sample[0] * C_loc

    # Extract EnKF posterior mean and covariance for the two plotted variables
    H_mat = np.zeros((250, 1000))
    for i, h_idx in enumerate(range(3, 1000, 4)):
        H_mat[i, h_idx] = 1.0
    H_tf = tf.constant(H_mat, dtype=dtype)
    R_tf = tf.eye(250, dtype=dtype) * (0.5 ** 2)

    HB = tf.matmul(H_tf, B_loc)
    S = tf.matmul(HB, H_tf, transpose_b=True) + R_tf
    K_mat = tf.matmul(B_loc, tf.matmul(H_tf, tf.linalg.inv(S), transpose_a=True))

    P_post = B_loc - tf.matmul(K_mat, HB)
    innov = y_obs[0] - tf.matmul(x_bar, H_tf, transpose_b=True)[0]
    mu_post = x_bar[0] + tf.matmul(K_mat, tf.expand_dims(innov, 1))[:, 0]

    mu_2d = [mu_post[idx_u].numpy(), mu_post[idx_o].numpy()]
    P_2d = np.array([[P_post[idx_u, idx_u].numpy(), P_post[idx_u, idx_o].numpy()],
                     [P_post[idx_o, idx_u].numpy(), P_post[idx_o, idx_o].numpy()]])

    # 5. Plotting identically to Fig 3
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def plot_res(ax, post, title):
        # EnKF Contours
        x_grid, y_grid = np.mgrid[-8:8:.1, -2:12:.1]
        pos = np.dstack((x_grid, y_grid))
        rv = multivariate_normal(mu_2d, P_2d)
        ax.contour(x_grid, y_grid, rv.pdf(pos), levels=8, cmap='jet', alpha=0.5, zorder=1)

        # Particles
        ax.scatter(prior[0, :, idx_u], prior[0, :, idx_o], facecolors='none', edgecolors='k', zorder=2)
        ax.scatter(post[0, :, idx_u], post[0, :, idx_o], facecolors='none', edgecolors='r', zorder=3)

        ax.set_title(title)
        ax.set_xlabel("x19")
        if ax == axes[0]: ax.set_ylabel("x20")

        ax.set_xlim([-8, 8])
        ax.set_ylim([-2, 12])
        ax.grid(True, alpha=0.3)

    plot_res(axes[0], post_mat, "matrix-valued")
    plot_res(axes[1], post_scal, "scalar")

    plt.savefig("posterior_kernelPFF.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    replicate_figure_3()