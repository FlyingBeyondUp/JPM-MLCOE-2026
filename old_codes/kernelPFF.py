import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Use float64 for numerical precision in High-D sums
dtype = tf.float64
tf.random.set_seed(42)
np.random.seed(42)


class Lorenz96:
    def __init__(self, dim=1000, F=8.0, dt=0.01):
        self.state_dim = dim
        self.F = F
        self.dt = dt
        # Observation indices: Every 4th variable (0-based: 3, 7, 11...)
        self.obs_indices = np.arange(3, dim, 4)
        self.obs_dim = len(self.obs_indices)

    def transition(self, x):
        """RK4 Integration for L96"""

        def f(s):
            s_p1 = tf.roll(s, shift=-1, axis=1)
            s_m2 = tf.roll(s, shift=2, axis=1)
            s_m1 = tf.roll(s, shift=1, axis=1)
            return (s_p1 - s_m2) * s_m1 - s + self.F

        k1 = f(x)
        k2 = f(x + 0.5 * self.dt * k1)
        k3 = f(x + 0.5 * self.dt * k2)
        k4 = f(x + self.dt * k3)
        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def observation(self, x):
        return tf.gather(x, self.obs_indices, axis=1)


class KernelPFF:
    def __init__(self, model, num_particles, kernel_type='matrix'):
        self.model = model
        self.num_particles = num_particles
        self.kernel_type = kernel_type
        # [cite_start]Alpha suggested in paper (Section 2.3) [cite: 259]
        self.alpha = 1.0 / num_particles

        # [cite_start]Localization Vector (Gaussian decay) [cite: 463]
        r_loc = 4.0
        d_range = np.arange(model.state_dim)
        dist = np.minimum(d_range, model.state_dim - d_range)  # Circular dist
        self.C_loc_vec = tf.constant(np.exp(-(dist / r_loc) ** 2), dtype=dtype)

        # Obs Noise Covariance (epsilon=0.5 -> R=0.25)
        self.R_inv_diag = tf.ones(model.obs_dim, dtype=dtype) * (1.0 / 0.5 ** 2)

    def _compute_gradients_log_posterior(self, particles, y, B_inv, x_bar):
        """
        Gradient of log Posterior = Grad(Prior) + Grad(Likelihood)
        """
        N = self.num_particles
        D = self.model.state_dim

        # [cite_start]1. Prior Gradient: - B_inv @ (x - x_bar) [cite: 213]
        diff = particles - x_bar
        grad_prior = -tf.matmul(diff, B_inv)

        # [cite_start]2. Likelihood Gradient: H^T R^-1 (y - Hx) [cite: 201]
        Hx = tf.gather(particles, self.model.obs_indices, axis=1)
        innov = y - Hx
        weighted_innov = innov * self.R_inv_diag

        # Scatter R^-1(y-Hx) back to state dimension (H^T operation)
        indices = tf.constant(self.model.obs_indices, dtype=tf.int32)
        p_idx = tf.repeat(tf.range(N), len(self.model.obs_indices))
        s_idx = tf.tile(indices, [N])
        scatter_indices = tf.stack([tf.cast(p_idx, tf.int32), tf.cast(s_idx, tf.int32)], axis=1)

        grad_like = tf.scatter_nd(scatter_indices, tf.reshape(weighted_innov, [-1]), shape=[N, D])

        return grad_prior + grad_like

    def _flow_update(self, particles, y, num_steps=200, step_size=0.05):
        """
        Runs the particle flow update.
        CRITICAL FIX: step_size=0.05 (from paper) and sufficient steps ensure convergence.
        """
        N = self.num_particles
        D = self.model.state_dim

        # [cite_start]--- 1. Compute Localized Prior Covariance B [cite: 338, 339] ---
        x_bar = tf.reduce_mean(particles, axis=0)
        centered = particles - x_bar
        B_sample = tf.matmul(centered, centered, transpose_a=True) / (tf.cast(N, dtype) - 1)

        # Localization (Row-wise approximation)
        idx = tf.range(D)
        diff_idx = tf.abs(tf.expand_dims(idx, 0) - tf.expand_dims(idx, 1))
        dist_mat = tf.minimum(diff_idx, D - diff_idx)
        C_mat = tf.exp(-(tf.cast(dist_mat, dtype) / 4.0) ** 2)
        B_loc = B_sample * C_mat

        # [cite_start]Preconditioner D = B_loc [cite: 129]
        D_matrix = B_loc
        B_inv = tf.linalg.inv(B_loc + 1e-5 * tf.eye(D, dtype=dtype))

        # --- 2. Bandwidth Calculation ---
        B_diag = tf.linalg.diag_part(B_loc)

        if self.kernel_type == 'matrix':
            # [cite_start]Matrix Kernel: Local Bandwidth [cite: 248]
            # sigma_d^2 ~ Var(x_d) -> Small
            sigma_sq = self.alpha * B_diag
            sigma_sq = tf.maximum(sigma_sq, 1e-6)
        else:
            # [cite_start]Scalar Kernel: Global Bandwidth [cite: 230]
            # We use the standard definition relative to mean variance.
            # IMPORTANT: In 1000D, the distance between particles is massive.
            # exp(-dist / sigma) becomes effectively 0.
            # This kills the repulsion, causing collapse.
            mean_var = tf.reduce_mean(B_diag)
            sigma_scalar_val = self.alpha * mean_var
            sigma_sq = tf.ones(D, dtype=dtype) * sigma_scalar_val

        curr_particles = particles

        for s in range(num_steps):
            # A. Gradient of Log Posterior (Attracting Force)
            grads = self._compute_gradients_log_posterior(curr_particles, y, B_inv, x_bar)

            # B. Compute Kernel Terms
            # xi: [N, 1, D], xj: [1, N, D]
            xi = tf.expand_dims(curr_particles, 1)
            xj = tf.expand_dims(curr_particles, 0)
            diff_ij = xi - xj
            diff_sq = diff_ij ** 2

            if self.kernel_type == 'matrix':
                # --- Matrix Kernel ---
                # [cite_start]Component-wise K [cite: 246]
                K_vals = tf.exp(-0.5 * diff_sq / sigma_sq)

                # [cite_start]Divergence (Repelling Force) [cite: 263]
                # Force = (x^i - x^j) / sigma_d^2 * K
                # sigma_d is small -> Repulsion is Strong -> No Collapse
                div_K = (diff_ij / sigma_sq) * K_vals

                # Attraction: K * grad(x^j)
                grads_j = tf.expand_dims(grads, 0)
                attract = K_vals * grads_j

                total_force = tf.reduce_mean(attract + div_K, axis=1)

            else:
                # --- Scalar Kernel ---
                # [cite_start]Global K [cite: 226]
                dist_sq_sum = tf.reduce_sum(diff_sq, axis=2)  # Sum over 1000 dims
                sig_val = sigma_sq[0]

                # In 1000D, dist_sq_sum is HUGE. K_scalar -> 0 for neighbors.
                K_scalar = tf.exp(-0.5 * dist_sq_sum / sig_val)
                K_scalar_exp = tf.expand_dims(K_scalar, -1)

                # [cite_start]Divergence (Repelling Force) [cite: 240]
                # Force = (x^i - x^j) / sigma * K
                # Since K -> 0, Force -> 0.
                div_K = (diff_ij / sig_val) * K_scalar_exp

                # Attraction
                # Since K -> 0 for neighbors, this term becomes just (1 * grad_i)
                # i.e., each particle moves independently to the mode.
                grads_j = tf.expand_dims(grads, 0)
                attract = K_scalar_exp * grads_j

                total_force = tf.reduce_mean(attract + div_K, axis=1)

            # [cite_start]C. Update: dx = D * force * dt [cite: 136]
            flow = tf.matmul(total_force, D_matrix)
            curr_particles = curr_particles + step_size * flow

        return curr_particles


def replicate_figure_3():
    # [cite_start]SETUP: D=1000 is critical for scalar failure [cite: 244]
    N = 20
    D = 1000
    F = 8.0

    print(f"Initializing Lorenz 96 (D={D})...")
    l96 = Lorenz96(dim=D, F=F)

    # 1. Truth & Obs (Spin up to chaotic state)
    x_true = tf.ones((1, D), dtype=dtype) * F
    x_true = x_true + tf.random.normal((1, D), stddev=0.5, dtype=dtype)
    for _ in range(200): x_true = l96.transition(x_true)

    # Find snapshot where x20 > 0 to match visual of Fig 3
    print("Finding suitable snapshot...")
    for _ in range(500):
        if x_true[0, 19] > 4.0: break
        x_true = l96.transition(x_true)

    y_full = l96.observation(x_true)
    y_obs = y_full + tf.random.normal(y_full.shape, stddev=0.5, dtype=dtype)
    obs_val = y_obs[0, 4]  # Obs for x20

    # 2. Prior (Forecast Ensemble)
    prior = x_true + tf.random.normal((N, D), stddev=2.0, dtype=dtype)

    # 3. Run Filters
    # Increased steps and step_size to allow full convergence
    print("Running Matrix Kernel PFF...")
    pff_mat = KernelPFF(l96, N, kernel_type='matrix')
    post_mat = pff_mat._flow_update(prior, y_obs, num_steps=200, step_size=0.05)

    print("Running Scalar Kernel PFF...")
    pff_scal = KernelPFF(l96, N, kernel_type='scalar')
    post_scal = pff_scal._flow_update(prior, y_obs, num_steps=200, step_size=0.05)

    # 4. Plotting
    idx_u = 18  # Unobserved x19
    idx_o = 19  # Observed x20

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Function
    def plot_res(ax, post, title, is_collapsed):
        ax.scatter(prior[:, idx_u], prior[:, idx_o],
                   facecolors='none', edgecolors='k', label='Prior')
        ax.scatter(post[:, idx_u], post[:, idx_o],
                   c='r', label=f'Posterior ({title})')
        ax.axhline(obs_val, color='b', linestyle='--', label='Observation')
        ax.set_title(f"{title} Kernel" + (" (Collapsed)" if is_collapsed else ""))
        ax.set_xlabel("Unobserved x19")
        ax.set_ylabel("Observed x20")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_res(axes[0], post_mat, "Matrix-Valued", False)
    plot_res(axes[1], post_scal, "Scalar", True)

    #plt.suptitle(f"Replication of Figure 3 (Hu & van Leeuwen 2021)\nEffect of Kernel Choice in High Dimensions (D={D})")
    plt.savefig("posterior_kernelPFF.pdf",bbox_inches='tight')
    plt.show()



def replicate_figure_2():
    sigmax_sq, sigmay_sq = 2, 2
    alpha_inv = 5
    sigma_sq_np = np.array([sigmax_sq, sigmay_sq])
    K_matrix = lambda p1, p2: np.array([
        np.exp(-((p1[0] - p2[0]) ** 2) / (sigmax_sq)),
        np.exp(-((p1[1] - p2[1]) ** 2) / (sigmay_sq))
    ])

    p1, p2 = np.array([9, 9.6]), np.array([10.4, 10.4])
    matrix_force1 = alpha_inv * ((p1 - p2) / sigma_sq_np) * K_matrix(p1, p2)
    matrix_force2 = -alpha_inv * ((p1 - p2) / sigma_sq_np) * K_matrix(p1, p2)

    sigma_sq = 2.0
    K_scalar = lambda p1, p2: np.exp(-np.sum((p1 - p2) ** 2) / (1.4 * sigma_sq))
    scalar_force1 = alpha_inv * (p1 - p2) * K_scalar(p1, p2) / sigma_sq
    scalar_force2 = -alpha_inv * (p1 - p2) * K_scalar(p1, p2) / sigma_sq

    p3, p4 = np.array([6, 9.6]), np.array([14, 10.4])
    matrix_force3 = alpha_inv * ((p3 - p4) / sigma_sq_np) * K_matrix(p3, p4)
    matrix_force4 = -alpha_inv * ((p3 - p4) / sigma_sq_np) * K_matrix(p3, p4)

    scalar_force3 = alpha_inv * (p3 - p4) * K_scalar(p3, p4) / sigma_sq
    scalar_force4 = -alpha_inv * (p3 - p4) * K_scalar(p3, p4) / sigma_sq

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    x_min, x_max = 3, 18
    y_min, y_max = 7, 13
    headwidth = 0.5

    labels = ['(a)', '(b)', '(c)', '(d)']
    for idx,ax in enumerate(axs.flat):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

        arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
        ax.annotate('', xy=(x_max - 0.5, y_min), xytext=(x_min, y_min),
                    arrowprops=arrow_props)
        ax.annotate('', xy=(x_min, y_max - 0.5), xytext=(x_min, y_min),
                    arrowprops=arrow_props)

        ax.text(x_max - 1, y_min - 1, 'x₁', fontsize=20, ha='center')
        ax.text(x_min - 1, y_max - 1, 'x₂', fontsize=20, va='center')

        ax.text(-0.1, 1.1, labels[idx], transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top', ha='left')

    axs[0, 0].scatter(*p1, color='black', s=300, zorder=5)
    axs[0, 0].scatter(*p2, color='black', s=300, zorder=5)
    axs[0, 0].arrow(p1[0], p1[1], matrix_force1[0], 0, color='black', head_width=headwidth, length_includes_head=True)
    axs[0, 0].arrow(p1[0], p1[1], 0, matrix_force1[1], color='black', head_width=headwidth, length_includes_head=True)
    axs[0, 0].arrow(p2[0], p2[1], matrix_force2[0], 0, color='black', head_width=headwidth, length_includes_head=True)
    axs[0, 0].arrow(p2[0], p2[1], 0, matrix_force2[1], color='black', head_width=headwidth, length_includes_head=True)
    for center in [p1, p2]:
        circle = plt.Circle(center, np.sqrt(sigmax_sq), color='gray', alpha=0.2, fill=True)
        axs[0, 0].add_patch(circle)
    axs[0, 0].axhline(y=(p1[1]+p2[1])/2, color='black', linestyle='--', linewidth=1)

    axs[0, 1].scatter(*p1, color='black', s=300, zorder=5)
    axs[0, 1].scatter(*p2, color='black', s=300, zorder=5)
    axs[0, 1].arrow(p1[0], p1[1], scalar_force1[0], scalar_force1[1], color='black', head_width=headwidth,
                    length_includes_head=True)
    axs[0, 1].arrow(p2[0], p2[1], scalar_force2[0], scalar_force2[1], color='black', head_width=headwidth,
                    length_includes_head=True)
    for center in [p1, p2]:
        circle = plt.Circle(center, np.sqrt(sigmax_sq), color='gray', alpha=0.2, fill=True)
        axs[0, 1].add_patch(circle)
    axs[0, 1].axhline(y=(p1[1] + p2[1]) / 2, color='black', linestyle='--', linewidth=1)

    axs[1, 0].scatter(*p3, color='black', s=300, zorder=5)
    axs[1, 0].scatter(*p4, color='black', s=300, zorder=5)
    axs[1, 0].arrow(p3[0], p3[1], 0, matrix_force3[1], color='black', head_width=0.5, length_includes_head=True)
    axs[1, 0].arrow(p4[0], p4[1], 0, matrix_force4[1], color='black', head_width=0.5, length_includes_head=True)
    for center in [p3, p4]:
        circle = plt.Circle(center, np.sqrt(sigmax_sq), color='gray', alpha=0.2, fill=True)
        axs[1, 0].add_patch(circle)
    axs[1, 0].axhline(y=(p2[1] + p3[1]) / 2, color='black', linestyle='--', linewidth=1)


    axs[1, 1].scatter(*p3, color='black', s=300, zorder=5)
    axs[1, 1].scatter(*p4, color='black', s=300, zorder=5)
    for center in [p3, p4]:
        circle = plt.Circle(center, np.sqrt(sigmax_sq), color='gray', alpha=0.2, fill=True)
        axs[1, 1].add_patch(circle)
    axs[1, 1].axhline(y=(p2[1] + p3[1]) / 2, color='black', linestyle='--', linewidth=1)

    def get_extended_line(p1, p2, extend_factor=2.0):
        direction = p2 - p1
        center = (p1 + p2) / 2
        half_length = np.linalg.norm(direction) * extend_factor / 2
        unit_dir = direction / np.linalg.norm(direction)

        start = center - half_length * unit_dir
        end = center + half_length * unit_dir
        return start, end


    start, end = get_extended_line(p1, p2, extend_factor=6.0)
    axs[0, 0].plot([start[0], end[0]], [start[1], end[1]],
                   color='black', linestyle='--', linewidth=1, zorder=3)

    #start, end = get_extended_line(p1, p2, extend_factor=3.0)
    axs[0, 1].plot([start[0], end[0]], [start[1], end[1]],
                   color='black', linestyle='--', linewidth=1, zorder=3)

    start, end = get_extended_line(p3, p4, extend_factor=1.5)
    axs[1, 0].plot([start[0], end[0]], [start[1], end[1]],
                   color='black', linestyle='--', linewidth=1, zorder=3)

    start, end = get_extended_line(p3, p4, extend_factor=1.5)
    axs[1, 1].plot([start[0], end[0]], [start[1], end[1]],
                   color='black', linestyle='--', linewidth=1, zorder=3)

    plt.tight_layout()
    plt.savefig("schematic_figure_matrix_scalar_kernels.pdf",bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    replicate_figure_3()

    #replicate_figure_2()