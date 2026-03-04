import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from .deterministic_flow import EDHFlow
from models import NLSSM

# Use float64 for numerical precision in High-D sums
dtype = tf.float64
tf.random.set_seed(42)
np.random.seed(42)
tfd=tfp.distributions





class KernelPFF(EDHFlow):
    def __init__(self, model, num_particles, kernel_type='matrix'):
        super().__init__(model, num_particles)
        self.kernel_type = kernel_type
        self.alpha = 1.0 / num_particles

        r_loc = 4.0
        idx = tf.range(model.state_dim)
        diff_idx = tf.abs(tf.expand_dims(idx, 0) - tf.expand_dims(idx, 1))
        dist_mat = tf.minimum(diff_idx, model.state_dim - diff_idx)
        self.C_loc_mat = tf.exp(-(tf.cast(dist_mat, dtype) / r_loc) ** 2)  # [D, D]

        # Obs Noise Covariance (epsilon=0.5 -> R=0.25)
        self.R_inv_diag = tf.ones(model.obs_dim, dtype=dtype) * (1.0 / 0.5 ** 2)

    def _compute_gradients_log_posterior(self, particles, y, B_inv, x_bar):
        """
        Gradient of log Posterior = Grad(Prior) + Grad(Likelihood)
        particles: [B, N, D]
        y: [B, ObsDim]
        B_inv: [B, D, D]
        x_bar: [B, 1, D]
        Returns:
            grads: [B, N, D]
        """
        B = tf.shape(particles)[0]
        N = self.num_particles
        D = self.model.state_dim
        Dy = self.model.obs_dim

        # Prior Gradient: - B_inv @ (x - x_bar)
        diff = particles - x_bar  # shape: [B, N, D]
        grad_prior = -tf.matmul(diff, B_inv)  # shape: [B, N, D]

        # Likelihood Gradient using vectorized gradient computation
        # Reshape to treat each particle independently
        particles_flat = tf.reshape(particles, [-1, D])  # [B*N, D]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(particles_flat)
            # Compute observation for all particles at once
            Hx_flat = self.model.observation_fn(
                particles_flat,
                tf.zeros((B*N,Dy),dtype=particles.dtype)
            )  # [B*N, ObsDim]

        # Reshape observations to match particle structure
        Hx = tf.reshape(Hx_flat, [B, N, -1])  # [B, N, ObsDim]

        # Innovation: y - Hx
        innov = tf.expand_dims(y, 1) - Hx  # [B, N, ObsDim]
        weighted_innov = innov * self.R_inv_diag  # [B, N, ObsDim]

        # Flatten weighted innovation to match particles_flat
        weighted_innov_flat = tf.reshape(weighted_innov, [-1, tf.shape(Hx)[2]])  # [B*N, ObsDim]

        # Use gradient with output_gradients to compute weighted Jacobian-vector product
        # This computes J^T @ weighted_innov in one pass, avoiding explicit Jacobian computation
        grad_like_flat = tape.gradient(
            Hx_flat,
            particles_flat,
            output_gradients=weighted_innov_flat
        )  # [B*N, D]

        # Reshape back to batch structure
        grad_like = tf.reshape(grad_like_flat, [B, N, D])  # [B, N, D]

        return grad_prior + grad_like

    def _flow_update(self,observations, particles, num_flow_steps,P_xx=None, step_sizes=None):
        """
        Runs the particle flow update for a batch of particles given observations.
        observations: [B, ObsDim]
        particles: [B,N, StateDim]
        Returns:
            Updated particles after flow: [B, N, StateDim]

        Steps:
        1. Compute Localized Prior Covariance B
        2. Compute Bandwidth (Matrix or Scalar)
        3. For each flow step:
            A. Compute Gradient of Log Posterior
            B. Compute Kernel and its Divergence
            C. Update Particles
        4. Return Updated Particles
        """
        N = self.num_particles
        D = self.model.state_dim

        # Compute Localized Prior Covariance B
        x_bar = tf.reduce_mean(particles, axis=1,keepdims=True)
        centered = particles - x_bar # shape: [B,N, D]
        B_sample = tf.matmul(centered, centered, transpose_a=True) / (tf.cast(N, dtype) - 1) # shape: [B,D,D]

        # Localization (Row-wise approximation)
        # idx = tf.range(D)
        # diff_idx = tf.abs(tf.expand_dims(idx, 0) - tf.expand_dims(idx, 1))
        # dist_mat = tf.minimum(diff_idx, D - diff_idx)
        # C_mat = tf.exp(-(tf.cast(dist_mat, dtype) / 4.0) ** 2) # shape: [D,D]
        # B_loc = B_sample * C_mat
        B_loc = B_sample * self.C_loc_mat # shape: [B,D,D]*[D,D] -> [B,D,D]

        # Preconditioner D = B_loc
        D_matrix = B_loc # shape: [B, D, D]
        B_inv = tf.linalg.inv(B_loc + 1e-5 * tf.eye(D, dtype=dtype))
        B_diag = tf.linalg.diag_part(B_loc)  # shape: [B, D]
        if self.kernel_type == 'matrix':
            # Matrix Kernel: Local Bandwidth
            # sigma_d^2 ~ Var(x_d) -> Small
            sigma_sq = self.alpha * B_diag
            sigma_sq = tf.maximum(sigma_sq, 1e-6)
        else:
            # Scalar Kernel: Global Bandwidth
            # We use the standard definition relative to mean variance.
            # IMPORTANT: In 1000D, the distance between particles is massive.
            # exp(-dist / sigma) becomes effectively 0.
            # This kills the repulsion, causing collapse.
            mean_var = tf.reduce_mean(B_diag,axis=1,keepdims=True) # shape: [B, 1]
            sigma_scalar_val = self.alpha * mean_var # shape: [B, 1]
            sigma_sq = sigma_scalar_val * tf.ones([1, D], dtype=dtype) # shape: [B, D]

        curr_particles = particles

        for s in tf.range(num_flow_steps):
            # Gradient of Log Posterior (Attracting Force)
            grads = self._compute_gradients_log_posterior(curr_particles, observations, B_inv, x_bar)

            # Compute Kernel Terms
            # xi: [B, N, 1, D], xj: [B, 1, N, D]
            xi = tf.expand_dims(curr_particles, 2)
            xj = tf.expand_dims(curr_particles, 1)
            diff_ij = xi - xj # shape: [B, N, N, D], for each batch, the difference between particle i and j along direction d
            diff_sq = diff_ij ** 2 # shape: [B, N, N, D], the squared distance between particle i and j along direction d

            if self.kernel_type == 'matrix':
                # --- Matrix Kernel ---
                # Component-wise K by broadcasting
                K_vals = tf.exp(-0.5 * diff_sq / sigma_sq) # shape : [B, N, N, D]

                # Divergence (Repelling Force)
                # Force = (x^i - x^j) / sigma_d^2 * K
                # sigma_d is small -> Repulsion is Strong -> No Collapse
                div_K = (diff_ij / sigma_sq) * K_vals # shape: [B, N, N, D]

                # Attraction: K * grad(x^j)
                grads_j = tf.expand_dims(grads, 1) # shape: [B,1, N, D]
                attract = K_vals * grads_j # shape: [B,N, N, D]


            else:
                # Scalar Kernel
                dist_sq_sum = tf.reduce_sum(diff_sq, axis=-1)  # Sum over state dims, shape: [B, N, N]
                sig_val = sigma_sq[:,0] # shape: [B]

                K_scalar = tf.exp(-0.5 * dist_sq_sum / sig_val) # shape: [B, N, N]
                K_scalar_exp = tf.expand_dims(K_scalar, -1)

                # Divergence (Repelling Force)
                # Force = (x^i - x^j) / sigma * K
                # Since K -> 0, Force -> 0.
                div_K = (diff_ij / sig_val) * K_scalar_exp # [B, N, N, D]

                # Attraction
                # Since K -> 0 for neighbors, this term becomes just (1 * grad_i)
                # i.e., each particle moves independently to the mode.
                grads_j = tf.expand_dims(grads, 1) # [B,1, N, D]
                attract = K_scalar_exp * grads_j

            total_force = tf.reduce_mean(attract + div_K, axis=2)

            # Update: dx = D * force * dt
            flow = tf.matmul(total_force, D_matrix)
            curr_particles = curr_particles + step_sizes[s] * flow

        return curr_particles

