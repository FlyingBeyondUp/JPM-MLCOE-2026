import tensorflow as tf
import tensorflow_probability as tfp
from .deterministic_flow import EDHFlow

# Standardize on float32 to match the rest of the framework and prevent TF TypeError crashes
dtype = tf.float32
tfd = tfp.distributions


class KernelPFF(EDHFlow):
    """
    Kernel Particle Flow Filter.

    Uses a kernel density approach (similar to SVGD) to migrate particles.
    """

    def __init__(self, model, num_particles, num_flow_steps=10, step_sizes=None,
                 ukf=None, resample_from_ukf=False, kernel_type='matrix'):
        # 1. API FIX: Pipe the standard flow arguments down to the EDHFlow parent
        super().__init__(model, num_particles, num_flow_steps, step_sizes, ukf, resample_from_ukf)

        self.kernel_type = kernel_type
        self.alpha = 1.0 / num_particles

        r_loc = 4.0
        idx = tf.range(model.state_dim)
        diff_idx = tf.abs(tf.expand_dims(idx, 0) - tf.expand_dims(idx, 1))
        dist_mat = tf.minimum(diff_idx, model.state_dim - diff_idx)
        self.C_loc_mat = tf.exp(-(tf.cast(dist_mat, dtype) / r_loc) ** 2)  # [D, D]

    @property
    def R_inv_diag(self):
        """API FIX: Dynamically fetch observation noise precision from the model."""
        R = self._get_cov(self.model.observation_noise)
        R_diag = tf.linalg.diag_part(R)
        return 1.0 / (R_diag + 1e-6)

    def _compute_gradients_log_posterior(self, particles, y, B_inv, x_bar):
        """
        Gradient of log Posterior = Grad(Prior) + Grad(Likelihood)
        """
        B = tf.shape(particles)[0]
        N = self.num_particles
        D = self.model.state_dim
        Dy = self.model.obs_dim

        # Prior Gradient: - B_inv @ (x - x_bar)
        diff = particles - x_bar  # shape: [B, N, D]
        grad_prior = -tf.matmul(diff, B_inv)  # shape: [B, N, D]

        # Likelihood Gradient using vectorized gradient computation
        particles_flat = tf.reshape(particles, [-1, D])  # [B*N, D]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(particles_flat)
            Hx_flat = self.model.observation_fn(
                particles_flat,
                tf.zeros((B * N, Dy), dtype=dtype)
            )

        Hx = tf.reshape(Hx_flat, [B, N, -1])  # [B, N, ObsDim]

        # Innovation: y - Hx
        innov = tf.expand_dims(y, 1) - Hx  # [B, N, ObsDim]
        weighted_innov = innov * self.R_inv_diag  # [B, N, ObsDim]

        weighted_innov_flat = tf.reshape(weighted_innov, [-1, tf.shape(Hx)[2]])  # [B*N, ObsDim]

        grad_like_flat = tape.gradient(
            Hx_flat,
            particles_flat,
            output_gradients=weighted_innov_flat
        )  # [B*N, D]

        grad_like = tf.reshape(grad_like_flat, [B, N, D])  # [B, N, D]

        return grad_prior + grad_like

    # 2. API FIX: Match the exact signature expected by EDHFlow.update()
    def _flow_update(self, observation: tf.Tensor, particles: tf.Tensor, P_xx=None) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor]:
        N = self.num_particles
        D = self.model.state_dim
        const_delta_lambda = tf.cast(1.0 / float(self.num_flow_steps), dtype)

        x_bar = tf.reduce_mean(particles, axis=1, keepdims=True)
        centered = particles - x_bar
        B_sample = tf.matmul(centered, centered, transpose_a=True) / (tf.cast(N, dtype) - 1.0)

        B_loc = B_sample * self.C_loc_mat

        D_matrix = B_loc
        B_inv = tf.linalg.inv(B_loc + 1e-5 * tf.eye(D, dtype=dtype))
        B_diag = tf.linalg.diag_part(B_loc)

        if self.kernel_type == 'matrix':
            sigma_sq = self.alpha * B_diag
            sigma_sq = tf.maximum(sigma_sq, 1e-6)
        else:
            mean_var = tf.reduce_mean(B_diag, axis=1, keepdims=True)
            sigma_scalar_val = self.alpha * mean_var
            sigma_sq = sigma_scalar_val * tf.ones([1, D], dtype=dtype)

        curr_particles = particles

        for s in tf.range(self.num_flow_steps):
            # 3. API FIX: Inject the AutoGraph dynamic shape loop options
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (curr_particles, tf.TensorShape([None, self.num_particles, self.model.state_dim]))
                ]
            )

            # Fetch step size safely
            dt = self.step_sizes[s] if self.step_sizes is not None else const_delta_lambda

            grads = self._compute_gradients_log_posterior(curr_particles, observation, B_inv, x_bar)

            xi = tf.expand_dims(curr_particles, 2)
            xj = tf.expand_dims(curr_particles, 1)
            diff_ij = xi - xj
            diff_sq = diff_ij ** 2

            if self.kernel_type == 'matrix':
                # FIX: Expand [B, D] to [B, 1, 1, D] for correct broadcasting
                sigma_sq_exp = sigma_sq[:, tf.newaxis, tf.newaxis, :]

                K_vals = tf.exp(-0.5 * diff_sq / sigma_sq_exp)
                div_K = (diff_ij / sigma_sq_exp) * K_vals
                grads_j = tf.expand_dims(grads, 1)
                attract = K_vals * grads_j
            else:
                dist_sq_sum = tf.reduce_sum(diff_sq, axis=-1)
                sig_val = sigma_sq[:, 0]

                # FIX: Expand [B] to [B, 1, 1]
                sig_val_exp = sig_val[:, tf.newaxis, tf.newaxis]

                K_scalar = tf.exp(-0.5 * dist_sq_sum / sig_val_exp)
                K_scalar_exp = tf.expand_dims(K_scalar, -1)

                # FIX: Expand [B] to [B, 1, 1, 1] for the 4D division
                sig_val_exp_D = sig_val[:, tf.newaxis, tf.newaxis, tf.newaxis]
                div_K = (diff_ij / sig_val_exp_D) * K_scalar_exp

                grads_j = tf.expand_dims(grads, 1)
                attract = K_scalar_exp * grads_j

            total_force = tf.reduce_mean(attract + div_K, axis=2)

            flow = tf.matmul(total_force, D_matrix)
            curr_particles = curr_particles + dt * flow

        # 4. API FIX: Extract analytical statistics to match the required return tuple
        x_filt = tf.reduce_mean(curr_particles, axis=1)
        P_filt = tfp.stats.covariance(curr_particles, sample_axis=1)

        return curr_particles, x_filt, P_filt