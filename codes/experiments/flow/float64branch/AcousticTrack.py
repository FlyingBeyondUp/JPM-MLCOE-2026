import os
import time
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.linalg

from codes.Filters.basic_filters import KalmanFilter, UnscentedKalmanFilter, ExtendedKalmanFilter, ParticleFilter
from models import LGSSM, NLSSM
from codes.Filters.flow_filters import EDHFlow, InvertiblePFPF, LEDHFlow

tfd = tfp.distributions
dtype=tf.float32


def batch_compute_omat(true_state, est_state, num_targets=4):
    """
    Inputs: [Batch, T, Dim]
    """
    B, T, D = true_state.shape
    # Reshape all at once: [B, T, Targets, 4] -> take pos [:, :, :, :2]
    t_pos = tf.reshape(true_state, [B, T, num_targets, 4])[:, :, :, :2]
    e_pos = tf.reshape(est_state, [B, T, num_targets, 4])[:, :, :, :2]

    # Euclidean Cost Matrix: [B, T, Targets, Targets]
    diff = tf.expand_dims(t_pos, -2) - tf.expand_dims(e_pos, -3)
    cost_matrix = tf.norm(diff, axis=-1).numpy()  # Move to CPU

    omat_batch = np.zeros((B, T))

    # Loop over batch/time for assignment (CPU bound, but unavoidable for Hungarian Algo)
    for b in range(B):
        for t in range(T):
            row_ind, col_ind = linear_sum_assignment(cost_matrix[b, t])
            omat_batch[b, t] = cost_matrix[b, t, row_ind, col_ind].sum() / num_targets

    return omat_batch  # [Batch, T]


def get_acoustic_model_experiment_A(init_means, is_generative=True):
    """
    Constructs the Multi-Target Acoustic Tracking Model described in Experiment A.
    """
    num_targets = 4
    state_dim = num_targets * 4
    obs_dim = 25
    dt = 1.0
    Psi = 10.0
    d0 = 0.1
    grid_size = 40.0

    # Sensor grid (5x5)
    coords = np.linspace(0, grid_size, 5)
    X_grid, Y_grid = np.meshgrid(coords, coords)
    sensor_locs = tf.constant(np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1), dtype=dtype)  # [25, 2]

    def transition_fn(x, noise):
        # Identify if x is an EKF column vector [16, 1]
        is_ekf_col = (len(x.shape) == 2 and x.shape[-1] == 1)

        # Safely reshape to [-1, targets, features] grouping all batch/particle dims into -1
        x_r = tf.reshape(x, [-1, num_targets, 4])
        pos = x_r[..., :2]
        vel = x_r[..., 2:]
        new_pos = pos + vel * dt

        # Reshape back to the exact incoming shape of x
        x_next_det = tf.reshape(tf.concat([new_pos, vel], axis=-1), tf.shape(x))

        # Prevent EKF [16, 1] + [16] broadcasting bug
        if is_ekf_col:
            noise = tf.reshape(noise, [-1, 1])
        return x_next_det + noise

    def observation_fn(x, noise):
        # Identify if x is an EKF column vector [16, 1]
        is_ekf_col = (len(x.shape) == 2 and x.shape[-1] == 1)

        x_r = tf.reshape(x, [-1, num_targets, 4])
        target_pos = x_r[..., :2]

        diff = tf.expand_dims(target_pos, -2) - tf.reshape(sensor_locs, [1, 1, 25, 2])
        dists = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-9)
        z_clean = tf.reduce_sum(Psi / (dists + d0), axis=-2)

        if is_ekf_col:
            # Reconstruct to [25, 1] for EKF
            z_clean = tf.reshape(z_clean, [-1, 1])
            noise = tf.reshape(noise, [-1, 1])
        else:
            # Reconstruct to [Batch, ..., 25] for PF/UKF
            batch_shape = tf.shape(x)[:-1]
            z_clean = tf.reshape(z_clean, tf.concat([batch_shape, [25]], axis=0))

        return z_clean + noise

    sigma_w = tf.constant(0.1, dtype=dtype)
    obs_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(obs_dim,dtype=dtype), scale_diag=tf.fill([obs_dim], sigma_w))

    if is_generative:
        block_4x4 = np.array([
            [1 / 3.0, 0.0, 0.5, 0.0],
            [0.0, 1 / 3.0, 0.0, 0.5],
            [0.5, 0.0, 1, 0.0],
            [0.0, 0.5, 0.0, 1]
        ], dtype=np.float64) / 20.0

        init_cov_diag = tf.tile(tf.constant([1., 1., 1., 1.],dtype=dtype), [4])
        init_noise_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(init_means,dtype=dtype),
            scale_diag=1e-4 * tf.sqrt(init_cov_diag))
    else:
        block_4x4 = np.array([
            [3.0, 0.0, 0.1, 0.0],
            [0.0, 3.0, 0.0, 0.1],
            [0.1, 0.0, 0.03, 0.0],
            [0.0, 0.1, 0.0, 0.03]
        ], dtype=np.float64)

        init_cov_diag = tf.tile(tf.constant([100., 1., 1., 1.],dtype=dtype), [4])
        init_noise_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(init_means,dtype=dtype),
            scale_diag=tf.sqrt(init_cov_diag))

    full_cov = scipy.linalg.block_diag(block_4x4, block_4x4, block_4x4, block_4x4)
    cov_tril = tf.linalg.cholesky(tf.constant(full_cov,dtype=dtype))
    proc_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(state_dim,dtype=dtype), scale_tril=cov_tril)

    return NLSSM(state_dim, obs_dim, transition_fn, observation_fn, proc_dist, obs_dist, init_noise_dist, init_means)


def get_exponential_schedule(num_steps: int = 29, q: float = 1.2):
    q = tf.constant(q, dtype=dtype)

    if num_steps == 1:
        return tf.constant([1.0], dtype=dtype)

    numerator = q - tf.constant(1.0, dtype=dtype)
    denominator = tf.pow(q, tf.cast(num_steps, dtype)) - tf.constant(1.0, dtype=dtype)
    epsilon_1 = numerator / denominator

    exponents = tf.range(num_steps, dtype=dtype)
    step_sizes = epsilon_1 * tf.pow(q, exponents)
    return step_sizes


def generate_bouncing_tracks(T, batch_size, gen_model, lower_bound=2.0, upper_bound=38.0, max_speed=1.0):
    X_valid = []
    Y_valid = []

    while len(X_valid) < batch_size:
        x_seq = [gen_model.x0.numpy()]

        for t in range(1, T + 1):
            prev_x = tf.constant([x_seq[-1]])
            noise = gen_model.process_noise.sample(1)
            next_x = gen_model.transition_fn(prev_x, noise)[0].numpy()

            # Apply Bouncing Logic for all 4 targets
            for i in range(4):
                # X position (index i*4 + 0) and X velocity (index i*4 + 2)
                if next_x[i*4 + 0] < lower_bound:
                    next_x[i*4 + 0] = 2 * lower_bound - next_x[i*4 + 0]
                    next_x[i*4 + 2] = -next_x[i*4 + 2]
                elif next_x[i*4 + 0] > upper_bound:
                    next_x[i*4 + 0] = 2 * upper_bound - next_x[i*4 + 0]
                    next_x[i*4 + 2] = -next_x[i*4 + 2]

                # Y position (index i*4 + 1) and Y velocity (index i*4 + 3)
                if next_x[i*4 + 1] < lower_bound:
                    next_x[i*4 + 1] = 2 * lower_bound - next_x[i*4 + 1]
                    next_x[i*4 + 3] = -next_x[i*4 + 3]
                elif next_x[i*4 + 1] > upper_bound:
                    next_x[i*4 + 1] = 2 * upper_bound - next_x[i*4 + 1]
                    next_x[i*4 + 3] = -next_x[i*4 + 3]

                # Speed limit X
                if next_x[i*4 + 2] > max_speed:
                    next_x[i*4 + 2] = 2 * max_speed - next_x[i*4 + 2]
                elif next_x[i*4 + 2] < -max_speed:
                    next_x[i*4 + 2] = -2 * max_speed - next_x[i*4 + 2]

                # Speed limit Y
                if next_x[i*4 + 3] > max_speed:
                    next_x[i*4 + 3] = 2 * max_speed - next_x[i*4 + 3]
                elif next_x[i*4 + 3] < -max_speed:
                    next_x[i*4 + 3] = -2 * max_speed - next_x[i*4 + 3]

            x_seq.append(next_x)

        # Convert to tensor, dropping the initial x0 state so it is shape [T, 16]
        x_seq_tensor = tf.constant(x_seq[1:], dtype=dtype)

        # Final Safety Rejection Check (Matches MATLAB exactly)
        pos_x = x_seq_tensor[:, 0::4]
        pos_y = x_seq_tensor[:, 1::4]
        vel_x = x_seq_tensor[:, 2::4]
        vel_y = x_seq_tensor[:, 3::4]

        if tf.reduce_any(pos_x < lower_bound) or tf.reduce_any(pos_x > upper_bound) or \
           tf.reduce_any(pos_y < lower_bound) or tf.reduce_any(pos_y > upper_bound):
            continue
        if tf.reduce_any(tf.abs(vel_x) > max_speed) or tf.reduce_any(tf.abs(vel_y) > max_speed):
            continue

        X_valid.append(x_seq_tensor)

        # Generate observations for the valid track
        noise_obs = gen_model.observation_noise.sample(T)
        y_seq_track = gen_model.observation_fn(x_seq_tensor, noise_obs)
        Y_valid.append(y_seq_track)

    return tf.stack(X_valid), tf.stack(Y_valid)

def run_experiment_A_replication(T=40, batch_size=10, num_particles_pf=500, num_particles_bpf=100000,
                                 iter_per_sample=5):
    """
    Replicates Experiment A: Multi-Target Acoustic Tracking.
    """
    print(f"Setting up Experiment A: {batch_size} trajectories, {iter_per_sample} runs each...")
    os.makedirs("Figures", exist_ok=True)

    true_init_means = tf.constant([
        12.0, 6.0, 0.001, 0.001,
        32.0, 32.0, -0.001, -0.005,
        20.0, 13.0, -0.1, 0.01,
        15.0, 35.0, 0.002, 0.002
    ], dtype=dtype)

    # 1. Generate True Data
    gen_model = get_acoustic_model_experiment_A(true_init_means, is_generative=True)
    print("Simulating true trajectories (with boundary rejection sampling)...")
    print("Simulating true trajectories (with boundary bouncing physics)...")
    X_true, Y_obs = generate_bouncing_tracks(T, batch_size, gen_model)
    print('Dataset generation complete.')

    print('Dataset generation complete.')

    # 2. Setup Filter Models
    # Using a tf.Variable for x0 allows us to update the initial state per run without retracing the tf.function
    filter_x0_var = tf.Variable(true_init_means, dtype=dtype)
    filter_model = get_acoustic_model_experiment_A(filter_x0_var, is_generative=False)

    ukf = UnscentedKalmanFilter(model=filter_model, alpha=5e-2, beta=2.0, kappa=0.0)
    ekf = ExtendedKalmanFilter(model=filter_model)

    pf_pf_ledh = InvertiblePFPF(model=filter_model, num_particles=num_particles_pf, ukf=ukf, flow_class=LEDHFlow,
                                resample_method='systematic',resample_threshold=0.5)
    pf_pf_edh = InvertiblePFPF(model=filter_model, num_particles=num_particles_pf, ukf=ukf, flow_class=EDHFlow,
                               resample_method='systematic',resample_threshold=0.5)
    ledh = LEDHFlow(model=filter_model, num_particles=num_particles_pf, ukf=ukf)
    edh = EDHFlow(model=filter_model, num_particles=num_particles_pf, ukf=ukf)
    bpf = ParticleFilter(model=filter_model, num_particles=num_particles_bpf, resample_method='systematic')

    step_sizes = get_exponential_schedule(29)

    def sample_valid_x0(base_means):
        # 10 for pos, 1 for vel -> Variance = 100 and 1
        std_dev = tf.sqrt(tf.tile(tf.constant([100., 100., 1., 1.], dtype=dtype), [4]))

        while True:
            eps = tf.random.normal([16], dtype=dtype)
            hat_x0 = base_means + eps * std_dev

            pos_x = hat_x0[0::4]
            pos_y = hat_x0[1::4]
            # Ensure within 40x40 tracking grid
            if tf.reduce_all(pos_x >= 0) and tf.reduce_all(pos_x <= 40) and \
                    tf.reduce_all(pos_y >= 0) and tf.reduce_all(pos_y <= 40):
                return hat_x0

    # Dictionaries to track results
    filters = ['PF-PF (LEDH)', 'PF-PF (EDH)', 'LEDH', 'EDH', 'UKF', 'EKF', 'BPF']
    omat_res = {k: np.zeros((batch_size, iter_per_sample, T)) for k in filters}
    ess_res = {k: np.zeros((batch_size, iter_per_sample, T)) for k in ['PF-PF (LEDH)', 'PF-PF (EDH)', 'BPF']}
    time_res = {k: 0.0 for k in filters}

    print("Beginning filter evaluations (this may take some time)...")
    for b in range(batch_size):
        x_true_b = X_true[b:b + 1]  # [1, T, 16]
        y_obs_b = Y_obs[b:b + 1]  # [1, T, 25]

        for i in range(iter_per_sample):
            # Sample starting location per the paper's specs and assign it to the model
            hat_x0 = sample_valid_x0(true_init_means)
            filter_x0_var.assign(hat_x0)

            # EKF
            t0 = time.time()
            # Expand dims so y is [T, 25, 1], preventing broadcasting during y[t] - h_val
            y_obs_ekf = tf.expand_dims(y_obs_b[0], axis=-1)
            ekf_out = ekf.filter(y_obs_ekf, T, requires_stabilization=True)
            time_res['EKF'] += (time.time() - t0)
            omat_res['EKF'][b, i, :] = batch_compute_omat(x_true_b, tf.expand_dims(ekf_out[0], 0))[0]

            # UKF
            t0 = time.time()
            x_filt_ukf, _, _ = ukf.filter(y_obs_b)
            time_res['UKF'] += (time.time() - t0)
            omat_res['UKF'][b, i, :] = batch_compute_omat(x_true_b, x_filt_ukf)[0]

            # EDH (using redraw strategy via resample=True)
            t0 = time.time()
            x_filt_edh, _, _ = edh.filter_with_ukf(y_obs_b, num_flow_steps=29, step_sizes=step_sizes, resample=True)
            time_res['EDH'] += (time.time() - t0)
            omat_res['EDH'][b, i, :] = batch_compute_omat(x_true_b, x_filt_edh)[0]

            # LEDH (using redraw strategy via resample=True)
            t0 = time.time()
            x_filt_ledh, _, _ = ledh.filter_with_ukf(y_obs_b, num_flow_steps=29, step_sizes=step_sizes, resample=True)
            time_res['LEDH'] += (time.time() - t0)
            omat_res['LEDH'][b, i, :] = batch_compute_omat(x_true_b, x_filt_ledh)[0]

            # PF-PF (EDH)
            t0 = time.time()
            x_filt_pfpf_edh, _, _, weights_edh = pf_pf_edh.filter(y_obs_b, num_flow_steps=29, step_sizes=step_sizes)
            time_res['PF-PF (EDH)'] += (time.time() - t0)
            omat_res['PF-PF (EDH)'][b, i, :] = batch_compute_omat(x_true_b, x_filt_pfpf_edh)[0]
            ess_res['PF-PF (EDH)'][b, i, :] = (1.0 / tf.reduce_sum(tf.square(weights_edh), axis=-1))[0]

            # PF-PF (LEDH)
            t0 = time.time()
            x_filt_pfpf_ledh, _, _, weights_ledh = pf_pf_ledh.filter(y_obs_b, num_flow_steps=29, step_sizes=step_sizes)
            time_res['PF-PF (LEDH)'] += (time.time() - t0)
            omat_res['PF-PF (LEDH)'][b, i, :] = batch_compute_omat(x_true_b, x_filt_pfpf_ledh)[0]
            ess_res['PF-PF (LEDH)'][b, i, :] = (1.0 / tf.reduce_sum(tf.square(weights_ledh), axis=-1))[0]

            # BPF
            t0 = time.time()
            x_filt_bpf, _, ess_bpf, _ = bpf.filter_summarized(y_obs_b)
            time_res['BPF'] += (time.time() - t0)
            omat_res['BPF'][b, i, :] = batch_compute_omat(x_true_b, x_filt_bpf)[0]
            ess_res['BPF'][b, i, :] = ess_bpf[0]

        print(f"Trajectory {b + 1}/{batch_size} completed.")

    total_runs = batch_size * iter_per_sample
    print("\n" + "=" * 60)
    print("Table I Metrics (Overall Averages)")
    print("=" * 60)
    for k in filters:
        avg_omat = np.mean(omat_res[k])
        avg_ess = f"{np.mean(ess_res[k]):.1f}" if k in ess_res else "N/A"
        avg_time = time_res[k] / total_runs / T
        print(f"{k:15s} | OMAT: {avg_omat:.2f} m | ESS: {avg_ess:>4s} | Exec/step: {avg_time:.4f} s")

    # Plot Figure 2: Average OMAT over time
    plt.figure(figsize=(10, 6))
    for k in filters:
        mean_omat_over_time = np.mean(omat_res[k], axis=(0, 1))
        plt.plot(range(1, T + 1), mean_omat_over_time, label=k, marker='o', markersize=4)
    plt.xlabel('Time Step')
    plt.ylabel('Average OMAT Error (m)')
    plt.title('Average OMAT Error at Each Time Step (Figure 2 Replication)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Figures/Fig2_OMAT.pdf", bbox_inches='tight')
    plt.show()

    # Plot Figure 4: Average ESS over time
    plt.figure(figsize=(10, 6))
    for k in ess_res.keys():
        mean_ess_over_time = np.mean(ess_res[k], axis=(0, 1))
        plt.plot(range(1, T + 1), mean_ess_over_time, label=k, marker='^', markersize=6, linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Average ESS')
    plt.title('Average Effective Sample Size at Each Time Step (Figure 4 Replication)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Figures/Fig4_ESS.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    # Paper uses batch=100 trajectories, T=40.
    run_experiment_A_replication(T=40, batch_size=10, num_particles_pf=500, num_particles_bpf=100000,
                                 iter_per_sample=5)