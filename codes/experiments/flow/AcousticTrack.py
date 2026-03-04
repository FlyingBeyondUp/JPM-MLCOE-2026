import os
import time
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.linalg
from Filters.basic_filters import KalmanFilter, UnscentedKalmanFilter, ExtendedKalmanFilter, ParticleFilter
from models import LGSSM, NLSSM
from Filters.flow_filters import EDHFlow, LEDHFlow
from Filters.flow_filters.invertible_flow import InvertiblePFPF

tfd = tfp.distributions


def plot_ledh_trajectory_sample(x_true, x_est):
    """
    Plots a single sample of the true and estimated trajectories for PF-PF-UKF (LEDH).
    """
    plt.figure(figsize=(10, 10))

    # 1. Plot Sensors (5x5 grid from 0 to 40)
    coords = np.linspace(0, 40.0, 5)
    X_grid, Y_grid = np.meshgrid(coords, coords)
    plt.scatter(X_grid.ravel(), Y_grid.ravel(), c='blue', marker='o', s=100, label='Sensors', zorder=5)
    colors = ['red', 'green', 'cyan', 'purple']
    # 2. Plot the 4 targets
    for tgt in range(4):
        x_idx = tgt * 4
        y_idx = tgt * 4 + 1

        # Ground Truth
        plt.plot(x_true[:, x_idx], x_true[:, y_idx], '-.', color=colors[tgt], linewidth=2,
                 label=f'True Track {tgt}', zorder=1)
        plt.scatter(x_true[0, x_idx], x_true[0, y_idx], c='k', marker='x', zorder=2, s=200)

        # PF-PF-UKF (LEDH) Estimate
        plt.plot(x_est[:, x_idx], x_est[:, y_idx], '.', color=colors[tgt], linewidth=2,
                 label=f'Est Track {tgt}', alpha=0.9, zorder=3)

    plt.title('Acoustic Target Tracking - PF-PF-UKF (LEDH) Trajectory Sample')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-2, 42)
    plt.ylim(-2, 42)

    #os.makedirs("Figures", exist_ok=True)
    #plt.savefig("Figures/Fig_Trajectory_Sample_LEDH.pdf", bbox_inches='tight')
    plt.show()


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
    sensor_locs = tf.constant(np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1), dtype=tf.float32)  # [25, 2]

    def transition_fn(x, noise):
        is_ekf_col = (len(x.shape) == 2 and x.shape[-1] == 1)

        x_r = tf.reshape(x, [-1, num_targets, 4])
        pos = x_r[..., :2]
        vel = x_r[..., 2:]
        new_pos = pos + vel * dt

        x_next_det = tf.reshape(tf.concat([new_pos, vel], axis=-1), tf.shape(x))

        if is_ekf_col:
            noise = tf.reshape(noise, [-1, 1])
        return x_next_det + noise

    def observation_fn(x, noise):
        is_ekf_col = (len(x.shape) == 2 and x.shape[-1] == 1)

        x_r = tf.reshape(x, [-1, num_targets, 4])
        target_pos = x_r[..., :2]

        diff = tf.expand_dims(target_pos, -2) - tf.reshape(sensor_locs, [1, 1, 25, 2])
        dists = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-9)
        z_clean = tf.reduce_sum(Psi / (dists + d0), axis=-2)

        if is_ekf_col:
            z_clean = tf.reshape(z_clean, [-1, 1])
            noise = tf.reshape(noise, [-1, 1])
        else:
            batch_shape = tf.shape(x)[:-1]
            z_clean = tf.reshape(z_clean, tf.concat([batch_shape, [25]], axis=0))

        return z_clean + noise

    sigma_w = 0.1
    obs_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(obs_dim), scale_diag=tf.fill([obs_dim], sigma_w))

    if is_generative:
        block_4x4 = np.array([
            [1 / 3.0, 0.0, 0.5, 0.0],
            [0.0, 1 / 3.0, 0.0, 0.5],
            [0.5, 0.0, 1, 0.0],
            [0.0, 0.5, 0.0, 1]
        ], dtype=np.float32) / 20.0

        init_cov_diag = tf.tile([1., 1., 1., 1.], [4])
        init_noise_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(init_means),
            scale_diag=1e-4 * tf.sqrt(init_cov_diag))
    else:
        block_4x4 = np.array([
            [3.0, 0.0, 0.1, 0.0],
            [0.0, 3.0, 0.0, 0.1],
            [0.1, 0.0, 0.03, 0.0],
            [0.0, 0.1, 0.0, 0.03]
        ], dtype=np.float32)

        init_cov_diag = tf.tile([100., 100., 1., 1.], [4])
        init_noise_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(init_means),
            scale_diag=tf.sqrt(init_cov_diag))

    full_cov = scipy.linalg.block_diag(block_4x4, block_4x4, block_4x4, block_4x4)
    cov_tril = tf.linalg.cholesky(tf.constant(full_cov))
    proc_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(state_dim), scale_tril=cov_tril)

    return NLSSM(state_dim, obs_dim, transition_fn, observation_fn, proc_dist, obs_dist, init_noise_dist, init_means)


def get_exponential_schedule(num_steps: int = 29, q: float = 1.2):
    if num_steps == 1:
        return tf.constant([1.0], dtype=tf.float32)

    numerator = q - 1.0
    denominator = tf.pow(q, float(num_steps)) - 1.0
    epsilon_1 = numerator / denominator
    exponents = tf.range(num_steps, dtype=tf.float32)
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

            for i in range(4):
                if next_x[i * 4 + 0] < lower_bound:
                    next_x[i * 4 + 0] = 2 * lower_bound - next_x[i * 4 + 0]
                    next_x[i * 4 + 2] = -next_x[i * 4 + 2]
                elif next_x[i * 4 + 0] > upper_bound:
                    next_x[i * 4 + 0] = 2 * upper_bound - next_x[i * 4 + 0]
                    next_x[i * 4 + 2] = -next_x[i * 4 + 2]

                if next_x[i * 4 + 1] < lower_bound:
                    next_x[i * 4 + 1] = 2 * lower_bound - next_x[i * 4 + 1]
                    next_x[i * 4 + 3] = -next_x[i * 4 + 3]
                elif next_x[i * 4 + 1] > upper_bound:
                    next_x[i * 4 + 1] = 2 * upper_bound - next_x[i * 4 + 1]
                    next_x[i * 4 + 3] = -next_x[i * 4 + 3]

                if next_x[i * 4 + 2] > max_speed:
                    next_x[i * 4 + 2] = 2 * max_speed - next_x[i * 4 + 2]
                elif next_x[i * 4 + 2] < -max_speed:
                    next_x[i * 4 + 2] = -2 * max_speed - next_x[i * 4 + 2]

                if next_x[i * 4 + 3] > max_speed:
                    next_x[i * 4 + 3] = 2 * max_speed - next_x[i * 4 + 3]
                elif next_x[i * 4 + 3] < -max_speed:
                    next_x[i * 4 + 3] = -2 * max_speed - next_x[i * 4 + 3]

            x_seq.append(next_x)

        x_seq_tensor = tf.constant(x_seq[1:], dtype=tf.float32)

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
    ], dtype=tf.float32)

    # 1. Generate True Data
    gen_model = get_acoustic_model_experiment_A(true_init_means, is_generative=True)
    print("Simulating true trajectories (with boundary bouncing physics)...")
    X_true, Y_obs = generate_bouncing_tracks(T, batch_size, gen_model)
    print('Dataset generation complete.')

    # 2. Setup Filter Models
    filter_x0_var = tf.Variable(true_init_means, dtype=tf.float32)
    filter_model = get_acoustic_model_experiment_A(filter_x0_var, is_generative=False)
    step_sizes = get_exponential_schedule(29)

    ukf = UnscentedKalmanFilter(model=filter_model, alpha=3e-2, beta=2.0, kappa=0.0)
    ekf = ExtendedKalmanFilter(model=filter_model)

    # API FIX: Pass step_sizes and num_flow_steps into constructors
    pf_pf_ledh_ukf = InvertiblePFPF(
        model=filter_model, num_particles=num_particles_pf, ukf=ukf,
        flow_class=LEDHFlow, num_flow_steps=29, step_sizes=step_sizes,
        resample_method='systematic', resample_threshold=0.5
    )
    pf_pf_edh_ukf = InvertiblePFPF(
        model=filter_model, num_particles=num_particles_pf, ukf=ukf,
        flow_class=EDHFlow, num_flow_steps=29, step_sizes=step_sizes,
        resample_method='systematic', resample_threshold=0.5
    )

    ledh = LEDHFlow(model=filter_model, num_particles=num_particles_pf, ukf=ukf,
                    num_flow_steps=29, step_sizes=step_sizes, resample_from_ukf=True)
    edh = EDHFlow(model=filter_model, num_particles=num_particles_pf, ukf=ukf,
                  num_flow_steps=29, step_sizes=step_sizes, resample_from_ukf=True)

    bpf = ParticleFilter(model=filter_model, num_particles=num_particles_bpf, resample_method='systematic')

    def sample_valid_x0(base_means):
        std_dev = tf.sqrt(tf.tile([100., 100., 1., 1.], [4]))
        while True:
            hat_x0 = base_means + tf.random.normal([16]) * std_dev
            pos_x = hat_x0[0::4]
            pos_y = hat_x0[1::4]
            if tf.reduce_all(pos_x >= 0) and tf.reduce_all(pos_x <= 40) and \
                    tf.reduce_all(pos_y >= 0) and tf.reduce_all(pos_y <= 40):
                return hat_x0

    filters = ['PF-PF-UKF (LEDH)', 'PF-PF-UKF (EDH)',  'LEDH', 'EDH', 'UKF',
               'EKF', 'BPF']
    omat_res = {k: np.zeros((batch_size, iter_per_sample, T)) for k in filters}
    ess_res = {k: np.zeros((batch_size, iter_per_sample, T)) for k in
               ['PF-PF-UKF (LEDH)', 'PF-PF-UKF (EDH)',  'BPF']}
    time_res = {k: 0.0 for k in filters}

    sample_x_true = None
    sample_x_est_ledh = None
    print("Beginning filter evaluations (this may take some time)...")
    for b in range(batch_size):
        x_true_b = X_true[b:b + 1]  # [1, T, 16]
        y_obs_b = Y_obs[b:b + 1]  # [1, T, 25]

        for i in range(iter_per_sample):
            hat_x0 = sample_valid_x0(true_init_means)
            filter_x0_var.assign(hat_x0)

            # API FIX: Unpacking standard dictionary returns natively across all filters

            # EKF
            t0 = time.time()
            if hasattr(ekf, 'filter'):
                # Handle EKF cleanly if it aligns with BaseFilter framework dict returns
                res_ekf = ekf.filter(y_obs_b)
                x_filt_ekf = res_ekf['x_filt'] if isinstance(res_ekf, dict) else res_ekf[0]
            time_res['EKF'] += (time.time() - t0)
            omat_res['EKF'][b, i, :] = batch_compute_omat(x_true_b, x_filt_ekf)[0]

            # UKF
            t0 = time.time()
            res_ukf = ukf.filter(y_obs_b)
            x_filt_ukf = res_ukf['x_filt']
            time_res['UKF'] += (time.time() - t0)
            omat_res['UKF'][b, i, :] = batch_compute_omat(x_true_b, x_filt_ukf)[0]

            # EDH
            t0 = time.time()
            res_edh = edh.filter(y_obs_b)
            x_filt_edh = res_edh['x_filt']
            time_res['EDH'] += (time.time() - t0)
            omat_res['EDH'][b, i, :] = batch_compute_omat(x_true_b, x_filt_edh)[0]

            # LEDH
            t0 = time.time()
            res_ledh = ledh.filter(y_obs_b)
            x_filt_ledh = res_ledh['x_filt']
            time_res['LEDH'] += (time.time() - t0)
            omat_res['LEDH'][b, i, :] = batch_compute_omat(x_true_b, x_filt_ledh)[0]

            # PF-PF (EDH) - UKF
            t0 = time.time()
            res_pfpf_edh = pf_pf_edh_ukf.filter(y_obs_b)
            x_filt_pfpf_edh = res_pfpf_edh['x_filt']
            time_res['PF-PF-UKF (EDH)'] += (time.time() - t0)
            omat_res['PF-PF-UKF (EDH)'][b, i, :] = batch_compute_omat(x_true_b, x_filt_pfpf_edh)[0]
            ess_res['PF-PF-UKF (EDH)'][b, i, :] = res_pfpf_edh['ess'][0]

            # PF-PF (LEDH) - UKF
            t0 = time.time()
            res_pfpf_ledh = pf_pf_ledh_ukf.filter(y_obs_b)
            x_filt_pfpf_ledh = res_pfpf_ledh['x_filt']
            time_res['PF-PF-UKF (LEDH)'] += (time.time() - t0)
            omat_res['PF-PF-UKF (LEDH)'][b, i, :] = batch_compute_omat(x_true_b, x_filt_pfpf_ledh)[0]
            ess_res['PF-PF-UKF (LEDH)'][b, i, :] = res_pfpf_ledh['ess'][0]

            # BPF
            t0 = time.time()
            res_bpf = bpf.filter(y_obs_b)
            x_filt_bpf = res_bpf['x_filt']
            time_res['BPF'] += (time.time() - t0)
            omat_res['BPF'][b, i, :] = batch_compute_omat(x_true_b, x_filt_bpf)[0]
            ess_res['BPF'][b, i, :] = res_bpf['ess'][0]

            if b == 0 and i == 0:
                sample_x_true = x_true_b[0].numpy()
                sample_x_est_ledh = x_filt_pfpf_ledh[0].numpy()

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

    if sample_x_true is not None and sample_x_est_ledh is not None:
        plot_ledh_trajectory_sample(sample_x_true, sample_x_est_ledh)

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
    #plt.savefig("Figures/Fig2_OMAT.pdf", bbox_inches='tight')
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
    #plt.savefig("Figures/Fig4_ESS.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    tf.config.experimental.enable_tensor_float_32_execution(False)

    run_experiment_A_replication(T=40, batch_size=10, num_particles_pf=500, num_particles_bpf=100000,
                                 iter_per_sample=5)