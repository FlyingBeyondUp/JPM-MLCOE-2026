import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.linalg
from Filters.basic_filters import KalmanFilter,UnscentedKalmanFilter, ExtendedKalmanFilter, ParticleFilter
from models import LGSSM,NLSSM
from Filters.flow_filters import EDHFlow, InvertiblePFPF, LEDHFlow
import time

tfd = tfp.distributions



def run_comparison_experiment(T: int = 100, num_particles: int = 100,batch_size:int=1):
    print(f"Running comparison for T={T} steps with {num_particles} particles...")

    # a static model with very small process noise
    # to show the advantage of particle flow methods
    # that can overcome sample impoverishment problem of particle filter
    true_state_value = 5.0

    def transition_fn(x, noise):
        return x + noise

    def observation_fn(x, noise):
        return x + noise

    static_model = NLSSM(
        state_dim=1, obs_dim=1,
        x0=tf.ones((1,)) * true_state_value,  # 初始状态
        init_noise=tfd.Normal(loc=0.0, scale=1.0),  # 初始分布很宽
        process_noise=tfd.Normal(loc=0.0, scale=1e-6),  # 过程噪声极小 -> 静态模型
        observation_noise=tfd.Normal(loc=0.0, scale=1.0),
        transition_fn=transition_fn,
        observation_fn=observation_fn
    )


    #x_true, y_obs = static_model.sample(T)
    x_true, y_obs = static_model.batch_sample(T,batch_size)
    # ensure y_obs shape: [Batch, T, obs_dim]
    if len(y_obs.shape) == 1:
        y_obs = tf.reshape(y_obs, [1, -1, 1])
    elif len(y_obs.shape) == 2:
        y_obs = tf.expand_dims(y_obs, axis=0)

    print("Running Standard Particle Filter...")
    pf = ParticleFilter(model=static_model, num_particles=num_particles, resample_method='multinomial')
    t0 = time.time()
    x_filt_pf, _, particles_pf, _ = pf.filter(y_obs)
    t1 = time.time()
    print(f"Standard PF took {t1 - t0:.4f} seconds.")

    print("Running EDH Particle Flow...")
    edh = EDHFlow(model=static_model, num_particles=num_particles)
    t2=time.time()
    x_filt_edh, _, particles_edh = edh.filter(y_obs)
    t3=time.time()
    print(f"EDH Flow took {t3 - t2:.4f} seconds.")

    mse_pf = tf.reduce_mean(tf.square(x_true[:, -1, :] - x_filt_pf[:, -1, :]))
    mse_edh = tf.reduce_mean(tf.square(x_true[:, -1, :] - x_filt_edh[:, -1, :]))

    # particles shape: [Batch, T, NumParticles, StateDim]
    # Variance across particles (axis 2) -> [Batch, T, StateDim]
    var_pf_all = tf.math.reduce_variance(particles_pf, axis=2)
    var_edh_all = tf.math.reduce_variance(particles_edh, axis=2)

    # Take the last time step (T-1) and average over batch
    last_var_pf_avg = tf.reduce_mean(var_pf_all[:, -1, :])
    last_var_edh_avg = tf.reduce_mean(var_edh_all[:, -1, :])

    print(f"\n=== Batch Comparison (Batch Size: {batch_size}) ===")
    print(f"Standard PF | MSE: {mse_pf:.5f} | Avg Terminal Var: {last_var_pf_avg:.5f}")
    print(f"EDH Flow    | MSE: {mse_edh:.5f} | Avg Terminal Var: {last_var_edh_avg:.5f}")
    print("===================================================\n")

    p_pf = particles_pf.numpy()[0, :, :, 0]
    p_edh = particles_edh.numpy()[0, :, :, 0]

    var_pf = np.var(p_pf, axis=1)
    var_edh = np.var(p_edh, axis=1)

    plt.figure(figsize=(14, 10))

    # compare trajectories and variances
    plt.subplot(2, 2, 1)
    plt.title("Standard PF: Particle Collapse")
    for t in range(T):
        plt.scatter(np.full(num_particles, t), p_pf[t], s=1, c='r', alpha=0.3)
    plt.plot(x_filt_pf[0, :, 0], 'k-', label='PF Estimate')
    plt.plot(x_true.numpy()[0,:,0], 'g--', label='True State')
    plt.ylabel('State Value')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("EDH Flow: Maintaining Diversity")
    for t in range(T):
        plt.scatter(np.full(num_particles, t), p_edh[t], s=1, c='b', alpha=0.3)
    plt.plot(x_filt_edh[0, :, 0], 'k-', label='EDH Estimate')
    plt.plot(x_true.numpy()[0,:,0], 'g--', label='True State')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(var_pf, 'r-o', label='Standard PF Variance')
    plt.plot(var_edh, 'b-o', label='EDH Flow Variance')
    plt.title('Particle Variance over Time (Log Scale)')
    plt.xlabel('Time Step')
    plt.ylabel('Variance')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig("Figures\comparison_flow_filters_sample_impov.pdf",bbox_inches='tight')
    plt.show()


def get_exponential_schedule(num_steps: int = 29, q: float = 1.2):
    """
    Computes exponentially spaced step sizes (epsilon) that sum to 1.0.

    According to the paper:
    epsilon_j = epsilon_{j-1} * q
    Sum(epsilon) = 1

    This forms a geometric series where the first term a = epsilon_1.
    Sum = a * (1 - q^N) / (1 - q) = 1
    Therefore: epsilon_1 = (1 - q) / (1 - q^N)
               (or equivalently (q - 1) / (q^N - 1))
    """
    # Handle edge case for a single step
    if num_steps == 1:
        return tf.constant([1.0], dtype=tf.float32)

    # Calculate the initial step size (epsilon_1) using geometric series sum formula
    # epsilon_1 = (q - 1) / (q^num_steps - 1)
    numerator = q - 1.0
    denominator = tf.pow(q, float(num_steps)) - 1.0
    epsilon_1 = numerator / denominator

    # Generate the exponents [0, 1, 2, ..., N-1]
    exponents = tf.range(num_steps, dtype=tf.float32)

    # Calculate all step sizes: epsilon_j = epsilon_1 * q^(j-1)
    step_sizes = epsilon_1 * tf.pow(q, exponents)

    return step_sizes



def get_linear_spatial_model(d=64, sigma_z=1.0):
    # Experiment B
    grid_side = int(np.sqrt(d))
    coords = np.array([(i, j) for i in range(grid_side) for j in range(grid_side)])
    dists_sq = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    Sigma = 3.0 * np.exp(-dists_sq / 20.0) + 0.01 * np.eye(d)
    Sigma_L = tf.linalg.cholesky(tf.cast(Sigma, tf.float32))

    trans_fn = lambda x, n: 0.9 * x + n
    obs_fn = lambda x, n: x + n

    return NLSSM(d, d, trans_fn, obs_fn,
                 tfd.MultivariateNormalTriL(loc=tf.zeros(d), scale_tril=Sigma_L),
                 tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.fill([d], float(sigma_z))),
                 tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=1e-6*tf.ones(d)), tf.zeros(d))


def get_skewed_t_poisson_model(d=144):
    # Experiment C
    # Reuse covariance from B
    model_b = get_linear_spatial_model(d, 1.0)

    # GH Skewed-t approximation using Student T for process noise
    proc_noise = tfd.MultivariateStudentTLinearOperator(
        df=5.0, loc=tf.zeros(d), scale=tf.linalg.LinearOperatorLowerTriangular(model_b.process_noise.scale.to_dense()))

    # Poisson Observation
    def obs_fn(x, n): return tfd.Poisson(rate=1.0 * tf.math.exp((1.0 / 3.0) * x)).sample()

    def obs_dist_fn(x): return tfd.Poisson(rate=1.0 * tf.math.exp((1.0 / 3.0) * x))

    return NLSSM(d, d, lambda x, n: 0.9 * x + n, obs_fn, proc_noise, tfd.Normal(0., 0.),
                 model_b.init_noise, model_b.x0, get_observation_dist=obs_dist_fn)


def run_exp_linear_spatial_model(sigma_z=1.0):

    T = 10
    num_particles = 200
    batch_size = 100

    # 1. GENERATION MODEL: Low variance init_noise to ensure x_true starts at ~0
    # This matches the paper's "All true states start with x0 = 0"
    gen_model = get_linear_spatial_model(d=64, sigma_z=sigma_z)
    # Tighter noise for truth generation
    gen_model.init_noise = tfd.MultivariateNormalDiag(
        loc=tf.zeros(64), scale_diag=1e-5 * tf.ones(64)
    )

    # Sample True Data
    x_true, y_obs = gen_model.batch_sample(T, batch_size)
    def get_stationary_cov(model, alpha=0.9):
        # Q is the process noise covariance
        Q = model.process_noise.covariance()

        # Calculate Stationary Covariance: P = Q / (1 - alpha^2)
        P_stationary = Q / (1.0 - alpha ** 2)
        return P_stationary

    # Usage in your experiment:
    # ... generate models ...
    P0_optimal = get_stationary_cov(gen_model, alpha=0.9)
    P0_scale_tril = tf.linalg.cholesky(P0_optimal)

    # Initialize Filter model with the correct Stationary Covariance
    filter_model = get_linear_spatial_model(d=64, sigma_z=sigma_z)
    filter_model.init_noise = tfd.MultivariateNormalTriL(loc=tf.zeros(64),scale_tril=P0_scale_tril)

    # --- WARM UP (Crucial for correct timing) ---
    print("Warming up JIT compilation...")
    warmup_obs = y_obs[:, :2, :]  # Small slice
    step_sizes = get_exponential_schedule(29)
    num_flow_steps = 29
    # step_sizes=None
    # num_flow_steps=200

    # Initialize dummy filters just to trigger compilation
    _ukf = UnscentedKalmanFilter(model=filter_model, alpha=1, beta=2.0, kappa=0.0)
    _pf = ParticleFilter(model=filter_model, num_particles=num_particles)
    _edh = EDHFlow(model=filter_model, num_particles=num_particles, ukf=_ukf)
    _pfpf = InvertiblePFPF(model=filter_model, num_particles=num_particles, ukf=_ukf, flow_class=EDHFlow)

    # Run once to compile graph
    _pf.filter(warmup_obs)
    _ukf.filter(warmup_obs)
    _edh.filter_with_ukf(warmup_obs, num_flow_steps=num_flow_steps, step_sizes=step_sizes)
    _pfpf.filter(warmup_obs, num_flow_steps=num_flow_steps, step_sizes=step_sizes)
    print("Warm up complete.\n")

    # --- EXPERIMENT START ---
    # A=0.9*I, C=I, Q=Stationary, R=Sigma_z
    # d = 64
    # A_lin = tf.eye(d) * 0.9
    # C_lin = tf.eye(d)
    # # Extract Q from the covariance matrix of the process noise distribution
    # Q_lin = filter_model.process_noise.covariance()
    # R_lin = filter_model.observation_noise.covariance()
    # x0_lin = tf.zeros((d, 1))
    # P0_lin = P0_optimal
    #
    # lgssm_params = [A_lin, C_lin, Q_lin, R_lin, x0_lin, P0_lin]
    # lgssm_model = LGSSM(state_dim=d, obs_dim=d, params=lgssm_params)
    # kf_optimal = KalmanFilter(lgssm_model)
    #
    # print("Running Optimal Kalman Filter (Baseline)...")
    # t_kf_start = time.time()
    # # Now using the batched filter method directly
    # x_filt_kf, _, _, _, _ = kf_optimal.filter(y_obs)
    # t_kf_end = time.time()
    # print(f"Optimal KF took {(t_kf_end - t_kf_start) / batch_size:.4f} seconds.")
    #
    # print("Running Standard Particle Filter...")
    # # Use filter_model
    # pf = ParticleFilter(model=filter_model, num_particles=200, resample_method='systematic')
    # t_pf=time.time()
    # x_filt_pf, _, _, weights_pf = pf.filter(y_obs)
    # print(f"Standard PF took {(time.time() - t_pf) / batch_size:.4f} seconds.")
    #
    # pf_multinomial = ParticleFilter(model=filter_model, num_particles=200, resample_method='multinomial')
    # t_pf_multi=time.time()
    # x_filt_pf_multi, _, _, weights_pf_multi = pf_multinomial.filter(y_obs)
    # print(f"Standard PF (Multinomial) took {(time.time() - t_pf_multi) / batch_size:.4f} seconds.")
    #
    # print("Running Standard Particle Filter with 10^5 particles (in splits)...")
    # pf_large = ParticleFilter(model=filter_model, num_particles=100_000, resample_method='systematic')
    num_splits = 10
    B = y_obs.shape[0]
    split_size = int(np.ceil(B / num_splits))  # 100 // 10 = 10

    # x_filt_list = []
    # # Rename variables to match actual return types from filter_summarized
    # P_filt_list = []
    # ess_list = []
    #
    # time_cost = 0.0
    # for i in range(num_splits):
    #     start = i * split_size
    #     end = min((i + 1) * split_size, B)
    #     if start >= B: break  # Safety break
    #     y_obs_chunk = y_obs[start:end, :, :]
    #     t0 = time.time()
    #     # filter_summarized returns: (Means, Covariances, ESS)
    #     x_filt_chunk, P_filt_chunk, ess_chunk = pf_large.filter_summarized(y_obs_chunk)
    #     time_cost += time.time() - t0
    #
    #     x_filt_list.append(x_filt_chunk)
    #     P_filt_list.append(P_filt_chunk)
    #     ess_list.append(ess_chunk)
    #
    # x_filt_pf_large = tf.concat(x_filt_list, axis=0)
    # # Reassemble results if needed
    # weights_pf_large = tf.concat(ess_list, axis=0)
    # print(f"Standard PF took {time_cost / batch_size:.4f} seconds.")
    #
    # UKF
    ukf = UnscentedKalmanFilter(model=filter_model, alpha=1e-1, beta=2.0, kappa=0.0)
    t4 = time.time()
    x_filt_ukf, _, _ = ukf.filter(y_obs)
    t5 = time.time()
    print(f"UKF took {(t5 - t4) / batch_size:.4f} seconds.")
    #
    # # EDH raw
    # edh_raw= EDHFlow(model=filter_model, num_particles=num_particles)
    # t2 = time.time()
    # x_filt_edh_raw, _, _ = edh_raw.filter(y_obs, num_flow_steps=num_flow_steps, step_sizes=step_sizes)
    # t3 = time.time()
    # print(f"EDH raw Flow took {(t3 - t2) / batch_size:.4f} seconds.")

    # EDH with ukf
    print("Running EDH Particle Flow...")
    edh = EDHFlow(model=filter_model, num_particles=num_particles, ukf=ukf)
    t2 = time.time()
    x_filt_edh, _, _ = edh.filter_with_ukf(y_obs, num_flow_steps=num_flow_steps, step_sizes=step_sizes)
    t3 = time.time()
    print(f"EDH Flow took {(t3 - t2) / batch_size:.4f} seconds.")

    # PF-PF, EDH
    print("Running PF-PF (EDH) Particle Flow...")
    pfpf = InvertiblePFPF(model=filter_model, ukf=ukf, flow_class=EDHFlow, num_particles=num_particles,
                          resample_method='systematic',resample_threshold=0.5)
    t6 = time.time()
    x_filt_pfpf, _, _, weights_pfpf = pfpf.filter(y_obs, num_flow_steps=num_flow_steps, step_sizes=step_sizes)
    t7 = time.time()
    print(f"PF-PF took {(t7 - t6) / batch_size:.4f} seconds.")

    # PF-PF EDH with 10^4 particles
    print("Running PF-PF (EDH) Particle Flow with 10^4 particles...")
    pfpf_large = InvertiblePFPF(model=filter_model, ukf=ukf, flow_class=EDHFlow, num_particles=10_000,
                          resample_method='systematic',resample_threshold=0.5)
    t6 = time.time()
    x_filt_pfpf_large, _, _, weights_pfpf_large = pfpf_large.filter(y_obs, num_flow_steps=num_flow_steps, step_sizes=step_sizes)
    t7 = time.time()
    print(f"PF-PF (10^4) took {(t7 - t6) / batch_size:.4f} seconds.")

    # PF-PF, LEDH
    print("Running PF-PF (LEDH) Particle Flow...")
    pfpf_LEDH = InvertiblePFPF(model=filter_model, ukf=ukf, flow_class=LEDHFlow, num_particles=num_particles,
                          resample_method='systematic',resample_threshold=0.5)
    # t6 = time.time()
    # x_filt_pfpf_LEDH, _, _, weights_pfpf_LEDH = pfpf_LEDH.filter(y_obs, num_flow_steps=num_flow_steps, step_sizes=step_sizes)
    # t7 = time.time()
    # print(f"PF-PF took {(t7 - t6) / batch_size:.4f} seconds.")

    x_filt_list = []
    # Rename variables to match actual return types from filter_summarized
    P_filt_list = []
    ess_list = []
    time_cost = 0.0
    for i in range(num_splits):
        print(f'batch index:{i}')
        start = i * split_size
        end = min((i + 1) * split_size, B)
        if start >= B: break  # Safety break
        y_obs_chunk = y_obs[start:end, :, :]
        t0 = time.time()
        # filter_summarized returns: (Means, Covariances, ESS)
        x_filt_chunk, P_filt_chunk, _,ess_chunk = pfpf_LEDH.filter(y_obs_chunk, num_flow_steps=num_flow_steps, step_sizes=step_sizes)
        time_cost += time.time() - t0

        x_filt_list.append(x_filt_chunk)
        P_filt_list.append(P_filt_chunk)
        ess_list.append(ess_chunk)

    x_filt_pfpf_LEDH = tf.concat(x_filt_list, axis=0)
    # Reassemble results if needed
    weights_pfpf_LEDH = tf.concat(ess_list, axis=0)
    print(f"PF-PF LEDH costs {time_cost / batch_size:.4f} seconds.")

    # Calculate MSE
    # mse_kf = tf.reduce_mean(tf.square(x_true - x_filt_kf))
    # mse_pf=tf.reduce_mean(tf.square(x_true - x_filt_pf))
    # mse_pf_large = tf.reduce_mean(tf.square(x_true - x_filt_pf_large))
    mse_ukf = tf.reduce_mean(tf.square(x_true - x_filt_ukf))
    mse_edh = tf.reduce_mean(tf.square(x_true - x_filt_edh))
    #mse_edh_raw = tf.reduce_mean(tf.square(x_true - x_filt_edh_raw))
    mse_pfpf_LEDH = tf.reduce_mean(tf.square(x_true - x_filt_pfpf_LEDH))
    mse_pfpf_EDH = tf.reduce_mean(tf.square(x_true - x_filt_pfpf))
    mse_pfpf_EDH_large = tf.reduce_mean(tf.square(x_true - x_filt_pfpf_large))

     # Print Results

    print(f"\n=== Linear Spatial Model Comparison ===")
    print(f"PF-PF (LEDH)      | MSE: {mse_pfpf_LEDH:.5f}")
    print(f"PF-PF  (EDH)     | MSE: {mse_pfpf_EDH:.5f}")
    print(f"PF-PF  (EDH 10^4)     | MSE: {mse_pfpf_EDH_large:.5f}")
    print(f"EDH Flow    | MSE: {mse_edh:.5f}")
    # print(f'EDH raw    | MSE: {mse_edh_raw:.5f}')
    # print(f"Optimal KF  | MSE: {mse_kf:.5f}")
    print(f"Standard UKF | MSE: {mse_ukf:.5f}")
    # print(f'Standard PF (Multinomial) | MSE: {tf.reduce_mean(tf.square(x_true - x_filt_pf_multi)):.5f}')
    # print(f"Standard PF | MSE: {mse_pf:.5f}")
    # print(f"Standard PF (10^5) | MSE: {mse_pf_large:.5f}")
    print("========================================\n")

    # effective sample size
    # ess_pf = 1.0 / tf.reduce_sum(tf.square(weights_pf), axis=-1)
    # ess_pf_multi = 1.0 / tf.reduce_sum(tf.square(weights_pf_multi), axis=-1)
    # mean_ess_pf_large = tf.reduce_mean(weights_pf_large)
    ess_pfpf_LEDH = 1.0 / tf.reduce_sum(tf.square(weights_pfpf_LEDH), axis=-1)
    ess_pfpf_EDH = 1.0 / tf.reduce_sum(tf.square(weights_pfpf), axis=-1)
    ess_pfpf_EDH_large = 1.0 / tf.reduce_sum(tf.square(weights_pfpf_large), axis=-1)

    # mean_ess_pf = tf.reduce_mean(ess_pf)
    # mean_ess_pf_multi = tf.reduce_mean(ess_pf_multi)
    mean_ess_pfpf_LEDH = tf.reduce_mean(ess_pfpf_LEDH)
    mean_ess_pfpf_EDH = tf.reduce_mean(ess_pfpf_EDH)
    mean_ess_pfpf_EDH_large = tf.reduce_mean(ess_pfpf_EDH_large)

    # print(f"Mean ESS Standard PF (Multinomial): {mean_ess_pf_multi:.2f}")
    # print(f"Mean ESS Standard PF: {mean_ess_pf:.2f}")
    # print(f"Mean ESS Standard PF (10^5): {mean_ess_pf_large:.2f}")
    print(f"Mean ESS PF-PF (LEDH): {mean_ess_pfpf_LEDH:.2f}")
    print(f"Mean ESS PF-PF (EDH): {mean_ess_pfpf_EDH:.2f}")
    print(f"Mean ESS PF-PF (EDH 10^4): {mean_ess_pfpf_EDH_large:.2f}")




if __name__ == "__main__":
    #run_comparison_experiment(T=100,batch_size=100)
    run_exp_linear_spatial_model(sigma_z=0.5)
    # run_exp_linear_spatial_model(sigma_z=1.0)
    # run_exp_linear_spatial_model(sigma_z=2.0)

    # Paper uses batch=100 (runs), T=40.
    #run_experiment_A_replication(T=40, batch_size=10, num_particles_pf=500, num_particles_bpf=10_0000)
