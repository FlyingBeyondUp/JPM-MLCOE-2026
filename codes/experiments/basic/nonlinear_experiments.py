from models import *
from Filters.basic_filters import KalmanFilter
from Filters.basic_filters import ExtendedKalmanFilter, UnscentedKalmanFilter
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import tensorflow_probability as tfp


def run_example_SVM(alpha, sigma, beta, T=500):
    model = get1DStochasticVolModel(alpha=alpha, sigma=sigma, beta=beta)
    model_st = get1DStochasticVolModel(alpha=alpha, sigma=sigma, beta=beta, heavy_tail=True)

    # Updated to unified sample API
    x, y = model.sample(batch_size=1, T=T)
    x_st, y_st = model_st.sample(batch_size=1, T=T)

    print('max range of y_t:', tf.reduce_max(y) - tf.reduce_min(y))
    print('max range of y_t (Heavy-Tailed):', tf.reduce_max(y_st) - tf.reduce_min(y_st))

    y_log_sq = tf.math.log(y ** 2 + 1e-8)

    # --- EKF ---
    ekf = ExtendedKalmanFilter(get1DLogSquaredSVM(alpha=alpha, sigma=sigma, beta=beta), requires_stabilization=True)
    res_ekf = ekf.filter(y_log_sq)
    x_filt_ekf = tf.squeeze(res_ekf['x_filt'], axis=[0, -1])

    # --- LGSSM ---
    lgssm_svm, bias = get_LogSVM_LGSSM(alpha=alpha, sigma=sigma, beta=beta)
    kf = KalmanFilter(lgssm_svm, requires_stabilization=True)
    y_log_sq_bias_corrected = y_log_sq - bias
    res_kf = kf.filter(tf.reshape(y_log_sq_bias_corrected, [1, -1, 1]))
    x_filt_kf = tf.squeeze(res_kf['x_filt'], axis=[0, -1])

    # --- UKF ---
    if len(y_log_sq.shape) == 1:
        y_log_sq = tf.reshape(y_log_sq, [1, -1, 1])  # Batch size 1

    ukf = UnscentedKalmanFilter(get1DLogSquaredSVM(alpha=alpha, sigma=sigma, beta=beta), alpha=1.0, beta=2.0, kappa=0.0)

    t0 = time.time()
    res_ukf = ukf.filter(y_log_sq)
    t1 = time.time()
    print(f'UKF time (tf.function): {t1 - t0:.4f} seconds')

    x_filt_ukf = tf.squeeze(res_ukf['x_filt'], axis=[0, -1])
    x_flat = tf.squeeze(x, axis=[0, -1])

    print('\n')
    print('EKF max error:', tf.reduce_max(tf.abs(x_flat - x_filt_ekf)).numpy())
    print('LGSSM max error:', tf.reduce_max(tf.abs(x_flat - x_filt_kf)).numpy())
    print('UKF max error:', tf.reduce_max(tf.abs(x_flat - x_filt_ukf)).numpy())
    print('EKF mean error:', tf.reduce_mean(tf.abs(x_flat - x_filt_ekf)).numpy())
    print('LGSSM mean error:', tf.reduce_mean(tf.abs(x_flat - x_filt_kf)).numpy())
    print('UKF mean error:', tf.reduce_mean(tf.abs(x_flat - x_filt_ukf)).numpy())

    plt.figure(figsize=(12, 6))
    plt.plot(x_flat.numpy(), label='Hidden States x_t', c='black', linestyle='--')
    plt.plot(x_filt_ekf.numpy(), label='Hidden States from Extended Kalman Filter', c='green')
    plt.plot(x_filt_kf.numpy(), label='Hidden States from LGSSM Approximation', c='blue')
    plt.plot(x_filt_ukf.numpy(), label='Hidden States from Unscented Kalman Filter', c='orange')
    plt.scatter(range(T), tf.squeeze(y).numpy(), c='red', s=10, label='Observations y_t')
    plt.title('Stochastic Volatility Model - Hidden States Estimation')
    plt.xlabel('Time Step')
    plt.ylabel('x_t')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(tf.squeeze(x_st).numpy(), label='Hidden State x_t (Heavy-Tailed)')
    plt.scatter(range(T), tf.squeeze(y_st).numpy(), c='red', s=10, label='Observations y_t (Heavy-Tailed)')
    plt.title('Stochastic Volatility Model with Heavy-Tailed Process Noise - Hidden States')
    plt.xlabel('Time Step')
    plt.ylabel('x_t')
    plt.legend()
    plt.show()


def run_experiment_Vasicek(kappa, theta, sigma, tau, dt, T=200, x0=tf.zeros((1,))):
    model_vasicek = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt, x0=x0)
    x, y = model_vasicek.sample(batch_size=1, T=T)
    print('max range of y_t:', tf.reduce_max(y) - tf.reduce_min(y))
    print('y_t shape:', y.shape)

    # --- EKF ---
    ekf = ExtendedKalmanFilter(model_vasicek, requires_stabilization=True)
    res_ekf = ekf.filter(y)
    x_filt_ekf = tf.squeeze(res_ekf['x_filt'], axis=[0, -1])

    # --- LGSSM ---
    lgssm_vasicek, bias = getVasicekLGSSM(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt, x0=x0)
    kf = KalmanFilter(lgssm_vasicek, requires_stabilization=True)
    y_bias_corrected = y - bias
    res_kf = kf.filter(y_bias_corrected)
    x_filt_kf = tf.squeeze(res_kf['x_filt'], axis=[0, -1])

    # --- UKF ---
    ukf = UnscentedKalmanFilter(model_vasicek, alpha=1.0, beta=0.0, kappa=0.0)
    res_ukf = ukf.filter(y)
    x_filt_ukf = tf.squeeze(res_ukf['x_filt'], axis=[0, -1])

    x_flat = tf.squeeze(x, axis=[0, -1])

    print('\n')
    print('EKF max error:', tf.reduce_max(tf.abs(x_flat - x_filt_ekf)).numpy())
    print('UKF max error:', tf.reduce_max(tf.abs(x_flat - x_filt_ukf)).numpy())
    print('LGSSM max error:', tf.reduce_max(tf.abs(x_flat - x_filt_kf)).numpy())
    print('\n')
    print('EKF mean error:', tf.reduce_mean(tf.abs(x_flat - x_filt_ekf)).numpy())
    print('UKF mean error:', tf.reduce_mean(tf.abs(x_flat - x_filt_ukf)).numpy())
    print('LGSSM mean error:', tf.reduce_mean(tf.abs(x_flat - x_filt_kf)).numpy())

    # --- Plotting ---
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    ax1.plot(x_flat.numpy(), label='True Hidden States (Rate Deviation)', c='black', linestyle='--')
    ax1.plot(x_filt_ekf.numpy(), label='EKF Estimate', c='green')
    ax1.plot(x_filt_kf.numpy(), label='LGSSM Estimate', c='blue')
    ax1.plot(x_filt_ukf.numpy(), label='UKF Estimate', c='orange')
    ax1.set_title('Hidden States: Interest Rate Deviation (x_t)')
    ax1.set_ylabel('Rate Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_experiment_Vasicek_EM(kappa, theta, sigma, tau, dt, T=50, batch_size=20, n_em_iter=20):
    model_vasicek = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt)
    x_batch, y_batch = model_vasicek.sample(batch_size=batch_size, T=T)
    print('max range of y_t in batch:', tf.reduce_max(y_batch) - tf.reduce_min(y_batch))

    # initiate EKF noise parameters to default values
    model_vasicek.process_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.state_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.state_dim,), dtype=tf.float32) * 0.01
    )
    model_vasicek.observation_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.obs_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.obs_dim,), dtype=tf.float32) * 0.02
    )
    ekf = ExtendedKalmanFilter(model_vasicek)
    log_likelihoods_ekf = ekf.fit(y_batch, n_iter=n_em_iter)

    model_vasicek_ukf = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt)
    model_vasicek_ukf.process_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.state_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.state_dim,), dtype=tf.float32) * 0.01
    )
    model_vasicek_ukf.observation_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.obs_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.obs_dim,), dtype=tf.float32) * 0.02
    )
    ukf = UnscentedKalmanFilter(model_vasicek_ukf, alpha=1.0, beta=2.0, kappa=0.0, train_noise=True)

    t0 = time.time()
    log_losses_ukf = ukf.fit(y_batch, n_iter=10 * n_em_iter, learning_rate=0.05)
    t1 = time.time()
    print(f'UKF fit time (tf.function BPTT): {t1 - t0:.4f} seconds')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(log_likelihoods_ekf) + 1), log_likelihoods_ekf, marker='o')
    plt.title('EM Log-Likelihood Progression (EKF)')
    plt.xlabel('EM Iteration')
    plt.ylabel('Average Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(log_losses_ukf) + 1), log_losses_ukf, marker='o', color='orange')
    plt.title('UKF Loss Progression (BPTT)')
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss (-Log-Likelihood)')
    plt.grid(True, alpha=0.3)
    plt.show()

    est_proc_std = ekf.model.process_noise.stddev().numpy().flatten()[0]
    est_obs_std = ekf.model.observation_noise.stddev().numpy().flatten()[0]
    print('\n')
    print(f'True Process Noise Std: {sigma * (dt ** 0.5):.4f}, Estimated (EKF): {est_proc_std:.4f}')
    print(f'True Observation Noise Std: {0.01:.4f}, Estimated (EKF): {est_obs_std:.4f}')

    ukf_proc_std = ukf.model.process_noise.stddev().numpy().flatten()[0]
    ukf_obs_std = ukf.model.observation_noise.stddev().numpy().flatten()[0]
    print('\n')
    print(f'UKF Estimated Process Noise Std: {ukf_proc_std:.4f}')
    print(f'UKF Estimated Observation Noise Std: {ukf_obs_std:.4f}')


if __name__ == '__main__':
    #run_example_SVM(alpha=0.9, sigma=0.3, beta=0.5)
    #run_experiment_Vasicek(kappa=0.1, theta=0.05, sigma=0.02, tau=20.0, dt=1/252, x0=tf.constant([0.05]))
    run_experiment_Vasicek_EM(kappa=0.2, theta=0.05, sigma=0.02, tau=15.0, dt=1/252, T=100, batch_size=50, n_em_iter=10)