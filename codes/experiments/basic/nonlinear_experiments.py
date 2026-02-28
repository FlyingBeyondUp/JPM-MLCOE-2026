from models import *
from Filters.basic_filters import KalmanFilter
from Filters.basic_filters import ExtendedKalmanFilter, UnscentedKalmanFilter
import matplotlib.pyplot as plt
import time
import tensorflow as tf



def run_example_SVM(alpha,sigma,beta,T=500):
    model = get1DStochasticVolModel(alpha=alpha, sigma=sigma, beta=beta)
    model_st = get1DStochasticVolModel(alpha=alpha, sigma=sigma, beta=beta, heavy_tail=True)
    x, y = model.sample(T)
    x_st, y_st = model_st.sample(T)
    print('max range of y_t:', tf.reduce_max(y) - tf.reduce_min(y))
    print('max range of y_t (Heavy-Tailed):', tf.reduce_max(y_st) - tf.reduce_min(y_st))

    y_log_sq = tf.math.log(y ** 2 + 1e-8)
    ekf = ExtendedKalmanFilter(get1DLogSquaredSVM(alpha=alpha, sigma=sigma, beta=beta))
    x_filt_ekf, P_filt_ekf, _, _, log_l_ekf = ekf.filter(y_log_sq, T, requires_stabilization=True)
    x_filt_ekf = tf.squeeze(x_filt_ekf, axis=-1)

    lgssm_svm, bias = get_LogSVM_LGSSM(alpha=alpha, sigma=sigma, beta=beta)
    kf = KalmanFilter(lgssm_svm)
    y_log_sq_bias_corrected = y_log_sq - bias
    x_filt_kf, P_filt_kf, _, _, log_l_kf = kf.filter(tf.reshape(y_log_sq_bias_corrected, [-1, 1]), T,
                                                     requires_stabilization=True)
    x_filt_kf = tf.squeeze(x_filt_kf, axis=-1)

    if len(y_log_sq.shape)==1:
        y_log_sq = tf.reshape(y_log_sq, [1, -1, 1])  # Batch size 1
    ukf= UnscentedKalmanFilter(get1DLogSquaredSVM(alpha=alpha, sigma=sigma, beta=beta), alpha=1, beta=2.0, kappa=0.0) # For 1D, alpha=1 is better than 1e-3?
    t0=time.time()
    x_filt_ukf, P_filt_ukf,_ = ukf.filter(y_log_sq)
    t1=time.time()
    x_filt_ukf, P_filt_ukf, _ = ukf.filter(y_log_sq, py_loop=True)
    t2=time.time()
    print(f'UKF time (tf.scan): {t1 - t0:.4f} seconds')
    print(f'UKF time (py loop): {t2 - t1:.4f} seconds')

    print('\n')
    print('EKF max error:', tf.reduce_max(tf.abs(x - x_filt_ekf)).numpy())
    print('LGSSM max error:', tf.reduce_max(tf.abs(x - x_filt_kf)).numpy())
    print('UKF max error:', tf.reduce_max(tf.abs(x - x_filt_ukf)).numpy())
    print('EKF mean error:', tf.reduce_mean(tf.abs(x - x_filt_ekf)).numpy())
    print('LGSSM mean error:', tf.reduce_mean(tf.abs(x - x_filt_kf)).numpy())
    print('UKF mean error:', tf.reduce_mean(tf.abs(x - x_filt_ukf)).numpy())

    plt.figure(figsize=(12, 6))
    plt.plot(x.numpy(), label='Hidden States x_t', c='black', linestyle='--')
    plt.plot(x_filt_ekf.numpy(), label='Hidden States from Extended Kalman Filter', c='green')
    plt.plot(x_filt_kf.numpy(), label='Hidden States from LGSSM Approximation', c='blue')
    plt.plot(x_filt_ukf.numpy().flatten(), label='Hidden States from Unscented Kalman Filter', c='orange')
    plt.title('Stochastic Volatility Model - Hidden States Estimation')
    plt.scatter(range(T), y.numpy(), c='red', s=10, label='Observations y_t')
    plt.title('Stochastic Volatility Model - Hidden States')
    plt.xlabel('Time Step')
    plt.ylabel('x_t')
    plt.legend()

    plt.figure(figsize=(12, 6))
    plt.plot(x_st.numpy(), label='Hidden State x_t (Heavy-Tailed)')
    plt.scatter(range(T), y_st.numpy(), c='red', s=10, label='Observations y_t (Heavy-Tailed)')
    plt.title('Stochastic Volatility Model with Heavy-Tailed Process Noise - Hidden States')
    plt.xlabel('Time Step')
    plt.ylabel('x_t')
    plt.legend()

    plt.show()


def run_experiment_Vasicek(kappa, theta, sigma, tau, dt, T=200,x0=tf.zeros((1,))):
    model_vasicek = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt,x0=x0)
    x, y = model_vasicek.sample(T)
    print('max range of y_t:', tf.reduce_max(y) - tf.reduce_min(y))
    print('y_t shape:', y.shape)

    # --- EKF ---
    ekf = ExtendedKalmanFilter(model_vasicek)
    x_filt_ekf_raw, P_filt_ekf, _, _, log_l_ekf = ekf.filter(y, T, requires_stabilization=True)
    x_filt_ekf = tf.squeeze(x_filt_ekf_raw, axis=-1)

    # --- LGSSM ---
    lgssm_vasicek, bias = getVasicekLGSSM(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt,x0=x0)
    kf = KalmanFilter(lgssm_vasicek)
    y_bias_corrected = y - bias
    x_filt_kf, P_filt_kf, _, _, log_l_kf = kf.filter(tf.reshape(y_bias_corrected, [-1, 1]), T,
                                                     requires_stabilization=True)
    x_filt_kf = tf.squeeze(x_filt_kf, axis=-1)

    ukf= UnscentedKalmanFilter(model_vasicek,alpha=1.0, beta=0.0, kappa=0.0) # For 1D, alpha=1 is better than 1e-3
    y=tf.reshape(y, [1, -1, 1])  # Batch size 1
    x_filt_ukf, P_filt_ukf,_ = ukf.filter(y)
    x_filt_ukf = x_filt_ukf.numpy().flatten()

    print('\n')
    print('EKF max error:', tf.reduce_max(tf.abs(x - x_filt_ekf)).numpy())
    print('UKF max error:', tf.reduce_max(tf.abs(x - x_filt_ukf)).numpy())
    print('LGSSM max error:', tf.reduce_max(tf.abs(x - x_filt_kf)).numpy())
    print('\n')
    print('EKF mean error:', tf.reduce_mean(tf.abs(x - x_filt_ekf)).numpy())
    print('UKF mean error:', tf.reduce_mean(tf.abs(x - x_filt_ukf)).numpy())
    print('LGSSM mean error:', tf.reduce_mean(tf.abs(x - x_filt_kf)).numpy())

    # y_pred = h(x_filt)
    zero_noise = tf.zeros([1])
    y_pred_ekf = []
    y_pred_kf=[]
    for t in range(T):
        state_val = tf.reshape(x_filt_ekf[t], [1])
        pred = model_vasicek.observation_fn(state_val, zero_noise)
        y_pred_ekf.append(pred)
        y_pred_kf.append(lgssm_vasicek.C @ tf.reshape(x_filt_kf[t],[-1,1]) + bias)
    y_pred_ekf = tf.stack(y_pred_ekf)
    y_pred_kf = tf.reshape(tf.stack(y_pred_kf), [-1])

    # --- Plotting ---
    fig, ax1= plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    ax1.plot(x.numpy(), label='True Hidden States (Rate Deviation)', c='black', linestyle='--')
    ax1.plot(x_filt_ekf.numpy(), label='EKF Estimate', c='green')
    ax1.plot(x_filt_kf.numpy(), label='LGSSM Estimate', c='blue')
    ax1.plot(x_filt_ukf, label='UKF Estimate', c='orange')
    ax1.set_title('Hidden States: Interest Rate Deviation (x_t)')
    ax1.set_ylabel('Rate Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ax2.scatter(range(T), y.numpy(), c='red', s=15, label='Observed Bond Prices (y_t)', alpha=0.6)
    # ax2.plot(y_pred_ekf.numpy(), label='EKF Implied Price', c='green', linewidth=2)
    # ax2.plot(y_pred_kf.numpy(), label='LGSSM Implied Price', c='blue', linewidth=2)
    # ax2.set_title('Observations: Bond Prices (y_t)')
    # ax2.set_xlabel('Time Step')
    # ax2.set_ylabel('Price')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    plt.savefig('Figures\Vasicek_hidden_state_estimation.pdf', bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def run_experiment_Vasicek_EM(kappa, theta, sigma, tau, dt, T=50, batch_size=20, n_em_iter=20):
    model_vasicek = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt)
    x_batch, y_batch = model_vasicek.batch_sample(T, batch_size)
    print('max range of y_t in batch:', tf.reduce_max(y_batch) - tf.reduce_min(y_batch))
    if len(y_batch.shape) == 2:
        y_batch = tf.expand_dims(y_batch, -1)
    print('y_batch shape:', y_batch.shape)

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
    log_likelihoods_ekf = ekf.fit_EM(y_batch, n_iter=n_em_iter)

    model_vasicek_ukf = getVasicekBondPriceModel(kappa=kappa, theta=theta, sigma=sigma, tau=tau, dt=dt)
    model_vasicek_ukf.process_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.state_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.state_dim,), dtype=tf.float32) * 0.01
    )
    model_vasicek_ukf.observation_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.obs_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.obs_dim,), dtype=tf.float32) * 0.02
    )
    ukf=UnscentedKalmanFilter(model_vasicek_ukf,alpha=1, beta=2.0, kappa=0.0,train_noise=True)
    t0=time.time()
    log_likelihoods_ukf = ukf.fit(y_batch, n_iter=10*n_em_iter, learning_rate=0.05)
    t1=time.time()

    model_vasicek_ukf.process_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.state_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.state_dim,), dtype=tf.float32) * 0.01
    )
    model_vasicek_ukf.observation_noise = tfp.distributions.Normal(
        loc=tf.zeros((model_vasicek.obs_dim,), dtype=tf.float32),
        scale=tf.ones((model_vasicek.obs_dim,), dtype=tf.float32) * 0.02
    )
    ukf = UnscentedKalmanFilter(model_vasicek_ukf, alpha=1, beta=2.0, kappa=0.0, train_noise=True)
    t2=time.time()
    log_likelihoods_ukf = ukf.fit(y_batch, n_iter=10*n_em_iter, learning_rate=0.05,py_loop=True)
    t3=time.time()
    print(f'UKF fit time (tf.scan): {t1 - t0:.4f} seconds')
    print(f'UKF fit time (py loop): {t3 - t2:.4f} seconds')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(log_likelihoods_ekf) + 1), log_likelihoods_ekf, marker='o')
    plt.title('EM Log-Likelihood Progression')
    plt.xlabel('EM Iteration')
    plt.ylabel('Average Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.savefig('EKF_EM_vasicek_loglikelihood.pdf',bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(log_likelihoods_ukf) + 1), log_likelihoods_ukf, marker='o', color='orange')
    plt.title('UKF Loss Progression')
    plt.xlabel('Training Iteration')
    plt.ylabel('Average Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.savefig('UKF_EM_vasicek_loglikelihood.pdf', bbox_inches='tight')
    plt.show()

    est_proc_std = ekf.model.process_noise.stddev().numpy().flatten()[0]
    est_obs_std = ekf.model.observation_noise.stddev().numpy().flatten()[0]
    print('\n')
    print(f'True Process Noise Std: {sigma * (dt ** 0.5):.4f}, Estimated: {est_proc_std:.4f}')
    print(f'True Observation Noise Std: {0.01:.4f}, Estimated: {est_obs_std:.4f}')

    ukf_proc_std = ukf.model.process_noise.stddev().numpy().flatten()[0]
    ukf_obs_std = ukf.model.observation_noise.stddev().numpy().flatten()[0]
    print('\n')
    print(f'UKF Estimated Process Noise Std: {ukf_proc_std:.4f}')
    print(f'UKF Estimated Observation Noise Std: {ukf_obs_std:.4f}')


if __name__ == '__main__':
    #run_example_SVM(alpha=0.9, sigma=0.3, beta=0.5)
    run_experiment_Vasicek(kappa=0.1, theta=0.05, sigma=0.02, tau=20.0, dt=1/252,x0=tf.constant((0.05,)))
    #run_experiment_Vasicek_EM(kappa=0.2, theta=0.05, sigma=0.02, tau=15.0, dt=1/252, T=100, batch_size=50, n_em_iter=10)
