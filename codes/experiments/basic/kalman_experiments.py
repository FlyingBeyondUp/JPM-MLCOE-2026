import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from models.base_models import LGSSM
from Filters.basic_filters.kalman_filter import KalmanFilter, EM_initializer, EM_solver
import math

tfd = tfp.distributions

def check_condition_number_impact(T=200, state_dim=10, obs_dim=10):
    # 1. Create a model with very low observation noise (High Precision)
    # This exacerbates numerical instability in the Standard Form
    model = LGSSM(state_dim, obs_dim)

    # Force R to be very small (1e-6) compared to P0 (1.0)
    # This makes K*C very close to Identity, causing cancellation in (I - KC)
    R_small = tf.eye(obs_dim) * 1e-8
    model.set_params([model.A, model.C, model.Q, R_small, model.x0, model.P0])

    names=['A', 'C', 'Q', 'R', 'x0', 'P0']
    for name, param in zip(names, model.get_params()):
        if name != 'x0':
            print(f"{name}:\n{tf.linalg.eigvals(param.numpy())}\n")
        else:
            print(f"{name}:\n{param.numpy()}\n")

    kf = KalmanFilter(model)

    # 2. Generate Data
    X, Y = model.sample(T)

    # 3. Run Filter WITH Stabilization (Joseph Form)
    X_stab, P_stab, _, _, _ = kf.filter(Y, T, requires_stabilization=True)

    # 4. Run Filter WITHOUT Stabilization (Standard Form)
    X_unstab, P_unstab, _, _, _ = kf.filter(Y, T, requires_stabilization=False)

    # 5. Compute Condition Numbers
    # Condition Number = max_eigenvalue / min_eigenvalue
    def get_condition_numbers(P_seq):
        conds = []
        for t in range(T):
            s = tf.linalg.svd(P_seq[t], compute_uv=False)
            # s is sorted largest to smallest
            cond = s[0] / s[-1]
            conds.append(cond)
        return conds

    cond_stab = get_condition_numbers(P_stab)
    cond_unstab = get_condition_numbers(P_unstab)

    # 6. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cond_stab, label='With Stabilization (Joseph)', linewidth=2)
    plt.plot(cond_unstab, label='Without Stabilization (Standard)', linestyle='--', linewidth=2)
    plt.yscale('log')
    plt.title('Condition Number of Covariance Matrix P_t (Log Scale)')
    plt.ylabel('Condition Number (Lower is Better)')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig("Figures\KF_Condition_Number_t.pdf",bbox_inches='tight')
    plt.show()

    # Check for asymmetry or non-PSD in unstabilized version
    last_P_unstab = P_unstab[-1]
    is_symmetric = tf.norm(last_P_unstab - tf.transpose(last_P_unstab)) < 1e-6
    eig_vals = tf.linalg.eigvals(last_P_unstab)
    min_eig = tf.reduce_min(tf.math.real(eig_vals))

    print(f"Final Unstabilized P Symmetric? {is_symmetric.numpy()}")
    print(f"Final Unstabilized P Min Eigenvalue: {min_eig.numpy()} (Should be > 0)")

    print(f"Max State Diff: {tf.reduce_max(tf.abs(X_stab - X_unstab)).numpy()}")

    # check the inferred states
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 0], X[:, 1], label='True State', linewidth=2)
    plt.plot(X_stab[:, 0], X_stab[:, 1], label='Inferred State (Stabilized)', linestyle='--', linewidth=2)
    plt.plot(X_unstab[:, 0], X_unstab[:, 1], label='Inferred State (Unstabilized)', linestyle=':', linewidth=2)
    plt.title('True vs Inferred States')
    plt.xlabel('State Dimension 1')
    plt.ylabel('State Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def show_EM_example(num_trials=64, T=100, state_dim=3, obs_dim=3, max_iters=50, stabilization=True):
    '''
    # compare the estimated parameters with the true parameters
    Due to the identifiability issue of LGSSM, the estimated parameters may differ from
    the true parameters by a linear transformation significantly even if the inference of latent variables is sound.

    true_params=data_generator.get_params()
    est_params=kf.model.get_params()
    param_names=['A','C','Q','R','x0','P0']
    for i in range(len(true_params)):
        true_param=true_params[i]
        est_param=est_params[i]
        param_name=param_names[i]
        param_error=tf.norm(true_param - est_param) / (tf.norm(true_param) + 1e-9)
        print(f'Parameter {param_name} relative error: {param_error.numpy():.4f}')
    '''

    data_generator = LGSSM(state_dim, obs_dim)
    X, Y = data_generator.batch_sample(T=T, batch_size=num_trials)
    kf = EM_initializer(Y, state_dim)
    # kf=KalmanFilter(LGSSM(state_dim,obs_dim))
    print('Successfully initialized EM Kalman Filter')

    list_log_l = EM_solver(kf, Y, max_iters, stabilization=stabilization)

    plt.figure()
    plt.plot(list_log_l)
    plt.xlabel('EM Iterations')
    plt.ylabel('Log Likelihood')
    plt.title('EM Convergence')
    plt.savefig('KF_EM_Convergence.pdf')
    plt.show()

    print("\n=== Model Evaluation ===")

    # Observation Reconstruction
    X_smooth, _, _, _ = kf.smooth_filter(Y)
    Y_pred = tf.matmul(X_smooth, kf.model.C, transpose_b=True)

    obs_rmse = tf.sqrt(tf.reduce_mean(tf.square(Y - Y_pred)))
    baseline_rmse = tf.sqrt(tf.reduce_mean(tf.square(Y)))

    print(f"Observation Reconstruction RMSE: {obs_rmse:.4f}")
    print(f"Baseline (Zero) RMSE:            {baseline_rmse:.4f}")
    print(f"Reconstruction R^2:              {1 - (obs_rmse ** 2 / baseline_rmse ** 2):.4f}")

    # Dynamics Matrix Comparison
    true_A = data_generator.A
    est_A = kf.model.A

    eig_true = tf.linalg.eigvals(true_A)
    eig_est = tf.linalg.eigvals(est_A)

    sort_idx_true = tf.argsort(tf.abs(eig_true), direction='DESCENDING')
    sort_idx_est = tf.argsort(tf.abs(eig_est), direction='DESCENDING')

    eig_true_sorted = tf.gather(eig_true, sort_idx_true)
    eig_est_sorted = tf.gather(eig_est, sort_idx_est)

    print("\n--- Eigenvalues of Transition Matrix A (Invariant to Rotation) ---")
    print("True Eigenvalues:     ", eig_true_sorted.numpy())
    print("Estimated Eigenvalues:", eig_est_sorted.numpy())

    print("True Moduli:          ", tf.abs(eig_true_sorted).numpy())
    print("Estimated Moduli:     ", tf.abs(eig_est_sorted).numpy())




def test_filter(num_trials=128, T=50, state_dim=3, obs_dim=3):
    model = LGSSM(state_dim, obs_dim)
    kf = KalmanFilter(model)

    X, Y = model.batch_sample(T=T, batch_size=num_trials)
    X_filt, _, _, _, _ = kf.filter(Y)
    X_smooth, _, _, _ = kf.smooth_filter(Y)

    err_filt = X - X_filt
    err_smooth = X - X_smooth

    per_mse_filt = tf.reduce_mean(tf.square(err_filt), axis=[1, 2])
    per_rmse_filt = tf.sqrt(per_mse_filt)

    per_mse_smooth = tf.reduce_mean(tf.square(err_smooth), axis=[1, 2])
    per_rmse_smooth = tf.sqrt(per_mse_smooth)

    mean_rmse_filt = tf.reduce_mean(per_rmse_filt)
    std_rmse_filt = tf.math.reduce_std(per_rmse_filt)

    mean_rmse_smooth = tf.reduce_mean(per_rmse_smooth)
    std_rmse_smooth = tf.math.reduce_std(per_rmse_smooth)

    print(f'Filtering RMSE: {mean_rmse_filt.numpy():.4f} ± {std_rmse_filt.numpy():.4f}')
    print(f'Smoothing RMSE: {mean_rmse_smooth.numpy():.4f} ± {std_rmse_smooth.numpy():.4f}')
    eval_rmse_and_baselines(X, X_filt, X_smooth, Y, model)


def eval_rmse_and_baselines(X, X_filt, X_smooth, Y, model):
    per_mse_filt = tf.reduce_mean(tf.square(X - X_filt), axis=[1, 2])
    per_rmse_filt = tf.sqrt(per_mse_filt)
    per_mse_smooth = tf.reduce_mean(tf.square(X - X_smooth), axis=[1, 2])
    per_rmse_smooth = tf.sqrt(per_mse_smooth)

    mean_rmse_filt = tf.reduce_mean(per_rmse_filt)
    std_rmse_filt = tf.math.reduce_std(per_rmse_filt)
    mean_rmse_smooth = tf.reduce_mean(per_rmse_smooth)
    std_rmse_smooth = tf.math.reduce_std(per_rmse_smooth)

    std_X = tf.math.reduce_std(X)

    baseline_zero_rmse = tf.sqrt(tf.reduce_mean(tf.square(X)))

    pinvC = tf.linalg.pinv(model.C)  # shape [state_dim, obs_dim]
    X_from_Y = tf.einsum('so,bto->bts', pinvC, Y)  # [B,T,state_dim]
    baseline_pinv_rmse = tf.sqrt(tf.reduce_mean(tf.square(X - X_from_Y)))

    print('\n')
    print(f'Filtering RMSE: {mean_rmse_filt.numpy():.4f} ± {std_rmse_filt.numpy():.4f}')
    print(f'Smoothing RMSE: {mean_rmse_smooth.numpy():.4f} ± {std_rmse_smooth.numpy():.4f}')
    print(f'State std (global): {std_X.numpy():.4f}')
    print(f'Normalized Filtering RMSE: {(mean_rmse_filt / std_X).numpy():.4f}')
    print(f'Baseline zero RMSE: {baseline_zero_rmse.numpy():.4f}')
    print(f'Baseline pinv(C) RMSE: {baseline_pinv_rmse.numpy():.4f}')


# eval_rmse_and_baselines(X, X_filt, X_smooth, Y, model)


def show_example(T=100, state_dim=2, obs_dim=1):
    model = LGSSM(state_dim, obs_dim)
    kf = KalmanFilter(model)

    x, y = model.sample(T=T)
    x_filt, _ = kf.filter(y, T)[:2]
    x_smooth, _, _, _ = kf.smooth_filter(y, T)

    time_axis = tf.range(T)

    plt.figure(figsize=(12, 8))
    for d in range(state_dim):
        plt.subplot(state_dim, 1, d + 1)
        plt.plot(time_axis, y[:, d], label='Observations', color='gray', linestyle='None', marker='o', markersize=5,
                 alpha=0.5)
        plt.plot(time_axis, x[:, d], label='True State', color='black')
        plt.plot(time_axis, x_filt[:, d], label='Filtered State', color='blue', linestyle='--')
        plt.plot(time_axis, x_smooth[:, d], label='Smoothed State', color='red', linestyle=':')
        plt.title(f'State Dimension {d + 1}')
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.legend()
    plt.tight_layout()
    plt.show()


def run_comparison_example():
    state_dim = 2  # [Position, Velocity]
    obs_dim = 1  # [Noisy Position]
    dt = 0.1

    # Physics: Rotation matrix for harmonic oscillator
    theta = 0.1
    A_val = tf.constant([[math.cos(theta), math.sin(theta)],
                         [-math.sin(theta), math.cos(theta)]], dtype=tf.float32)

    # Observation: We only see Position
    C_val = tf.constant([[1.0, 0.0]], dtype=tf.float32)

    # Noise: Low process noise (smooth dynamics), HIGH observation noise
    Q_val = tf.eye(state_dim) * 0.01
    R_val = tf.eye(obs_dim) * 1.0  # <--- High Noise!

    x0_val = tf.zeros([state_dim, 1])
    P0_val = tf.eye(state_dim)

    # Initialize Custom Model
    params = [A_val, C_val, Q_val, R_val, x0_val, P0_val]
    my_model = LGSSM(state_dim, obs_dim, params)
    my_kf = KalmanFilter(my_model)

    # --- 2. Generate Data ---
    T = 100
    true_states, observations = my_model.sample(T)

    # --- 3. Run Custom Implementation ---
    # Filter
    custom_filt_x, _, _, _, _ = my_kf.filter(observations, T)
    # custom_filt_x is already [T, state_dim] for non-batched input

    # Smooth
    custom_smooth_x, _, _, _ = my_kf.smooth_filter(observations, T)
    # custom_smooth_x is already [T, state_dim] for non-batched input

    # --- 4. Run TensorFlow Probability (TFP) Implementation ---
    tfp_lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=T,
        transition_matrix=A_val,
        transition_noise=tfd.MultivariateNormalTriL(scale_tril=tf.linalg.cholesky(Q_val)),
        observation_matrix=C_val,
        observation_noise=tfd.MultivariateNormalTriL(scale_tril=tf.linalg.cholesky(R_val)),
        initial_state_prior=tfd.MultivariateNormalTriL(loc=tf.squeeze(x0_val), scale_tril=tf.linalg.cholesky(P0_val))
    )

    # TFP Filter & Smooth
    _, filtered_means, _, _, _, _, _ = tfp_lgssm.forward_filter(observations)
    tfp_smoothed_means, _ = tfp_lgssm.posterior_marginals(observations)

    # --- 5. Visualization ---
    plt.figure(figsize=(14, 10))

    # Plot Position (Dim 0)
    plt.plot(observations[:, 0], 'k.', alpha=0.3, label='Noisy Observations')
    plt.plot(true_states[:, 0], 'k-', linewidth=2, label='True State')

    plt.plot(custom_filt_x[:, 0], 'g-', label='Custom Filter')
    plt.plot(custom_smooth_x[:, 0], 'r-', linewidth=2, label='Custom Smoother')

    # Overlay TFP to prove match
    plt.plot(filtered_means[:, 0], 'o:', linewidth=3, label='TFP Filter')
    plt.plot(tfp_smoothed_means[:, 0], 'y:', linewidth=3, label='TFP Smoother')

    plt.title("Kalman Filter vs Smoother (High Observation Noise Scenario)")
    plt.legend(fontsize=16)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("State 0", fontsize=20)
    plt.grid(True, alpha=0.3)

    # Calculate RMSE
    rmse_filt = tf.sqrt(tf.reduce_mean((true_states - custom_filt_x) ** 2))
    rmse_smooth = tf.sqrt(tf.reduce_mean((true_states - custom_smooth_x) ** 2))

    print(f"Filter RMSE:   {rmse_filt:.4f}")
    print(f"Smoother RMSE: {rmse_smooth:.4f}")
    print("If the yellow dotted line perfectly overlaps the red line, your custom implementation matches TFP.")
    plt.savefig('Figures/kalman_filter_smoother_comparison.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    check_condition_number_impact()
    show_EM_example()