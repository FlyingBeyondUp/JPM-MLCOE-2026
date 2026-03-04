import tensorflow as tf
import tensorflow_probability as tfp

from models.base_models import NLSSM
from Filters.basic_filters.nonlinear_filters import ExtendedKalmanFilter, UnscentedKalmanFilter

tfd = tfp.distributions
dtype = tf.float32


def create_stress_nlssm(
    state_dim: int = 4,
    obs_dim: int = 2,
    nonlin_gain: float = 2.5,
    fold_gain: float = 1.8,
    process_scale: float = 0.15,
    obs_scale: float = 0.10,
):
    def transition_fn(state, noise):
        x0 = state[..., 0]
        x1 = state[..., 1]
        x2 = state[..., 2]
        x3 = state[..., 3]

        nx0 = 0.65 * x0 + 0.35 * x1 + nonlin_gain * tf.sin(1.7 * x2)
        nx1 = 0.55 * x1 - 0.25 * x2 + 0.20 * (x0 ** 3)
        nx2 = 0.50 * x2 + 0.30 * x3 + 0.15 * tf.sin(2.3 * x0 * x1)
        nx3 = 0.75 * x3 + 0.25 * tf.tanh(2.0 * x0) + 0.10 * (x1 ** 2)

        x_next = tf.stack([nx0, nx1, nx2, nx3], axis=-1)
        return x_next + noise

    def observation_fn(state, noise):
        z0 = state[..., 0]
        z1 = state[..., 1]
        z2 = state[..., 2]
        z3 = state[..., 3]

        y0 = fold_gain * tf.square(z0) + 0.8 * tf.sin(3.0 * z1) - 0.6 * z2
        y1 = 1.5 * tf.tanh(2.2 * z2) + 0.7 * tf.sin(4.0 * z0 * z3) + 0.3 * z1 * z3

        y = tf.stack([y0, y1], axis=-1)
        return y + noise

    process_noise = tfd.MultivariateNormalDiag(
        loc=tf.zeros([state_dim], dtype=dtype),
        scale_diag=process_scale * tf.ones([state_dim], dtype=dtype),
    )
    observation_noise = tfd.MultivariateNormalDiag(
        loc=tf.zeros([obs_dim], dtype=dtype),
        scale_diag=obs_scale * tf.ones([obs_dim], dtype=dtype),
    )
    init_noise = tfd.MultivariateNormalDiag(
        loc=tf.zeros([state_dim], dtype=dtype),
        scale_diag=0.6 * tf.ones([state_dim], dtype=dtype),
    )

    x0 = tf.constant([1.2, -1.0, 0.8, -0.6], dtype=dtype)

    return NLSSM(
        state_dim=state_dim,
        obs_dim=obs_dim,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        process_noise=process_noise,
        observation_noise=observation_noise,
        init_noise=init_noise,
        x0=x0,
    )


def rmse(x_hat: tf.Tensor, x_true: tf.Tensor) -> float:
    return float(tf.sqrt(tf.reduce_mean(tf.square(x_hat - x_true))).numpy())


def nll_per_step(log_likelihood: tf.Tensor, T: int) -> float:
    return float((-tf.reduce_mean(log_likelihood) / float(T)).numpy())


def run_one_setting(nonlin_gain, fold_gain, batch_size=64, T=80, seed=42):
    tf.keras.utils.set_random_seed(seed)

    model = create_stress_nlssm(
        nonlin_gain=nonlin_gain,
        fold_gain=fold_gain,
        process_scale=0.15,
        obs_scale=0.10,
    )

    x_true, y_obs = model.sample(batch_size=batch_size, T=T)

    ekf = ExtendedKalmanFilter(model=model, requires_stabilization=True)
    ukf = UnscentedKalmanFilter(model=model, alpha=1e-1, beta=2.0, kappa=0.0, train_noise=False)

    ekf_res = ekf.filter(y_obs)
    ukf_res = ukf.filter(y_obs)

    ekf_rmse = rmse(ekf_res["x_filt"], x_true)
    ukf_rmse = rmse(ukf_res["x_filt"], x_true)

    ekf_nll = nll_per_step(ekf_res["log_likelihood"], T)
    ukf_nll = nll_per_step(ukf_res["log_likelihood"], T)

    return {
        "nonlin_gain": nonlin_gain,
        "fold_gain": fold_gain,
        "ekf_rmse": ekf_rmse,
        "ukf_rmse": ukf_rmse,
        "ekf_nll": ekf_nll,
        "ukf_nll": ukf_nll,
    }


def reveal_failure():
    nonlin_grid = [0.6, 1.0, 1.6, 2.2, 2.8, 3.4]
    fold_grid = [0.6, 1.0, 1.4, 1.8, 2.2]

    rows = []
    for ng in nonlin_grid:
        for fg in fold_grid:
            rows.append(run_one_setting(ng, fg))

    print("\n=== Failure Map \\(EKF vs UKF\\) ===")
    print("nonlin\\tfold\\tEKF\\_RMSE\\tUKF\\_RMSE\\tEKF\\_NLL\\tUKF\\_NLL")
    for r in rows:
        print(
            f"{r['nonlin_gain']:.2f}\t{r['fold_gain']:.2f}\t"
            f"{r['ekf_rmse']:.4f}\t{r['ukf_rmse']:.4f}\t"
            f"{r['ekf_nll']:.4f}\t{r['ukf_nll']:.4f}"
        )

    # 简单失败判据: 相比低非线性基线恶化超过 2 倍
    base = [r for r in rows if abs(r["nonlin_gain"] - 0.6) < 1e-9 and abs(r["fold_gain"] - 0.6) < 1e-9][0]
    ekf_thr = 2.0 * base["ekf_rmse"]
    ukf_thr = 2.0 * base["ukf_rmse"]

    ekf_fail = next((r for r in rows if r["ekf_rmse"] >= ekf_thr), None)
    ukf_fail = next((r for r in rows if r["ukf_rmse"] >= ukf_thr), None)

    print("\n=== Estimated Failure Threshold \\(RMSE x2 baseline\\) ===")
    print(f"EKF baseline RMSE={base['ekf_rmse']:.4f}, fail\\_thr={ekf_thr:.4f}")
    print(f"UKF baseline RMSE={base['ukf_rmse']:.4f}, fail\\_thr={ukf_thr:.4f}")

    if ekf_fail is not None:
        print(
            f"EKF first fail at nonlin={ekf_fail['nonlin_gain']:.2f}, "
            f"fold={ekf_fail['fold_gain']:.2f}, rmse={ekf_fail['ekf_rmse']:.4f}"
        )
    else:
        print("EKF no fail point found on current grid.")

    if ukf_fail is not None:
        print(
            f"UKF first fail at nonlin={ukf_fail['nonlin_gain']:.2f}, "
            f"fold={ukf_fail['fold_gain']:.2f}, rmse={ukf_fail['ukf_rmse']:.4f}"
        )
    else:
        print("UKF no fail point found on current grid.")


if __name__ == "__main__":
    reveal_failure()
