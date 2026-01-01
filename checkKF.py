import tensorflow as tf
import matplotlib.pyplot as plt
from linearSSM import LGSSM, KalmanFilter


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


if __name__ == "__main__":
    check_condition_number_impact()
