import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from models.base_models import LGSSM, NLSSM,LearnableSSM
from Filters.basic_filters import KalmanFilter
from Filters.basic_filters import ParticleFilter
from Filters.differentiable_filters import DifferentiableParticleFilter

tfd = tfp.distributions


def create_models(theta_val):
    """
    Creates both the LGSSM (for Kalman Filter and data generation)
    and NLSSM (for Particle Filters) for a given theta = (theta_1, theta_2).
    """
    state_dim = 2
    obs_dim = 2

    # 1. Setup LGSSM Parameters
    A = tf.eye(state_dim) * theta_val
    C = tf.eye(state_dim)  # Y_t | X_t ~ N(x, ...) implies C is Identity
    Q = tf.eye(state_dim) * 0.5
    R = tf.eye(state_dim) * 0.1
    x0 = tf.zeros([state_dim, 1])
    P0 = tf.eye(state_dim) * 1.0  # Standard normal prior

    lgssm = LGSSM(state_dim, obs_dim, params=[A, C, Q, R, x0, P0])

    # 2. Setup equivalent NLSSM for the Particle Filters
    def transition_fn(x, noise):
        return x * theta_val + noise

    def observation_fn(x, noise):
        return x + noise

    process_noise = tfd.MultivariateNormalDiag(
        scale_diag=tf.ones(state_dim) * tf.sqrt(0.5)
    )
    observation_noise = tfd.MultivariateNormalDiag(
        scale_diag=tf.ones(state_dim) * tf.sqrt(0.1)
    )
    init_noise = tfd.MultivariateNormalDiag(
        scale_diag=tf.ones(state_dim) * 1.0
    )
    x0_nlssm = tf.zeros([state_dim])

    nlssm = NLSSM(state_dim, obs_dim, transition_fn, observation_fn,
                  process_noise, observation_noise, init_noise, x0_nlssm)

    return lgssm, nlssm


def run_section_5_1_experiment():
    T = 150
    N = 25
    num_realizations = 100

    print(f"Simulating 1 sequence of T={T} steps, evaluating {num_realizations} realizations of U...")

    # Generate True Data using theta = (0.5, 0.5)
    lgssm_true, _ = create_models(theta_val=0.5)

    # Generate exactly ONE sequence.
    _, single_y = lgssm_true.sample(T=T)

    # Duplicate this single sequence 100 times to create the batch
    batch_y = tf.tile(tf.expand_dims(single_y, 0), [num_realizations, 1, 1])

    # Evaluation target thetas
    eval_thetas = [0.25, 0.5, 0.75]
    epsilons = [0.25, 0.5, 0.75]

    results = []

    for theta in eval_thetas:
        print(f"Evaluating at theta = ({theta}, {theta})...")
        lgssm_eval, nlssm_eval = create_models(theta)

        # 1. Exact Log-Likelihood via Kalman Filter
        kf = KalmanFilter(lgssm_eval)
        _, _, _, _, exact_log_l = kf.filter(batch_y, requires_stabilization=True)

        # 2. Standard Particle Filter
        pf = ParticleFilter(nlssm_eval, num_particles=N, resample_method='multinomial')
        _, _, _, pf_log_l = pf.filter_summarized(batch_y)

        pf_diff = (pf_log_l - exact_log_l) / float(T)
        results.append({
            'Theta': str(theta),
            'Method': 'PF',
            'Mean': tf.reduce_mean(pf_diff).numpy(),
            'Std': tf.math.reduce_std(pf_diff).numpy()
        })

        # 3. Differentiable Particle Filters (Varying Epsilon)
        for eps in epsilons:
            dpf = DifferentiableParticleFilter(nlssm_eval, num_particles=N, epsilon=eps)
            dpf.use_differentiable_resample = True

            _, _, _, dpf_log_l = dpf.filter_summarized(batch_y)
            dpf_diff = (dpf_log_l - exact_log_l) / float(T)

            results.append({
                'Theta': str(theta),
                'Method': f'DPF (\u03b5 = {eps})',  # Using the epsilon unicode character
                'Mean': tf.reduce_mean(dpf_diff).numpy(),
                'Std': tf.math.reduce_std(dpf_diff).numpy()
            })

    # --- Custom Printing Logic to match the Paper ---
    print("\nTable 1. Mean & std of 1/T (hat_l(theta; U) - l(theta))")
    print("-" * 55)

    # Print Header
    print(f"{'θ1, θ2':>16} {'':>6} {0.25:>8} {0.5:>8} {0.75:>8}")
    print("-" * 55)

    # Reorganize results for row-by-row printing
    structured_results = {}
    for r in results:
        method = r['Method']
        theta = r['Theta']
        if method not in structured_results:
            structured_results[method] = {}
        structured_results[method][theta] = {'mean': r['Mean'], 'std': r['Std']}

    # Define row order
    methods_order = ['PF', 'DPF (ε = 0.25)', 'DPF (ε = 0.5)', 'DPF (ε = 0.75)']
    thetas_order = ['0.25', '0.5', '0.75']

    # Print Body
    for method in methods_order:
        means = [structured_results[method][th]['mean'] for th in thetas_order]
        stds = [structured_results[method][th]['std'] for th in thetas_order]

        # [cite_start]Format the numbers to 2 decimal places [cite: 3]
        mean_str = " ".join([f"{m:>8.2f}" for m in means])
        std_str = " ".join([f"{s:>8.2f}" for s in stds])

        # [cite_start]Print mean and std on separate lines, with the method name vertically "centered" [cite: 3]
        print(f"{method:>16}   mean {mean_str}")
        print(f"{'':>16}    std {std_str}")
        print("-" * 55)


class ProposalNetwork(tf.keras.layers.Layer):
    def __init__(self, dx, dy, A_mat, **kwargs):
        super().__init__(**kwargs)
        self.dx = dx
        self.dy = dy
        self.A_mat = tf.constant(A_mat, dtype=tf.float32)

        initial_log_phi = np.log(2.0).astype(np.float32)
        self.log_phi = self.add_weight(
            shape=(dx + dy,),
            initializer=tf.constant_initializer(initial_log_phi),
            trainable=True,
            name='log_phi'
        )

    def call(self, x_prev, y_t):
        log_phi_clipped = tf.clip_by_value(self.log_phi, -4.0, 2.0)
        phi = tf.exp(log_phi_clipped)

        # 1. Compute A * x_{t-1} using batched matmul (x_prev @ A^T)
        Ax = tf.matmul(x_prev, self.A_mat, transpose_b=True)

        # 2. Compute Gamma_phi * y_t
        Gy = y_t * phi[-self.dy:]

        # CORRECTED: Pad with dx - dy zeros instead of dx zeros
        paddings = [[0, 0]] * (len(y_t.shape) - 1) + [[0, self.dx - self.dy]]
        Gy_padded = tf.pad(Gy, paddings)

        # 3. Compute Mean: Delta_phi^{-1} * (Ax + Gy)
        mean = (Ax + Gy_padded) / phi[:self.dx]

        # 4. Compute Standard Deviation
        std = tf.sqrt(phi[:self.dx] + 1e-8)
        std = tf.broadcast_to(std, tf.shape(mean))

        return mean, std


def run_section_5_2_experiment():
    dx = 25
    dy = 1
    T = 100

    M = 10  # CORRECTED: Set M = 100 realizations

    print(f"Setting up Experiment 5.2 (dx={dx}, dy={dy}, T={T}, M={M})...")

    # 1. Build the specific Transition and Observation Matrices
    A_mat = np.array([[0.42 ** (abs(i - j) + 1) for j in range(dx)] for i in range(dx)], dtype=np.float32)

    C_mat = np.zeros((dy, dx), dtype=np.float32)
    C_mat[0, 0] = 1.0  # I_{dy, dx} structure

    Q_mat = np.eye(dx, dtype=np.float32)
    R_mat = np.eye(dy, dtype=np.float32)

    # 2. Generate True Data using LGSSM
    lgssm = LGSSM(dx, dy, params=[A_mat, C_mat, Q_mat, R_mat, np.zeros((dx, 1)), np.eye(dx)])
    _, batch_y = lgssm.batch_sample(T, M)  # Shape: [M, T, dy]

    # 3. Setup the Filter Model (LearnableSSM)
    transition_fn = lambda x: tf.matmul(x, tf.constant(A_mat, dtype=tf.float32), transpose_b=True)
    observation_fn = lambda x: tf.matmul(x, tf.constant(C_mat, dtype=tf.float32), transpose_b=True)

    proposal_net = ProposalNetwork(dx, dy, A_mat)
    init_noise_dist = tfd.MultivariateNormalDiag(scale_diag=tf.ones(dx))

    filter_model = LearnableSSM(
        state_dim=dx, obs_dim=dy,
        transition_layers=transition_fn,
        observation_layers=observation_fn,
        proposal_layers=proposal_net,
        init_noise=init_noise_dist,
        learn_noise=False,
        learn_init_state=False,
        init_noise_scale=1.0  # Forces Q and R strictly to Identity
    )

    # 4. Instantiate DPF with N=25
    dpf = DifferentiableParticleFilter(
        model=filter_model,
        num_particles=25,
        epsilon=0.5,
        # CORRECTED: Use SGD with learning rate 0.1
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, clipnorm=1.0)
    )

    # 5. Training Loop
    epochs = 100
    print(f"\nStarting SGD optimization for {epochs} steps...")
    print("Initial log_phi values (first 5):", proposal_net.log_phi.numpy()[:5])

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Enforce DET resampling during the forward pass
            dpf.use_differentiable_resample = True

            loss = 0.0
            ess_batch_total = 0.0

            # CORRECTED: Average the loss and ESS over 4 independent filters
            for _ in range(4):
                _, _, ess_batch, log_l = dpf.filter_summarized(batch_y)
                loss += -tf.reduce_mean(log_l) / 4.0
                ess_batch_total += ess_batch / 4.0

        # Apply gradients exclusively to the proposal network's phi variable
        grads = tape.gradient(loss, proposal_net.trainable_variables)
        dpf.optimizer.apply_gradients(zip(grads, proposal_net.trainable_variables))

        # Extract the actual phi values for metrics
        current_phi = tf.exp(proposal_net.log_phi)
        # RMSE of phi against the optimal vector of 1s
        rmse = tf.sqrt(tf.reduce_mean(tf.square(current_phi - 1.0)))

        # Average ESS percentage
        avg_ess = tf.reduce_mean(ess_batch_total)
        ess_pct = (avg_ess / 25.0) * 100

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Step {epoch + 1:03d} | Loss: {loss:.2f} | φ RMSE: {rmse:.4f} | Avg ESS: {ess_pct:.1f}%")

    print("\nFinal phi values:  ", tf.exp(proposal_net.log_phi).numpy())

if __name__ == "__main__":
    # Ensure memory doesn't blow up for the batched operations
    tf.config.optimizer.set_jit(True)
    #run_section_5_1_experiment()
    run_section_5_2_experiment()