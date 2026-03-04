import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import wishart
from scipy.linalg import fractional_matrix_power

# ==========================================
# 1. High-Dimensional Distribution Setup
# ==========================================
BATCH_SIZE = 512
ITERATIONS = 1500  # Adjusted for execution time, can be increased for convergence
TEST_SAMPLES = 1000


def generate_distributions(d):
    """Generates source (random) and target (standard) Gaussian parameters."""
    # Source: Mean from standard normal, Covariance from Wishart(d+1, I)
    mu_0 = np.random.randn(d).astype(np.float32)
    cov_0 = wishart.rvs(df=d + 1, scale=np.eye(d)).astype(np.float32)

    # Target: Standard normal
    mu_1 = np.zeros(d, dtype=np.float32)
    cov_1 = np.eye(d, dtype=np.float32)

    # Precompute inverses and log determinants for stability
    inv_cov_0 = np.linalg.inv(cov_0).astype(np.float32)
    inv_cov_1 = np.linalg.inv(cov_1).astype(np.float32)

    sign_0, logdet_0 = np.linalg.slogdet(cov_0)
    sign_1, logdet_1 = np.linalg.slogdet(cov_1)

    return mu_0, cov_0, inv_cov_0, logdet_0, mu_1, cov_1, inv_cov_1, logdet_1


def exact_whitening_transform(X, mu_0, cov_0):
    """Ground truth OT map to standard Gaussian: T(x) = (x - mu_0) * Sigma_0^{-1/2}"""
    cov0_inv_sqrt = fractional_matrix_power(cov_0, -0.5).real.astype(np.float32)
    return np.dot(X - mu_0, cov0_inv_sqrt)


# ==========================================
# 2. Architectures (Dynamically Scaled)
# ==========================================

class mGradNet_C(tf.keras.Model):
    def __init__(self, dim, hidden_dim, L=3):
        super().__init__()
        self.L = L
        self.W = self.add_weight(shape=(hidden_dim, dim), initializer="glorot_normal", trainable=True)
        self.alpha = [self.add_weight(shape=(1, hidden_dim), initializer="ones", trainable=True) for _ in range(L)]
        self.beta = [self.add_weight(shape=(1, hidden_dim), initializer="ones", trainable=True) for _ in range(L)]
        self.b = [self.add_weight(shape=(1, hidden_dim), initializer="zeros", trainable=True) for _ in range(L)]
        self.b_out = self.add_weight(shape=(1, dim), initializer="zeros", trainable=True)

    def call(self, x):
        W_x = tf.matmul(x, self.W, transpose_b=True)
        z = tf.nn.softplus(self.beta[0]) * W_x + self.b[0]
        for i in range(1, self.L):
            z = tf.nn.softplus(self.beta[i]) * W_x + tf.nn.softplus(self.alpha[i - 1]) * tf.math.tanh(z) + self.b[i]
        out = tf.matmul(tf.nn.softplus(self.alpha[-1]) * tf.math.tanh(z), self.W) + self.b_out
        return out


class mGradNet_M(tf.keras.Model):
    def __init__(self, dim, hidden_dim, M=4):
        super().__init__()
        self.M_modules = M
        self.W_m = [self.add_weight(shape=(hidden_dim, dim), initializer="glorot_normal", trainable=True) for _ in range(M)]
        self.b_m = [self.add_weight(shape=(1, hidden_dim), initializer="zeros", trainable=True) for _ in range(M)]
        self.c_m = [self.add_weight(shape=(1, 1), initializer="ones", trainable=True) for _ in range(M)]
        self.a = self.add_weight(shape=(1, dim), initializer="zeros", trainable=True)
        # Temperature scaling factor t based on dimension
        self.t = tf.sqrt(tf.cast(dim, tf.float32))

    def call(self, x):
        out = self.a
        for i in range(self.M_modules):
            z_m = tf.matmul(x, self.W_m[i], transpose_b=True) + self.b_m[i]
            # Apply the temperature scaling t to prevent softmax saturation
            sigma_m = tf.nn.softmax(z_m / self.t, axis=-1)
            out += tf.nn.softplus(self.c_m[i]) * tf.matmul(sigma_m, self.W_m[i])
        return out


# ==========================================
# 3. Training & Evaluation Pipeline
# ==========================================

def evaluate_dimension(d):
    print(f"\n--- Evaluating Dimension: {d} ---")

    # 1. Setup Data
    mu_0, cov_0, inv_cov_0, logdet_0, mu_1, cov_1, inv_cov_1, logdet_1 = generate_distributions(d)

    # 2. Initialize Models (Scaling hidden capacity with dimension)
    hidden_dim = max(32, 4 * d)
    model_c = mGradNet_C(dim=d, hidden_dim=hidden_dim)
    model_m = mGradNet_M(dim=d, hidden_dim=hidden_dim)

    model_c.build(input_shape=(None, d))
    model_m.build(input_shape=(None, d))

    # 3. Training Function Factory (Captures distribution stats for this dimension)
    def train_routine(model, name):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        optimizer.build(model.trainable_variables)

        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(x)
                    y = model(x)

                J = inner_tape.batch_jacobian(y, x)

                # Numerically stable log determinant for high-D
                sign, log_det_J = tf.linalg.slogdet(J)

                # Log PDFs
                diff_0 = x - mu_0
                quad_0 = tf.reduce_sum(diff_0 * tf.matmul(diff_0, inv_cov_0), axis=1)
                log_p = -0.5 * quad_0 - 0.5 * logdet_0 - (d / 2) * tf.math.log(2 * np.pi)

                diff_1 = y - mu_1
                quad_1 = tf.reduce_sum(diff_1 * tf.matmul(diff_1, inv_cov_1), axis=1)
                log_q = -0.5 * quad_1 - 0.5 * logdet_1 - (d / 2) * tf.math.log(2 * np.pi)

                loss = tf.reduce_mean(tf.square(log_det_J - (log_p - log_q)))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # Training Loop
        print(f"Training {name}...")
        for i in range(ITERATIONS):
            x_batch = np.random.multivariate_normal(mu_0, cov_0, BATCH_SIZE).astype(np.float32)
            train_step(x_batch)

        # Evaluation
        X_test = np.random.multivariate_normal(mu_0, cov_0, TEST_SAMPLES).astype(np.float32)
        Y_pred = model(X_test).numpy()
        Y_true = exact_whitening_transform(X_test, mu_0, cov_0)

        mse = np.mean(np.square(Y_pred - Y_true))
        print(f"  Final MSE: {mse:.6f}")
        return mse

    mse_c = train_routine(model_c, "mGradNet-C")
    mse_m = train_routine(model_m, "mGradNet-M")

    return mse_c, mse_m


# ==========================================
# 4. Run Experiment & Plot Figure 2
# ==========================================
dimensions = [2, 4, 8, 16, 32, 64]
mse_results_c = []
mse_results_m = []

for d in dimensions:
    mse_c, mse_m = evaluate_dimension(d)
    mse_results_c.append(mse_c)
    mse_results_m.append(mse_m)

# Plotting to match Figure 2
x_pos = np.arange(len(dimensions))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x_pos - width / 2, mse_results_c, width, label='mGradNet-C')
ax.bar(x_pos + width / 2, mse_results_m, width, label='mGradNet-M')

ax.set_yscale('log')  # Log scale as in the paper
ax.set_ylabel('MSE')
ax.set_xlabel('Dimension')
ax.set_title('MSE between Learned and Optimal OT Maps')
ax.set_xticks(x_pos)
ax.set_xticklabels(dimensions)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()