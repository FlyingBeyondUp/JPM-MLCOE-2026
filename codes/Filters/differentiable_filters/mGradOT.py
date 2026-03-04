import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

# ==========================================
# 1. Setup: Distributions and Ground Truth
# ==========================================
DIM = 2
BATCH_SIZE = 1000
ITERATIONS = 2000

# Source Skewed Gaussian
MU_0 = np.array([-0.5, -0.5], dtype=np.float32)
COV_0 = np.array([[2.0, 1.5], [1.5, 2.0]], dtype=np.float32)

# Target Standard Gaussian
MU_1 = np.array([0.0, 0.0], dtype=np.float32)
COV_1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

INV_COV_0 = np.linalg.inv(COV_0)
INV_COV_1 = np.linalg.inv(COV_1)
DET_COV_0 = np.linalg.det(COV_0)
DET_COV_1 = np.linalg.det(COV_1)

# Whitening Transform (Ground Truth OT Map)
# T(x) = Sigma_0^{-1/2} (x - mu_0)
cov0_inv_sqrt = fractional_matrix_power(COV_0, -0.5).real.astype(np.float32)


def whitening_transform(x):
    return np.dot(x - MU_0, cov0_inv_sqrt)


# Log PDF functions
def log_pdf_source(x):
    diff = x - MU_0
    quad = tf.reduce_sum(diff * tf.matmul(diff, INV_COV_0), axis=1)
    return -0.5 * quad - 0.5 * np.log(DET_COV_0) - DIM / 2 * np.log(2 * np.pi)


def log_pdf_target(y):
    diff = y - MU_1
    quad = tf.reduce_sum(diff * tf.matmul(diff, INV_COV_1), axis=1)
    return -0.5 * quad - 0.5 * np.log(DET_COV_1) - DIM / 2 * np.log(2 * np.pi)


# ==========================================
# 2. Architectures
# ==========================================

# Baseline: Standard MLP
def build_baseline():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh', input_shape=(DIM,)),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(DIM)
    ])


# mGradNet-C: Cascaded Gradient Network
class mGradNet_C(tf.keras.Model):
    def __init__(self, dim=2, hidden_dim=7, L=3):
        super().__init__()
        self.L = L
        self.W = self.add_weight(shape=(hidden_dim, dim), initializer="random_normal", trainable=True)
        self.alpha = [self.add_weight(shape=(1, hidden_dim), initializer="ones", trainable=True) for _ in range(L)]
        self.beta = [self.add_weight(shape=(1, hidden_dim), initializer="ones", trainable=True) for _ in range(L)]
        self.b = [self.add_weight(shape=(1, hidden_dim), initializer="zeros", trainable=True) for _ in range(L)]
        self.b_out = self.add_weight(shape=(1, dim), initializer="zeros", trainable=True)

    def call(self, x):
        W_x = tf.matmul(x, self.W, transpose_b=True)
        # z_0 = beta_0 * Wx + b_0
        z = tf.nn.softplus(self.beta[0]) * W_x + self.b[0]
        # Hidden layers
        for i in range(1, self.L):
            z = tf.nn.softplus(self.beta[i]) * W_x + tf.nn.softplus(self.alpha[i - 1]) * tf.math.tanh(z) + self.b[i]
        # Output layer: W^T [alpha_L * sigma(z_{L-1})] + b_L
        out = tf.matmul(tf.nn.softplus(self.alpha[-1]) * tf.math.tanh(z), self.W) + self.b_out
        return out


# mGradNet-M: Modular Gradient Network
class mGradNet_M(tf.keras.Model):
    def __init__(self, dim=2, hidden_dim=7, M=4):
        super().__init__()
        self.M_modules = M
        self.W_m = [self.add_weight(shape=(hidden_dim, dim), initializer="random_normal", trainable=True) for _ in
                    range(M)]
        self.b_m = [self.add_weight(shape=(1, hidden_dim), initializer="zeros", trainable=True) for _ in range(M)]
        self.c_m = [self.add_weight(shape=(1, 1), initializer="ones", trainable=True) for _ in range(M)]
        self.a = self.add_weight(shape=(1, dim), initializer="zeros", trainable=True)

    def call(self, x):
        out = self.a
        for i in range(self.M_modules):
            z_m = tf.matmul(x, self.W_m[i], transpose_b=True) + self.b_m[i]
            sigma_m = tf.nn.softmax(z_m, axis=-1)
            # Conical combination: T(x) += c_m * W_m^T softmax(W_m x + b_m)
            out += tf.nn.softplus(self.c_m[i]) * tf.matmul(sigma_m, self.W_m[i])
        return out


# ==========================================
# 3. Training Routine (Monge-Ampere Loss)
# ==========================================

def train_model(model, name="Model"):
    # 1. Explicitly build the model to initialize all weights
    model.build(input_shape=(None, DIM))

    # 2. Instantiate and explicitly build the optimizer's variables
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005,clipnorm=10)
    optimizer.build(model.trainable_variables)

    # 3. Define the tf.function as a closure so it captures the model and optimizer
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(x)
                y = model(x)

            # Compute Jacobian J_T(x)
            J = inner_tape.batch_jacobian(y, x)

            # Compute log |det J_T(x)|
            # det_J = tf.linalg.det(J)
            # log_det_J = tf.math.log(tf.math.abs(det_J) + 1e-6)
            sign, log_det_J = tf.linalg.slogdet(J)

            # Compute log p(x) and log q(y)
            log_p = log_pdf_source(x)
            log_q = log_pdf_target(y)

            # Monge-Ampere Loss
            loss = tf.reduce_mean(tf.square(log_det_J - (log_p - log_q)))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print(f"Training {name}...")
    for i in range(ITERATIONS):
        x_batch = np.random.multivariate_normal(MU_0, COV_0, BATCH_SIZE).astype(np.float32)
        loss = train_step(x_batch)
        if i % 500 == 0:
            print(f"  Step {i}, Loss: {loss.numpy():.4f}")


# ==========================================
# 4. Execution and Plotting
# ==========================================

# Instantiate and Train
baseline_model = build_baseline()
mgradnet_c_model = mGradNet_C()
mgradnet_m_model = mGradNet_M()

train_model(baseline_model, "Baseline")
train_model(mgradnet_c_model, "mGradNet-C")
train_model(mgradnet_m_model, "mGradNet-M")

# Generate Test Data
X_test = np.random.multivariate_normal(MU_0, COV_0, 1000).astype(np.float32)
color_metric = X_test[:, 0] + X_test[:, 1]  # Used to track point displacements

# Get mapped predictions
Y_baseline = baseline_model(X_test).numpy()
Y_c = mgradnet_c_model(X_test).numpy()
Y_m = mgradnet_m_model(X_test).numpy()
Y_whitening = whitening_transform(X_test)

# Plotting to match Figure 1
fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)

titles = ["(a) Source Samples", "(b) Baseline", "(c) mGradNet-C", "(d) mGradNet-M", "(e) Whitening Transform"]
data = [X_test, Y_baseline, Y_c, Y_m, Y_whitening]

for i, ax in enumerate(axs):
    ax.scatter(data[i][:, 0], data[i][:, 1], c=color_metric, cmap='rainbow', s=5, alpha=0.8)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_title(titles[i], y=-0.2,fontsize=18)
    ax.grid(True)

# plt.suptitle(
#     "OT from skewed to standard Gaussian. Points are colored based on their positions in the source distribution.",
#     y=0.05, fontsize=14)
plt.subplots_adjust(bottom=0.25)
#plt.savefig("Gaussian_OT.pdf", bbox_inches='tight')
plt.show()