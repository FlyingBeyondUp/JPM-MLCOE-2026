import tensorflow as tf
import tensorflow_probability as tfp
import time
import matplotlib.pyplot as plt
import math

tfd = tfp.distributions

class LGSSM:
    def __init__(self,state_dim,obs_dim,params=None):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        if params is None:
            A0=tf.eye(state_dim)*0.95  # State transition matrix
            A0+=tf.random.normal([state_dim,state_dim],stddev=0.1,dtype=tf.float32)
            s=tf.linalg.svd(A0,compute_uv=False)
            scale=tf.minimum(1,1/(s[0]+1e-9)) # Ensure stability
            self.A=tf.Variable(A0*tf.cast(scale,tf.float32))
            self.C=tf.Variable(tf.random.normal([obs_dim,state_dim],stddev=1.0,dtype=tf.float32))  # Observation matrix
            self.Q=tf.Variable(tf.eye(state_dim)*0.2)  # Process noise covariance
            self.R=tf.Variable(tf.eye(obs_dim)*0.1)    # Observation noise covariance
            self.x0=tf.Variable(tf.zeros([state_dim,1],dtype=tf.float32))  # Initial state mean
            self.P0=tf.Variable(tf.eye(state_dim)*1.0)  # Initial state covariance
        else:
            self.A, self.C, self.Q, self.R, self.x0, self.P0 = [
                p if isinstance(p, tf.Variable) else tf.Variable(p) for p in params
            ]
        self.params = [self.A, self.C, self.Q, self.R, self.x0, self.P0]
        self.LQ = tf.linalg.cholesky(self.Q)
        self.LR = tf.linalg.cholesky(self.R)
        self.LP0 = tf.linalg.cholesky(self.P0)

    def sample(self,T):
        '''
        Generate a sample sequence of length T
        :param T: The length of the sequence
        :return: sequence of states x and observations y
                 with shape [T,state_dim] and [T,obs_dim]
        '''
        x = tf.TensorArray(dtype=tf.float32, size=T)
        y = tf.TensorArray(dtype=tf.float32, size=T)

        x_t = self.x0 + self.LP0 @ tf.random.normal([self.state_dim,1],dtype=tf.float32)

        for t in range(T):
            process_noise =  self.LQ@ tf.random.normal([self.state_dim,1],dtype=tf.float32)
            x_t = self.A @ x_t + process_noise

            observation_noise = self.LR @ tf.random.normal([self.obs_dim,1],dtype=tf.float32)
            y_t = self.C @ x_t + observation_noise

            x = x.write(t, x_t)
            y = y.write(t, y_t)
        x,y=x.stack(),y.stack()
        return tf.squeeze(x,axis=-1),tf.squeeze(y,axis=-1)

    def batch_sample(self,T,batch_size):
        '''
        :param T: The length of the sequences
        :return: x: [batch_size,T,state_dim], y: [batch_size,T,obs_dim]
        '''
        return tf.vectorized_map(lambda _: self.sample(T), tf.range(batch_size))

    def get_params(self):
        return self.params

    def update_cholesky(self):
        self.LQ = tf.linalg.cholesky(self.Q)
        self.LR = tf.linalg.cholesky(self.R)
        self.LP0 = tf.linalg.cholesky(self.P0)

    def set_params(self, params, update_cholesky=True):
        new_A, new_C, new_Q, new_R, new_x0, new_P0 = params

        self.A.assign(new_A)
        self.C.assign(new_C)
        self.Q.assign(new_Q)
        self.R.assign(new_R)
        self.x0.assign(new_x0)
        self.P0.assign(new_P0)

        if update_cholesky:
            self.update_cholesky()



class KalmanFilter:
    def __init__(self,model:LGSSM):
        self.model=model

    def filter(self,y,T=None,requires_stabilization=True):
        '''
        :param y: [B,T,obs_dim]
        :requires_stablization: Whether to use Joseph form for numerical stability
        :return: x_filtered: [B,T,state_dim], P_filtered: [B,T,state_dim,state_dim]

        Implement the Kalman filter algorithm to estimate the hidden states
        x_t_pred=A*x_{t-1}_filt
        P_t_pred=A*P_{t-1}_filt*A^T+Q

        S_t=C*P_t_pred*C^T+R
        K_t=P_t_pred*C^T*S_t^{-1} # Kalman gain

        x_t_filt=x_t_pred+K_t*(y_t-C*x_t_pred)
        Standard form: P_t_filt=(I-K_t*C)*P_t_pred
        Joseph form: P_t_filt=(I-K_t*C)*P_t_pred*(I-K_t*C)^T+K_t*R*K_t^T

        The returned x_filtered and P_filtered contain the filtered state means and covariances at each time step.
        '''
        state_dim=self.model.state_dim
        obs_dim=self.model.obs_dim

        rank = len(y.shape)
        if rank == 2: # y shape: [T, Obs]
            is_batched = False
            y_in = tf.expand_dims(y, 0)  # [1, T, Obs]
        elif rank == 3:
            is_batched = True
            y_in = y
        elif rank==1:
            is_batched = False
            y_in = tf.expand_dims(tf.expand_dims(y, 0), -1)  # [1, T, 1]
        else:
            raise ValueError(f"y must be rank 1, 2 or 3, got {rank}")

        batch_size = tf.shape(y_in)[0]
        if T is None:
            T = tf.shape(y_in)[1]

        A = tf.expand_dims(self.model.A, 0)
        C = tf.expand_dims(self.model.C, 0)
        Q = tf.expand_dims(self.model.Q, 0)
        R = tf.expand_dims(self.model.R, 0)
        I = tf.eye(state_dim, dtype=tf.float32)

        # Initial State [B, S, 1], [B, S, S]
        # Ensure x0 and P0 have correct rank before expanding and tiling
        x0_reshaped = tf.reshape(self.model.x0, [state_dim, 1])
        P0_reshaped = tf.reshape(self.model.P0, [state_dim, state_dim])
        x_t = tf.tile(tf.expand_dims(x0_reshaped, 0), [batch_size, 1, 1])
        P_t = tf.tile(tf.expand_dims(P0_reshaped, 0), [batch_size, 1, 1])

        x_pred_ta = tf.TensorArray(tf.float32, size=T)
        P_pred_ta = tf.TensorArray(tf.float32, size=T)
        x_filt_ta = tf.TensorArray(tf.float32, size=T)
        P_filt_ta = tf.TensorArray(tf.float32, size=T)

        log_l = tf.zeros([batch_size], dtype=tf.float32)
        const_term=-0.5*obs_dim*tf.math.log(2*math.pi)
        for t in range(T):
            # Prediction
            if t == 0:
                x_t_pred = x_t
                P_t_pred = P_t
            else:
                x_t_pred = A @ x_t
                P_t_pred = A @ P_t @ tf.transpose(A, perm=[0, 2, 1]) + Q

            # Update
            y_t = tf.expand_dims(y_in[:, t, :], -1)  # [B, O, 1]
            innov = y_t - C @ x_t_pred

            PCt = P_t_pred @ tf.transpose(C, perm=[0, 2, 1])
            S_t = C @ PCt + R + 1e-6 * tf.eye(obs_dim)

            S_chol = tf.linalg.cholesky(S_t)
            # K = P C^T S^-1 => K^T = S^-T C P^T
            K_t = tf.transpose(tf.linalg.cholesky_solve(S_chol, tf.transpose(PCt, perm=[0, 2, 1])), perm=[0, 2, 1])
            x_t = x_t_pred + K_t @ innov

            I_KC = I - K_t @ C
            if requires_stabilization:
                P_t = I_KC @ P_t_pred @ tf.transpose(I_KC, perm=[0, 2, 1]) + K_t @ R @ tf.transpose(K_t, perm=[0, 2, 1])
            else:
                P_t = I_KC @ P_t_pred

                # Likelihood
            log_det_S = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(S_chol)), axis=-1)
            quad_term = tf.squeeze(tf.transpose(innov, perm=[0, 2, 1]) @ tf.linalg.cholesky_solve(S_chol, innov),
                                   axis=[-2, -1])
            log_l += (const_term - 0.5 * log_det_S - 0.5 * quad_term)

            x_pred_ta = x_pred_ta.write(t, x_t_pred)
            P_pred_ta = P_pred_ta.write(t, P_t_pred)
            x_filt_ta = x_filt_ta.write(t, x_t)
            P_filt_ta = P_filt_ta.write(t, P_t)

        x_filt = tf.transpose(x_filt_ta.stack(), perm=[1, 0, 2, 3])
        P_filt = tf.transpose(P_filt_ta.stack(), perm=[1, 0, 2, 3])
        x_pred = tf.transpose(x_pred_ta.stack(), perm=[1, 0, 2, 3])
        P_pred = tf.transpose(P_pred_ta.stack(), perm=[1, 0, 2, 3])

        # Squeeze last dim of x [B, T, S]
        x_filt = tf.squeeze(x_filt, axis=-1)
        x_pred = tf.squeeze(x_pred, axis=-1)

        if not is_batched:
            return x_filt[0], P_filt[0], x_pred[0], P_pred[0], log_l[0]
        else:
            return x_filt, P_filt, x_pred, P_pred, log_l

    def smooth_filter(self, y, T=None, stabilization=True):
        """
        Batched RTS Smoother.
        """
        # 1. Run Filter
        x_filt, P_filt, x_pred, P_pred, log_l = self.filter(y, T, stabilization)

        # 2. Check shapes to determine if input was batched
        rank = len(x_filt.shape)
        if rank == 2:  # [T, S]
            is_batched = False
            # Expand to [1, T, S]
            x_filt_in = tf.expand_dims(x_filt, 0)
            P_filt_in = tf.expand_dims(P_filt, 0)
            x_pred_in = tf.expand_dims(x_pred, 0)
            P_pred_in = tf.expand_dims(P_pred, 0)
            batch_size = 1
        else:
            is_batched = True
            x_filt_in, P_filt_in, x_pred_in, P_pred_in = x_filt, P_filt, x_pred, P_pred
            batch_size = tf.shape(x_filt)[0]

        if T is None:
            T = tf.shape(x_filt_in)[1]

        state_dim = self.model.state_dim
        A = tf.expand_dims(self.model.A, 0)  # [1, S, S]

        # Prepare TensorArrays for smoothing
        x_smooth_ta = tf.TensorArray(tf.float32, size=T, clear_after_read=False)
        P_smooth_ta = tf.TensorArray(tf.float32, size=T, clear_after_read=False)
        J_ta = tf.TensorArray(tf.float32, size=max(1, T - 1))  # Size T-1

        # Initialize at T-1
        # x_filt_in: [B, T, S], P_filt_in: [B, T, S, S]
        x_last = tf.expand_dims(x_filt_in[:, T - 1, :], -1)
        P_last = P_filt_in[:, T - 1, :, :]

        x_smooth_ta = x_smooth_ta.write(T - 1, x_last)
        P_smooth_ta = P_smooth_ta.write(T - 1, P_last)

        x_smooth_next = x_last
        P_smooth_next = P_last

        # Backward Loop
        for t in tf.range(T - 2, -1, -1):
            P_filt_t = P_filt_in[:, t, :, :]  # [B, S, S]
            P_pred_next = P_pred_in[:, t + 1, :, :]  # [B, S, S]

            x_pred_next = tf.expand_dims(x_pred_in[:, t + 1, :], -1)  # [B, S, 1]
            x_filt_t = tf.expand_dims(x_filt_in[:, t, :], -1)  # [B, S, 1]

            # J_t = P_t|t A^T P_t+1|t^-1
            # Solve J_t^T = P_next^-1 (P_filt A^T)^T = P_next^-1 (A P_filt)
            # We use A @ P_filt directly because P_filt is symmetric.
            # Dimensions: A [1, S, S] @ P_filt_t [B, S, S] -> [B, S, S]

            P_pred_next_chol = tf.linalg.cholesky(P_pred_next + 1e-6 * tf.eye(state_dim))
            rhs = A @ P_filt_t

            J_t_T = tf.linalg.cholesky_solve(P_pred_next_chol, tf.transpose(rhs, perm=[0, 2, 1]))
            J_t = tf.transpose(J_t_T, perm=[0, 2, 1])

            dx = x_smooth_next - x_pred_next
            x_smooth_t = x_filt_t + J_t @ dx

            dP = P_smooth_next - P_pred_next
            P_smooth_t = P_filt_t + J_t @ dP @ tf.transpose(J_t, perm=[0, 2, 1])

            x_smooth_next = x_smooth_t
            P_smooth_next = P_smooth_t

            x_smooth_ta = x_smooth_ta.write(t, x_smooth_t)
            P_smooth_ta = P_smooth_ta.write(t, P_smooth_t)
            J_ta = J_ta.write(t, J_t)

        # Stack and Transpose
        x_smooth = tf.transpose(x_smooth_ta.stack(), perm=[1, 0, 2, 3])
        P_smooth = tf.transpose(P_smooth_ta.stack(), perm=[1, 0, 2, 3])
        J_ts = tf.transpose(J_ta.stack(), perm=[1, 0, 2, 3])  # [B, T-1, S, S]

        x_smooth = tf.squeeze(x_smooth, axis=-1)

        if not is_batched:
            return x_smooth[0], P_smooth[0], J_ts[0], log_l
        else:
            return x_smooth, P_smooth, J_ts, log_l

    def real_time_predict(self,y_past,steps_ahead):
        '''
        :param y_past: [T,obs_dim] past observations
        :param steps_ahead: number of steps to predict ahead
        :return: x_pred: [steps_ahead,state_dim], P_pred: [steps_ahead,state_dim,state_dim]
        '''
        ...


def EM_solver(kf: KalmanFilter, Y, max_iters=10, tol=1e-3, stabilization=True):
    # Y: [batch_size,T,obs_dim]

    T = int(tf.shape(Y)[1])
    list_log_l = []
    for i in range(max_iters):
        # E-step: run Kalman smoother to get expected sufficient statistics
        # X_smooth: [batch_size,T,state_dim]
        # P_smooth: [batch_size,T,state_dim,state_dim]
        # J_ts: [batch_size,T-1,state_dim,state_dim]
        X_s, P_s, J_s, log_L = kf.smooth_filter(Y, stabilization=stabilization)

        # Aggregate batch log-likelihoods
        total_log_L = tf.reduce_mean(log_L)  # Sum over batch
        print(f'EM Iteration {i}, Log Likelihood: {total_log_L.numpy():.4f}')

        list_log_l.append(total_log_L)
        if len(list_log_l) > 1 and abs(list_log_l[-1] - list_log_l[-2]) < tol:
            kf.model.update_cholesky()
            print(f'EM converged at iteration {i}')
            break

        Exx = P_s + tf.matmul(tf.expand_dims(X_s, -1), tf.expand_dims(X_s, -2))  # E[x_t x_t^T]
        Exx1 = tf.matmul(P_s[:, 1:, :, :], J_s, transpose_b=True) + tf.matmul(tf.expand_dims(X_s[:, 1:, :], -1),
                                                                              tf.expand_dims(X_s[:, :-1, :],
                                                                                             -2))  # E[x_t x_{t-1}^T]
        Sigma_xx = tf.reduce_sum(Exx[:, :-1, :, :], axis=[0, 1])  # sum_t E[x_t x_t^T] from t=0 to T-2 over batch
        Gamma_xx = Sigma_xx + tf.reduce_sum(Exx[:, -1, :, :], axis=0)  # sum_t E[x_t x_t^T] from t=0 to T-1 over batch
        Sigma_x1x1 = Gamma_xx - tf.reduce_sum(Exx[:, 0, :, :], axis=0)  # sum_t E[x_t x_t^T] from t=1 to T-1 over batch
        Sigma_xx1 = tf.reduce_sum(Exx1, axis=[0, 1])  # sum_t E[x_t x_{t-1}^T] from t=1 to T-1 over batch
        Gamma_yy = tf.reduce_sum(tf.matmul(tf.expand_dims(Y, -1), tf.expand_dims(Y, -2)),
                                 axis=[0, 1])  # sum_t y_t y_t^T from t=0 to T-1 over batch
        Gamma_yx = tf.reduce_sum(tf.matmul(tf.expand_dims(Y, -1), tf.expand_dims(X_s, -2)),
                                 axis=[0, 1])  # sum_t y_t x_t^T from t=0 to T-1 over batch

        # M-step: update model parameters using the expected sufficient statistics
        Sigma_xx_chol = tf.linalg.cholesky(Sigma_xx + 1e-6 * tf.eye(kf.model.state_dim, dtype=Sigma_xx.dtype))
        Sigma_x1x1_chol = tf.linalg.cholesky(Sigma_x1x1 + 1e-6 * tf.eye(kf.model.state_dim, dtype=Sigma_x1x1.dtype))

        A_new = tf.transpose(tf.linalg.cholesky_solve(Sigma_x1x1_chol, tf.transpose(Sigma_xx1)))
        C_new = tf.transpose(tf.linalg.cholesky_solve(Sigma_xx_chol, tf.transpose(Gamma_yx)))

        # The update of R and Q can be simplified by using the new A and C
        R_term = Gamma_yy - C_new @ tf.transpose(Gamma_yx) - Gamma_yx @ tf.transpose(
            C_new) + C_new @ Gamma_xx @ tf.transpose(C_new)
        R_new = R_term / (tf.cast(tf.shape(Y)[0] * T, tf.float32))
        R_new = 0.5 * (R_new + tf.transpose(R_new)) + 1e-6 * tf.eye(kf.model.obs_dim)

        # Robust Q update: (Sigma_x1x1 - A Sigma_xx1^T - Sigma_xx1 A^T + A Sigma_xx A^T) / N
        Q_term = Sigma_x1x1 - A_new @ tf.transpose(Sigma_xx1) - Sigma_xx1 @ tf.transpose(
            A_new) + A_new @ Sigma_xx @ tf.transpose(A_new)
        Q_new = Q_term / (tf.cast(tf.shape(Y)[0] * (T - 1), tf.float32))
        Q_new = 0.5 * (Q_new + tf.transpose(Q_new)) + 1e-6 * tf.eye(kf.model.state_dim)

        x0_new = tf.reduce_mean(X_s[:, 0, :], axis=0, keepdims=True)

        centered_x0 = X_s[:, 0, :] - x0_new  # [batch, state_dim]
        cov_means = tf.matmul(tf.expand_dims(centered_x0, -1),
                              tf.expand_dims(centered_x0, -2))  # [batch, state_dim, state_dim]

        # P0_new = Mean( Posterior_Covariance + Covariance_of_Means )
        P0_new = tf.reduce_mean(P_s[:, 0, :, :] + cov_means, axis=0)
        P0_new = 0.5 * (P0_new + tf.transpose(P0_new)) + 1e-6 * tf.eye(kf.model.state_dim)

        kf.model.set_params([A_new, C_new, Q_new, R_new,
                             tf.transpose(x0_new), P0_new])

    kf.model.update_cholesky()
    return list_log_l


def EM_initializer(Y,state_dim):
    '''
    :param Y: observations with shape [batch_size,T,obs_dim]
    :param state_dim: dimension of the hidden state
    :return: Kalman Filter initialized with parameters estimated from data

    For obs_dim>=state_dim:
    The initilization is done by PCA on the observations to estimate C, and R.
    C is initialized as the top principal components scaled by the square root of the singular values.
    R is initialized as the covariance of the residuals after projecting Y onto the subspace spanned by C.
    x0 is initialized as the mean of the observations projected onto the subspace spanned by C.
    '''
    Y_flat=tf.reshape(Y,[-1,Y.shape[-1]])
    S,U,V=tf.linalg.svd(Y_flat- tf.reduce_mean(Y_flat,axis=0,keepdims=True),full_matrices=False)
    top_s=tf.linalg.diag(tf.sqrt(S[:state_dim])) # [state_dim,state_dim]
    top_v=V[:,:state_dim] # [obs_dim,state_dim]
    C_init=tf.matmul(top_v,top_s) # [obs_dim,state_dim]

    residuals=Y_flat@(tf.eye(top_v.shape[0],dtype=Y.dtype)-tf.matmul(top_v,tf.transpose(top_v))) # [N,obs_dim]
    residual_variance=tf.math.reduce_variance(residuals,axis=0)
    R_init=tf.linalg.diag(residual_variance+1e-6)

    x0_init=tf.reshape(tf.reduce_mean(Y,axis=[0,1],keepdims=True),(1,-1)) @ top_v  # [1,state_dim]
    P0_init=tf.eye(state_dim,dtype=Y.dtype) * 1.0
    A_init=tf.eye(state_dim,dtype=Y.dtype)*0.9
    Q_init=tf.eye(state_dim,dtype=Y.dtype)*0.1
    model=LGSSM(state_dim,Y.shape[-1],params=[A_init,C_init,Q_init,R_init,tf.transpose(x0_init),P0_init])
    kf=KalmanFilter(model)
    return kf

def show_EM_example(num_trials=64,T=100,state_dim=3,obs_dim=3,max_iters=50,stabilization=True):
    data_generator=LGSSM(state_dim,obs_dim)
    X,Y=data_generator.batch_sample(T=T,batch_size=num_trials)
    kf=EM_initializer(Y,state_dim)
    #kf=KalmanFilter(LGSSM(state_dim,obs_dim))
    print('Successfully initialized EM Kalman Filter')

    list_log_l=EM_solver(kf,Y,max_iters,stabilization=stabilization)

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




def test_filter(num_trials=128,T=50,state_dim=3,obs_dim=3):
    model = LGSSM(state_dim, obs_dim)
    kf = KalmanFilter(model)

    X,Y=model.batch_sample(T=T,batch_size=num_trials)
    X_filt,_,_,_,_=kf.filter(Y)
    X_smooth,_,_,_=kf.smooth_filter(Y)


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

    pinvC = tf.linalg.pinv(model.C)               # shape [state_dim, obs_dim]
    X_from_Y = tf.einsum('so,bto->bts', pinvC, Y) # [B,T,state_dim]
    baseline_pinv_rmse = tf.sqrt(tf.reduce_mean(tf.square(X - X_from_Y)))

    print('\n')
    print(f'Filtering RMSE: {mean_rmse_filt.numpy():.4f} ± {std_rmse_filt.numpy():.4f}')
    print(f'Smoothing RMSE: {mean_rmse_smooth.numpy():.4f} ± {std_rmse_smooth.numpy():.4f}')
    print(f'State std (global): {std_X.numpy():.4f}')
    print(f'Normalized Filtering RMSE: {(mean_rmse_filt/std_X).numpy():.4f}')
    print(f'Baseline zero RMSE: {baseline_zero_rmse.numpy():.4f}')
    print(f'Baseline pinv(C) RMSE: {baseline_pinv_rmse.numpy():.4f}')

# eval_rmse_and_baselines(X, X_filt, X_smooth, Y, model)



def show_example(T=100,state_dim=2,obs_dim=1):
    model = LGSSM(state_dim, obs_dim)
    kf = KalmanFilter(model)

    x,y=model.sample(T=T)
    x_filt,_=kf.filter(y,T)[:2]
    x_smooth,_,_,_=kf.smooth_filter(y,T)

    time_axis=tf.range(T)

    plt.figure(figsize=(12,8))
    for d in range(state_dim):
        plt.subplot(state_dim,1,d+1)
        plt.plot(time_axis,y[:,d],label='Observations',color='gray',linestyle='None',marker='o',markersize=5,alpha=0.5)
        plt.plot(time_axis,x[:,d],label='True State',color='black')
        plt.plot(time_axis,x_filt[:,d],label='Filtered State',color='blue',linestyle='--')
        plt.plot(time_axis,x_smooth[:,d],label='Smoothed State',color='red',linestyle=':')
        plt.title(f'State Dimension {d+1}')
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
    plt.plot(filtered_means[:, 0], 'o:',linewidth=3, label='TFP Filter')
    plt.plot(tfp_smoothed_means[:, 0], 'y:', linewidth=3, label='TFP Smoother')

    plt.title("Kalman Filter vs Smoother (High Observation Noise Scenario)")
    plt.legend(fontsize=16)
    plt.xlabel("Time",fontsize=20)
    plt.ylabel("State 0",fontsize=20)
    plt.grid(True, alpha=0.3)

    # Calculate RMSE
    rmse_filt = tf.sqrt(tf.reduce_mean((true_states - custom_filt_x) ** 2))
    rmse_smooth = tf.sqrt(tf.reduce_mean((true_states - custom_smooth_x) ** 2))

    print(f"Filter RMSE:   {rmse_filt:.4f}")
    print(f"Smoother RMSE: {rmse_smooth:.4f}")
    print("If the yellow dotted line perfectly overlaps the red line, your custom implementation matches TFP.")
    plt.savefig('Figures/kalman_filter_smoother_comparison.pdf',bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    #test_filter(num_trials=128,T=50,state_dim=3,obs_dim=3)
    #show_example()
    #run_comparison_example()
    show_EM_example(num_trials=128,T=100,state_dim=3,obs_dim=3,max_iters=10,stabilization=False)











