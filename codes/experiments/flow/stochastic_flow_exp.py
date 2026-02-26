import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from codes.Filters.flow_filters import StochasticFlow
from codes.Filters.flow_filters import EDHFlow
from codes.models import NLSSM
from codes.Filters.basic_filters import ParticleFilter

tfd = tfp.distributions

def get_exponential_schedule(num_steps: int = 29, q: float = 1.2):
    if num_steps == 1:
        return tf.constant([1.0], dtype=tf.float32)

    numerator = q - 1.0
    denominator = tf.pow(q, float(num_steps)) - 1.0
    epsilon_1 = numerator / denominator
    exponents = tf.range(num_steps, dtype=tf.float32)
    step_sizes = epsilon_1 * tf.pow(q, exponents)

    return step_sizes

def run_exp1():
    x0 = tf.constant([3.0, 5.0], dtype=tf.float32)
    P0 = tf.linalg.diag([100.0,2.0])
    P0 = tf.expand_dims(P0, 0)
    Q=tf.linalg.diag([2,0.4])

    x_true = tf.constant([4.0, 4.0], dtype=tf.float32)

    def angle_observation_fn(state, noise):
        # State is [batch, 2] or [2]
        x = state[..., 0]
        y = state[..., 1]
        # Sensor 1 at [3.5, 0]， Sensor 2 at [-3.5, 0]
        z1 = tf.math.atan2(y - 0.0, x - 3.5)
        z2 = tf.math.atan2(y - 0.0, x - (-3.5))

        z = tf.stack([z2, z1], axis=-1)
        return z + noise

    model = NLSSM(
        state_dim=2,
        obs_dim=2,
        transition_fn=lambda s, v: s,  # Identity transition
        observation_fn=angle_observation_fn,
        process_noise=tfd.MultivariateNormalDiag(loc=tf.zeros(2), scale_diag=tf.zeros(2)),
        observation_noise=tfd.MultivariateNormalDiag(loc=tf.zeros(2), scale_diag=0.2*tf.ones(2)),
        # Note below
        init_noise=tfd.MultivariateNormalDiag(loc=tf.zeros(2), scale_diag=tf.zeros(2)),
        x0=x0
    )
    base_flow=StochasticFlow(model=model,num_particles=50,stiffness_weight=0,
                             diffusion_q=tf.linalg.diag([2.0,tf.sqrt(0.4)]),diffusion_dim=2,requires_stiffness=True)
    # base_flow = StochasticFlow(model=model, num_particles=200, stiffness_weight=0,
    #                            diffusion_q=None, getQ=lambda Hh, Hp_inv: tf.linalg.cholesky(Hp_inv @ Hh @ Hp_inv),
    #                            diffusion_dim=2, requires_stiffness=True)
    optimal_flow=StochasticFlow(model=model,num_particles=50,stiffness_weight=0.2,
                                diffusion_q=tf.linalg.diag([2.0,tf.sqrt(0.4)]), diffusion_dim=2,requires_stiffness=True)
    EDH_flow = EDHFlow(model=model,num_particles=50)

    observation=tf.constant([0.4754,1.1868])
    observation=tf.expand_dims(observation,0) # [1,2] for [B,Obs]

    _, Hh = base_flow.getObsHessian(x_true[tf.newaxis,:])
    print("Hh eigenvalues:", tf.linalg.eigvalsh(Hh).numpy())

    base_flow.update_path(num_flow_steps=1000,P_xx=P0,Hh=Hh)
    optimal_flow.update_path(num_flow_steps=1000,P_xx=P0,Hh=Hh)
    base_path=base_flow.betas
    optimal_path=tf.reshape(optimal_flow.betas,(-1,))
    optimal_u=tf.reshape(optimal_flow.beta_dots,(-1,))
    print(optimal_path)
    print(optimal_u)

    scale_diag = tf.sqrt(tf.constant([100.0,2.0], dtype=tf.float32))
    particles = tfd.MultivariateNormalDiag(
        loc=x0,
        scale_diag=scale_diag
    ).sample(50)
    particles = tf.expand_dims(particles, 0) # [1,50,2] for [B,N,D]


    _, x_filt_opt, P_filt_opt = optimal_flow._flow_update(observation, particles, num_flow_steps=1000, P_xx=P0)
    _,x_filt_base,P_filt_base=base_flow._flow_update(observation,particles,num_flow_steps=1000,P_xx=P0)


    _,x_filt_edh,P_filt_edh=EDH_flow._flow_update(observation,particles,num_flow_steps=200,P_xx=P0)
    # base_stiffness=[stiffness[0] for stiffness in base_flow.stiffness_records]
    # optimal_stiffness=[stiffness[0] for stiffness in optimal_flow.stiffness_records]
    print('Filtered state and covariance at lambda=1:')
    print('Base flow: x_filt =', x_filt_base.numpy(), ', P_filt =', P_filt_base.numpy())
    print('Optimal flow: x_filt =', x_filt_opt.numpy(), ', P_filt =', P_filt_opt.numpy())
    print('EDH flow: x_filt =', x_filt_edh.numpy(), ', P_filt =', P_filt_edh.numpy())
    print('\n')

    plt.figure()
    plt.plot(tf.linspace(0,1,len(base_path)),base_path,label='Base Flow',linestyle='--',color='blue')
    plt.plot(tf.linspace(0,1,len(optimal_path)),optimal_path,label='Optimal Flow',color='red')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\beta(\lambda)$')
    plt.legend()

    plt.figure()
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$e^*=\beta(\lambda)-\lambda$')
    plt.plot(tf.linspace(0,1,len(optimal_path)),optimal_path-base_path,label='Optimal Flow',color='blue')

    plt.figure()
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$u^*$')
    plt.plot(tf.linspace(0,1,len(optimal_u)),optimal_u,label='Optimal Control',color='blue')

    # plt.figure()
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$R$')
    # plt.yscale('log')
    # plt.plot(tf.linspace(0,1,len(optimal_stiffness)),optimal_stiffness,label='Optimal Flow',color='red')
    # plt.plot(tf.linspace(0,1,len(base_stiffness)),base_stiffness,label='Base Flow',color='blue')
    # plt.legend()

    # Hg = -tf.linalg.inv(P0 + 1e-6 * tf.eye(2))
    # Mh = -Hh  # positive definite [1, D, D]
    # M0 = tf.linalg.inv(P0 + 1e-6 * tf.eye(2))  # [1, D, D]
    #
    # num_points = 400
    # base_stiffness = []
    # optimal_stiffness = []
    # for i in range(num_points):
    #     lam = (i + 1) / num_points  # avoid lambda=0 where M0 is very ill-conditioned
    #
    #     # Base flow: beta = lambda
    #     beta_base = lam
    #     M_base = M0 + beta_base * Mh
    #     M_base_inv = tf.linalg.inv(M_base)
    #     F_base = -1/2*(Q@M_base+M_base_inv @ Mh)
    #     eig_base = tf.abs(tf.math.real(tf.linalg.eigvals(tf.cast(F_base, tf.complex64))))
    #     R_base = tf.reduce_max(eig_base) / tf.maximum(tf.reduce_min(eig_base), 1e-10)
    #     base_stiffness.append(R_base.numpy())
    #
    #     # Optimal flow: beta = beta*(lambda)
    #     beta_opt, u_opt = optimal_flow._get_beta(lam, B=1)
    #     M_opt = M0 + beta_opt * Mh
    #     M_opt_inv = tf.linalg.inv(M_opt)
    #     F_opt = -1/2*(Q@M_opt+u_opt*M_opt_inv @ Mh)
    #     eig_opt = tf.abs(tf.math.real(tf.linalg.eigvals(tf.cast(F_opt, tf.complex64))))
    #     R_opt = tf.reduce_max(eig_opt) / tf.maximum(tf.reduce_min(eig_opt), 1e-10)
    #     optimal_stiffness.append(R_opt.numpy())
    #
    # lambdas = [(i + 1) / num_points for i in range(num_points)]
    # plt.figure()
    # plt.yscale('log')
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$R_{stiff}$')
    # plt.plot(lambdas, base_stiffness, label=r'$\beta(\lambda)=\lambda$',
    #          color='blue', linestyle='--')
    # plt.plot(lambdas, optimal_stiffness, label=r'optimal $\beta^*(\lambda)$',
    #          color='red')
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    run_exp1()
