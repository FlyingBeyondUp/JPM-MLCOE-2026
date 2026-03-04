import tensorflow as tf
import tensorflow_probability as tfp
from models import NLSSM,get1DStochasticVolModel
from Filters.basic_filters import ParticleFilter
import matplotlib.pyplot as plt
import numpy as np

tfd = tfp.distributions

def run_resampling_experiment(model: NLSSM, num_particles: int, x_true:tf.Tensor,y_obs: tf.Tensor,num_runs: int = 20):
    results={'multinomial':[], 'systematic':[]}
    for method in results.keys():
        pf = ParticleFilter(model=model, num_particles=num_particles, resample_method=method)
        estimates=[]
        for run in range(num_runs):
            x_filt, _,_,_ = pf.filter(y_obs)
            estimates.append(x_filt[0, :, 0].numpy())
        results[method]=np.array(estimates)
    x_true_np=x_true.numpy()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    var_mult = np.var(results['multinomial'], axis=0)
    var_syst = np.var(results['systematic'], axis=0)
    plt.plot(var_mult, label='Multinomial Variance', color='red', alpha=0.7)
    plt.plot(var_syst, label='Systematic Variance', color='blue', alpha=0.7)
    plt.title('Variance of Estimator across Trials') #  (Lower is Better)
    plt.xlabel('Time Step')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    rmse_mult = np.sqrt(np.mean((results['multinomial'] - x_true_np) ** 2, axis=1))
    rmse_syst = np.sqrt(np.mean((results['systematic'] - x_true_np) ** 2, axis=1))
    plt.boxplot([rmse_mult, rmse_syst], labels=['Multinomial', 'Systematic'])
    plt.title(f'RMSE Distribution over {num_runs} Trials')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Resampling_Methods_Comparison.pdf",bbox_inches='tight')
    plt.show()

    print(f"Multinomial Mean Variance: {np.mean(var_mult):.6f}")
    print(f"Systematic Mean Variance:  {np.mean(var_syst):.6f}")


def experiment_weight_degeneracy(model, y_obs,method='none'):
    print("\n--- Running Weight Degeneracy Experiment (No Resampling) ---")
    # 使用 'none' 方法禁用重采样
    pf = ParticleFilter(model=model, num_particles=1000, resample_method='systematic')

    _, _, _, all_weights = pf.filter(y_obs)

    # all_weights shape: [Batch, T, Num_Particles]
    weights_np = all_weights.numpy()[0]

    # 计算 N_eff = 1 / sum(w^2)
    n_eff = 1.0 / np.sum(weights_np ** 2, axis=1) # sum over particles

    pf = ParticleFilter(model=model, num_particles=1000, resample_method='none')

    _, _, _, all_weights = pf.filter(y_obs)

    # all_weights shape: [Batch, T, Num_Particles]
    weights_np = all_weights.numpy()[0]

    # 计算 N_eff = 1 / sum(w^2)
    n_eff_no_resample = 1.0 / np.sum(weights_np ** 2, axis=1)  # sum over particles

    plt.figure(figsize=(10, 5))
    plt.plot(n_eff, label='Effective Sample Size ($N_{eff}$)')
    plt.plot(n_eff_no_resample, label='Effective Sample Size without Resampling ($N_{eff}$)', linestyle='--')
    plt.axhline(y=1000, color='r', linestyle='--', label='Total Particles (Ideal)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Weight Degeneracy: $N_{eff}$ over Time, {method}')
    plt.xlabel('Time Step')
    plt.ylabel('$N_{eff}$')
    #plt.yscale('log')  # 使用对数坐标更明显
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Weight_Degeneracy.pdf",bbox_inches='tight')
    plt.show()


def experiment_static_impoverishment_wrong_convergence(T: int = 100):
    true_state_value = 0.0

    # define a static model where the state does not change
    def transition_fn(x, noise):
        return x + noise

    def observation_fn(x, noise):
        return x + noise

    static_model = NLSSM(
        state_dim=1, obs_dim=1,
        x0=tf.ones([1, 1]) * true_state_value,
        init_noise=tfd.Normal(loc=0.0, scale=0.1),
        process_noise=tfd.Normal(loc=0.0, scale=1e-10),
        observation_noise=tfd.Normal(loc=0.0, scale=2.0),
        transition_fn=transition_fn,
        observation_fn=observation_fn
    )

    x_true, y_obs = static_model.sample(T)

    # Ensure y_obs has shape [Batch, T, obs_dim]
    if len(y_obs.shape) == 2:
        y_obs = tf.expand_dims(y_obs, axis=0)
    elif len(y_obs.shape) == 1:
        y_obs = tf.reshape(y_obs, [1, -1, 1])

    pf = ParticleFilter(model=static_model, num_particles=100, resample_method='multinomial')
    x_filt, P_filt, all_particles, _ = pf.filter(y_obs)

    particles_np = all_particles.numpy()[0, :, :, 0]  # [T, N]
    estimated_mean = x_filt.numpy()[0, :, 0]  # [T]
    estimated_var = P_filt.numpy()[0, :, 0, 0]  # [T]

    final_estimate = estimated_mean[-1]
    final_error = abs(final_estimate - true_state_value)
    final_variance = estimated_var[-1]

    print(f"True State: {true_state_value}")
    print(f"Final Estimate: {final_estimate:.4f}")
    print(f"Final Error: {final_error:.4f}")
    print(f"Final Variance (Uncertainty): {final_variance:.10f}")

    plt.figure(figsize=(12, 6))
    plt.plot(estimated_mean, color='blue', linewidth=2, label='Estimated Mean')
    plt.plot(x_true.numpy().reshape(-1), color='red', linewidth=2, label='True State')
    plt.scatter(range(T), y_obs.numpy().flatten(), color='grey', marker='o', label='Observations', s=10, alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('State Value/Observations')
    plt.title('Estimated Mean vs True State with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 6))
    for t in range(T):
        plt.scatter(np.full_like(particles_np[t], t), particles_np[t],
                    s=5, alpha=0.1, color='k', marker='.')
    plt.plot(estimated_mean, color='blue', linewidth=2, label='PF Estimate')
    estimated_std = np.sqrt(estimated_var)
    plt.fill_between(range(T),
                     estimated_mean - 2 * estimated_std,  # 2 standard deviations for 95% CI
                     estimated_mean + 2 * estimated_std,
                     color='blue', alpha=0.2, label='Uncertainty (95% CI)')
    plt.axhline(y=x_true.numpy().reshape(-1)[0], color='red', linestyle='--', linewidth=2, label='True State (0.0)')
    plt.title(f'Static Impoverishment: Converged to {final_estimate:.2f} instead of {true_state_value}')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Example usage with 1D Stochastic Volatility Model
    sv_model = get1DStochasticVolModel(alpha=0.9, beta=0.5, sigma=1)
    pf = ParticleFilter(model=sv_model, num_particles=10000, resample_method='multinomial')

    # Simulate data
    T = 200
    x_true, y_obs = sv_model.sample(T)
    if len(y_obs.shape) == 1:
        y_obs = tf.reshape(y_obs, [1,-1, 1])  # Ensure y_obs has shape [B, T, obs_dim]

    #Run particle filter
    t0=tf.timestamp()
    x_filt, P_filt, all_particles, all_weights = pf.filter(y_obs)
    t1=tf.timestamp()
    print(f"Particle filter completed in {t1 - t0:.2f} seconds.")

    # # Plot results
    # plt.figure(figsize=(12, 6))
    # plt.plot(x_true.numpy(), label='True State', color='g')
    # plt.plot(x_filt[0,:,0].numpy(), label='Filtered State', color='b')
    # plt.fill_between(range(T),
    #                  x_filt[0,:,0].numpy() - 2*tf.sqrt(P_filt[0,:,0,0]).numpy(),
    #                  x_filt[0,:,0].numpy() + 2*tf.sqrt(P_filt[0,:,0,0]).numpy(),
    #                  color='b', alpha=0.2, label='95% Confidence Interval')
    # #plt.scatter(range(T), y_obs.numpy(), label='Observations', color='r', s=10)
    # plt.legend()
    # plt.title('Particle Filter on 1D Stochastic Volatility Model')
    # plt.xlabel('Time')
    # plt.ylabel('State / Observation')
    # plt.savefig("ParticleFilter_1DSVM.pdf",bbox_inches='tight')
    # plt.show()

    #run_resampling_experiment(sv_model, num_particles=1000, x_true=x_true, y_obs=y_obs, num_runs=50)
    experiment_weight_degeneracy(sv_model, y_obs, method='none')
    #experiment_static_impoverishment_wrong_convergence(T)