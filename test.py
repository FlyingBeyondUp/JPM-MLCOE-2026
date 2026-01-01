import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from nonlinearSSM import NLSSM, get1DStochasticVolModel, getVasicekBondPriceModel, UnscentedKalmanFilter
from particle_filter import ParticleFilter
from particle_flow import EDHFlow, LEDHFlow, InvertiblePFPF


class TestNonlinearSSM(unittest.TestCase):
    def setUp(self):
        # Create a simple 1D Stochastic Volatility Model for testing
        self.svm_model = get1DStochasticVolModel(alpha=0.9, sigma=0.1, beta=0.5)

    def test_svm_initialization(self):
        self.assertEqual(self.svm_model.state_dim, 1)
        self.assertEqual(self.svm_model.obs_dim, 1)

    def test_svm_sampling(self):
        T = 10
        x, y = self.svm_model.sample(T)
        self.assertEqual(x.shape, (T,))
        self.assertEqual(y.shape, (T,))

    def test_vasicek_initialization(self):
        vasicek = getVasicekBondPriceModel(kappa=0.5, theta=0.05, sigma=0.01, tau=1.0, dt=0.1)
        self.assertEqual(vasicek.state_dim, 1)
        self.assertEqual(vasicek.obs_dim, 1)

    def test_batch_sample(self):
        # Test batch sampling capability
        T = 5
        batch_size = 3
        x, y = self.svm_model.batch_sample(T, batch_size)
        # Expected shapes: [Batch, T, Dim] (based on typical TF conventions in the code)
        self.assertEqual(x.shape, (batch_size, T, 1))
        self.assertEqual(y.shape, (batch_size, T, 1))


class TestParticleFilter(unittest.TestCase):
    def setUp(self):
        self.model = get1DStochasticVolModel(alpha=0.9, sigma=0.1, beta=0.5)
        self.pf = ParticleFilter(model=self.model, num_particles=100, resample_method='multinomial')

    def test_filter_execution(self):
        # Generate fake observations: [Batch=2, Time=5, ObsDim=1]
        observations = tf.random.normal((2, 5, 1))

        x_filt, P_filt, all_particles, all_weights = self.pf.filter(observations)

        # Check output shapes
        # x_filt: [Batch, Time, StateDim]
        self.assertEqual(x_filt.shape, (2, 5, 1))
        # P_filt: [Batch, Time, StateDim, StateDim]
        self.assertEqual(P_filt.shape, (2, 5, 1, 1))
        # weights: [Batch, Time, NumParticles]
        self.assertEqual(all_weights.shape, (2, 5, 100))


class TestParticleFlow(unittest.TestCase):
    def setUp(self):
        self.model = get1DStochasticVolModel(alpha=0.9, sigma=0.1, beta=0.5)
        self.obs = tf.random.normal((2, 5, 1))  # Batch=2, T=5

        # UKF is often required for flow initialization/guidance
        self.ukf = UnscentedKalmanFilter(self.model, alpha=1e-3, beta=2.0, kappa=0.0)

    def test_edh_flow_execution(self):
        # Exact Daum-Huang Flow
        edh = EDHFlow(model=self.model, num_particles=50, ukf=self.ukf)

        # Test filter method
        # Note: EDHFlow.filter might require specific args like num_flow_steps
        x_filt, P_filt, particles = edh.filter(self.obs, num_flow_steps=5)

        self.assertEqual(x_filt.shape, (2, 5, 1))
        self.assertEqual(P_filt.shape, (2, 5, 1, 1))

    def test_ledh_flow_execution(self):
        # Local EDH Flow
        ledh = LEDHFlow(model=self.model, num_particles=50, ukf=self.ukf)

        # LEDH is computationally heavier, use small steps
        x_filt, P_filt, particles = ledh.filter(self.obs, num_flow_steps=2)

        self.assertEqual(x_filt.shape, (2, 5, 1))
        self.assertEqual(P_filt.shape, (2, 5, 1, 1))

    def test_invertible_pfpf_execution(self):
        # Invertible Particle Flow Particle Filter
        # This uses EDHFlow internally by default
        pfpf = InvertiblePFPF(
            model=self.model,
            num_particles=50,
            ukf=self.ukf,
            flow_class=EDHFlow,
            num_flow_steps=2
        )

        # Run filter
        x_filt, P_filt, particles, weights = pfpf.filter(self.obs)

        # Check shapes
        self.assertEqual(x_filt.shape, (2, 5, 1))
        self.assertEqual(weights.shape, (2, 5, 50))

        # Basic check: Weights should sum to 1 (approx)
        weight_sums = tf.reduce_sum(weights, axis=-1)
        self.assertTrue(np.allclose(weight_sums.numpy(), 1.0, atol=1e-5))


if __name__ == '__main__':
    unittest.main()