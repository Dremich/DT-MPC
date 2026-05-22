"""Unit tests for the DubinsCar dynamics."""
"""Initialized by Copilot on 2026-05-19"""
"""Verified by Andrew Tolton on 2026-05-19"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from dynamics.dubins_car import DubinsCar


class TestDubinsCarDynamics(unittest.TestCase):
    def setUp(self):
        self.car = DubinsCar(wheelbase=0.5)
        self.x = np.array([1.0, 2.0, 0.3])
        self.u = np.array([1.2, 0.1])
        self.dt = 0.01
        self.eps = 1e-6

    def test_dimensions(self):
        self.assertEqual(self.car.state_dim, 3)
        self.assertEqual(self.car.control_dim, 2)

    def test_dynamics_values(self):
        v, omega = self.u[0], self.u[1]
        theta = self.x[2]

        expected = np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            (v / self.car.L) * np.tan(omega)
        ])

        f = self.car.dynamics(self.x, self.u)
        assert_allclose(f, expected, rtol=1e-7, atol=1e-9)

    def test_step_euler_integration(self):
        x_next = self.car.step(self.x, self.u, self.dt)
        expected = self.x + self.car.dynamics(self.x, self.u) * self.dt
        assert_allclose(x_next, expected, rtol=1e-7)

    def numerical_jacobian_x(self, x, u):
        f0 = self.car.dynamics(x, u)
        A_num = np.zeros((self.car.state_dim, self.car.state_dim))
        for i in range(self.car.state_dim):
            xp = x.copy()
            xm = x.copy()
            xp[i] += self.eps
            xm[i] -= self.eps
            fp = self.car.dynamics(xp, u)
            fm = self.car.dynamics(xm, u)
            A_num[:, i] = (fp - fm) / (2 * self.eps)
        return A_num

    def numerical_jacobian_u(self, x, u):
        f0 = self.car.dynamics(x, u)
        B_num = np.zeros((self.car.state_dim, self.car.control_dim))
        for i in range(self.car.control_dim):
            up = u.copy()
            um = u.copy()
            up[i] += self.eps
            um[i] -= self.eps
            fp = self.car.dynamics(x, up)
            fm = self.car.dynamics(x, um)
            B_num[:, i] = (fp - fm) / (2 * self.eps)
        return B_num

    def test_continuous_jacobians_shapes(self):
        A, B = self.car.continuous_jacobians(self.x, self.u)
        self.assertEqual(A.shape, (3, 3))
        self.assertEqual(B.shape, (3, 2))

    def test_continuous_jacobians_numerical(self):
        A, B = self.car.continuous_jacobians(self.x, self.u)
        A_num = self.numerical_jacobian_x(self.x, self.u)
        B_num = self.numerical_jacobian_u(self.x, self.u)

        assert_allclose(A, A_num, rtol=1e-4, atol=1e-6)
        assert_allclose(B, B_num, rtol=1e-4, atol=1e-6)

    def test_jacobians(self):
        # Initialize your OOP car
        car = DubinsCar()
        dt = 0.1
        L = car.L if hasattr(car, 'L') else 1.0  # Assuming wheelbase L is defined
        
        # 1. Pick a random, non-zero state and control to test
        # (Don't use zeros, or missing terms might accidentally multiply to zero and hide the bug!)
        x_test = np.array([-5.0, 5.0, -1.5]) 
        u_test = np.array([2.0, 0.5]) # v=2.0, omega=0.5
        
        # 2. Define the EXACT physics of your OOP class in JAX
        @jax.jit
        def jax_discrete_step(x, u):
            v, omega = u[0], u[1]
            theta = x[2]
            
            # Kinematic Bicycle Model
            return jnp.array([
                x[0] + (v * jnp.cos(theta)) * dt,
                x[1] + (v * jnp.sin(theta)) * dt,
                x[2] + (v / L * jnp.tan(omega)) * dt
            ])
            
        # 3. Ask JAX to compute the ground-truth Jacobians
        get_A_jax = jax.jacobian(jax_discrete_step, argnums=0)
        get_B_jax = jax.jacobian(jax_discrete_step, argnums=1)
        
        A_jax = np.array(get_A_jax(x_test, u_test))
        B_jax = np.array(get_B_jax(x_test, u_test))
        
        # 4. Ask your handwritten code for its Jacobians
        A_oop, B_oop = car.discrete_jacobians(x_test, u_test, dt)
        
        # 5. Compare them!
        print("--- Testing Matrix A (State Jacobian df/dx) ---")
        if np.allclose(A_jax, A_oop, atol=1e-5):
            print("✅ Matrix A matches JAX perfectly!")
        else:
            print("❌ Matrix A has an error.")
            print("JAX Ground Truth:\n", np.round(A_jax, 4))
            print("Your Code:\n", np.round(A_oop, 4))
            print("Difference:\n", np.round(A_jax - A_oop, 4))
            
        print("\n--- Testing Matrix B (Control Jacobian df/du) ---")
        if np.allclose(B_jax, B_oop, atol=1e-5):
            print("✅ Matrix B matches JAX perfectly!")
        else:
            print("❌ Matrix B has an error.")
            print("JAX Ground Truth:\n", np.round(B_jax, 4))
            print("Your Code:\n", np.round(B_oop, 4))
            print("Difference:\n", np.round(B_jax - B_oop, 4))


if __name__ == '__main__':
    unittest.main()
