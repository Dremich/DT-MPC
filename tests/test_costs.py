"""Unit tests for the cost functions module."""
"""Initialized via Copilot on 2026-05-18."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from solvers.costs import BaseCost, QuadraticCost, TerminalCost

jax.config.update("jax_enable_x64", True) # Tests fail without this due to numerical precision issues in finite difference gradient checks.

class TestQuadraticCost(unittest.TestCase):
    """Test suite for QuadraticCost class."""

    def setUp(self):
        """Initialize test fixtures."""
        self.nx = 3  # State dimension
        self.nu = 2  # Control dimension
        
        # Create symmetric positive definite cost matrices
        self.Q = jnp.eye(self.nx)
        self.R = jnp.eye(self.nu)
        
        # Test states and controls
        self.x = jnp.array([1.0, 2.0, 3.0])
        self.u = jnp.array([0.5, 1.5])

    def test_quadratic_cost_without_reference(self):
        """Test quadratic cost evaluation without reference tracking."""
        cost_fn = QuadraticCost(self.Q, self.R)
        cost = cost_fn.evaluate(self.x, self.u)
        
        # Expected: x^T Q x + u^T R u
        expected = (
            self.x @ self.Q @ self.x + 
            self.u @ self.R @ self.u
        )
        
        self.assertAlmostEqual(float(cost), float(expected), places=6)

    def test_quadratic_cost_with_state_reference(self):
        """Test quadratic cost evaluation with state reference."""
        x_ref = jnp.array([0.5, 1.0, 1.5])
        cost_fn = QuadraticCost(self.Q, self.R, x_ref=x_ref)
        cost = cost_fn.evaluate(self.x, self.u)
        
        # Expected: (x - x_ref)^T Q (x - x_ref) + u^T R u
        dx = self.x - x_ref
        expected = dx @ self.Q @ dx + self.u @ self.R @ self.u
        
        self.assertAlmostEqual(float(cost), float(expected), places=6)

    def test_quadratic_cost_with_control_reference(self):
        """Test quadratic cost evaluation with control reference."""
        u_ref = jnp.array([0.1, 0.2])
        cost_fn = QuadraticCost(self.Q, self.R, u_ref=u_ref)
        cost = cost_fn.evaluate(self.x, self.u)
        
        # Expected: x^T Q x + (u - u_ref)^T R (u - u_ref)
        du = self.u - u_ref
        expected = self.x @ self.Q @ self.x + du @ self.R @ du
        
        self.assertAlmostEqual(float(cost), float(expected), places=6)

    def test_quadratic_cost_with_both_references(self):
        """Test quadratic cost evaluation with both state and control references."""
        x_ref = jnp.array([0.5, 1.0, 1.5])
        u_ref = jnp.array([0.1, 0.2])
        cost_fn = QuadraticCost(self.Q, self.R, x_ref=x_ref, u_ref=u_ref)
        cost = cost_fn.evaluate(self.x, self.u)
        
        # Expected: (x - x_ref)^T Q (x - x_ref) + (u - u_ref)^T R (u - u_ref)
        dx = self.x - x_ref
        du = self.u - u_ref
        expected = dx @ self.Q @ dx + du @ self.R @ du
        
        self.assertAlmostEqual(float(cost), float(expected), places=6)

    def test_quadratic_cost_derivatives_shape(self):
        """Test that get_derivatives returns correct shapes."""
        cost_fn = QuadraticCost(self.Q, self.R)
        l_x, l_u, l_xx, l_uu, l_xu = cost_fn.get_derivatives(self.x, self.u)
        
        # Check shapes
        self.assertEqual(l_x.shape, (self.nx,))
        self.assertEqual(l_u.shape, (self.nu,))
        self.assertEqual(l_xx.shape, (self.nx, self.nx))
        self.assertEqual(l_uu.shape, (self.nu, self.nu))
        self.assertEqual(l_xu.shape, (self.nx, self.nu))

    def test_quadratic_cost_derivatives_values(self):
        """Test that derivatives have correct values for quadratic cost."""
        cost_fn = QuadraticCost(self.Q, self.R)
        l_x, l_u, l_xx, l_uu, l_xu = cost_fn.get_derivatives(self.x, self.u)
        
        # For quadratic cost: l_x = 2 Q x, l_u = 2 R u
        expected_l_x = 2 * self.Q @ self.x
        expected_l_u = 2 * self.R @ self.u
        expected_l_xx = 2 * self.Q
        expected_l_uu = 2 * self.R
        expected_l_xu = jnp.zeros((self.nx, self.nu))  # No cross terms
        
        np.testing.assert_allclose(l_x, expected_l_x, rtol=1e-6)
        np.testing.assert_allclose(l_u, expected_l_u, rtol=1e-6)
        np.testing.assert_allclose(l_xx, expected_l_xx, rtol=1e-6)
        np.testing.assert_allclose(l_uu, expected_l_uu, rtol=1e-6)
        np.testing.assert_allclose(l_xu, expected_l_xu, rtol=1e-6)

    def test_quadratic_cost_derivatives_with_reference(self):
        """Test derivatives with reference trajectories."""
        x_ref = jnp.array([0.5, 1.0, 1.5])
        u_ref = jnp.array([0.1, 0.2])
        cost_fn = QuadraticCost(self.Q, self.R, x_ref=x_ref, u_ref=u_ref)
        l_x, l_u, l_xx, l_uu, l_xu = cost_fn.get_derivatives(self.x, self.u)
        
        # Derivatives should be w.r.t. deviation from reference
        dx = self.x - x_ref
        du = self.u - u_ref
        expected_l_x = 2 * self.Q @ dx
        expected_l_u = 2 * self.R @ du
        
        np.testing.assert_allclose(l_x, expected_l_x, rtol=1e-6)
        np.testing.assert_allclose(l_u, expected_l_u, rtol=1e-6)

    def test_quadratic_cost_zero_cost_at_reference(self):
        """Test that cost is zero when state and control are at references."""
        x_ref = jnp.array([1.0, 2.0, 3.0])
        u_ref = jnp.array([0.5, 1.5])
        cost_fn = QuadraticCost(self.Q, self.R, x_ref=x_ref, u_ref=u_ref)
        
        cost = cost_fn.evaluate(self.x, self.u)
        self.assertAlmostEqual(float(cost), 0.0, places=6)


class TestTerminalCost(unittest.TestCase):
    """Test suite for TerminalCost class."""

    def setUp(self):
        """Initialize test fixtures."""
        self.nx = 3  # State dimension
        
        # Create symmetric positive definite cost matrix
        self.P = jnp.eye(self.nx)
        
        # Test state
        self.x = jnp.array([1.0, 2.0, 3.0])

    def test_terminal_cost_without_reference(self):
        """Test terminal cost evaluation without reference."""
        cost_fn = TerminalCost(self.P)
        cost = cost_fn.evaluate(self.x)
        
        # Expected: x^T P x
        expected = self.x @ self.P @ self.x
        
        self.assertAlmostEqual(float(cost), float(expected), places=6)

    def test_terminal_cost_with_reference(self):
        """Test terminal cost evaluation with reference."""
        x_ref = jnp.array([0.5, 1.0, 1.5])
        cost_fn = TerminalCost(self.P, x_ref=x_ref)
        cost = cost_fn.evaluate(self.x)
        
        # Expected: (x - x_ref)^T P (x - x_ref)
        dx = self.x - x_ref
        expected = dx @ self.P @ dx
        
        self.assertAlmostEqual(float(cost), float(expected), places=6)

    def test_terminal_cost_derivatives_shape(self):
        """Test that get_derivatives returns correct shapes for terminal cost."""
        cost_fn = TerminalCost(self.P)
        phi_x, phi_u, phi_xx, phi_uu, phi_xu = cost_fn.get_derivatives(self.x)
        
        # Check shapes (terminal cost only has state derivatives)
        self.assertEqual(phi_x.shape, (self.nx,))
        self.assertIsNone(phi_u)
        self.assertEqual(phi_xx.shape, (self.nx, self.nx))
        self.assertIsNone(phi_uu)
        self.assertIsNone(phi_xu)

    def test_terminal_cost_derivatives_values(self):
        """Test that terminal cost derivatives have correct values."""
        cost_fn = TerminalCost(self.P)
        phi_x, phi_u, phi_xx, phi_uu, phi_xu = cost_fn.get_derivatives(self.x)
        
        # For quadratic terminal cost: phi_x = 2 P x
        expected_phi_x = 2 * self.P @ self.x
        expected_phi_xx = 2 * self.P
        
        np.testing.assert_allclose(phi_x, expected_phi_x, rtol=1e-6)
        np.testing.assert_allclose(phi_xx, expected_phi_xx, rtol=1e-6)

    def test_terminal_cost_derivatives_with_reference(self):
        """Test terminal cost derivatives with reference."""
        x_ref = jnp.array([0.5, 1.0, 1.5])
        cost_fn = TerminalCost(self.P, x_ref=x_ref)
        phi_x, _, phi_xx, _, _ = cost_fn.get_derivatives(self.x)
        
        # Derivatives should be w.r.t. deviation from reference
        dx = self.x - x_ref
        expected_phi_x = 2 * self.P @ dx
        
        np.testing.assert_allclose(phi_x, expected_phi_x, rtol=1e-6)

    def test_terminal_cost_zero_cost_at_reference(self):
        """Test that terminal cost is zero when state is at reference."""
        x_ref = jnp.array([1.0, 2.0, 3.0])
        cost_fn = TerminalCost(self.P, x_ref=x_ref)
        
        cost = cost_fn.evaluate(self.x)
        self.assertAlmostEqual(float(cost), 0.0, places=6)


class TestCostMatrices(unittest.TestCase):
    """Test suite for different cost matrix configurations."""

    def test_diagonal_cost_matrices(self):
        """Test with diagonal cost matrices."""
        nx, nu = 4, 2
        Q = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
        R = jnp.diag(jnp.array([0.5, 0.5]))
        
        x = jnp.ones(nx)
        u = jnp.ones(nu)
        
        cost_fn = QuadraticCost(Q, R)
        cost = cost_fn.evaluate(x, u)
        
        # Expected: sum of diagonal Q * 1 + sum of diagonal R * 1
        expected = float(jnp.sum(Q) + jnp.sum(R))
        
        self.assertAlmostEqual(float(cost), expected, places=6)

    def test_full_cost_matrices(self):
        """Test with full (non-diagonal) cost matrices."""
        Q = jnp.array([[2.0, 1.0, 0.5], 
                       [1.0, 3.0, 0.2], 
                       [0.5, 0.2, 1.0]])
        R = jnp.array([[1.0, 0.1], 
                       [0.1, 2.0]])
        
        x = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([0.5, 1.5])
        
        cost_fn = QuadraticCost(Q, R)
        cost = cost_fn.evaluate(x, u)
        
        expected = float(x @ Q @ x + u @ R @ u)
        self.assertAlmostEqual(float(cost), expected, places=6)


class TestNumericalGradientVerification(unittest.TestCase):
    """Verify analytical derivatives against numerical gradients."""

    def setUp(self):
        """Initialize test fixtures."""
        self.nx = 3
        self.nu = 2
        self.Q = jnp.eye(self.nx) * 2.0
        self.R = jnp.eye(self.nu) * 0.5
        self.x = jnp.array([1.0, -0.5, 2.0])
        self.u = jnp.array([0.3, -0.2])
        self.epsilon = 1e-5

    def numerical_gradient_x(self, cost_fn, x, u):
        """Compute numerical gradient w.r.t. x using finite differences."""
        grad = jnp.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.at[i].add(self.epsilon)
            x_minus = x.at[i].add(-self.epsilon)
            grad = grad.at[i].set(
                (cost_fn.evaluate(x_plus, u) - cost_fn.evaluate(x_minus, u)) / (2 * self.epsilon)
            )
        return grad

    def numerical_gradient_u(self, cost_fn, x, u):
        """Compute numerical gradient w.r.t. u using finite differences."""
        grad = jnp.zeros_like(u)
        for i in range(len(u)):
            u_plus = u.at[i].add(self.epsilon)
            u_minus = u.at[i].add(-self.epsilon)
            grad = grad.at[i].set(
                (cost_fn.evaluate(x, u_plus) - cost_fn.evaluate(x, u_minus)) / (2 * self.epsilon)
            )
        return grad

    def test_quadratic_cost_gradient_x(self):
        """Verify analytical gradient w.r.t. x matches numerical gradient."""
        cost_fn = QuadraticCost(self.Q, self.R)
        l_x, _, _, _, _ = cost_fn.get_derivatives(self.x, self.u)
        
        numerical_grad = self.numerical_gradient_x(cost_fn, self.x, self.u)
        
        np.testing.assert_allclose(l_x, numerical_grad, rtol=1e-4)

    def test_quadratic_cost_gradient_u(self):
        """Verify analytical gradient w.r.t. u matches numerical gradient."""
        cost_fn = QuadraticCost(self.Q, self.R)
        _, l_u, _, _, _ = cost_fn.get_derivatives(self.x, self.u)
        
        numerical_grad = self.numerical_gradient_u(cost_fn, self.x, self.u)
        
        np.testing.assert_allclose(l_u, numerical_grad, rtol=1e-4)

    def test_terminal_cost_gradient(self):
        """Verify terminal cost gradient matches numerical gradient."""
        P = jnp.eye(self.nx) * 3.0
        cost_fn = TerminalCost(P)
        phi_x, _, _, _, _ = cost_fn.get_derivatives(self.x)
        
        numerical_grad = self.numerical_gradient_x(cost_fn, self.x, None)
        
        np.testing.assert_allclose(phi_x, numerical_grad, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
