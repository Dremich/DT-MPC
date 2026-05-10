"""Smoke tests for the initial DT-MPC repository structure."""

import unittest

import numpy as np

from dynamics.base_system import DubinsCar
from dynamics.safety_embedded import SafetyEmbeddedDynamics
from learning.doc_engine import DifferentiableOptimalControl
from learning.dt_mpc_loop import DTMPCTrainer
from solvers.optimal_control import DDPSolver


class RepoStructureTests(unittest.TestCase):
    def test_dubins_car_interface(self):
        car = DubinsCar(dt=0.1)

        self.assertEqual(car.dt, 0.1)
        self.assertEqual(car.state_dim, 3)
        self.assertEqual(car.control_dim, 2)
        with self.assertRaises(NotImplementedError):
            car.step(np.zeros(3), np.zeros(2))

    def test_safety_embedded_dynamics_interface(self):
        dynamics = SafetyEmbeddedDynamics(DubinsCar(dt=0.1), {"gamma": 1.0, "alpha": 0.5})

        self.assertEqual(dynamics.theta["gamma"], 1.0)
        with self.assertRaises(NotImplementedError):
            dynamics.step(np.zeros(4), np.zeros(2))
        with self.assertRaises(NotImplementedError):
            dynamics.get_derivatives(np.zeros(4), np.zeros(2))

    def test_solver_and_learning_interfaces(self):
        dynamics = SafetyEmbeddedDynamics(DubinsCar(dt=0.1), {"gamma": 1.0})
        solver = DDPSolver(dynamics=dynamics, horizon=10)
        doc = DifferentiableOptimalControl()
        trainer = DTMPCTrainer(
            plant=dynamics.plant,
            solver=solver,
            doc_engine=doc,
            learning_rate=1e-2,
            horizon_H=10,
        )

        self.assertEqual(solver.horizon, 10)
        self.assertEqual(trainer.horizon_H, 10)

        with self.assertRaises(NotImplementedError):
            solver.solve(np.zeros(4), is_nominal=True)
        with self.assertRaises(NotImplementedError):
            doc.backward_pass({}, {})
        with self.assertRaises(NotImplementedError):
            doc.forward_pass({}, {})
        with self.assertRaises(NotImplementedError):
            doc.compute_gradient({}, {}, dynamics)
        with self.assertRaises(NotImplementedError):
            trainer.compute_upper_level_loss({}, {})
        with self.assertRaises(NotImplementedError):
            trainer.train_step(np.zeros(4))


if __name__ == "__main__":
    unittest.main()
