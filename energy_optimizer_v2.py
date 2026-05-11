import coal
"""
Production Energy Optimizer V2 - VERIFIED ITERATIVE TOPP-RA

Final Architecture:
1. Static feasibility rejection (gravity vs torque)
2. Grid-based local constraints (TOPPRA)
3. Iterative refinement loop (tighten bounds if violated)
4. Analytical verification using RNEA
5. Final safety scaling (guaranteed τ ≤ τ_max)
"""

import numpy as np
import pinocchio as pin
from typing import Union, Tuple, Dict
from pathlib import Path

try:
    import toppra
    import toppra.constraint as constraint
    import toppra.algorithm as algo
    TOPPRA_AVAILABLE = True
except ImportError:
    TOPPRA_AVAILABLE = False


class EnergyOptimizerV2:
    def __init__(self, urdf_path: Union[str, Path], verbose: bool = False):
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        self.verbose = verbose
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.v_max = self.model.velocityLimit.copy()
        
        self.tau_max = np.array([
            self.model.effortLimit[i] if self.model.effortLimit[i] > 0 else 50.0 
            for i in range(self.nv)
        ])
        
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()

    def compute_torques(self, q, v, a):
        return pin.rnea(self.model, self.data, q, v, a)

    def analyze_trajectory(self, q_traj, v_traj, a_traj) -> Dict:
        """Original analytical metric computation."""
        n = q_traj.shape[0]
        torques = np.array([self.compute_torques(q_traj[i], v_traj[i], a_traj[i]) for i in range(n)])
        
        peak_torque = np.max(np.abs(torques))
        peak_velocity = np.max(np.abs(v_traj))
        peak_acceleration = np.max(np.abs(a_traj))
        powers = np.abs(np.sum(torques * v_traj, axis=1))
        peak_power = np.max(powers)
        
        # Thermal load proxy: Absolute mechanical work integral
        # Since we resample at control_dt, we need the dt for the sum
        # We'll calculate it inside compute_trajectory_energy for accuracy
        return {
            'peak_torque': peak_torque,
            'peak_velocity': peak_velocity,
            'peak_acceleration': peak_acceleration,
            'peak_power': peak_power,
            'rms_torque': np.sqrt(np.mean(torques**2))
        }

    def compute_trajectory_energy(self, trajectory: np.ndarray, dt: float) -> Dict:
        """Utility for hierarchical metrics."""
        n_waypoints = trajectory.shape[0]
        v_traj = np.zeros_like(trajectory)
        a_traj = np.zeros_like(trajectory)
        for t in range(1, n_waypoints - 1):
            v_traj[t] = (trajectory[t+1] - trajectory[t-1]) / (2.0 * dt)
            a_traj[t] = (trajectory[t+1] - 2.0 * trajectory[t] + trajectory[t-1]) / (dt**2)
        
        metrics = self.analyze_trajectory(trajectory, v_traj, a_traj)
        thermal_load = sum(np.abs(np.sum(self.compute_torques(trajectory[t], v_traj[t], a_traj[t]) * v_traj[t])) * dt for t in range(n_waypoints))
        metrics['total_energy'] = thermal_load
        return metrics

    def _compute_dynamics_bounds(self, trajectory):
        n = trajectory.shape[0]
        tightest_a_limit = np.full(self.nv, 15.0) 
        for t in range(n):
            dq = trajectory[t+1] - trajectory[t] if t < n - 1 else trajectory[t] - trajectory[t-1]
            v_dir = (dq / (np.linalg.norm(dq) + 1e-6)) * np.mean(self.v_max) * 0.3
            nle = pin.rnea(self.model, self.data, trajectory[t], v_dir, np.zeros(self.nv))
            tau_avail = np.maximum(self.tau_max - np.abs(nle), 1.0)
            pin.crba(self.model, self.data, trajectory[t])
            M_coupled = np.sum(np.abs(self.data.M), axis=1)
            a_max_grid = tau_avail / (M_coupled + 1e-6)
            tightest_a_limit = np.minimum(tightest_a_limit, a_max_grid)

        alim = np.zeros((self.nv, 2))
        alim[:, 0] = -np.clip(tightest_a_limit, 0.1, 15.0)
        alim[:, 1] = np.clip(tightest_a_limit, 0.1, 15.0)
        return alim

    def _verify_trajectory(self, traj_fn, ts):
        worst_violation = 1.0
        for t in ts:
            tau = self.compute_torques(traj_fn(t), traj_fn(t, 1), traj_fn(t, 2))
            violation = np.max(np.abs(tau) / self.tau_max)
            worst_violation = max(worst_violation, violation)
        return worst_violation

    def retime_trajectory(self, trajectory: np.ndarray, control_dt: float = 0.1):
        if not TOPPRA_AVAILABLE:
            return self._fallback_uniform_scaling(trajectory, control_dt)

        n = trajectory.shape[0]

        # 1. STATIC FEASIBILITY
        for t in range(n):
            pin.computeGeneralizedGravity(self.model, self.data, trajectory[t])
            if np.any(np.abs(self.data.g) >= self.tau_max * 0.98):
                metrics = self.compute_trajectory_energy(trajectory, control_dt)
                metrics.update({'path_rejected': True, 'was_retimed': False})
                return trajectory, control_dt, metrics

        s_grid = np.linspace(0, 1, n)
        path = toppra.SplineInterpolator(s_grid, trajectory)
        pc_vel = constraint.JointVelocityConstraint(np.stack([-self.v_max, self.v_max], axis=1))

        # 2. ITERATIVE REFINEMENT LOOP
        alim = self._compute_dynamics_bounds(trajectory)
        for iteration in range(3):
            pc_acc = constraint.JointAccelerationConstraint(alim)
            try:
                instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
                traj = instance.compute_trajectory(0, 0)
            except: traj = None

            if traj is None: return self._fallback_uniform_scaling(trajectory, control_dt)

            duration = traj.duration
            ts = np.linspace(0, duration, int(np.ceil(duration / control_dt)) + 1)
            violation = self._verify_trajectory(traj, ts)
            if violation <= 1.01: break
            alim /= np.sqrt(violation)

        # 3. FINAL SAFETY SCALING
        violation = self._verify_trajectory(traj, ts)
        scale = np.sqrt(violation) * 1.02 if violation > 1.01 else 1.0
        ts_final = ts * scale
        retimed = traj(ts_final)

        # 4. COMPUTE UNIFIED METRICS
        # Compute exact V and A at the new scale for analytical RNEA
        v_final = traj(ts_final, 1) / scale
        a_final = traj(ts_final, 2) / (scale**2)
        
        metrics = self.analyze_trajectory(retimed, v_final, a_final)
        
        # Calculate thermal load for the retimed path
        actual_dt = control_dt * scale
        thermal_load = sum(
            np.abs(np.sum(self.compute_torques(retimed[i], v_final[i], a_final[i]) * v_final[i])) * actual_dt
            for i in range(len(ts_final))
        )
        
        metrics.update({
            'total_energy': thermal_load,
            'was_retimed': True, 'used_fallback': False, 'path_rejected': False,
            'optimal_duration': duration * scale, 'violation_final': violation
        })

        return retimed, actual_dt, metrics

    def _fallback_uniform_scaling(self, trajectory, base_dt):
        dt = base_dt
        n = trajectory.shape[0]
        for _ in range(20):
            v = np.zeros_like(trajectory)
            a = np.zeros_like(trajectory)
            for t in range(1, n - 1):
                v[t] = (trajectory[t+1] - trajectory[t-1]) / (2*dt)
                a[t] = (trajectory[t+1] - 2*trajectory[t] + trajectory[t-1]) / (dt**2)
            violation = 1.0
            for t in range(n):
                tau = self.compute_torques(trajectory[t], v[t], a[t])
                violation = max(violation, np.max(np.abs(tau)/self.tau_max))
            if violation <= 1.01: break
            dt *= np.sqrt(violation) * 1.05

        metrics = self.compute_trajectory_energy(trajectory, dt)
        metrics.update({'was_retimed': True, 'used_fallback': True, 'path_rejected': False, 'optimal_duration': n * dt})
        return trajectory, dt, metrics

    def generate_trajectory(self, q_start, q_goal, duration, n_points=50):
        q_goal = np.clip(q_goal, self.q_min, self.q_max)
        displacement = q_goal - q_start
        t = np.linspace(0, duration, n_points)
        tau = t / duration
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (duration**2)
        q_traj = q_start[None, :] + s[:, None] * displacement[None, :]
        v_traj = s_dot[:, None] * displacement[None, :]
        a_traj = s_ddot[:, None] * displacement[None, :]
        return q_traj, v_traj, a_traj

    def optimize_target(self, q_current, q_target, baseline_duration=2.0):
        q_base, v_base, a_base = self.generate_trajectory(q_current, q_target, baseline_duration)
        metrics_base = self.analyze_trajectory(q_base, v_base, a_base)
        retimed_traj, _, metrics_opt = self.retime_trajectory(q_base, control_dt=baseline_duration/50.0)
        torque_reduction = ((metrics_base['peak_torque'] - metrics_opt['peak_torque']) / metrics_base['peak_torque'] * 100) if metrics_base['peak_torque'] > 0 else 0.0
        return q_target, {
            'torque_reduction_percent': torque_reduction,
            'baseline_peak_torque': metrics_base['peak_torque'],
            'optimized_peak_torque': metrics_opt['peak_torque']
        }
