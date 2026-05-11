import coal
"""
ProxQP 

Key fixes:
1. Proper acceleration cost via finite differences
2. Don't double-constrain start/goal (equality overrides limits)
3. Correct constraint ordering
"""

import numpy as np
import pinocchio as pin
import proxsuite
from typing import Dict, Tuple, Union
from pathlib import Path


class TrajectoryOptimizerProxQPV2:
    
    def __init__(self, urdf_path: Union[str, Path], collision_detector=None, verbose: bool = False):
        urdf_path = Path(urdf_path)
        
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        
        self.nq = self.model.nq
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()
        self.v_max = self.model.velocityLimit.copy()
        self.verbose = verbose
    
    def optimize_trajectory(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        n_waypoints: int = 10,
        dt: float = 0.1,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        
        n_vars = n_waypoints * self.nq
        
        # Acceleration cost
        D = np.zeros((n_waypoints - 2, n_waypoints))
        for t in range(n_waypoints - 2):
            D[t, t] = 1.0 / dt**2
            D[t, t+1] = -2.0 / dt**2
            D[t, t+2] = 1.0 / dt**2
        
        D_full = np.zeros(((n_waypoints - 2) * self.nq, n_vars))
        for i in range(self.nq):
            for t in range(n_waypoints - 2):
                for k in range(n_waypoints):
                    D_full[t * self.nq + i, k * self.nq + i] = D[t, k]
        
        H = D_full.T @ D_full + np.eye(n_vars) * 1e-8
        g = np.zeros(n_vars)
        
        # Constraints
        C_list = []
        lb_list = []
        ub_list = []
        
        # Position limits for INTERMEDIATE waypoints only (not start/goal)
        for t in range(1, n_waypoints - 1):
            for i in range(self.nq):
                row = np.zeros(n_vars)
                row[t * self.nq + i] = 1.0
                C_list.append(row)
                lb_list.append(self.q_min[i])
                ub_list.append(self.q_max[i])
        
        # Velocity limits
        for t in range(n_waypoints - 1):
            for i in range(self.nq):
                row = np.zeros(n_vars)
                row[t * self.nq + i] = -1.0 / dt
                row[(t+1) * self.nq + i] = 1.0 / dt
                C_list.append(row)
                lb_list.append(-self.v_max[i])
                ub_list.append(self.v_max[i])
        
        # EQUALITY constraints for start and goal (these override limits)
        for i in range(self.nq):
            row = np.zeros(n_vars)
            row[i] = 1.0
            C_list.append(row)
            lb_list.append(q_start[i])
            ub_list.append(q_start[i])
            
            row = np.zeros(n_vars)
            row[-self.nq + i] = 1.0
            C_list.append(row)
            lb_list.append(q_goal[i])
            ub_list.append(q_goal[i])
        
        C = np.vstack(C_list)
        lb = np.array(lb_list)
        ub = np.array(ub_list)
        
        # Solve
        qp = proxsuite.proxqp.dense.QP(n_vars, 0, C.shape[0])
        qp.settings.eps_abs = 1e-6
        qp.settings.max_iter = 1000
        qp.init(H, g, None, None, C, lb, ub)
        qp.solve()
        
        trajectory = qp.results.x.reshape(n_waypoints, self.nq)
        
        return trajectory, {
            'converged': qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED,
            'iterations': qp.results.info.iter,
            'goal_error': np.linalg.norm(trajectory[-1] - q_goal),
            'objective': qp.results.info.objValue
        }


if __name__ == "__main__":
    
    urdf = "/scratch/anshb3/ovla/robots/franka/franka_panda_with_inertia.urdf"
    optimizer = TrajectoryOptimizerProxQPV2(urdf)
    
    q_start = np.zeros(optimizer.nq)
    q_range = optimizer.q_max - optimizer.q_min
    q_goal = optimizer.q_min + np.random.rand(optimizer.nq) * q_range * 0.4 + q_range * 0.3
    
    trajectory, metrics = optimizer.optimize_trajectory(q_start, q_goal, n_waypoints=10, dt=0.2)
    
    print(f"\nResult:")
    print(f"  Converged: {metrics['converged']}")
    print(f"  Goal error: {metrics['goal_error']:.9f}")
    
    motion = np.sum(np.abs(np.diff(trajectory, axis=0)))
    print(f"  Motion: {motion:.3f} rad")
    
    # Energy
    import pinocchio as pin
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    
    velocities = np.diff(trajectory, axis=0) / 0.2
    accelerations = np.diff(velocities, axis=0) / 0.2
    
    energy = 0.0
    for i in range(len(trajectory) - 1):
        v = velocities[i] if i < len(velocities) else velocities[-1]
        a = accelerations[i] if i < len(accelerations) else np.zeros(model.nv)
        tau = pin.rnea(model, data, trajectory[i], v, a)
        power = np.abs(np.dot(tau, v))
        energy += power * 0.2
    
    print(f"  Energy: {energy:.3f} J")
    
    if metrics['converged'] and metrics['goal_error'] < 1e-3 and motion > 0.5 and energy > 0.1:
        print("\n PROXQP WORKS!")
