import coal
"""
CasADi Trajectory Optimizer V6 - Multi-Obstacle Support

Extends V6 with support for multiple analytic obstacle types:
- Spheres (point obstacles)
- Capsules (line segment obstacles, like poles/limbs)
- Boxes (rectangular obstacles, like tables/walls)

All obstacles have exact analytic distance functions → smooth gradients.
"""

import numpy as np
import pinocchio as pin
import casadi as ca
import time
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import pinocchio.casadi as cpin
except ImportError:
    raise ImportError("pinocchio.casadi not found. Install via: conda install -c conda-forge pinocchio")


class CasadiOptimizerV6:
    """
    Unified trajectory optimizer using CasADi + IPOPT.
    
    Supports multiple obstacle types with exact gradients.
    """
    
    def __init__(self, urdf_path: str, verbose: bool = True):
        self.urdf_path = Path(urdf_path)
        self.verbose = verbose
        
        # Standard Pinocchio model
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        # Limits
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit
        self.v_max = self.model.velocityLimit
        self.tau_max = np.array([
            self.model.effortLimit[i] if self.model.effortLimit[i] > 0 else 50.0
            for i in range(self.nv)
        ])
        
        # Build CasADi symbolic models
        self._build_casadi_functions()
        
        if self.verbose:
            print(f"CasADi Optimizer V6 initialized:")
            print(f"  Robot: {self.urdf_path.name}")
            print(f"  DOF: {self.nq}")
    
    def _build_casadi_functions(self):
        """Build exact computational graphs for dynamics and kinematics"""
        if self.verbose:
            print("Building CasADi computational graphs...")
        
        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        
        # Symbolic variables
        cq = ca.SX.sym('q', self.nq, 1)
        cv = ca.SX.sym('v', self.nv, 1)
        ca_acc = ca.SX.sym('a', self.nv, 1)
        
        # Inverse Dynamics (RNEA)
        ctau = cpin.rnea(cmodel, cdata, cq, cv, ca_acc)
        self.rnea_func = ca.Function('rnea', [cq, cv, ca_acc], [ctau])
        
        # Forward Kinematics for multiple frames
        # We'll create FK functions for all collision-relevant links
        self.fk_funcs = {}
        
        # Get end-effector (last frame)
        ee_frame_id = self.model.nframes - 1
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        ee_pos = cdata.oMf[ee_frame_id].translation
        self.fk_funcs['ee'] = ca.Function('fk_ee', [cq], [ee_pos])
        
        # For more complex collision checking, we'd add FK for all links
        # For now, we'll use end-effector as primary collision point
        
        if self.verbose:
            print("  ✓ RNEA function built")
            print("  ✓ FK functions built")
    
    def _distance_point_to_sphere(self, point: ca.SX, sphere_center: np.ndarray, sphere_radius: float) -> ca.SX:
        """
        Exact distance from point to sphere surface.
        
        Returns: distance (positive = outside, negative = inside)
        """
        dist_to_center = ca.norm_2(point - sphere_center)
        return dist_to_center - sphere_radius
    
    def _distance_point_to_capsule(
        self, 
        point: ca.SX, 
        capsule_start: np.ndarray, 
        capsule_end: np.ndarray, 
        capsule_radius: float
    ) -> ca.SX:
        """
        Exact distance from point to capsule (line segment with radius).
        
        Capsule = sphere swept along line segment
        """
        # Vector from start to end
        segment = capsule_end - capsule_start
        segment_length = np.linalg.norm(segment)
        
        if segment_length < 1e-6:
            # Degenerate case: capsule is a sphere
            return self._distance_point_to_sphere(point, capsule_start, capsule_radius)
        
        # Project point onto line segment
        t = ca.dot(point - capsule_start, segment) / (segment_length ** 2)
        t_clamped = ca.fmin(ca.fmax(t, 0), 1)  # Clamp to [0, 1]
        
        # Closest point on segment
        closest_point = capsule_start + t_clamped * segment
        
        # Distance to capsule surface
        dist_to_axis = ca.norm_2(point - closest_point)
        return dist_to_axis - capsule_radius
    
    def _distance_point_to_box(
        self, 
        point: ca.SX, 
        box_center: np.ndarray, 
        box_dims: np.ndarray
    ) -> ca.SX:
        """
        Exact signed distance from point to axis-aligned box.
        
        box_dims = [width, depth, height] (full dimensions, not half-extents)
        """
        half_dims = box_dims / 2.0
        
        # Distance in each dimension
        dx = ca.fabs(point[0] - box_center[0]) - half_dims[0]
        dy = ca.fabs(point[1] - box_center[1]) - half_dims[1]
        dz = ca.fabs(point[2] - box_center[2]) - half_dims[2]
        
        # Outside distance
        outside_dist = ca.norm_2(ca.vertcat(
            ca.fmax(dx, 0),
            ca.fmax(dy, 0),
            ca.fmax(dz, 0)
        ))
        
        # Inside distance (negative if inside)
        inside_dist = ca.fmin(ca.fmax(dx, ca.fmax(dy, dz)), 0)
        
        return outside_dist + inside_dist
    
    def optimize(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        obstacles: List[Dict] = None,
        n_waypoints: int = 50,
        weights: Dict[str, float] = None,
        safety_margin: float = 0.02
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Optimize trajectory with multi-objective cost and obstacle avoidance.
        
        Args:
            q_start: Start configuration
            q_goal: Goal configuration
            obstacles: List of obstacle dicts, each with:
                - 'type': 'sphere', 'capsule', or 'box'
                - 'sphere': {'center': [x,y,z], 'radius': r}
                - 'capsule': {'start': [x,y,z], 'end': [x,y,z], 'radius': r}
                - 'box': {'center': [x,y,z], 'dims': [w,d,h]}
            n_waypoints: Number of trajectory waypoints
            weights: Cost function weights
            safety_margin: Minimum clearance to obstacles (m)
        
        Returns:
            q_traj: Optimized trajectory (N × DOF)
            dt: Optimized time step
            metrics: Dictionary of results
        """
        
        if obstacles is None:
            obstacles = []
        
        # Default weights
        default_weights = {
            'smoothness': 1.0,
            'torque': 0.1,
            'jerk': 0.1,
            'duration': 10.0,
            'collision': 1000.0  # High penalty for collision violations
        }
        
        if weights is not None:
            default_weights.update(weights)
        weights = default_weights
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"CASADI OPTIMIZER V6 - MULTI-OBJECTIVE OPTIMIZATION")
            print(f"{'='*80}")
            print(f"Waypoints: {n_waypoints}")
            print(f"Obstacles: {len(obstacles)}")
            print(f"Weights: {weights}")
            print(f"Safety margin: {safety_margin}m")
        
        t_start = time.time()
        
        # =====================================================================
        # BUILD OPTIMIZATION PROBLEM
        # =====================================================================
        
        opti = ca.Opti()
        
        # Decision variables
        Q = opti.variable(self.nq, n_waypoints)  # Trajectory
        dt = opti.variable()                      # Time step
        
        # Initial guess (linear interpolation)
        Q_guess = np.linspace(q_start, q_goal, n_waypoints).T
        opti.set_initial(Q, Q_guess)
        opti.set_initial(dt, 0.05)
        
        # Cost components
        cost_smoothness = 0
        cost_torque = 0
        cost_jerk = 0
        cost_duration = n_waypoints * dt
        
        # =====================================================================
        # DYNAMICS AND SMOOTHNESS
        # =====================================================================
        
        for t in range(1, n_waypoints - 1):
            # Velocity and acceleration (finite differences)
            v_t = (Q[:, t+1] - Q[:, t-1]) / (2 * dt)
            a_t = (Q[:, t+1] - 2*Q[:, t] + Q[:, t-1]) / (dt**2)
            
            # Inverse dynamics
            tau_t = self.rnea_func(Q[:, t], v_t, a_t)
            
            # Smoothness cost (acceleration magnitude)
            cost_smoothness += ca.sumsqr(a_t) * dt
            
            # Torque cost
            cost_torque += ca.sumsqr(tau_t) * dt
            
            # Constraints
            opti.subject_to(opti.bounded(-self.v_max, v_t, self.v_max))
            opti.subject_to(opti.bounded(-self.tau_max, tau_t, self.tau_max))
        
        # Jerk cost
        for t in range(1, n_waypoints - 2):
            a_t = (Q[:, t+1] - 2*Q[:, t] + Q[:, t-1]) / (dt**2)
            a_t1 = (Q[:, t+2] - 2*Q[:, t+1] + Q[:, t]) / (dt**2)
            jerk_t = (a_t1 - a_t) / dt
            cost_jerk += ca.sumsqr(jerk_t) * dt
        
        # =====================================================================
        # COLLISION AVOIDANCE
        # =====================================================================
        
        collision_violations = 0
        
        for obs in obstacles:
            obs_type = obs['type']
            
            for t in range(n_waypoints):
                # Get end-effector position at this waypoint
                ee_pos = self.fk_funcs['ee'](Q[:, t])
                
                # Compute distance based on obstacle type
                if obs_type == 'sphere':
                    dist = self._distance_point_to_sphere(
                        ee_pos,
                        np.array(obs['center']),
                        obs['radius']
                    )
                
                elif obs_type == 'capsule':
                    dist = self._distance_point_to_capsule(
                        ee_pos,
                        np.array(obs['start']),
                        np.array(obs['end']),
                        obs['radius']
                    )
                
                elif obs_type == 'box':
                    dist = self._distance_point_to_box(
                        ee_pos,
                        np.array(obs['center']),
                        np.array(obs['dims'])
                    )
                
                else:
                    raise ValueError(f"Unknown obstacle type: {obs_type}")
                
                # Hard constraint: maintain safety margin
                opti.subject_to(dist >= safety_margin)
                
                # Soft penalty for being too close (helps convergence)
                penalty_threshold = safety_margin + 0.05
                penalty = ca.fmax(0, penalty_threshold - dist) ** 2
                collision_violations += penalty
        
        # =====================================================================
        # TOTAL COST
        # =====================================================================
        
        total_cost = (
            weights['smoothness'] * cost_smoothness +
            weights['torque'] * cost_torque +
            weights['jerk'] * cost_jerk +
            weights['duration'] * cost_duration +
            weights['collision'] * collision_violations
        )
        
        opti.minimize(total_cost)
        
        # =====================================================================
        # BOUNDARY CONDITIONS AND LIMITS
        # =====================================================================
        
        # Fixed start and goal
        opti.subject_to(Q[:, 0] == q_start)
        opti.subject_to(Q[:, -1] == q_goal)
        
        # Joint limits
        for i in range(self.nq):
            opti.subject_to(opti.bounded(self.q_min[i], Q[i, :], self.q_max[i]))
        
        # Time step bounds
        opti.subject_to(dt >= 0.001)
        opti.subject_to(dt <= 0.5)
        
        # =====================================================================
        # SOLVE
        # =====================================================================
        
        # IPOPT options
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 500,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "print_level": 5 if self.verbose else 0,
            "sb": "yes" if not self.verbose else "no"  # Suppress banner
        }
        
        opti.solver("ipopt", p_opts, s_opts)
        
        try:
            sol = opti.solve()
            q_opt = sol.value(Q).T
            dt_opt = sol.value(dt)
            success = True
            final_cost = sol.value(total_cost)
            
            # Extract cost breakdown
            cost_breakdown = {
                'smoothness': sol.value(cost_smoothness),
                'torque': sol.value(cost_torque),
                'jerk': sol.value(cost_jerk),
                'duration': sol.value(cost_duration),
                'collision': sol.value(collision_violations)
            }
            
        except RuntimeError as e:
            if self.verbose:
                print(f"\n⚠️  Solver did not fully converge: {e}")
                print("Returning best available solution...")
            
            q_opt = opti.debug.value(Q).T
            dt_opt = opti.debug.value(dt)
            success = False
            final_cost = opti.debug.value(total_cost)
            
            cost_breakdown = {
                'smoothness': opti.debug.value(cost_smoothness),
                'torque': opti.debug.value(cost_torque),
                'jerk': opti.debug.value(cost_jerk),
                'duration': opti.debug.value(cost_duration),
                'collision': opti.debug.value(collision_violations)
            }
        
        solve_time = time.time() - t_start
        
        # =====================================================================
        # COMPUTE FINAL METRICS
        # =====================================================================
        
        # Compute actual energy, peak torque, etc.
        energy = 0.0
        peak_torque = 0.0
        
        for t in range(1, n_waypoints - 1):
            v_t = (q_opt[t+1] - q_opt[t-1]) / (2 * dt_opt)
            a_t = (q_opt[t+1] - 2*q_opt[t] + q_opt[t-1]) / (dt_opt**2)
            
            tau_t = pin.rnea(self.model, self.model.createData(), q_opt[t], v_t, a_t)
            power = np.abs(np.dot(tau_t, v_t))
            energy += power * dt_opt
            peak_torque = max(peak_torque, np.max(np.abs(tau_t)))
        
        metrics = {
            'success': success,
            'solve_time': solve_time,
            'duration': n_waypoints * dt_opt,
            'dt': dt_opt,
            'final_cost': final_cost,
            'cost_breakdown': cost_breakdown,
            'energy': energy,
            'peak_torque': peak_torque,
            'n_waypoints': n_waypoints,
            'n_obstacles': len(obstacles)
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"  Success: {success}")
            print(f"  Solve time: {solve_time:.2f}s")
            print(f"  Duration: {metrics['duration']:.2f}s")
            print(f"  Energy: {energy:.2f} J")
            print(f"  Peak torque: {peak_torque:.2f} Nm")
            print(f"  Final cost: {final_cost:.2f}")
            print(f"{'='*80}\n")
        
        return q_opt, dt_opt, metrics


if __name__ == "__main__":
    print("="*80)
    print("TESTING CASADI OPTIMIZER V6 - MULTI-OBSTACLE")
    print("="*80)
    
    urdf = "/scratch/anshb3/ovla/robots/franka/franka_panda_with_inertia.urdf"
    
    # Initialize optimizer
    optimizer = CasadiOptimizerV6(urdf, verbose=True)
    
    # FIX: Use the valid neutral pose instead of np.zeros!
    q_start = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    q_goal = np.array([0.5, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5])
    
    # Define obstacles
    obstacles = [
        {
            'type': 'sphere',
            'center': [0.4, 0.0, 0.4],
            'radius': 0.12
        },
        {
            'type': 'box',
            'center': [0.3, 0.3, 0.3],
            'dims': [0.1, 0.1, 0.3]
        }
    ]
    
    print(f"\nTest case:")
    print(f"  Obstacles: 1 sphere + 1 box")
    print(f"  Start: {q_start}")
    print(f"  Goal: {q_goal}")
    
    # Optimize
    q_traj, dt, metrics = optimizer.optimize(
        q_start, 
        q_goal,
        obstacles=obstacles,
        n_waypoints=50
    )
    
    print("\n✅ TEST COMPLETE")
    print(f"Generated trajectory: {q_traj.shape}")
