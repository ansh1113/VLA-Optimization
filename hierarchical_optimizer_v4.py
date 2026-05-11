import coal
"""
HierarchicalOptimizerV4

Complete pipeline: ProxQP → CHOMP → TOPP-RA
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

from collision_detector_v3 import CollisionDetectorV3
from trajectory_optimizer_proxqp_v2 import TrajectoryOptimizerProxQPV2
from collision_avoidance_chomp_v2 import CHOMPCollisionAvoidance
from balance_checker_v2 import BalanceCheckerV2
from energy_optimizer_v2 import EnergyOptimizerV2
from workspace_optimizer_v2 import WorkspaceOptimizerV2


class HierarchicalOptimizerV4:
    """
    Layer 3: Physics-based trajectory optimization.
    
    Input: q_start, q_goal, URDF
    Output: Collision-free, constraint-satisfying, energy-optimal trajectory
    
    Works for ANY robot morphology.
    """
    
    def __init__(
        self, 
        urdf_path: Union[str, Path], 
        ee_frame_name: Optional[str] = None,
        verbose: bool = False
    ):
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        self.urdf_path = urdf_path
        self.verbose = verbose
        
        # Initialize all components
        self.collision_detector = CollisionDetectorV3(
            urdf_path, 
            safety_margin=0.0, 
            verbose=False
        )
        
        self.trajectory_optimizer = TrajectoryOptimizerProxQPV2(
            urdf_path, 
            collision_detector=self.collision_detector,
            verbose=False
        )
        
        self.collision_avoidance = CHOMPCollisionAvoidance(
            urdf_path,
            self.collision_detector,
            verbose=self.verbose
        )
        
        self.balance_checker = BalanceCheckerV2(urdf_path, verbose=False)
        self.energy_optimizer = EnergyOptimizerV2(urdf_path, verbose=False)
        self.workspace_optimizer = WorkspaceOptimizerV2(urdf_path, ee_frame_name, verbose=False)
        
        self.nq = self.collision_detector.nq
        
        if self.verbose:
            print(f"HierarchicalOptimizerV4 initialized:")
            print(f"  Robot: {urdf_path.name}")
            print(f"  DOF: {self.nq}")
    
    def optimize_trajectory(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        n_waypoints: int = 20,
        dt: float = 0.1,
        enable_collision_avoidance: bool = True,
        chomp_params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete optimization pipeline.
        
        Args:
            q_start: Start configuration
            q_goal: Goal configuration
            n_waypoints: Number of waypoints
            dt: Time step
            enable_collision_avoidance: Enable CHOMP
            chomp_params: CHOMP hyperparameters
        
        Returns:
            trajectory: Optimized trajectory
            metrics: Detailed metrics
        """
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"LAYER 3 OPTIMIZATION")
            print(f"{'='*80}")
        
        metrics = {}
        
        # STEP 1: ProxQP - Constraint-satisfying smooth initialization
        if self.verbose:
            print(f"\nStep 1: ProxQP constraint satisfaction")
        
        trajectory, proxqp_metrics = self.trajectory_optimizer.optimize_trajectory(
            q_start, q_goal, n_waypoints, dt
        )
        
        metrics['proxqp'] = proxqp_metrics
        
        if self.verbose:
            print(f" ProxQP converged in {proxqp_metrics['iterations']} iterations")
        
        # Count initial collisions
        init_collisions = sum(
            1 for t in range(n_waypoints)
            if self.collision_detector.get_collision_report(trajectory[t])['has_collision']
        )
        
        if self.verbose:
            print(f"  Initial collisions: {init_collisions}/{n_waypoints}")
        
        # STEP 2: CHOMP - Collision avoidance
        if enable_collision_avoidance and init_collisions > 0:
            if self.verbose:
                print(f"\nStep 2: CHOMP collision avoidance...")
            
            # Default CHOMP parameters
            chomp_defaults = {
                'max_iter': 150,
                'learning_rate': 1.0,
                'lr_decay': 0.99,
                'epsilon': 0.08,
                'max_grad_norm': 10.0,
                'momentum': 0.8,
                'perturbation_scale': 0.05
            }
            
            if chomp_params is not None:
                chomp_defaults.update(chomp_params)
            
            trajectory, chomp_metrics = self.collision_avoidance.optimize(
                trajectory,
                **chomp_defaults
            )
            
            metrics['chomp'] = chomp_metrics
            
            if self.verbose:
                if chomp_metrics['converged']:
                    print(f" CHOMP: Collision-free in {chomp_metrics['iterations']} iterations")
                else:
                    print(f" CHOMP: {chomp_metrics['final_collision_count']}/{n_waypoints} collisions remain")
        
        elif self.verbose:
            print(f"\nStep 2: Skipping CHOMP (already collision-free)")
            metrics['chomp'] = {'converged': True, 'skipped': True}
        else:
            metrics['chomp'] = {'converged': True, 'skipped': True}
        
        # STEP 3: TOPP-RA - Time-optimal retiming
        if self.verbose:
            print(f"\nStep 3: TOPP-RA time-optimal retiming")
        
        trajectory, final_dt, energy_metrics = self.energy_optimizer.retime_trajectory(
            trajectory, dt
        )
        
        metrics['energy'] = energy_metrics
        
        if self.verbose:
            print(f"  TOPP-RA complete")
            print(f"  Duration: {energy_metrics.get('optimal_duration', n_waypoints * dt):.2f}s")
            print(f"  Peak torque: {energy_metrics['peak_torque']:.1f} Nm")
        
        # FINAL METRICS
        final_collisions = sum(
            1 for t in range(trajectory.shape[0])
            if self.collision_detector.get_collision_report(trajectory[t])['has_collision']
        )
        
        metrics['final'] = {
            'collision_free': final_collisions == 0,
            'collision_count': final_collisions,
            'goal_reached': np.linalg.norm(trajectory[-1] - q_goal) < 1e-3
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"  Collision-free: {metrics['final']['collision_free']}")
            print(f"  Goal reached: {metrics['final']['goal_reached']}")
            print(f"  Energy: {metrics['energy']['total_energy']:.2f} J")
            print(f"{'='*80}\n")
        
        return trajectory, metrics


if __name__ == "__main__":
    
    urdf = "/scratch/anshb3/ovla/robots/franka/franka_panda_with_inertia.urdf"
    optimizer = HierarchicalOptimizerV4(urdf, verbose=True)
    
    q_start = np.zeros(optimizer.nq)
    q_range = optimizer.trajectory_optimizer.model.upperPositionLimit - optimizer.trajectory_optimizer.model.lowerPositionLimit
    q_goal = optimizer.trajectory_optimizer.model.lowerPositionLimit + np.random.rand(optimizer.nq) * q_range * 0.5 + q_range * 0.25
    
    trajectory, metrics = optimizer.optimize_trajectory(q_start, q_goal)
