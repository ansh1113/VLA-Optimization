import coal
"""
CHOMP (Covariant Hamiltonian Optimization for Motion Planning) - V2

Production-grade collision avoidance optimizer.

Reference:
Ratliff et al. "CHOMP: Gradient Optimization Techniques for 
Efficient Motion Planning" (ICRA 2009)
"""

import numpy as np
import pinocchio as pin
from typing import Dict, Tuple, Union
from pathlib import Path


class CHOMPCollisionAvoidance:
    """
    CHOMP-based trajectory optimization for collision avoidance.
    
    Uses covariant gradient descent to find smooth, collision-free paths.
    """
    
    def __init__(
        self, 
        urdf_path: Union[str, Path], 
        collision_detector,
        verbose: bool = False
    ):
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.collision_detector = collision_detector
        self.verbose = verbose
        
        self.nq = self.model.nq

    def _compute_smoothness_matrix(self, n_waypoints: int) -> np.ndarray:
        """
        Compute smoothness metric matrix A for covariant gradient.
        
        A is derived from K^T K where K is the finite-difference
        acceleration operator.
        """
        n_free = n_waypoints - 2
        A = np.zeros((n_free, n_free))
        
        for i in range(n_free):
            A[i, i] = 6.0
            if i + 1 < n_free:
                A[i, i+1] = -4.0
                A[i+1, i] = -4.0
            if i + 2 < n_free:
                A[i, i+2] = 1.0
                A[i+2, i] = 1.0
        
        # Regularization for numerical stability
        A += np.eye(n_free) * 1e-4
        return A

    def obstacle_cost(self, q: np.ndarray, epsilon: float = 0.05) -> float:
        """
        Compute obstacle cost for a configuration.
        
        Uses smooth potential field:
        - c(d) = 0                if d >= 0 (no collision)
        - c(d) = (d+ε)²/2ε       if -ε < d < 0 (near collision)
        - c(d) = -d + ε/2        if d <= -ε (deep penetration)
        
        Args:
            q: Configuration
            epsilon: Safety margin (m)
        
        Returns:
            Total obstacle cost
        """
        report = self.collision_detector.get_collision_report(q)
        
        if not report['has_collision']:
            return 0.0
        
        total_cost = 0.0
        for collision in report['collisions']:
            dist = collision['distance']  # Negative if penetrating
            
            if dist < -epsilon:
                cost = -dist + epsilon / 2.0
            elif dist < 0:
                cost = (dist + epsilon) ** 2 / (2.0 * epsilon)
            else:
                cost = 0.0
            
            total_cost += cost
        
        return total_cost
    
    def obstacle_gradient(
        self, 
        trajectory: np.ndarray, 
        epsilon: float = 0.05,
        delta: float = 1e-3
    ) -> np.ndarray:
        """
        Compute obstacle cost gradient via finite differences.
        
        Uses central differences for accuracy.
        """
        n_waypoints = trajectory.shape[0]
        gradient = np.zeros_like(trajectory)
        
        for t in range(n_waypoints):
            for i in range(self.nq):
                q_plus = trajectory[t].copy()
                q_plus[i] += delta
                
                q_minus = trajectory[t].copy()
                q_minus[i] -= delta
                
                cost_plus = self.obstacle_cost(q_plus, epsilon)
                cost_minus = self.obstacle_cost(q_minus, epsilon)
                
                gradient[t, i] = (cost_plus - cost_minus) / (2.0 * delta)
        
        return gradient
    
    def optimize(
        self,
        trajectory_init: np.ndarray,
        max_iter: int = 150,
        learning_rate: float = 1.0,
        lr_decay: float = 0.99,
        epsilon: float = 0.05,
        max_grad_norm: float = 10.0,
        momentum: float = 0.8,
        perturbation_scale: float = 0.05
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize trajectory to avoid collisions using CHOMP.
        """
        trajectory = trajectory_init.copy()
        n_waypoints = trajectory.shape[0]
        
        # Best-state tracker to ensure we don't drift into worse collisions
        best_trajectory = trajectory.copy()
        best_collision_count = sum(1 for t in range(n_waypoints) if self.collision_detector.get_collision_report(trajectory[t])['has_collision'])
        
        # Fix start and goal
        q_start = trajectory[0].copy()
        q_goal = trajectory[-1].copy()
        
        # Compute smoothness metric
        A = self._compute_smoothness_matrix(n_waypoints)
        A_inv = np.linalg.inv(A)
        
        # Velocity for Nesterov Momentum
        vel = np.zeros_like(trajectory)
        
        history = {'obs_cost': [], 'collision_count': []}
        stuck_counter = 0
        
        for iteration in range(max_iter):
            current_lr = learning_rate * (lr_decay ** iteration)
            
            # --- BETTER: ACCELERATED GRADIENT ---
            traj_lookahead = trajectory + momentum * vel
            grad_obs = self.obstacle_gradient(traj_lookahead, epsilon)
            
            g_norm = np.linalg.norm(grad_obs[1:-1])
            if g_norm > max_grad_norm:
                grad_obs[1:-1] *= (max_grad_norm / g_norm)
            
            projected_grad = A_inv @ grad_obs[1:-1]
            
            # Update trajectory
            vel[1:-1] = momentum * vel[1:-1] - current_lr * projected_grad
            trajectory[1:-1] += vel[1:-1]
            
            # Joint limit enforcement
            for t in range(1, n_waypoints - 1):
                trajectory[t] = np.clip(trajectory[t], self.model.lowerPositionLimit, self.model.upperPositionLimit)
            
            trajectory[0], trajectory[-1] = q_start, q_goal
            
            # --- BETTER: BEST-STATE TRACKING & STOCHASTIC RECOVERY ---
            collision_count = sum(1 for t in range(n_waypoints) if self.collision_detector.get_collision_report(trajectory[t])['has_collision'])
            
            if collision_count < best_collision_count:
                best_collision_count = collision_count
                best_trajectory = trajectory.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1

            # If we are stuck or getting worse, revert to best and apply a smooth wiggle
            if (stuck_counter > 10 or collision_count > best_collision_count) and collision_count > 0:
                trajectory = best_trajectory.copy()
                # Apply a random covariant nudge (smoothed by A_inv)
                noise = np.random.normal(0, perturbation_scale, trajectory[1:-1].shape)
                trajectory[1:-1] += A_inv @ noise 
                vel.fill(0) # Reset momentum on restart
                stuck_counter = 0

            # Logging
            obs_cost = sum(self.obstacle_cost(trajectory[t], epsilon) for t in range(n_waypoints))
            history['obs_cost'].append(obs_cost)
            history['collision_count'].append(collision_count)
            
            if self.verbose and iteration % 10 == 0:
                print(f"  CHOMP iter {iteration:03d}: cost={obs_cost:.4f}, collisions={collision_count}/{n_waypoints}")
            
            if collision_count == 0:
                if self.verbose:
                    print(f"  ✅ Collision-free at iteration {iteration}")
                break
        
        return best_trajectory if best_collision_count < collision_count else trajectory, {
            'converged': best_collision_count == 0,
            'iterations': iteration + 1,
            'final_collision_count': min(collision_count, best_collision_count),
            'final_cost': obs_cost,
            'history': history
        }
