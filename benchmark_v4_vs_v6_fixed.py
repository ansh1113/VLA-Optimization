import coal
"""
BENCHMARK V4 vs V6 - FIXED QUADRUPED INITIALIZATION

Uses safe starting poses for all robots.
"""

import numpy as np
import json
import time
from pathlib import Path

from hierarchical_optimizer_v4 import HierarchicalOptimizerV4
from casadi_optimizer_v6_multi_obstacle import CasadiOptimizerV6

np.random.seed(42)


def get_safe_start_pose(robot_name, model):
    """
    Get safe starting pose for each robot type.
    
    Arms: Midpoint of limits (neutral pose)
    Quadrupeds: Crouched stance (legs bent, stable)
    """
    
    if robot_name in ["Laikago", "Go1"]:
        # Quadruped: Crouched stance
        # Joint order: [hip_abd, hip_pitch, knee] × 4 legs
        # Safe pose: hips slightly abducted, legs bent
        q_start = np.zeros(model.nq)
        
        for leg in range(4):
            base_idx = leg * 3
            q_start[base_idx + 0] = 0.0      # Hip abduction: neutral
            q_start[base_idx + 1] = 0.9      # Hip pitch: bent forward
            q_start[base_idx + 2] = -1.8     # Knee: bent (negative = bent)
        
        # Clip to limits just in case
        q_start = np.clip(q_start, model.lowerPositionLimit, model.upperPositionLimit)
        
    else:
        # Arms & humanoids: Midpoint of limits
        q_start = (model.lowerPositionLimit + model.upperPositionLimit) / 2.0
    
    return q_start


class BenchmarkV4vsV6Fixed:
    
    def __init__(self):
        self.results = {}
    
    def benchmark_robot(
        self,
        robot_name: str,
        urdf_path: str,
        n_trials: int = 10
    ):
        print(f"\n{'='*80}")
        print(f"BENCHMARKING: {robot_name}")
        print(f"{'='*80}")
        
        import pinocchio as pin
        model = pin.buildModelFromUrdf(urdf_path)
        
        print(f"Initializing V4 (Pipeline)...")
        v4_optimizer = HierarchicalOptimizerV4(urdf_path, verbose=False)
        
        print(f"Initializing V6 (CasADi)...")
        v6_optimizer = CasadiOptimizerV6(urdf_path, verbose=False)
        
        v4_results = []
        v6_results = []
        
        print(f"\nRunning {n_trials} trials...")
        
        for trial in range(n_trials):
            # Safe starting pose
            q_start = get_safe_start_pose(robot_name, model)
            
            # Random goal
            q_range = model.upperPositionLimit - model.lowerPositionLimit
            q_target = model.lowerPositionLimit + np.random.rand(model.nq) * q_range * 0.4 + q_range * 0.3
            q_target = np.clip(q_target, model.lowerPositionLimit, model.upperPositionLimit)
            
            print(f"  Trial {trial+1}/{n_trials}...", end='', flush=True)
            
            # V4
            t_start = time.time()
            try:
                v4_traj, v4_metrics = v4_optimizer.optimize_trajectory(
                    q_start, q_target, n_waypoints=50
                )
                v4_time = time.time() - t_start
                
                v4_results.append({
                    'success': v4_metrics['final']['collision_free'],
                    'energy': v4_metrics['energy']['total_energy'],
                    'peak_torque': v4_metrics['energy']['peak_torque'],
                    'duration': v4_metrics['energy'].get('optimal_duration', 5.0),
                    'solve_time': v4_time
                })
            except Exception as e:
                print(f" V4-ERR", end='')
                v4_results.append({
                    'success': False, 'energy': 9999, 'peak_torque': 9999,
                    'duration': 9999, 'solve_time': 9999
                })
            
            # V6
            t_start = time.time()
            try:
                v6_traj, v6_dt, v6_metrics = v6_optimizer.optimize(
                    q_start, q_target, obstacles=[], n_waypoints=50
                )
                v6_time = time.time() - t_start
                
                v6_results.append({
                    'success': v6_metrics['success'],
                    'energy': v6_metrics['energy'],
                    'peak_torque': v6_metrics['peak_torque'],
                    'duration': v6_metrics['duration'],
                    'solve_time': v6_metrics['solve_time']
                })
            except Exception as e:
                print(f" V6-ERR", end='')
                v6_results.append({
                    'success': False, 'energy': 9999, 'peak_torque': 9999,
                    'duration': 9999, 'solve_time': 9999
                })
            
            print(" ✓")
        
        # Aggregate
        def aggregate(results):
            successes = [r for r in results if r['success']]
            if len(successes) == 0:
                return {
                    'success_rate': 0.0,
                    'energy': {'mean': 9999, 'std': 0},
                    'peak_torque': {'mean': 9999, 'std': 0},
                    'solve_time': {'mean': 9999, 'std': 0}
                }
            
            return {
                'success_rate': len(successes) / len(results),
                'energy': {
                    'mean': np.mean([r['energy'] for r in successes]),
                    'std': np.std([r['energy'] for r in successes])
                },
                'peak_torque': {
                    'mean': np.mean([r['peak_torque'] for r in successes]),
                    'std': np.std([r['peak_torque'] for r in successes])
                },
                'solve_time': {
                    'mean': np.mean([r['solve_time'] for r in successes]),
                    'std': np.std([r['solve_time'] for r in successes])
                }
            }
        
        v4_agg = aggregate(v4_results)
        v6_agg = aggregate(v6_results)
        
        # Print
        print(f"\n{robot_name} ({model.nq} DOF):")
        print(f"\n  V4: {v4_agg['success_rate']*100:.0f}% success, {v4_agg['energy']['mean']:.1f}J, {v4_agg['peak_torque']['mean']:.1f}Nm, {v4_agg['solve_time']['mean']:.2f}s")
        print(f"  V6: {v6_agg['success_rate']*100:.0f}% success, {v6_agg['energy']['mean']:.1f}J, {v6_agg['peak_torque']['mean']:.1f}Nm, {v6_agg['solve_time']['mean']:.2f}s")
        
        if v4_agg['success_rate'] > 0 and v6_agg['success_rate'] > 0:
            energy_imp = ((v4_agg['energy']['mean'] - v6_agg['energy']['mean']) / v4_agg['energy']['mean']) * 100
            torque_imp = ((v4_agg['peak_torque']['mean'] - v6_agg['peak_torque']['mean']) / v4_agg['peak_torque']['mean']) * 100
            print(f"\n  📊 V6 IMPROVEMENTS: Energy {energy_imp:+.1f}%, Torque {torque_imp:+.1f}%")
        
        self.results[robot_name] = {'v4': v4_agg, 'v6': v6_agg, 'dof': model.nq}
    
    def run(self):
        print("="*80)
        print("BENCHMARK: V4 vs V6 (FIXED QUADRUPED INIT)")
        print("="*80)
        
        robots = [
            ("Franka Panda", "/scratch/anshb3/ovla/robots/franka/franka_panda_with_inertia.urdf"),
            ("UR5e", "/scratch/anshb3/ovla/robots/ur5e/ur5e.urdf"),
            ("G1 Humanoid", "/scratch/anshb3/ovla/robots/unitree_ros/robots/g1_description/g1_23dof.urdf"),
            ("Laikago", "/scratch/anshb3/ovla/robots/unitree_ros/robots/laikago_description/urdf/laikago.urdf"),
            ("Go1", "/scratch/anshb3/ovla/robots/unitree_ros/robots/go1_description/urdf/go1.urdf"),
        ]
        
        for robot_name, urdf_path in robots:
            try:
                self.benchmark_robot(robot_name, urdf_path, n_trials=10)
            except KeyboardInterrupt:
                print("\n\nInterrupted")
                break
            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        with open('v4_vs_v6_fixed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ RESULTS: v4_vs_v6_fixed_results.json")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    BenchmarkV4vsV6Fixed().run()
