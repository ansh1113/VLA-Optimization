"""
BENCHMARK: Multiple random scenarios to get robust statistics
"""

import numpy as np
import json
from hierarchical_optimizer_v4 import HierarchicalOptimizerV4
from casadi_optimizer_v6_multi_obstacle import CasadiOptimizerV6
import pinocchio as pin
import time


def random_valid_config(model, seed=None):
    """Generate random valid configuration"""
    if seed is not None:
        np.random.seed(seed)
    
    q_range = model.upperPositionLimit - model.lowerPositionLimit
    q = model.lowerPositionLimit + np.random.rand(model.nq) * q_range
    
    return q


def benchmark_robot_multi_scenario(urdf_path, robot_name, n_scenarios=10):
    """Run multiple random scenarios"""
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {robot_name} ({n_scenarios} scenarios)")
    print(f"{'='*80}")
    
    model = pin.buildModelFromUrdf(urdf_path)
    
    results = {
        'robot': robot_name,
        'dof': int(model.nq),
        'scenarios': []
    }
    
    v4_opt = HierarchicalOptimizerV4(urdf_path, verbose=False)
    
    try:
        v6_opt = CasadiOptimizerV6(urdf_path, verbose=False)
        v6_available = True
    except:
        v6_available = False
        print("V6 initialization failed")
    
    for scenario_idx in range(n_scenarios):
        print(f"\nScenario {scenario_idx+1}/{n_scenarios}:")
        
        # Generate random start/goal
        q_start = random_valid_config(model, seed=scenario_idx*2)
        q_goal = random_valid_config(model, seed=scenario_idx*2+1)
        
        scenario = {
            'scenario_id': scenario_idx,
            'v4': {},
            'v6': {}
        }
        
        # V4
        try:
            start_time = time.time()
            traj_v4, metrics_v4 = v4_opt.optimize_trajectory(q_start, q_goal, n_waypoints=50)
            v4_time = time.time() - start_time
            
            scenario['v4'] = {
                'energy': float(metrics_v4['energy']['total_energy']),
                'time': float(v4_time),
                'success': True
            }
            print(f"  V4: {metrics_v4['energy']['total_energy']:.1f} J, {v4_time:.2f}s")
        except Exception as e:
            scenario['v4'] = {'success': False, 'error': str(e)}
            print(f"  V4 FAILED: {e}")
        
        # V6
        if v6_available and scenario['v4']['success']:
            try:
                start_time = time.time()
                traj_v6, dt, metrics_v6 = v6_opt.optimize(q_start, q_goal, n_waypoints=50)
                v6_time = time.time() - start_time
                
                scenario['v6'] = {
                    'energy': float(metrics_v6['energy']),
                    'time': float(v6_time),
                    'success': True
                }
                
                reduction = (scenario['v4']['energy'] - scenario['v6']['energy']) / scenario['v4']['energy'] * 100
                scenario['v6']['reduction'] = float(reduction)
                
                print(f"  V6: {metrics_v6['energy']:.1f} J, {v6_time:.2f}s ({reduction:+.1f}%)")
                
            except Exception as e:
                scenario['v6'] = {'success': False, 'error': str(e)[:100]}
                print(f"  V6 FAILED: {str(e)[:80]}")
        else:
            scenario['v6'] = {'success': False}
        
        results['scenarios'].append(scenario)
    
    # Calculate statistics
    v4_energies = [s['v4']['energy'] for s in results['scenarios'] if s['v4']['success']]
    v4_times = [s['v4']['time'] for s in results['scenarios'] if s['v4']['success']]
    
    v6_energies = [s['v6']['energy'] for s in results['scenarios'] if s['v6'].get('success', False)]
    v6_times = [s['v6']['time'] for s in results['scenarios'] if s['v6'].get('success', False)]
    v6_reductions = [s['v6']['reduction'] for s in results['scenarios'] if s['v6'].get('success', False)]
    
    results['summary'] = {
        'v4': {
            'mean_energy': float(np.mean(v4_energies)) if v4_energies else None,
            'std_energy': float(np.std(v4_energies)) if v4_energies else None,
            'mean_time': float(np.mean(v4_times)) if v4_times else None,
            'success_rate': len(v4_energies) / n_scenarios
        },
        'v6': {
            'mean_energy': float(np.mean(v6_energies)) if v6_energies else None,
            'std_energy': float(np.std(v6_energies)) if v6_energies else None,
            'mean_time': float(np.mean(v6_times)) if v6_times else None,
            'mean_reduction': float(np.mean(v6_reductions)) if v6_reductions else None,
            'std_reduction': float(np.std(v6_reductions)) if v6_reductions else None,
            'success_rate': len(v6_energies) / n_scenarios
        }
    }
    
    # Print summary
    print(f"SUMMARY: {robot_name}")
    print(f"V4: {results['summary']['v4']['mean_energy']:.1f} ± {results['summary']['v4']['std_energy']:.1f} J")
    print(f"    Time: {results['summary']['v4']['mean_time']:.2f}s")
    print(f"    Success: {results['summary']['v4']['success_rate']*100:.0f}%")
    
    if results['summary']['v6']['mean_energy'] is not None:
        print(f"\nV6: {results['summary']['v6']['mean_energy']:.1f} ± {results['summary']['v6']['std_energy']:.1f} J")
        print(f"    Time: {results['summary']['v6']['mean_time']:.2f}s")
        print(f"    Reduction: {results['summary']['v6']['mean_reduction']:+.1f}% ± {results['summary']['v6']['std_reduction']:.1f}%")
        print(f"    Success: {results['summary']['v6']['success_rate']*100:.0f}%")
    else:
        print(f"\nV6: No successful runs")
    
    return results


if __name__ == "__main__":
    
    robots = [
        ("Franka Panda", "../robots/franka/franka_panda_proper.urdf"),
        ("UR5e", "../robots/ur5e/ur5e.urdf"),
        ("G1 Humanoid", "../robots/unitree_ros/robots/g1_description/g1_23dof.urdf"),
    ]
    
    all_results = []
    
    for name, urdf in robots:
        results = benchmark_robot_multi_scenario(urdf, name, n_scenarios=10)
        all_results.append(results)
    
    # Save
    with open('benchmark_multi_scenario_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Final table
    print(f"\n{'='*80}")
    print("FINAL RESULTS (Mean ± Std)")
    print(f"{'='*80}")
    print(f"{'Robot':<20} {'DOF':<5} {'V4 Energy':<15} {'V6 Energy':<15} {'Reduction':<15}")
    print("-"*80)
    
    for r in all_results:
        v4_str = f"{r['summary']['v4']['mean_energy']:.1f}±{r['summary']['v4']['std_energy']:.1f}"
        
        if r['summary']['v6']['mean_energy'] is not None:
            v6_str = f"{r['summary']['v6']['mean_energy']:.1f}±{r['summary']['v6']['std_energy']:.1f}"
            red_str = f"{r['summary']['v6']['mean_reduction']:+.1f}±{r['summary']['v6']['std_reduction']:.1f}%"
        else:
            v6_str = "FAILED"
            red_str = "N/A"
        
        print(f"{r['robot']:<20} {r['dof']:<5} {v4_str:<15} {v6_str:<15} {red_str:<15}")
    
    print("\nResults saved to: benchmark_multi_scenario_results.json")

