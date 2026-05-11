# Repository Structure

## Core Optimization Systems

### V4 Hierarchical Pipeline
- `hierarchical_optimizer_v4.py` - Main V4 orchestrator
- `collision_detector_v3.py` - Layer 0: HPP-FCL collision detection
- `trajectory_optimizer_proxqp_v2.py` - Layer 1: ProxQP constraint satisfaction
- `collision_avoidance_chomp_v2.py` - Layer 2: CHOMP obstacle avoidance
- `energy_optimizer_v2.py` - Layer 4: Energy minimization

### V6 Unified Optimizer
- `casadi_optimizer_v6_multi_obstacle.py` - CasADi NLP with coupled RNEA dynamics

## Benchmarking
- `benchmark_v4_vs_v6_fixed.py` - Single-scenario comparison
- `benchmark_multiple_scenarios.py` - Multi-scenario statistics
- `benchmark_final_clean.py` - Standardized motion benchmark

## Demonstrations
- `demo_guaranteed_violations.py` - PyBullet visualization
- `demo_perfect.py` - Various test scenarios
- `demo_final_fixed.py` - Interactive GUI demo

## Results
- `v4_vs_v6_fixed_results.json` - Cluster benchmark data
- `benchmark_final_results.json` - Standardized motion results
- `benchmark_multi_scenario_results.json` - Statistical analysis

## Robot Models
robots/
├── franka/
│   └── franka_panda_proper.urdf
├── ur5e/
│   └── ur5e.urdf
└── unitree_ros/
└── robots/g1_description/g1_23dof.urdf

## Documentation
- `README.md` - Main documentation
- `STRUCTURE.md` - This file
- `LICENSE` - MIT License
