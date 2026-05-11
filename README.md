# O-VLA Phase 2: Morphology-Agnostic Trajectory Optimization

**Trajectory optimization layer for Vision-Language-Action models that ensures safe, constraint-satisfying robot motion across diverse morphologies.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Vision-Language-Action (VLA) models can generate robot commands from demonstrations, but their outputs often violate joint limits and safety constraints. **O-VLA Phase 2** addresses this by providing two trajectory optimization systems:

- **V4 (Hierarchical Pipeline)**: ProxQP + CHOMP + TOPP-RA for guaranteed constraint satisfaction
- **V6 (Unified Optimizer)**: CasADi + IPOPT with coupled dynamics for energy-optimal trajectories

Both systems are **morphology-agnostic** and work on diverse robots without per-robot tuning.

## Key Results

| Robot | DOF | V4 Energy | V6 Energy | V4 Time | V6 Time |
|-------|-----|-----------|-----------|---------|---------|
| Franka Panda | 7 | 17.4 J | 9.5 J (+45%) | 0.06s | 3.3s |
| UR5e | 6 | 44.3 J | 29.4 J (+34%) | 0.03s | 1.4s |
| G1 Humanoid | 23 | 11.5 J | 6.6 J (+42%) | 0.44s | 13.0s |

**Key Insight**: V6's use of coupled dynamics via RNEA reveals optimization opportunities that decoupled models miss, achieving 21-46% energy reduction at the cost of 50x longer planning time.

## Installation

### Prerequisites
- Python 3.8+
- Ubuntu 20.04/22.04 (tested)

### Dependencies

```bash
conda create -n ovla python=3.9
conda activate ovla

# Core dependencies
conda install -c conda-forge pinocchio=3.9.0
pip install casadi proxsuite pybullet numpy scipy

# For visualization
pip install matplotlib
```

### Robot Models

Download required URDF files:

```bash
# Franka Panda
git clone https://github.com/frankaemika/franka_ros.git robots/franka

# UR5e
git clone https://github.com/ros-industrial/universal_robot.git robots/ur5e

# Unitree G1
git clone https://github.com/unitreerobotics/unitree_ros.git robots/unitree_ros
```

## Usage

### Quick Start

```python
from hierarchical_optimizer_v4 import HierarchicalOptimizerV4
import numpy as np

# Initialize optimizer
optimizer = HierarchicalOptimizerV4("path/to/robot.urdf")

# Define start and goal configurations
q_start = np.zeros(7)  # 7-DOF robot
q_goal = np.array([0.5, -0.5, 0.3, -1.2, 0.8, 1.5, 0.0])

# Optimize trajectory
trajectory, metrics = optimizer.optimize_trajectory(
    q_start, q_goal, n_waypoints=50
)

print(f"Energy: {metrics['energy']['total_energy']:.2f} J")
```


## System Architecture

### V4: Hierarchical Pipeline

1. **Layer 0**: Collision detection (HPP-FCL)
2. **Layer 1**: Constraint satisfaction (ProxQP)
3. **Layer 2**: Collision avoidance (CHOMP)
4. **Layer 3**: Time-optimal retiming (TOPP-RA)
5. **Layer 4**: Energy optimization (gradient descent)

**Advantages**: Fast (0.03-0.44s), guaranteed constraint satisfaction
**Limitations**: Decoupled dynamics, sequential optimization

### V6: Unified CasADi Optimizer

Single NLP formulation with:
- Coupled dynamics via Recursive Newton-Euler Algorithm (RNEA)
- Joint limits, velocity limits, torque limits as hard constraints
- Energy minimization objective: min ∫ τ² dt
- IPOPT solver with algorithmic differentiation

**Advantages**: 21-46% energy reduction, exploits inertial coupling
**Limitations**: Slower (1-13s), occasional numerical tolerance violations

## Technical Details

### Coupled Dynamics (RNEA)

V6 uses the full rigid-body dynamics equation:
τ = M(q)q̈ + C(q,q̇) + g(q)

where:
- `M(q)`: 7×7 coupled inertia matrix
- `C(q,q̇)`: Coriolis and centrifugal terms
- `g(q)`: Gravity compensation

This is computed via Pinocchio's CasADi interface:

```python
import pinocchio.casadi as cpin

tau = cpin.rnea(model, data, q, v, a)
```

The optimizer sees exact derivatives of how joint accelerations couple through the inertia matrix, enabling exploitation of momentum transfer.

### Constraint Handling

**V4**: Geometric projection onto constraint manifold via ProxQP
- Quadratic program: min ||q - q_nominal||² s.t. q_min ≤ q ≤ q_max
- Guaranteed feasibility

**V6**: Penalty functions in NLP objective
- IPOPT convergence tolerance: 1e-6
- Occasionally produces small violations (~2 per trajectory at endpoints)

## License

MIT License - see LICENSE file for details

## Contact

**Ansh Bhansali**
- Website: [anshbhansali.com](https://anshbhansali.com)
- LinkedIn: [linkedin.com/in/anshbhansali](https://linkedin.com/in/anshbhansali)
- GitHub: [github.com/ansh1113](https://github.com/ansh1113)

## Acknowledgments

- Built on top of [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for rigid-body dynamics
- Uses [CasADi](https://web.casadi.org/) for symbolic optimization
- Collision detection via [HPP-FCL](https://github.com/humanoid-path-planner/hpp-fcl)
- TOPP-RA implementation from [toppra](https://github.com/hungpham2511/toppra)

