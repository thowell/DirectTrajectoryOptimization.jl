# DirectTrajectoryOptimization.jl
[![CI](https://github.com/thowell/DirectTrajectoryOptimization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/thowell/DirectTrajectoryOptimization.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/thowell/DirectTrajectoryOptimization.jl/branch/main/graph/badge.svg?token=821EI7HJEL)](https://codecov.io/gh/thowell/DirectTrajectoryOptimization.jl)

A Julia package for solving constrained trajectory optimization problems: 

```
minimize        cost_T(state_T; parameter_T) + sum(cost_t(state_t, action_t; parameter_t))
states, actions
subject to      dynamics_t(state_t, action_t, state_t+1; parameter_t),         t = 1,...,T-1 
                constraint_t(state_t, action_t; parameter_t) {<,=} 0,          t = 1,...,T
                state_lower_t < state_t < state_upper_t,                       t = 1,...,T 
                action_lower_t < action_t < action_upper_t,                    t = 1,...,T-1.
```

with direct trajectory optimization. 

Fast and allocation-free gradients and sparse Jacobians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs, constraints, and dynamics. The problem is solved with [Ipopt](https://coin-or.github.io/Ipopt/).

## Installation
```
Pkg.add("DirectTrajectoryOptimization")
```

## Quick Start
