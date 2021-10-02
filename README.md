# DirectTrajectoryOptimization.jl
[![CI](https://github.com/thowell/DirectTrajectoryOptimization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/thowell/DirectTrajectoryOptimization.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/thowell/DirectTrajectoryOptimization.jl/branch/main/graph/badge.svg?token=821EI7HJEL)](https://codecov.io/gh/thowell/DirectTrajectoryOptimization.jl)

This package solves constrained trajectory optimization problems: 

```
minimize        gT(xT; wT) + sum(gt(xt, ut; wt))
xt, ut
subject to      xt+1 = ft(xt, ut; wt) , t = 1,...,T-1 
                ct(xt, ut; wt) {>,=} 0, t = 1,...,T
                xt_L <= xt <= xt_U, t = 1,...,T 
                ut_L <= ut <= ut_U, t = 1,...,T-1.
```

Fast and allocation-free gradients and sparse Jacobians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs, constraints, and dynamics. The problem is solved with [Ipopt](https://coin-or.github.io/Ipopt/).

## Installation
```
Pkg.add("DirectTrajectoryOptimization")
```