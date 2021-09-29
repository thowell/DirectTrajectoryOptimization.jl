# DirectTrajectoryOptimization.jl
[![codecov](https://codecov.io/gh/thowell/DirectTrajectoryOptimization.jl/branch/main/graph/badge.svg?token=821EI7HJEL)](https://codecov.io/gh/thowell/DirectTrajectoryOptimization.jl)

This package solves constrained trajectory optimization problems: 

```
minimize        gT(xT) + sum(gt(xt, ut))
xt, ut
subject to      xt+1 = ft(xt, ut) , t = 1,...,T-1 
                ct(xt, ut) {>,=} 0, t = 1,...,T
                xt_L <= xt <= xt_U, t = 1,...,T 
                ut_L <= ut <= ut_U, t = 1,...,T-1.
```

Fast and allocation-free gradients and sparse Jacobians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs, constraints, and dynamics. The problem is solved with [Ipopt](https://coin-or.github.io/Ipopt/).

