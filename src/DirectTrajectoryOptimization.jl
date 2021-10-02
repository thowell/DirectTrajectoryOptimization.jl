module DirectTrajectoryOptimization

using LinearAlgebra
using SparseArrays
using Symbolics, IfElse
using Parameters
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

include(joinpath(pwd(), "src/objective.jl"))
include(joinpath(pwd(), "src/constraints.jl"))
include(joinpath(pwd(), "src/dynamics.jl"))
include(joinpath(pwd(), "src/problem.jl"))
include(joinpath(pwd(), "src/moi.jl"))
include(joinpath(pwd(), "src/solver.jl"))
include(joinpath(pwd(), "src/utils.jl"))

# objective 
export Cost

# constraints 
export Bound, StageConstraint, ConstraintSet

# dynamics 
export Dynamics, DynamicsModel

# problem 
export TrajectoryOptimizationProblem

# solver 
export Solver, initialize!, solve!

# utils 
export linear_interpolation

end # module
