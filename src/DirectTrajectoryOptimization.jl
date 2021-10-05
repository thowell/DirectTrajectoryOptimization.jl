module DirectTrajectoryOptimization

using LinearAlgebra
using SparseArrays
using Symbolics, IfElse
using Parameters
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

include("objective.jl")
include("constraints.jl")
include("dynamics.jl")
include("problem.jl")
include("moi.jl")
include("solver.jl")
include("utils.jl")

# objective 
export Cost

# constraints 
export Bound, StageConstraint, ConstraintSet

# dynamics 
export Dynamics, DynamicsModel

# problem 
export TrajectoryOptimizationProblem

# solver 
export Solver, Options, initialize!, solve!

# utils 
export linear_interpolation

end # module
