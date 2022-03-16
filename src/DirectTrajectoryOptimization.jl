module DirectTrajectoryOptimization

using LinearAlgebra
using SparseArrays
using Symbolics, IfElse
using Parameters
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

include("costs.jl")
include("constraints.jl")
include("bounds.jl")
include("general_constraint.jl")
include("dynamics.jl")
include("solver.jl")
include("data.jl")
include("moi.jl")
include("utils.jl")

# objective 
export Cost

# constraints 
export Bound, Bounds, Constraint, Constraints, GeneralConstraint

# dynamics 
export Dynamics

# problem 
export ProblemData

# solver 
export Solver, Options, initialize_states!, initialize_controls!, solve!, get_trajectory

# utils 
export linear_interpolation

end # module
