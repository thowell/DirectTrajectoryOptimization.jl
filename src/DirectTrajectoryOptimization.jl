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
include("bounds.jl")
include("general_constraint.jl")
include("dynamics.jl")
include("data.jl")
include("solver.jl")
include("moi.jl")
include("utils.jl")

# objective 
export Cost

# constraints 
export Bound, Bounds, Constraint, Constraints, GeneralConstraint

# dynamics 
export Dynamics

# solver 
export solver, Solver, Options, initialize_states!, initialize_controls!, solve!, get_trajectory

# utils 
export linear_interpolation

end # module
