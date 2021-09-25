module DirectTrajectoryOptimization

using LinearAlgebra
using StaticArrays
using SparseArrays
using Symbolics, IfElse
using ForwardDiff
using BenchmarkTools, InteractiveUtils
using MathOptInterface
const MOI = MathOptInterface
using Parameters
using JLD2
using Ipopt

include(joinpath(pwd(), "src/objective.jl"))
include(joinpath(pwd(), "src/constraints.jl"))
include(joinpath(pwd(), "src/dynamics.jl"))
include(joinpath(pwd(), "src/problem.jl"))
include(joinpath(pwd(), "src/moi.jl"))
include(joinpath(pwd(), "src/utils.jl"))

end # module
