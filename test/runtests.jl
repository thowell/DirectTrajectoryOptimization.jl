using Test
using Symbolics
using ForwardDiff
using LinearAlgebra
using SparseArrays
using DirectTrajectoryOptimization

include("objective.jl")
include("dynamics.jl")
include("constraints.jl")
include("hessian_lagrangian.jl") #TODO: fix 
include("solve.jl")