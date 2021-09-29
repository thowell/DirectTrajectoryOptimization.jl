using Test
using BenchmarkTools
using ForwardDiff
using LinearAlgebra
using DirectTrajectoryOptimization

include("objective.jl")
include("dynamics.jl")
include("constraints.jl")