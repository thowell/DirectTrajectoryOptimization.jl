using Test
using BenchmarkTools
using ForwardDiff

include(joinpath(pwd(), "test/objective.jl"))
include(joinpath(pwd(), "test/dynamics.jl"))
include(joinpath(pwd(), "test/constraints.jl"))