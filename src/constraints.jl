abstract type Constraint end 

struct Bound{T} <: Constraint
    xl::Vector{T} 
    xu::Vector{T}
    ul::Vector{T}
    uu::Vector{T}
    idx::Vector{Int}
end

function Bound(nx::Int=0, nu::Int=0, idx=[0]; 
    xl=-Inf * ones(nx), xu=Inf * ones(nx), ul=-Inf * ones(nu), uu=Inf * ones(nu)) 
    return Bound(xl, xu, ul, uu, idx)
end

const Bounds{T} = Vector{Bound{T}}

struct StageConstraint{T} <: Constraint
    val 
    jac 
    nx::Int 
    nu::Int 
    nc::Int
    nj::Int 
    sparsity::Vector{Vector{Int}}
    val_cache::Vector{T} 
    jac_cache::Vector{T}
    type::Symbol
    idx::Vector{Int}
end

StageConstraints{T} = Vector{StageConstraint{T}} where T

function StageConstraint(f::Function, nx::Int, nu::Int, idx::Vector{Int}, type::Symbol)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu]
    val = f(x, u)
    jac = Symbolics.sparsejacobian(val, [x; u])
    val_func = eval(Symbolics.build_function(val, x, u)[2])
    jac_func = eval(Symbolics.build_function(jac.nzval, x, u)[2])
    nc = length(val) 
    nj = length(jac.nzval)
    sparsity = [findnz(jac)[1:2]...]
    return StageConstraint(val_func, jac_func, 
        nx, nu, nc, nj, sparsity, zeros(nc), zeros(nj), type, idx)
end

function StageConstraint() 
    return StageConstraint((c, x, u) -> nothing, (j, x, u) -> nothing, 0, 0, 0, 0, [Int[]], Float64[], Float64[], :empty, Int[])
end

function eval_con!(c, idx, cons::StageConstraints{T}, x, u) where T
    fill!(c, 0.0)
    i = 1
    for con in cons
        for t in con.idx
            con.val(con.val_cache, x[t], u[t])
            @views c[idx[i]] .= con.val_cache
            fill!(con.val_cache, 0.0) # TODO: confirm this is necessary 
            i += 1
        end
    end
end

function eval_jac!(j, idx, cons::StageConstraints{T}, x, u) where T
    fill!(j, 0.0)
    i = 1
    for con in cons
        for t in con.idx
            con.jac(con.jac_cache, x[t], u[t])
            @views j[idx[i]] .= con.jac_cache
            fill!(con.jac_cache, 0.0) # TODO: confirm this is necessary
            i += 1
        end
    end
end

function sparsity(cons::StageConstraints{T}, nx::Vector{Int}, nu::Vector{Int}; row_shift=0) where T
    row = Int[]
    col = Int[]
    for con in cons
        for t in con.idx
            col_shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
            push!(row, (con.sparsity[1] .+ row_shift)...) 
            push!(col, (con.sparsity[2] .+ col_shift)...) 
            row_shift += con.nc
        end
    end
    return collect(zip(row, col))
end

num_con(cons::StageConstraints{T}) where T = sum([con.nc * length(con.idx) for con in cons])
num_jac(cons::StageConstraints{T}) where T = sum([con.nj * length(con.idx) for con in cons])

function constraint_indices(cons::StageConstraints{T}; shift=0) where T
    idx = Vector{Int}[]
    for con in cons 
        for t in con.idx 
            idx = [idx..., collect(shift .+ (1:con.nc)),]
            shift += con.nc
        end 
    end
    return idx
end 

function jacobian_indices(cons::StageConstraints{T}; shift=0) where T
    idx = Vector{Int}[]
    for con in cons 
        for t in con.idx 
            push!(idx, collect(shift .+ (1:con.nj)))
            shift += con.nj
        end 
    end
    return idx
end

struct ConstraintSet{T} 
    bounds::Bounds{T}
    stage::StageConstraints{T} 
end

ConstraintSet() = ConstraintSet([Bound()], [StageConstraint()]) 
ConstraintSet(stage::StageConstraints{T}) where T = ConstraintSet([Bound()], stage) 
ConstraintSet(bounds::Bounds{T}) where T = ConstraintSet(bounds, [StageConstraint()]) 

num_con(cons::ConstraintSet{T}) where T = sum([con.nc * length(con.idx) for con in cons.stage])
num_jac(cons::ConstraintSet{T}) where T = sum([con.nj * length(con.idx) for con in cons.stage])
