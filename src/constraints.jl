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
    hess
    nx::Int 
    nu::Int 
    nc::Int
    nj::Int 
    nh::Int
    sp_jac::Vector{Vector{Int}}
    sp_hess::Vector{Vector{Int}}
    val_cache::Vector{T} 
    jac_cache::Vector{T}
    hess_cache::Vector{T}
    type::Symbol
    idx::Vector{Int}
end

StageConstraints{T} = Vector{StageConstraint{T}} where T

function StageConstraint(f::Function, nx::Int, nu::Int, idx::Vector{Int}, type::Symbol; eval_hess=false)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu]
    val = f(x, u)
    jac = Symbolics.sparsejacobian(val, [x; u])
    val_func = eval(Symbolics.build_function(val, x, u)[2])
    jac_func = eval(Symbolics.build_function(jac.nzval, x, u)[2])
    nc = length(val) 
    nj = length(jac.nzval)
    sp_jac = [findnz(jac)[1:2]...]
    if eval_hess
        @variables λ[1:nc]
        lag_con = dot(λ, val) 
        hess = Symbolics.sparsehessian(lag_con, [x; u])
        hess_func = eval(Symbolics.build_function(hess.nzval, x, u, λ)[2])
        sp_hess = [findnz(hess)[1:2]...]
        nh = length(hess.nzval)
    else 
        hess_func = Expr(:null) 
        sp_hess = [Int[]]
        nh = 0
    end
    return StageConstraint(val_func, jac_func, hess_func,
        nx, nu, nc, nj, nh, sp_jac, sp_hess, zeros(nc), zeros(nj), zeros(nh), type, idx)
end

function StageConstraint()
    return StageConstraint((c, x, u) -> nothing, (j, x, u) -> nothing, (h, x, u) -> nothing, 0, 0, 0, 0, 0, [Int[]], [Int[]], Float64[], Float64[], Float64[], :empty, Int[])
end

function eval_con!(c, idx, cons::StageConstraints{T}, x, u) where T
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

function eval_hess_lag!(h, idx, cons::StageConstraints{T}, x, u, λ) where T
    i = 1
    for con in cons
        for t in con.idx
            con.hess(con.hess_cache, x[t], u[t], λ[t])
            # @views h[idx[i]] .= con.hess_cache
            fill!(con.hess_cache, 0.0) # TODO: confirm this is necessary
            i += 1
        end
    end
end

function sparsity_jacobian(cons::StageConstraints{T}, nx::Vector{Int}, nu::Vector{Int}; row_shift=0) where T
    row = Int[]
    col = Int[]
    for con in cons
        for t in con.idx
            col_shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
            push!(row, (con.sp_jac[1] .+ row_shift)...) 
            push!(col, (con.sp_jac[2] .+ col_shift)...) 
            row_shift += con.nc
        end
    end
    return collect(zip(row, col))
end

function sparsity_hessian(cons::StageConstraints{T}, nx::Vector{Int}, nu::Vector{Int}) where T
    row = Int[]
    col = Int[]
    for con in cons
        if !isempty(con.sp_hess[1])
            for t in con.idx
                shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
                push!(row, (con.sp_hess[1] .+ shift)...) 
                push!(col, (con.sp_hess[2] .+ shift)...) 
            end
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

function hessian_indices(cons::StageConstraints{T}, key::Vector{Tuple{Int,Int}}, nx::Vector{Int}, nu::Vector{Int}) where T
    idx = Vector{Int}[]
    for con in cons 
        if !isempty(con.sp_hess[1])
            for t in con.idx
                row = Int[]
                col = Int[]
                shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
                push!(row, (con.sp_hess[1] .+ shift)...) 
                push!(col, (con.sp_hess[2] .+ shift)...) 
                rc = collect(zip(row, col))
                push!(idx, [findfirst(x -> x == i, key) for i in rc])
            end
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
