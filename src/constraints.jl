struct Constraint{T}
    val 
    jac 
    hess
    nx::Int 
    nu::Int 
    nw::Int
    nc::Int
    nj::Int 
    nh::Int
    sp_jac::Vector{Vector{Int}}
    sp_hess::Vector{Vector{Int}}
    val_cache::Vector{T} 
    jac_cache::Vector{T}
    hess_cache::Vector{T}
    idx_ineq::Vector{Int}
end

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(f::Function, nx::Int, nu::Int; nw::Int=0, idx_ineq=collect(1:0), eval_hess=false)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu], w[1:nw]
    val = f(x, u, w)
    jac = Symbolics.sparsejacobian(val, [x; u])
    val_func = eval(Symbolics.build_function(val, x, u, w)[2])
    jac_func = eval(Symbolics.build_function(jac.nzval, x, u, w)[2])
    nc = length(val) 
    nj = length(jac.nzval)
    sp_jac = [findnz(jac)[1:2]...]
    if eval_hess
        @variables λ[1:nc]
        lag_con = dot(λ, val) 
        hess = Symbolics.sparsehessian(lag_con, [x; u])
        hess_func = eval(Symbolics.build_function(hess.nzval, x, u, w, λ)[2])
        sp_hess = [findnz(hess)[1:2]...]
        nh = length(hess.nzval)
    else 
        hess_func = Expr(:null) 
        sp_hess = [Int[]]
        nh = 0
    end
    return Constraint(val_func, jac_func, hess_func,
        nx, nu, nw, nc, nj, nh, sp_jac, sp_hess, zeros(nc), zeros(nj), zeros(nh), idx_ineq)
end

function Constraint()
    return Constraint((c, x, u, w) -> nothing, (j, x, u, w) -> nothing, (h, x, u, w) -> nothing, 
        0, 0, 0, 0, 0, 0, [Int[], Int[]], [Int[], Int[]], Float64[], Float64[], Float64[], collect(1:0))
end

function eval_con!(c, idx, cons::Constraints{T}, x, u, w) where T
    for (t, con) in enumerate(cons)
        con.val(con.val_cache, x[t], u[t], w[t])
        @views c[idx[t]] .= con.val_cache
        fill!(con.val_cache, 0.0) # TODO: confirm this is necessary 
    end
end

function eval_jac!(j, idx, cons::Constraints{T}, x, u, w) where T
    for (t, con) in enumerate(cons)
        con.jac(con.jac_cache, x[t], u[t], w[t])
        @views j[idx[t]] .= con.jac_cache
        fill!(con.jac_cache, 0.0) # TODO: confirm this is necessary
    end
end

function eval_hess_lag!(h, idx, cons::Constraints{T}, x, u, w, λ) where T
    for (t, con) in enumerate(cons)
        if !isempty(con.hess_cache)
            con.hess(con.hess_cache, x[t], u[t], w[t], λ[t])
            @views h[idx[t]] .+= con.hess_cache
            fill!(con.hess_cache, 0.0) # TODO: confirm this is necessary
        end
    end
end

function sparsity_jacobian(cons::Constraints{T}, nx::Vector{Int}, nu::Vector{Int}; row_shift=0) where T
    row = Int[]
    col = Int[]
    for (t, con) in enumerate(cons)
        col_shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
        push!(row, (con.sp_jac[1] .+ row_shift)...) 
        push!(col, (con.sp_jac[2] .+ col_shift)...) 
        row_shift += con.nc
    end
    return collect(zip(row, col))
end

function sparsity_hessian(cons::Constraints{T}, nx::Vector{Int}, nu::Vector{Int}) where T
    row = Int[]
    col = Int[]
    for (t, con) in enumerate(cons)
        if !isempty(con.sp_hess[1])
            shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
            push!(row, (con.sp_hess[1] .+ shift)...) 
            push!(col, (con.sp_hess[2] .+ shift)...) 
        end
    end
    return collect(zip(row, col))
end

num_con(cons::Constraints{T}) where T = sum([con.nc for con in cons])
num_jac(cons::Constraints{T}) where T = sum([con.nj for con in cons])

function constraint_indices(cons::Constraints{T}; shift=0) where T
    idx = Vector{Int}[]
    for (t, con) in enumerate(cons)
        idx = [idx..., collect(shift .+ (1:con.nc)),]
        shift += con.nc
    end
    return idx
end 

function jacobian_indices(cons::Constraints{T}; shift=0) where T
    idx = Vector{Int}[]
    for (t, con) in enumerate(cons) 
        push!(idx, collect(shift .+ (1:con.nj)))
        shift += con.nj
    end
    return idx
end

function hessian_indices(cons::Constraints{T}, key::Vector{Tuple{Int,Int}}, nx::Vector{Int}, nu::Vector{Int}) where T
    idx = Vector{Int}[]
    for (t, con) in enumerate(cons) 
        if !isempty(con.sp_hess[1])
            row = Int[]
            col = Int[]
            shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
            push!(row, (con.sp_hess[1] .+ shift)...) 
            push!(col, (con.sp_hess[2] .+ shift)...) 
            rc = collect(zip(row, col))
            push!(idx, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return idx
end
