struct GeneralConstraint{T}
    val 
    jac 
    hess
    nz::Int 
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

function GeneralConstraint(f::Function, nz::Int; nw::Int=0, idx_ineq=collect(1:0), eval_hess=false)
    #TODO: option to load/save methods
    @variables z[1:nz], w[1:nw]
    val = f(z, w)
    jac = Symbolics.sparsejacobian(val, z)
    val_func = eval(Symbolics.build_function(val, z, w)[2])
    jac_func = eval(Symbolics.build_function(jac.nzval, z, w)[2])
    nc = length(val) 
    nj = length(jac.nzval)
    sp_jac = [findnz(jac)[1:2]...]
    if eval_hess
        @variables λ[1:nc]
        lag_con = dot(λ, val) 
        hess = Symbolics.sparsehessian(lag_con, z)
        hess_func = eval(Symbolics.build_function(hess.nzval, z, w, λ)[2])
        sp_hess = [findnz(hess)[1:2]...]
        nh = length(hess.nzval)
    else 
        hess_func = Expr(:null) 
        sp_hess = [Int[]]
        nh = 0
    end
    return GeneralConstraint(val_func, jac_func, hess_func,
        nz, nw, nc, nj, nh, sp_jac, sp_hess, zeros(nc), zeros(nj), zeros(nh), idx_ineq)
end

function GeneralConstraint()
    return GeneralConstraint((c, z, w) -> nothing, (j, z, w) -> nothing, (h, z, w) -> nothing, 
        0, 0, 0, 0, 0, [Int[], Int[]], [Int[], Int[]], Float64[], Float64[], Float64[], collect(1:0))
end

function eval_con!(c, idx, con::GeneralConstraint{T}, z, w) where T
    con.val(con.val_cache, z, w)
    @views c[idx] .= con.val_cache
    fill!(con.val_cache, 0.0) # TODO: confirm this is necessary 
end

function eval_jac!(j, idx, con::GeneralConstraint{T}, z, w) where T
    con.jac(con.jac_cache, z, w)
    @views j[idx] .= con.jac_cache
    fill!(con.jac_cache, 0.0) # TODO: confirm this is necessary
end

function eval_hess_lag!(h, idx, con::GeneralConstraint{T}, z, w, λ) where T
    if !isempty(con.hess_cache)
        con.hess(con.hess_cache, z, λ)
        @views h[idx] .+= con.hess_cache
        fill!(con.hess_cache, 0.0) # TODO: confirm this is necessary
    end
end

function sparsity_jacobian(con::GeneralConstraint{T}, nz::Int; row_shift=0) where T
    row = Int[]
    col = Int[]
   
    push!(row, (con.sp_jac[1] .+ row_shift)...) 
    push!(col, con.sp_jac[2]...) 
    
    return collect(zip(row, col))
end

function sparsity_hessian(con::GeneralConstraint{T}, nz::Int) where T
    row = Int[]
    col = Int[]
    if !isempty(con.sp_hess[1])
        shift = 0
        push!(row, (con.sp_hess[1] .+ shift)...) 
        push!(col, (con.sp_hess[2] .+ shift)...) 
    end
    return collect(zip(row, col))
end

num_con(con::GeneralConstraint{T}) where T = con.nc
num_jac(con::GeneralConstraint{T}) where T = con.nj

constraint_indices(con::GeneralConstraint{T}; shift=0) where T = collect(shift .+ (1:con.nc))

jacobian_indices(con::GeneralConstraint{T}; shift=0) where T = collect(shift .+ (1:con.nj))
   
function hessian_indices(con::GeneralConstraint{T}, key::Vector{Tuple{Int,Int}}, nz::Int) where T
    if !isempty(con.sp_hess[1])
        row = Int[]
        col = Int[]
        shift = 0
        push!(row, (con.sp_hess[1] .+ shift)...) 
        push!(col, (con.sp_hess[2] .+ shift)...) 
        rc = collect(zip(row, col))
        idx = [findfirst(x -> x == i, key) for i in rc]
    else 
        idx = Vector{Int}[]
    end
    return idx
end
