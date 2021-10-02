struct TrajectoryOptimizationProblem{T}
    obj::Objective{T}
    model::DynamicsModel{T}
    con::ConstraintSet{T}
    x::Vector{Vector{T}}
    u::Vector{Vector{T}}
    w::Vector{Vector{T}}
    λ_dyn::Vector{Vector{T}} 
    λ_stage::Vector{Vector{T}}
end

function TrajectoryOptimizationProblem(obj::Objective, model::DynamicsModel, con::ConstraintSet; 
    w = [zeros(nw) for nw in model.dim.w])

    x = [zeros(nx) for nx in model.dim.x]
    u = [[zeros(nu) for nu in model.dim.u]..., zeros(0)]

    λ_dyn = [zeros(nx) for nx in model.dim.x[2:end]]
    λ_stage = [zeros(stage.nc) for stage in con.stage for t in stage.idx]
   
    TrajectoryOptimizationProblem(obj, model, con, x, u, w, λ_dyn, λ_stage)
end

TrajectoryOptimizationProblem(obj::Objective, model::DynamicsModel) = TrajectoryOptimizationProblem(obj, model, ConstraintSet())

struct TrajectoryOptimizationIndices 
    obj_hess::Vector{Vector{Int}}
    dyn_con::Vector{Vector{Int}} 
    dyn_jac::Vector{Vector{Int}} 
    dyn_hess::Vector{Vector{Int}}
    stage_con::Vector{Vector{Int}} 
    stage_jac::Vector{Vector{Int}} 
    stage_hess::Vector{Vector{Int}}
end

function indices(obj::Objective, dyn::Vector{Dynamics{T}}, stage::StageConstraints{T}, key::Vector{Tuple{Int,Int}},
        nx::Vector{Int}, nu::Vector{Int}) where T 
    # Jacobians
    dyn_con = constraint_indices(dyn, shift=0)
    dyn_jac = jacobian_indices(dyn, shift=0)
    stage_con = constraint_indices(stage, shift=num_con(dyn))
    stage_jac = jacobian_indices(stage, shift=num_jac(dyn)) 

    # Hessian of Lagrangian 
    obj_hess = hessian_indices(obj, key, nx, nu)
    dyn_hess = hessian_indices(dyn, key, nx, nu)
    stage_hess = hessian_indices(stage, key, nx, nu)

    return TrajectoryOptimizationIndices(obj_hess, dyn_con, dyn_jac, dyn_hess, stage_con, stage_jac, stage_hess) 
end

struct Problem{T} <: MOI.AbstractNLPEvaluator
    trajopt::TrajectoryOptimizationProblem{T}
    num_var::Int                 
    num_con::Int 
    num_jac::Int 
    num_hess_lag::Int               
    var_bnds::Vector{Vector{T}}
    con_bnds::Vector{Vector{T}}
    idx::TrajectoryOptimizationIndices
    sp_jac
    sp_hess_lag
    hess_lag::Bool 
end

function primal_bounds(bnds::Bounds{T}, nz::Int, x_idx::Vector{Vector{Int}}, u_idx::Vector{Vector{Int}}) where T 
    zl, zu = -Inf * ones(nz), Inf * ones(nz) 
    for bnd in bnds 
        for t in bnd.idx
            length(bnd.xl) > 0 && (zl[x_idx[t]] = bnd.xl)
            length(bnd.xu) > 0 && (zu[x_idx[t]] = bnd.xu)
            length(bnd.ul) > 0 && (zl[u_idx[t]] = bnd.ul)
            length(bnd.uu) > 0 && (zu[u_idx[t]] = bnd.uu)
        end
    end
    return zl, zu
end

function constraint_bounds(dyn::Vector{Dynamics{T}}, stage::StageConstraints{T}, nc::Int, idx::TrajectoryOptimizationIndices) where T
    cl, cu = zeros(nc), zeros(nc) 
    i = 1
    for con in stage 
        for t in con.idx 
            if con.type == :inequality
                cu[idx.stage_con[i]] .= Inf
            end
            i += 1
        end 
    end
    return cl, cu
end 

function Problem(trajopt::TrajectoryOptimizationProblem; eval_hess=false) 
    # number of variables
    nz = sum(trajopt.model.dim.x) + sum(trajopt.model.dim.u)
    
    # number of constraints
    nc_dyn = num_con(trajopt.model.dyn)
    nc_con = num_con(trajopt.con.stage)  
    nc = nc_dyn + nc_con

    # number of nonzeros in constraint Jacobian
    nj_dyn = num_jac(trajopt.model.dyn)
    nj_con = num_jac(trajopt.con.stage)  
    nj = nj_dyn + nj_con

    # number of nonzeros in Hessian of Lagrangian
    nh = 0

    # primal variable bounds
    zl, zu = primal_bounds(trajopt.con.bounds, nz, trajopt.model.idx.x, trajopt.model.idx.u) 

    # constraint Jacobian sparsity
    sp_dyn = sparsity_jacobian(trajopt.model.dyn, trajopt.model.dim.x, trajopt.model.dim.u, row_shift=0)
    sp_con = sparsity_jacobian(trajopt.con.stage, trajopt.model.dim.x, trajopt.model.dim.u, row_shift=nc_dyn)
    sp_jac = collect([sp_dyn..., sp_con...]) 

    # Hessian of Lagrangian sparsity 
    sp_obj_hess = sparsity_hessian(trajopt.obj, trajopt.model.dim.x, trajopt.model.dim.u)
    sp_dyn_hess = sparsity_hessian(trajopt.model.dyn, trajopt.model.dim.x, trajopt.model.dim.u)
    sp_con_hess = sparsity_hessian(trajopt.con.stage, trajopt.model.dim.x, trajopt.model.dim.u)
    sp_hess_lag = [sp_obj_hess..., sp_dyn_hess..., sp_con_hess...]
    sp_hess_lag = !isempty(sp_hess_lag) ? sp_hess_lag : Tuple{Int,Int}[]
    sp_hess_key = sort(unique(sp_hess_lag))
    sp_hess_status = [!isempty(sp_obj_hess), !isempty(sp_dyn_hess), !isempty(sp_con_hess)]

    # indices 
    idx = indices(trajopt.obj, trajopt.model.dyn, trajopt.con.stage, sp_hess_key, trajopt.model.dim.x, trajopt.model.dim.u)

    # nonlinear constraint bounds
    cl, cu = constraint_bounds(trajopt.model.dyn, trajopt.con.stage, nc, idx) 

    Problem(trajopt, nz, nc, nj, nh, [zl, zu], [cl, cu], idx, sp_jac, sp_hess_key, eval_hess)
end

function trajectory!(x::Vector{Vector{T}}, u::Vector{Vector{T}}, z::Vector{T}, 
    idx_x::Vector{Vector{Int}}, idx_u::Vector{Vector{Int}}) where T
    for (t, idx) in enumerate(idx_x)
        x[t] .= @views z[idx]
    end 
    for (t, idx) in enumerate(idx_u)
        u[t] .= @views z[idx]
    end
end

function duals!(λ_dyn::Vector{Vector{T}}, λ_stage::Vector{Vector{T}}, λ, 
    idx_dyn::Vector{Vector{Int}}, idx_stage::Vector{Vector{Int}}) where T
    for (t, idx) in enumerate(idx_dyn)
        λ_dyn[t] .= @views λ[idx]
    end 
    for (t, idx) in enumerate(idx_stage)
        λ_stage[t] .= @views λ[idx]
    end
end