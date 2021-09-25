struct TrajectoryOptimizationProblem{T}
    obj::Objective{T}
    model::DynamicsModel{T}
    con::ConstraintSet{T}
    x::Vector{Vector{T}}
    u::Vector{Vector{T}}
    w::Vector{Vector{T}}
end

function TrajectoryOptimizationProblem(obj::Objective, model::DynamicsModel, con::ConstraintSet; 
    w = [zeros(nw) for nw in model.dim.w])

    x = [zeros(nx) for nx in model.dim.x]
    u = [[zeros(nu) for nu in model.dim.u]..., zeros(0)]
   
    TrajectoryOptimizationProblem(obj, model, con, x, u, w)
end

TrajectoryOptimizationProblem(obj::Objective, model::DynamicsModel) = TrajectoryOptimizationProblem(obj, model, ConstraintSet())

struct TrajectoryOptimizationIndices
    dyn_con::Vector{Vector{Int}} 
    dyn_jac::Vector{Vector{Int}} 
    stage_con::Vector{Vector{Int}} 
    stage_jac::Vector{Vector{Int}} 
end

function indices(dyn::Vector{Dynamics{T}}, stage::StageConstraints{T}) where T 
    dyn_con = constraint_indices(dyn, shift=0)
    dyn_jac = jacobian_indices(dyn, shift=0)
    stage_con = constraint_indices(stage, shift=num_con(dyn))
    stage_jac = jacobian_indices(stage, shift=num_jac(dyn)) 
    return TrajectoryOptimizationIndices(dyn_con, dyn_jac, stage_con, stage_jac) 
end

struct Problem{T} <: MOI.AbstractNLPEvaluator
    trajopt::TrajectoryOptimizationProblem{T}
    num_var::Int                 
    num_con::Int 
    num_jac::Int 
    num_hess_lag::Int               
    var_bnds::Vector{Vector{T}}
    con_bnds::Vector{Vector{T}}
    con_idx::TrajectoryOptimizationIndices
    sparsity_jacobian
    sparsity_hessian_lagrangian
    hessian_lagrangian::Bool 
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
        end 
    end
    
    return cl, cu
end 

function Problem(trajopt::TrajectoryOptimizationProblem) 
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

    # nonlinear constraint indices 
    con_idx = indices(trajopt.model.dyn, trajopt.con.stage)

    # nonlinear constraint bounds
    cl, cu = constraint_bounds(trajopt.model.dyn, trajopt.con.stage, nc, con_idx) 

    # constraint Jacobian sparsity
    sp_dyn = sparsity(trajopt.model.dyn, trajopt.model.dim.x, trajopt.model.dim.u, row_shift=0)
    sp_con = sparsity(trajopt.con.stage, trajopt.model.dim.x, trajopt.model.dim.u, row_shift=nc_dyn)
    sp_jac = collect([sp_dyn..., sp_con...]) 

    # Hessian of Lagrangian sparsity
    sp_hess_lag = [] 

    Problem(trajopt, nz, nc, nj, nh, [zl, zu], [cl, cu], con_idx, sp_jac, sp_hess_lag, false)
end