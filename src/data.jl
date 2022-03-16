struct TrajectoryOptimizationData{T}
    x::Vector{Vector{T}}
    u::Vector{Vector{T}}
    w::Vector{Vector{T}}
    obj::Objective{T}
    dyn::Vector{Dynamics{T}}
    cons::Constraints{T}
    bnds::Bounds{T}
    λ_dyn::Vector{Vector{T}} 
    λ_stage::Vector{Vector{T}}
    x_dim::Vector{Int}
    u_dim::Vector{Int}
    w_dim::Vector{Int}
end

function TrajectoryOptimizationData(obj::Objective{T}, dyn::Vector{Dynamics{T}}, cons::Constraints{T}, bnds::Bounds{T}; 
    w = [zeros(nw) for nw in dimensions(dyn)[3]]) where T

    x_dim, u_dim, w_dim = dimensions(dyn)

    x = [zeros(num_state) for num_state in x_dim]
    u = [zeros(nu) for nu in u_dim]

    λ_dyn = [zeros(num_state) for num_state in x_dim[2:end]]
    λ_stage = [zeros(con.nc) for con in cons]
   
    TrajectoryOptimizationData(x, u, w, obj, dyn, cons, bnds, λ_dyn, λ_stage, x_dim, u_dim, w_dim)
end

TrajectoryOptimizationData(obj::Objective, dyn::Vector{Dynamics}) = TrajectoryOptimizationData(obj, dyn, [Constraint() for t = 1:length(dyn)], [Bound() for t = 1:length(dyn)])

struct TrajectoryOptimizationIndices 
    obj_hess::Vector{Vector{Int}}
    dyn_con::Vector{Vector{Int}} 
    dyn_jac::Vector{Vector{Int}} 
    dyn_hess::Vector{Vector{Int}}
    stage_con::Vector{Vector{Int}} 
    stage_jac::Vector{Vector{Int}} 
    stage_hess::Vector{Vector{Int}}
    gen_con::Vector{Int}
    gen_jac::Vector{Int}
    gen_hess::Vector{Int}
    x::Vector{Vector{Int}}
    u::Vector{Vector{Int}}
    xu::Vector{Vector{Int}}
    xuy::Vector{Vector{Int}}
end

function indices(obj::Objective{T}, dyn::Vector{Dynamics{T}}, cons::Constraints{T}, gc::GeneralConstraint{T},
     key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, nu::Vector{Int}, nz::Int) where T 
    # Jacobians
    dyn_con = constraint_indices(dyn, shift=0)
    dyn_jac = jacobian_indices(dyn, shift=0)
    stage_con = constraint_indices(cons, shift=num_con(dyn))
    stage_jac = jacobian_indices(cons, shift=num_jac(dyn)) 
    gen_con = constraint_indices(gc, shift=(num_con(dyn) + num_con(cons)))
    gen_jac = jacobian_indices(gc, shift=(num_jac(dyn) + num_jac(cons))) 

    # Hessian of Lagrangian 
    obj_hess = hessian_indices(obj, key, num_state, nu)
    dyn_hess = hessian_indices(dyn, key, num_state, nu)
    stage_hess = hessian_indices(cons, key, num_state, nu)
    gen_hess = hessian_indices(gc, key, nz)

    # indices
    x_idx = x_indices(dyn)
    u_idx = u_indices(dyn)
    xu_idx = xu_indices(dyn)
    xuy_idx = xuy_indices(dyn)

    return TrajectoryOptimizationIndices(
        obj_hess, 
        dyn_con, dyn_jac, dyn_hess, 
        stage_con, stage_jac, stage_hess,
        gen_con, gen_jac, gen_hess,
        x_idx, u_idx, xu_idx, xuy_idx) 
end

struct NLPData{T} <: MOI.AbstractNLPEvaluator
    trajopt::TrajectoryOptimizationData{T}
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
    gc::GeneralConstraint{T}
    w::Vector{T}
    λ_gen::Vector{T}
end

function primal_bounds(bnds::Bounds{T}, nz::Int, x_idx::Vector{Vector{Int}}, u_idx::Vector{Vector{Int}}) where T 
    zl, zu = -Inf * ones(nz), Inf * ones(nz) 
    for (t, bnd) in enumerate(bnds)
        length(bnd.state_lower) > 0 && (zl[x_idx[t]] = bnd.state_lower)
        length(bnd.xu) > 0 && (zu[x_idx[t]] = bnd.xu)
        length(bnd.ul) > 0 && (zl[u_idx[t]] = bnd.ul)
        length(bnd.uu) > 0 && (zu[u_idx[t]] = bnd.uu)
    end
    return zl, zu
end

function constraint_bounds(cons::Constraints{T}, gc::GeneralConstraint{T}, 
    nc_dyn::Int, nc_con::Int, idx::TrajectoryOptimizationIndices) where T
    # total constraints
    nc = nc_dyn + nc_con + gc.nc 
    # bounds
    cl, cu = zeros(nc), zeros(nc) 
    # stage
    for (t, con) in enumerate(cons) 
        cl[idx.stage_con[t][con.idx_ineq]] .= -Inf
    end
    # general
    cl[collect(nc_dyn + nc_con .+ gc.idx_ineq)] .= -Inf
    return cl, cu
end 

function NLPData(trajopt::TrajectoryOptimizationData; 
    eval_hess=false, 
    general_constraint=GeneralConstraint()) 

    # number of variables
    nz = sum(trajopt.x_dim) + sum(trajopt.u_dim)

    # number of constraints
    nc_dyn = num_con(trajopt.dyn)
    nc_con = num_con(trajopt.cons)  
    nc_gen = num_con(general_constraint)
    nc = nc_dyn + nc_con + nc_gen

    # number of nonzeros in constraint Jacobian
    nj_dyn = num_jac(trajopt.dyn)
    nj_con = num_jac(trajopt.cons)  
    nj_gen = num_jac(general_constraint)
    nj = nj_dyn + nj_con + nj_gen

    # number of nonzeros in Hessian of Lagrangian
    nh = 0

    # constraint Jacobian sparsity
    sp_dyn = sparsity_jacobian(trajopt.dyn, trajopt.x_dim, trajopt.u_dim, row_shift=0)
    sp_con = sparsity_jacobian(trajopt.cons, trajopt.x_dim, trajopt.u_dim, row_shift=nc_dyn)
    sp_gen = sparsity_jacobian(general_constraint, nz, row_shift=(nc_dyn + nc_con))
    sp_jac = collect([sp_dyn..., sp_con..., sp_gen...]) 

    # Hessian of Lagrangian sparsity 
    sp_obj_hess = sparsity_hessian(trajopt.obj, trajopt.x_dim, trajopt.u_dim)
    sp_dyn_hess = sparsity_hessian(trajopt.dyn, trajopt.x_dim, trajopt.u_dim)
    sp_con_hess = sparsity_hessian(trajopt.cons, trajopt.x_dim, trajopt.u_dim)
    sp_gen_hess = sparsity_hessian(general_constraint, nz)
    sp_hess_lag = [sp_obj_hess..., sp_dyn_hess..., sp_con_hess..., sp_gen_hess...]
    sp_hess_lag = !isempty(sp_hess_lag) ? sp_hess_lag : Tuple{Int,Int}[]
    sp_hess_key = sort(unique(sp_hess_lag))

    # indices 
    idx = indices(trajopt.obj, trajopt.dyn, trajopt.cons, 
        general_constraint, 
        sp_hess_key, 
        trajopt.x_dim, trajopt.u_dim, nz)

    # primal variable bounds
    zl, zu = primal_bounds(trajopt.bnds, nz, idx.x, idx.u) 

    # nonlinear constraint bounds
    cl, cu = constraint_bounds(trajopt.cons, general_constraint, nc_dyn, nc_con, idx) 

    NLPData(trajopt, 
        nz, nc, nj, nh, 
        [zl, zu], [cl, cu], 
        idx, 
        sp_jac, sp_hess_key,
        eval_hess, 
        general_constraint,
        vcat(trajopt.w...),
        zeros(general_constraint.nc))
end

struct SolverData 
    nlp_bounds::Vector{MOI.NLPBoundsPair}
    block_data::MOI.NLPBlockData
    solver::Ipopt.Optimizer
    z::Vector{MOI.VariableIndex} 
end

function SolverData(nlp::NLPData; options=Options()) 
    # solver
    nlp_bounds = MOI.NLPBoundsPair.(nlp.con_bnds...)
    block_data = MOI.NLPBlockData(nlp_bounds, nlp, true)
    
    # instantiate NLP solver
    solver = Ipopt.Optimizer()

    # set NLP solver options
    for name in fieldnames(typeof(options))
        solver.options[String(name)] = getfield(options, name)
    end
    
    z = MOI.add_variables(solver, nlp.num_var)
    
    for i = 1:nlp.num_var
        MOI.add_constraint(solver, z[i], MOI.LessThan(nlp.var_bnds[2][i]))
        MOI.add_constraint(solver, z[i], MOI.GreaterThan(nlp.var_bnds[1][i]))
    end
    
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE) 

    return SolverData(nlp_bounds, block_data, solver, z)
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

function duals!(λ_dyn::Vector{Vector{T}}, λ_stage::Vector{Vector{T}}, λ_gen::Vector{T}, λ, 
    idx_dyn::Vector{Vector{Int}}, idx_stage::Vector{Vector{Int}}, idx_gen::Vector{Int}) where T
    for (t, idx) in enumerate(idx_dyn)
        λ_dyn[t] .= @views λ[idx]
    end 
    for (t, idx) in enumerate(idx_stage)
        λ_stage[t] .= @views λ[idx]
    end
    λ_gen .= λ[idx_gen]
end

struct ProblemData{T} <: MOI.AbstractNLPEvaluator
    nlp::NLPData{T}
    s_data::SolverData
end

function ProblemData(obj::Objective{T}, dyn::Vector{Dynamics{T}}, cons::Constraints{T}, bnds::Bounds{T}; 
    eval_hess=false, 
    general_constraint=GeneralConstraint(),
    options=Options(),
    w=[[zeros(nw) for nw in dimensions(dyn)[3]]..., zeros(0)]) where T

    trajopt = TrajectoryOptimizationData(obj, dyn, cons, bnds, w=w)
    nlp = NLPData(trajopt, general_constraint=general_constraint, eval_hess=eval_hess) 
    s_data = SolverData(nlp, options=options)

    ProblemData(nlp, s_data) 
end

function initialize_states!(p::ProblemData, x) 
    for (t, xt) in enumerate(x) 
        n = length(xt)
        for i = 1:n
            MOI.set(p.s_data.solver, MOI.VariablePrimalStart(), p.s_data.z[p.nlp.idx.x[t][i]], xt[i])
        end
    end
end 

function initialize_controls!(p::ProblemData, u)
    for (t, ut) in enumerate(u) 
        m = length(ut) 
        for j = 1:m
            MOI.set(p.s_data.solver, MOI.VariablePrimalStart(), p.s_data.z[p.nlp.idx.u[t][j]], ut[j])
        end
    end
end

function get_trajectory(p::ProblemData) 
    return p.nlp.trajopt.x, p.nlp.trajopt.u[1:end-1]
end

function solve!(p::ProblemData) 
    MOI.optimize!(p.s_data.solver) 
end