struct Solver{T}
    p::Problem{T}
    nlp_bounds::Vector{MOI.NLPBoundsPair}
    block_data::MOI.NLPBlockData
    solver::Ipopt.Optimizer
    z::Vector{MOI.VariableIndex}
end

function Solver(trajopt::TrajectoryOptimizationProblem; eval_hess=false, options=nothing) 
    p = Problem(trajopt, eval_hess=eval_hess) 
    
    nlp_bounds = MOI.NLPBoundsPair.(p.con_bnds...)
    block_data = MOI.NLPBlockData(nlp_bounds, p, true)
    
    solver = Ipopt.Optimizer()
    
    # TODO: options 
    solver.options["max_iter"] = 1000
    solver.options["tol"] = 1.0e-3
    solver.options["constr_viol_tol"] = 1.0e-3
    
    z = MOI.add_variables(solver, p.num_var)
    
    for i = 1:p.num_var
        MOI.add_constraint(solver, z[i], MOI.LessThan(p.var_bnds[2][i]))
        MOI.add_constraint(solver, z[i], MOI.GreaterThan(p.var_bnds[1][i]))
    end
    
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    
    return Solver(p, nlp_bounds, block_data, solver, z) 
end

function initialize!(s::Solver, z) 
    for i = 1:s.p.num_var
        MOI.set(s.solver, MOI.VariablePrimalStart(), s.z[i], z[i])
    end
end

function solve!(s::Solver) 
    MOI.optimize!(s.solver) 
end

