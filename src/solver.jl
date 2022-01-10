@with_kw mutable struct Options{T} 
    # Ipopt settings: https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_print_options_documentation
    tol::T = 1e-6
    s_max::T = 100.0
    max_iter::Int = 1000
    # max_wall_time = 300.0
    max_cpu_time = 300.0
    dual_inf_tol::T = 1.0
    constr_viol_tol::T = 1.0e-3
    compl_inf_tol::T = 1.0e-3
    acceptable_tol::T = 1.0e-6
    acceptable_iter::Int = 15
    acceptable_dual_inf_tol::T = 1.0e10
    acceptable_constr_viol_tol::T = 1.0e-2
    acceptable_compl_inf_tol::T = 1.0e-2
    acceptable_obj_change_tol::T = 1.0e-5
    diverging_iterates_tol::T = 1.0e8
    mu_target::T = 1.0e-4
    print_level::Int = 5
    output_file = "output.txt"
    print_user_options = "no"
    # print_options_documentation = "no"
    # print_timing_statistics = "no"
    print_options_mode = "text"
    # print_advanced_options = "no"
    print_info_string = "no"
    inf_pr_output = "original"
    print_frequency_iter = 1
    print_frequency_time = 0.0
    skip_finalize_solution_call = "no"
    # timing_statistics = :no
end

# struct Solver{T}
#     p::Problem{T}
#     nlp_bounds::Vector{MOI.NLPBoundsPair}
#     block_data::MOI.NLPBlockData
#     solver::Ipopt.Optimizer
#     z::Vector{MOI.VariableIndex}
# end

# function Solver(trajopt::TrajectoryOptimizationProblem; eval_hess=false, options=Options()) 
#     p = Problem(trajopt, eval_hess=eval_hess) 
    
#     nlp_bounds = MOI.NLPBoundsPair.(p.con_bnds...)
#     block_data = MOI.NLPBlockData(nlp_bounds, p, true)
    
#     # instantiate NLP solver
#     solver = Ipopt.Optimizer()

#     # set NLP solver options
#     for name in fieldnames(typeof(options))
#         solver.options[String(name)] = getfield(options, name)
#     end
    
#     z = MOI.add_variables(solver, p.num_var)
    
#     for i = 1:p.num_var
#         MOI.add_constraint(solver, z[i], MOI.LessThan(p.var_bnds[2][i]))
#         MOI.add_constraint(solver, z[i], MOI.GreaterThan(p.var_bnds[1][i]))
#     end
    
#     MOI.set(solver, MOI.NLPBlock(), block_data)
#     MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    
#     return Solver(p, nlp_bounds, block_data, solver, z) 
# end

# function initialize!(s::Solver, z) 
#     for i = 1:s.p.num_var
#         MOI.set(s.solver, MOI.VariablePrimalStart(), s.z[i], z[i])
#     end
# end


