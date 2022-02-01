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

struct Solver{T} <: MOI.AbstractNLPEvaluator
    nlp::NLPData{T}
    s_data::SolverData
end

function solver(dyn::Vector{Dynamics{T}}, obj::Objective{T}, cons::Constraints{T}, bnds::Bounds{T}; 
    options=Options{T}(),
    w=[[zeros(nw) for nw in dimensions(dyn)[3]]..., zeros(0)],
    eval_hess=false, 
    general_constraint=GeneralConstraint()) where T

    trajopt = TrajectoryOptimizationData(obj, dyn, cons, bnds, w=w)
    nlp = NLPData(trajopt, general_constraint=general_constraint, eval_hess=eval_hess) 
    s_data = SolverData(nlp, options=options)

    Solver(nlp, s_data) 
end

function initialize_states!(p::Solver, x) 
    for (t, xt) in enumerate(x) 
        n = length(xt)
        for i = 1:n
            MOI.set(p.s_data.solver, MOI.VariablePrimalStart(), p.s_data.z[p.nlp.idx.x[t][i]], xt[i])
        end
    end
end 

function initialize_controls!(p::Solver, u)
    for (t, ut) in enumerate(u) 
        m = length(ut) 
        for j = 1:m
            MOI.set(p.s_data.solver, MOI.VariablePrimalStart(), p.s_data.z[p.nlp.idx.u[t][j]], ut[j])
        end
    end
end

function get_trajectory(p::Solver) 
    return p.nlp.trajopt.x, p.nlp.trajopt.u[1:end-1]
end

function solve!(p::Solver) 
    MOI.optimize!(p.s_data.solver) 
end
