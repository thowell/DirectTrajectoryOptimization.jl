function MOI.eval_objective(nlp::NLPData{T}, z::Vector{T}) where T
    trajectory!(nlp.trajopt.x, nlp.trajopt.u, z, 
        nlp.idx.x, nlp.idx.u)
    eval_obj(nlp.trajopt.obj, nlp.trajopt.x, nlp.trajopt.u, nlp.trajopt.w) 
end

function MOI.eval_objective_gradient(nlp::NLPData{T}, grad, z) where T
    fill!(grad, 0.0)
    trajectory!(nlp.trajopt.x, nlp.trajopt.u, z, 
        nlp.idx.x, nlp.idx.u)
    eval_obj_grad!(grad, nlp.idx.xu, nlp.trajopt.obj, nlp.trajopt.x, nlp.trajopt.u, nlp.trajopt.w) 
end

function MOI.eval_constraint(nlp::NLPData{T}, con, z) where T
    fill!(con, 0.0)
    trajectory!(nlp.trajopt.x, nlp.trajopt.u, z, 
        nlp.idx.x, nlp.idx.u)
    eval_con!(con, nlp.idx.dyn_con, nlp.trajopt.dyn, nlp.trajopt.x, nlp.trajopt.u, nlp.trajopt.w)
    !isempty(nlp.idx.stage_con) && eval_con!(con, nlp.idx.stage_con, nlp.trajopt.cons, nlp.trajopt.x, nlp.trajopt.u, nlp.trajopt.w)
    nlp.gc.nc != 0 && eval_con!(con, nlp.idx.gen_con, nlp.gc, z, nlp.w)
end

function MOI.eval_constraint_jacobian(nlp::NLPData{T}, jac, z) where T
    fill!(jac, 0.0)
    trajectory!(nlp.trajopt.x, nlp.trajopt.u, z, 
        nlp.idx.x, nlp.idx.u)
    eval_jac!(jac, nlp.idx.dyn_jac, nlp.trajopt.dyn, nlp.trajopt.x, nlp.trajopt.u, nlp.trajopt.w)
    !isempty(nlp.idx.stage_jac) && eval_jac!(jac, nlp.idx.stage_jac, nlp.trajopt.cons, nlp.trajopt.x, nlp.trajopt.u, nlp.trajopt.w)
    nlp.gc.nc != 0 && eval_jac!(jac, nlp.idx.gen_jac, nlp.gc, z, nlp.w)
    return nothing
end

function MOI.eval_hessian_lagrangian(nlp::MOI.AbstractNLPEvaluator, hess, z, σ, λ)
    fill!(hess, 0.0)
    trajectory!(nlp.trajopt.x, nlp.trajopt.u, z, 
        nlp.idx.x, nlp.idx.u)
    duals!(nlp.trajopt.λ_dyn, nlp.trajopt.λ_stage, nlp.λ_gen, λ, nlp.idx.dyn_con, nlp.idx.stage_con, nlp.idx.gen_con)
    eval_obj_hess!(hess, nlp.idx.obj_hess, nlp.trajopt.obj, nlp.trajopt.x, nlp.trajopt.u,nlp.trajopt.w, σ)
    eval_hess_lag!(hess, nlp.idx.dyn_hess, nlp.trajopt.dyn, nlp.trajopt.x, nlp.trajopt.u, nlp.trajopt.w, nlp.trajopt.λ_dyn)
    eval_hess_lag!(hess, nlp.idx.stage_hess, nlp.trajopt.cons, nlp.trajopt.x, nlp.trajopt.u, nlp.trajopt.w, nlp.trajopt.λ_stage)
    eval_hess_lag!(hess, nlp.idx.gen_hess, nlp.gc, z, nlp.w, nlp.λ_gen)
end

MOI.features_available(nlp::MOI.AbstractNLPEvaluator) = nlp.hess_lag ? [:Grad, :Jac, :Hess] : [:Grad, :Jac]
MOI.initialize(nlp::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(nlp::MOI.AbstractNLPEvaluator) = nlp.sp_jac
MOI.hessian_lagrangian_structure(nlp::MOI.AbstractNLPEvaluator) = nlp.sp_hess_lag


