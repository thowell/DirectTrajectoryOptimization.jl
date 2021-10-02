
function MOI.eval_objective(p::Problem{T}, z::Vector{T}) where T
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    eval_obj(p.trajopt.obj, p.trajopt.x, p.trajopt.u, p.trajopt.w) 
end

function MOI.eval_objective_gradient(p::Problem{T}, grad, z) where T
    fill!(grad, 0.0)
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    eval_obj_grad!(grad, p.trajopt.model.idx.xu, p.trajopt.obj, p.trajopt.x, p.trajopt.u, p.trajopt.w) 
end

function MOI.eval_constraint(p::Problem{T}, con, z) where T
    fill!(con, 0.0)
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    eval_con!(con, p.idx.dyn_con, p.trajopt.model.dyn, p.trajopt.x, p.trajopt.u, p.trajopt.w)
    !isempty(p.idx.stage_con) && eval_con!(con, p.idx.stage_con, p.trajopt.con.stage, p.trajopt.x, p.trajopt.u, p.trajopt.w)
end

function MOI.eval_constraint_jacobian(p::Problem{T}, jac, z) where T
    fill!(jac, 0.0)
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    eval_jac!(jac, p.idx.dyn_jac, p.trajopt.model.dyn, p.trajopt.x, p.trajopt.u, p.trajopt.w)
    !isempty(p.idx.stage_jac) && eval_jac!(jac, p.idx.stage_jac, p.trajopt.con.stage, p.trajopt.x, p.trajopt.u, p.trajopt.w)
    return nothing
end

function MOI.eval_hessian_lagrangian(p::MOI.AbstractNLPEvaluator, hess, z, σ, λ)
    fill!(hess, 0.0)
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    duals!(p.trajopt.λ_dyn, p.trajopt.λ_stage, λ, p.idx.dyn_con, p.idx.stage_con)
    eval_obj_hess!(hess, p.idx.obj_hess, p.trajopt.obj, p.trajopt.x, p.trajopt.u,p.trajopt.w, σ)
    eval_hess_lag!(hess, p.idx.dyn_hess, p.trajopt.model.dyn, p.trajopt.x, p.trajopt.u, p.trajopt.w, p.trajopt.λ_dyn)
    eval_hess_lag!(hess, p.idx.stage_hess, p.trajopt.con.stage, p.trajopt.x, p.trajopt.u, p.trajopt.w, p.trajopt.λ_stage)
end

MOI.features_available(p::MOI.AbstractNLPEvaluator) = p.hess_lag ? [:Grad, :Jac, :Hess] : [:Grad, :Jac]
MOI.initialize(p::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(p::MOI.AbstractNLPEvaluator) = p.sp_jac
MOI.hessian_lagrangian_structure(p::MOI.AbstractNLPEvaluator) = p.sp_hess_lag


