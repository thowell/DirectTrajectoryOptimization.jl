
function MOI.eval_objective(p::Problem{T}, z::Vector{T}) where T
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    eval_obj(p.trajopt.obj, p.trajopt.x, p.trajopt.u) 
end

function MOI.eval_objective_gradient(p::Problem{T}, grad, z) where T
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    eval_obj_grad!(grad, p.trajopt.model.idx.xu, p.trajopt.obj, p.trajopt.x, p.trajopt.u) 
end

function MOI.eval_constraint(p::Problem{T}, con, z) where T
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    eval_con!(con, p.con_idx.dyn_con, p.trajopt.model.dyn, p.trajopt.x, p.trajopt.u, p.trajopt.w)
    !isempty(p.con_idx.stage_con) && eval_con!(con, p.con_idx.stage_con, p.trajopt.con.stage, p.trajopt.x, p.trajopt.u)
end

function MOI.eval_constraint_jacobian(p::Problem{T}, jac, z) where T
    trajectory!(p.trajopt.x, p.trajopt.u, z, 
        p.trajopt.model.idx.x, p.trajopt.model.idx.u)
    eval_jac!(jac, p.con_idx.dyn_jac, p.trajopt.model.dyn, p.trajopt.x, p.trajopt.u, p.trajopt.w)
    !isempty(p.con_idx.stage_jac) && eval_jac!(jac, p.con_idx.stage_jac, p.trajopt.con.stage, p.trajopt.x, p.trajopt.u)
    return nothing
end

MOI.eval_hessian_lagrangian(p::MOI.AbstractNLPEvaluator, H, x, σ, μ) = nothing

MOI.features_available(p::MOI.AbstractNLPEvaluator) = p.hessian_lagrangian ? [:Grad, :Jac, :Hess] : [:Grad, :Jac]
MOI.initialize(p::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(p::MOI.AbstractNLPEvaluator) = p.sparsity_jacobian
MOI.hessian_lagrangian_structure(p::MOI.AbstractNLPEvaluator) = p.sparsity_hessian_lagrangian


