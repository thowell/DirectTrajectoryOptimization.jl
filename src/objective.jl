struct Cost{T}
    #TODO: types for methods
    val
    grad
    hess
    sparsity::Vector{Vector{Int}}
    val_cache::Vector{T}
    grad_cache::Vector{T}
    hess_cache::Vector{T}
    idx::Vector{Int}
end

function Cost(f::Function, nx::Int, nu::Int, idx::Vector{Int}; eval_hess=false)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu]
    val = f(x, u)
    grad = Symbolics.gradient(val, [x; u])
    val_func = eval(Symbolics.build_function([val], x, u)[2])
    grad_func = eval(Symbolics.build_function(grad, x, u)[2])
    if eval_hess 
        hess = Symbolics.sparsehessian(val, [x; u])
        hess_func = eval(Symbolics.build_function(hess.nzval, x, u)[2])
        sparsity = [findnz(jac)[1:2]...]
    else 
        hess_func = Expr(:null) 
        sparsity = [Int[]]
    end

    return Cost(val_func, grad_func, hess_func, sparsity,
        zeros(1), zeros(nx + nu), zeros((nx + nu) * (nx + nu)), idx)
end

Objective{T} = Vector{Cost{T}} where T

function eval_obj(obj::Objective, x, u) 
    J = 0.0
    for cost in obj
        for t in cost.idx
            cost.val(cost.val_cache, x[t], u[t])
            J += cost.val_cache[1]
        end
    end
    return J 
end

function eval_obj_grad!(grad, idx, obj::Objective, x, u)
    fill!(grad, 0.0)
    for cost in obj
        for t in cost.idx
            cost.grad(cost.grad_cache, x[t], u[t])
            @views grad[idx[t]] .+= cost.grad_cache
            # fill!(cost.grad_cache, 0.0) # TODO: confirm this is necessary
        end
    end
end

#TODO: test
function eval_obj_hessian!(hess, idx, obj::Objective, x, u, σ)
    fill!(hess, 0.0)
    for cost in obj
        for t in cost.idx
            cost.hess(cost.hess_cache, x[t], u[t])
            @views hess[idx[t]] .+= cost.hess_cache * σ
            # fill!(cost.grad_cache, 0.0) # TODO: confirm this is necessary
        end
    end
end