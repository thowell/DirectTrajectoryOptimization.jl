struct Cost{T}
    #TODO: types for methods
    val
    grad
    hess
    sp::Vector{Vector{Int}}
    val_cache::Vector{T}
    grad_cache::Vector{T}
    hess_cache::Vector{T}
    idx::Vector{Int}
end

function Cost(f::Function, nx::Int, nu::Int, nw::Int, idx::Vector{Int}; eval_hess=false)
    #TODO: option to load/save methods
    @variables x[1:nx], u[1:nu], w[1:nw]
    val = f(x, u, w)
    grad = Symbolics.gradient(val, [x; u])
    val_func = eval(Symbolics.build_function([val], x, u, w)[2])
    grad_func = eval(Symbolics.build_function(grad, x, u, w)[2])
    if eval_hess 
        hess = Symbolics.sparsehessian(val, [x; u])
        hess_func = eval(Symbolics.build_function(hess.nzval, x, u, w)[2])
        sparsity = [findnz(hess)[1:2]...]
        nh = length(hess.nzval)
    else 
        hess_func = Expr(:null) 
        sparsity = [Int[]]
        nh = 0
    end

    return Cost(val_func, grad_func, hess_func, sparsity,
        zeros(1), zeros(nx + nu), zeros(nh), idx)
end

Objective{T} = Vector{Cost{T}} where T

function eval_obj(obj::Objective, x, u, w) 
    J = 0.0
    for cost in obj
        for t in cost.idx
            cost.val(cost.val_cache, x[t], u[t], w[t])
            J += cost.val_cache[1]
        end
    end
    return J 
end

function eval_obj_grad!(grad, idx, obj::Objective, x, u, w)
    for cost in obj
        for t in cost.idx
            cost.grad(cost.grad_cache, x[t], u[t], w[t])
            @views grad[idx[t]] .+= cost.grad_cache
            # fill!(cost.grad_cache, 0.0) # TODO: confirm this is necessary
        end
    end
end

#TODO: test
function eval_obj_hess!(hess, idx, obj::Objective, x, u, w, σ)
    i = 1
    for cost in obj
        for t in cost.idx
            cost.hess(cost.hess_cache, x[t], u[t], w[t])
            cost.hess_cache .*= σ
            @views hess[idx[i]] .+= cost.hess_cache
            i += 1
            fill!(cost.hess_cache, 0.0) # TODO: confirm this is necessary
        end
    end
end

function sparsity_hessian(obj::Objective, nx::Vector{Int}, nu::Vector{Int})
    row = Int[]
    col = Int[]
    for cost in obj
        if !isempty(cost.sp[1])
            for t in cost.idx
                shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
                push!(row, (cost.sp[1] .+ shift)...) 
                push!(col, (cost.sp[2] .+ shift)...) 
            end
        end
    end
    return collect(zip(row, col))
end

function hessian_indices(obj::Objective, key::Vector{Tuple{Int,Int}}, nx::Vector{Int}, nu::Vector{Int})
    idx = Vector{Int}[]
    for cost in obj
        if !isempty(cost.sp[1])
            for t in cost.idx
                row = Int[]
                col = Int[]
                shift = (t > 1 ? (sum(nx[1:t-1]) + sum(nu[1:t-1])) : 0)
                push!(row, (cost.sp[1] .+ shift)...) 
                push!(col, (cost.sp[2] .+ shift)...) 
                rc = collect(zip(row, col))
                push!(idx, [findfirst(x -> x == i, key) for i in rc])
            end
        end
    end
    return idx
end