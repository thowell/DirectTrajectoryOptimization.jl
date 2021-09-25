function linear_interpolation(x1, xT, T)
    n = length(x1)
    X = [copy(Array(x1)) for t = 1:T]
    for t = 1:T
        for i = 1:n
            X[t][i] = (xT[i] - x1[i]) / (T - 1) * (t - 1) + x1[i]
        end
    end
    return X
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
