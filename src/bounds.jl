struct Bound{T}
    xl::Vector{T} 
    xu::Vector{T}
    ul::Vector{T}
    uu::Vector{T}
end

function Bound(nx::Int=0, nu::Int=0; 
    xl=-Inf * ones(nx), xu=Inf * ones(nx), ul=-Inf * ones(nu), uu=Inf * ones(nu)) 
    return Bound(xl, xu, ul, uu)
end

const Bounds{T} = Vector{Bound{T}}
