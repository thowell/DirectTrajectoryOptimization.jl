using LinearAlgebra 
using Symbolics 
using SparseArrays 

f(x) = dot(x[2:3], x[2:3])
c(x) = cos.(x)
L(x, y) = f(x) + dot(y, c(x))

@variables x[1:3], y[1:3]

Lsym = L(x, y)
hess = Symbolics.hessian(Lsym, x)
hess_sp = Symbolics.sparsehessian(Lsym, x)

hess_sp.nzval
sparsity = [findnz(hess_sp)[1:2]...]
findnz(hess_sp)
hess_sp.nzval[1]
hess_sp.nzval[2]
hess_sp[1]


n = 5 
m = 3
r_idx = collect(1:n) 
c_idx = collect(1:m)
A = rand(n, m)
Avec = vec(A)

v_idx = Tuple{Int,Int}[]
for j in c_idx 
    for i in r_idx
        push!(v_idx, (i, j))
    end
end
v_idx 

