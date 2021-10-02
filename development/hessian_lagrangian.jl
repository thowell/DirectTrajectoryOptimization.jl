using LinearAlgebra 
using Symbolics 
using SparseArrays 

f(x) = dot(x[2:3], x[2:3])
c(x) = cos.(x)
L(x, y) = f(x) + dot(y, c(x))

@variables x[1:3], y[1:3]

Lsym = L(x, y)
hess = Symbolics.hessian(Lsym, x)
hess_func = eval(Symbolics.build_function(hess, x, y)[1])
hess_sp = Symbolics.sparsehessian(Lsym, x)
hess_sp_func = eval(Symbolics.build_function(hess_sp.nzval, x, y)[2])
hess_sp.nzval
sparsity = [findnz(hess_sp)[1:2]...]
# findnz(hess_sp)
# hess_sp.nzval[1]
# hess_sp.nzval[2]
# hess_sp[1]

x0 = rand(3) 
y0 = rand(3)
h0 = zeros(size(hess))
h0_sp = zeros(length(hess_sp.nzval))

hess_sp_func(h0_sp, x0, y0)
for (i, h) in enumerate(h0_sp)
    h0[sparsity[1][i], sparsity[2][i]] = h
end

h0 - hess_func(x0, y0)