@testset "Constraints" begin 
    T = 5
    nx = 2
    nu = 1 
    nw = 0
    dim_x = [nx for t = 1:T] 
    dim_u = [nu for t = 1:T-1]
    dim_w = [nw for t = 1:T]
    x = [rand(dim_x[t]) for t = 1:T] 
    u = [[rand(dim_u[t]) for t = 1:T-1]..., zeros(0)]
    w = [rand(dim_w[t]) for t = 1:T]

    ct = (x, u, w) -> [-ones(nx) - x; x - ones(nx)]
    cT = (x, u, w) -> x

    cont = Constraint(ct, nx, nu, nw=nw, idx_ineq=collect(1:2nx))
    conT = Constraint(cT, nx, 0, nw=0)

    cons = [[cont for t = 1:T-1]..., conT]
    nc = DTO.num_con(cons)
    nj = DTO.num_jac(cons)
    idx_c = DTO.constraint_indices(cons)
    idx_j = DTO.jacobian_indices(cons)
    c = zeros(nc) 
    j = zeros(nj)
    cont.val(c[idx_c[1]], x[1], u[1], w[1])
    conT.val(c[idx_c[T]], x[T], u[T], w[T])

    DTO.eval_con!(c, idx_c, cons, x, u, w)
    # info = @benchmark DTO.eval_con!($c, $idx_c, $cons, $x, $u, $w)

    @test norm(c - vcat([ct(x[t], u[t], w[t]) for t = 1:T-1]..., cT(x[T], u[T], w[T]))) < 1.0e-8
    DTO.eval_jac!(j, idx_j, cons, x, u, w)
    # info = @benchmark DTO.eval_jac!($j, $idx_j, $cons, $x, $u, $w)

    dct = [-I zeros(nx, nu); I zeros(nx, nu)]
    dcT = Diagonal(ones(nx))
    dc = cat([dct for t = 1:T-1]..., dcT, dims=(1,2))
    sp = DTO.sparsity_jacobian(cons, dim_x, dim_u) 
    j_dense = zeros(nc, sum(dim_x) + sum(dim_u)) 
    for (i, v) in enumerate(sp)
        j_dense[v[1], v[2]] = j[i]
    end
    @test norm(j_dense - dc) < 1.0e-8
end