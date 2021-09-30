@testset "Constraints" begin 
    T = 5
    nx = 2
    nu = 1 
    nw = 0
    dim_x = [nx for t = 1:T] 
    dim_u = [nu for t = 1:T-1]
    x = [rand(dim_x[t]) for t = 1:T] 
    u = [[rand(dim_u[t]) for t = 1:T-1]..., zeros(0)]
    ft = (x, u) -> [-ones(nx) - x; x - ones(nx)]
    fT = (x, u) -> x

    cont = StageConstraint(ft, nx, nu, [t for t = 1:T-1], :inequality)
    conT = StageConstraint(fT, nx, 0, [T], :equality)

    cons = [cont, conT]
    nc = DirectTrajectoryOptimization.num_con(cons)
    nj = DirectTrajectoryOptimization.num_jac(cons)
    idx_c = DirectTrajectoryOptimization.constraint_indices(cons)
    idx_j = DirectTrajectoryOptimization.jacobian_indices(cons)
    c = zeros(nc) 
    j = zeros(nj)
    cont.val(c[idx_c[1]], x[1], u[1])
    conT.val(c[idx_c[T]], x[T], u[T])

    DirectTrajectoryOptimization.eval_con!(c, idx_c, cons, x, u)
    info = @benchmark DirectTrajectoryOptimization.eval_con!($c, $idx_c, $cons, $x, $u)

    @test norm(c - vcat([ft(x[t], u[t]) for t = 1:T-1]..., fT(x[T], u[T]))) < 1.0e-8
    DirectTrajectoryOptimization.eval_jac!(j, idx_j, cons, x, u)
    info = @benchmark DirectTrajectoryOptimization.eval_jac!($j, $idx_j, $cons, $x, $u)

    dct = [-I zeros(nx, nu); I zeros(nx, nu)]
    dcT = Diagonal(ones(nx))
    dc = cat([dct for t = 1:T-1]..., dcT, dims=(1,2))
    sp = DirectTrajectoryOptimization.sparsity_jacobian(cons, dim_x, dim_u) 
    j_dense = zeros(nc, sum(dim_x) + sum(dim_u)) 
    for (i, v) in enumerate(sp)
        j_dense[v[1], v[2]] = j[i]
    end
    @test norm(j_dense - dc) < 1.0e-8
end