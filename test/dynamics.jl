@testset "Dynamics" begin 
    T = 3
    nx = 2 
    nu = 1 
    nw = 0 

    function pendulum(z, u, w) 
        mass = 1.0 
        lc = 1.0 
        gravity = 9.81 
        damping = 0.1
        [z[2], (u[1] / ((mass * lc * lc)) - gravity * sin(z[1]) / lc - damping * z[2] / (mass * lc * lc))]
    end

    function euler_implicit(y, x, u, w)
        h = 0.1
        y - (x + h * pendulum(y, u, w))
    end

    dt = Dynamics(euler_implicit, nx, nx, nu, nw=nw);
    dyn = [dt for t = 1:T-1] 
    model = DynamicsModel(dyn)

    
    x1 = ones(nx) 
    u1 = ones(nu)
    w1 = zeros(nw)
    X = [x1 for t = 1:T]
    U = [u1 for t = 1:T]
    W = [w1 for t = 1:T]
    idx_dyn = DirectTrajectoryOptimization.constraint_indices(dyn)
    idx_jac = DirectTrajectoryOptimization.jacobian_indices(dyn)
    dim_x = model.dim.x 
    dim_u = model.dim.u
    d = zeros(DirectTrajectoryOptimization.num_con(dyn))
    j = zeros(DirectTrajectoryOptimization.num_jac(dyn))

    dt.val(dt.val_cache, x1, x1, u1, w1) 
    @test norm(dt.val_cache - euler_implicit(x1, x1, u1, w1)) < 1.0e-8
    dt.jac(dt.jac_cache, x1, x1, u1, w1) 
    jac_dense = zeros(dt.ny, dt.nx + dt.nu + dt.ny)
    for (i, ji) in enumerate(dt.jac_cache)
        jac_dense[dt.sparsity[1][i], dt.sparsity[2][i]] = ji
    end
    jac_fd = ForwardDiff.jacobian(a -> euler_implicit(a[nx + nu .+ (1:nx)], a[1:nx], a[nx .+ (1:nu)], w1), [x1; u1; x1])
    @test norm(jac_dense - jac_fd) < 1.0e-8

    DirectTrajectoryOptimization.eval_con!(d, idx_dyn, dyn, X, U, W)
    @test norm(vcat(d...) - vcat([euler_implicit(X[t+1], X[t], U[t], W[t]) for t = 1:T-1]...)) < 1.0e-8
    info = @benchmark DirectTrajectoryOptimization.eval_con!($d, $idx_dyn, $dyn, $X, $U, $W) 

    DirectTrajectoryOptimization.eval_jac!(j, idx_jac, dyn, X, U, W) 
    s = DirectTrajectoryOptimization.sparsity(dyn, dim_x, dim_u)
    jac_dense = zeros(DirectTrajectoryOptimization.num_con(dyn), DirectTrajectoryOptimization.num_xuy(dyn))
    for (i, ji) in enumerate(j)
        jac_dense[s[i][1], s[i][2]] = ji
    end

    @test norm(jac_dense - [jac_fd zeros(dyn[2].nx, dyn[2].nu + dyn[2].ny); zeros(dyn[2].ny, dyn[1].nx + dyn[1].nu) jac_fd]) < 1.0e-8
    info = @benchmark DirectTrajectoryOptimization.eval_jac!($j, $idx_jac, $dyn, $X, $U, $W) 

    idx_x = model.idx.x
    idx_u = model.idx.u
    idx_xu = model.idx.xu
    idx_xuy = model.idx.xuy

    nz = sum([t < T ? dyn[t].nx : dyn[t-1].ny for t = 1:T]) + sum([dyn[t].nu for t = 1:T-1])
    z = rand(nz)
    x = [zero(z[idx_x[t]]) for t = 1:T]
    u = [[zero(z[idx_u[t]]) for t = 1:T-1]..., zeros(0)]

    DirectTrajectoryOptimization.trajectory!(x, u, z, idx_x, idx_u)
    z̄ = zero(z)
    for (t, idx) in enumerate(idx_x) 
        z̄[idx] .= x[t] 
    end
    for (t, idx) in enumerate(idx_u) 
        z̄[idx] .= u[t] 
    end

    @test norm(z - z̄) < 1.0e-8
    info = @benchmark DirectTrajectoryOptimization.trajectory!($x, $u, $z, $idx_x, $idx_u)
end

