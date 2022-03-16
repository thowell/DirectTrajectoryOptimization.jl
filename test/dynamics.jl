@testset "Dynamics" begin 
    T = 3
    num_state = 2 
    nu = 1 
    nw = 0 
    w_dim = [nw for t = 1:T]

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

    dt = Dynamics(euler_implicit, num_state, num_state, nu, nw=nw);
    dyn = [dt for t = 1:T-1] 

    x1 = ones(num_state) 
    u1 = ones(nu)
    w1 = zeros(nw)
    X = [x1 for t = 1:T]
    U = [u1 for t = 1:T]
    W = [w1 for t = 1:T]
    idx_dyn = DTO.constraint_indices(dyn)
    idx_jac = DTO.jacobian_indices(dyn)

    d = zeros(DTO.num_con(dyn))
    j = zeros(DTO.num_jac(dyn))

    dt.val(dt.val_cache, x1, x1, u1, w1) 
    # @benchmark $dt.val($dt.val_cache, $x1, $x1, $u1, $w1) 
    @test norm(dt.val_cache - euler_implicit(x1, x1, u1, w1)) < 1.0e-8
    dt.jac(dt.jac_cache, x1, x1, u1, w1) 
    jac_dense = zeros(dt.ny, dt.num_state + dt.nu + dt.ny)
    for (i, ji) in enumerate(dt.jac_cache)
        jac_dense[dt.sp_jac[1][i], dt.sp_jac[2][i]] = ji
    end
    jac_fd = ForwardDiff.jacobian(a -> euler_implicit(a[num_state + nu .+ (1:num_state)], a[1:num_state], a[num_state .+ (1:nu)], w1), [x1; u1; x1])
    @test norm(jac_dense - jac_fd) < 1.0e-8

    DTO.eval_con!(d, idx_dyn, dyn, X, U, W)
    @test norm(vcat(d...) - vcat([euler_implicit(X[t+1], X[t], U[t], W[t]) for t = 1:T-1]...)) < 1.0e-8
    # info = @benchmark DTO.eval_con!($d, $idx_dyn, $dyn, $X, $U, $W) 

    DTO.eval_jac!(j, idx_jac, dyn, X, U, W) 
    s = DTO.sparsity_jacobian(dyn, DTO.dimensions(dyn)[1:2]...)
    jac_dense = zeros(DTO.num_con(dyn), DTO.num_xuy(dyn))
    for (i, ji) in enumerate(j)
        jac_dense[s[i][1], s[i][2]] = ji
    end

    @test norm(jac_dense - [jac_fd zeros(dyn[2].num_state, dyn[2].nu + dyn[2].ny); zeros(dyn[2].ny, dyn[1].num_state + dyn[1].nu) jac_fd]) < 1.0e-8
    # info = @benchmark DTO.eval_jac!($j, $idx_jac, $dyn, $X, $U, $W) 

    x_idx = DTO.x_indices(dyn)
    u_idx = DTO.u_indices(dyn)
    xu_idx = DTO.xu_indices(dyn)
    xuy_idx = DTO.xuy_indices(dyn)

    nz = sum([t < T ? dyn[t].num_state : dyn[t-1].ny for t = 1:T]) + sum([dyn[t].nu for t = 1:T-1])
    z = rand(nz)
    x = [zero(z[x_idx[t]]) for t = 1:T]
    u = [[zero(z[u_idx[t]]) for t = 1:T-1]..., zeros(0)]

    DTO.trajectory!(x, u, z, x_idx, u_idx)
    z̄ = zero(z)
    for (t, idx) in enumerate(x_idx) 
        z̄[idx] .= x[t] 
    end
    for (t, idx) in enumerate(u_idx) 
        z̄[idx] .= u[t] 
    end

    @test norm(z - z̄) < 1.0e-8
    # info = @benchmark DTO.trajectory!($x, $u, $z, $x_idx, $u_idx)
end

