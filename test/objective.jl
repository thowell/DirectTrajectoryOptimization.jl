@testset "Objective" begin
    T = 3
    nx = 2
    nu = 1 
    ft = (x, u) -> dot(x, x) + 0.1 * dot(u, u)
    fT = (x, u) -> 10.0 * dot(x, x)
    ct = Cost(ft, nx, nu, [t for t = 1:T-1])
    cT = Cost(fT, nx, 0, [T])
    obj = [ct, cT]

    J = [0.0]
    grad = zeros((T - 1) * (nx + nu) + nx)
    idx_xu = [collect((t - 1) * (nx + nu) .+ (1:(nx + (t == T ? 0 : nu)))) for t = 1:T]
    x1 = ones(nx) 
    u1 = ones(nu)
    X = [x1 for t = 1:T]
    U = [t < T ? u1 : zeros(0) for t = 1:T]

    ct.val(ct.val_cache, x1, u1)
    ct.grad(ct.grad_cache, x1, u1)
    @test ct.val_cache[1] ≈ ft(x1, u1)
    @test norm(ct.grad_cache - [2.0 * x1; 0.2 * u1]) < 1.0e-8

    cT.val(cT.val_cache, x1, u1)
    cT.grad(cT.grad_cache, x1, u1)
    @test cT.val_cache[1] ≈ fT(x1, u1)
    @test norm(cT.grad_cache - 20.0 * x1) < 1.0e-8

    @test eval_obj(obj, X, U) - sum([ft(X[t], U[t]) for t = 1:T-1]) - fT(X[T], U[T]) ≈ 0.0
    eval_obj_grad!(grad, idx_xu, obj, X, U) 
    @test norm(grad - vcat([[2.0 * x1; 0.2 * u1] for t = 1:T-1]..., 20.0 * x1)) < 1.0e-8

    info = @benchmark eval_obj($obj, $X, $U)
    info = @benchmark eval_obj_grad!($grad, $idx_xu, $obj, $X, $U)
end