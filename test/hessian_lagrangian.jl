@testset "Hessian of Lagrangian" begin
    MOI = DTO.MOI
    # horizon 
    T = 3

    # acrobot 
    num_state = 4 
    nu = 1 
    nw = 0 
    w_dim = [nw for t = 1:T]

    function acrobot(x, u, w)
        # dimensions
        n = 4
        m = 1
        d = 0

        # link 1
        mass1 = 1.0  
        inertia1 = 0.33  
        length1 = 1.0 
        lengthcom1 = 0.5 

        # link 2
        mass2 = 1.0  
        inertia2 = 0.33  
        length2 = 1.0 
        lengthcom2 = 0.5 

        gravity = 9.81 
        friction1 = 0.1 
        friction2 = 0.1

        # mass matrix
        function M(x, w)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

        return [a b; b c]
        end

        # dynamics bias
        function τ(x, w)
            a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
                - mass2 * gravity * (length1 * sin(x[1])
                + lengthcom2 * sin(x[1] + x[2])))

            b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

            return [a; b]
        end

        function C(x, w)
            a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
            d = 0.0

            return [a b; c d]
        end

        # input Jacobian
        function B(x, w)
            [0.0; 1.0]
        end

        # dynamics
        q = view(x, 1:2)
        v = view(x, 3:4)

        qdd = M(q, w) \ (-1.0 * C(x, w) * v
                + τ(q, w) + B(q, w) * u[1] - [friction1; friction2] .* v)

        return [x[3]; x[4]; qdd[1]; qdd[2]]
    end

    function midpoint_implicit(y, x, u, w)
        h = 0.05 # timestep 
        y - (x + h * acrobot(0.5 * (x + y), u, w))
    end

    dt = Dynamics(midpoint_implicit, num_state, num_state, nu, nw=nw, eval_hess=true)
    dyn = [dt for t = 1:T-1] 

    # initial state 
    x1 = [0.0; 0.0; 0.0; 0.0] 

    # goal state
    xT = [0.0; π; 0.0; 0.0] 

    # objective 
    ot = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4])
    objt = Cost(ot, num_state, nu, nw, eval_hess=true)
    objT = Cost(oT, num_state, 0, nw, eval_hess=true)
    obj = [[objt for t = 1:T-1]..., objT]

    # constraints
    bnd1 = Bound(num_state, nu, state_lower=x1, xu=x1)
    bndt = Bound(num_state, nu)
    bndT = Bound(num_state, 0, state_lower=xT, xu=xT)
    bnds = [bnd1, [bndt for t = 2:T-1]..., bndT]

    ct = (x, u, w) -> [-5.0 * ones(nu) - cos.(u) .* sum(x.^2); cos.(x) .* tan.(u) - 5.0 * ones(num_state)]
    cT = (x, u, w) -> sin.(x.^3.0)
    cont = Constraint(ct, num_state, nu, nw, idx_ineq=collect(1:(nu + num_state)), eval_hess=true)
    conT = Constraint(cT, num_state, 0, nw, eval_hess=true)
    cons = [[cont for t = 1:T-1]..., conT]

    # data 
    # trajopt = DTO.TrajectoryOptimizationData(obj, dyn, cons, bnds)
    # nlp = DTO.NLPData(trajopt, eval_hess=true)
    p = ProblemData(obj, dyn, cons, bnds, eval_hess=true)

    # Lagrangian
    function lagrangian(z) 
        x1 = z[1:num_state] 
        u1 = z[num_state .+ (1:nu)] 
        x2 = z[num_state + nu .+ (1:num_state)] 
        u2 = z[num_state + nu + num_state .+ (1:nu)] 
        x3 = z[num_state + nu + num_state + nu .+ (1:num_state)]
        λ1_dyn = z[num_state + nu + num_state + nu + num_state .+ (1:num_state)] 
        λ2_dyn = z[num_state + nu + num_state + nu + num_state + num_state .+ (1:num_state)] 

        λ1_stage = z[num_state + nu + num_state + nu + num_state + num_state + num_state .+ (1:(nu + num_state))] 
        λ2_stage = z[num_state + nu + num_state + nu + num_state + num_state + num_state + nu + num_state .+ (1:(nu + num_state))] 
        λ3_stage = z[num_state + nu + num_state + nu + num_state + num_state + num_state + nu + num_state + nu + num_state .+ (1:num_state)]

        L = 0.0 
        L += ot(x1, u1, zeros(nw)) 
        L += ot(x2, u2, zeros(nw)) 
        L += oT(x3, zeros(0), zeros(nw))
        L += dot(λ1_dyn, midpoint_implicit(x2, x1, u1, zeros(nw))) 
        L += dot(λ2_dyn, midpoint_implicit(x3, x2, u2, zeros(nw))) 
        L += dot(λ1_stage, ct(x1, u1, zeros(nw)))
        L += dot(λ2_stage, ct(x2, u2, zeros(nw)))
        L += dot(λ3_stage, cT(x3, zeros(0), zeros(nw)))
        return L
    end

    nz = num_state + nu + num_state + nu + num_state + num_state + num_state + nu + num_state + nu + num_state + num_state
    np = num_state + nu + num_state + nu + num_state
    nd = num_state + num_state + nu + num_state + nu + num_state + num_state
    @variables z[1:nz]
    L = lagrangian(z)
    Lxx = Symbolics.hessian(L, z[1:np])
    Lxx_sp = Symbolics.sparsehessian(L, z[1:np])
    spar = [findnz(Lxx_sp)[1:2]...]
    Lxx_func = eval(Symbolics.build_function(Lxx, z)[1])
    Lxx_sp_func = eval(Symbolics.build_function(Lxx_sp.nzval, z)[1])

    z0 = rand(nz)
    nh = length(p.nlp.sp_hess_lag)
    h0 = zeros(nh)

    σ = 1.0
    fill!(h0, 0.0)
    DTO.trajectory!(p.nlp.trajopt.x, p.nlp.trajopt.u, z0[1:np], 
        p.nlp.idx.x, p.nlp.idx.u)
    DTO.duals!(p.nlp.trajopt.λ_dyn, p.nlp.trajopt.λ_stage, p.nlp.λ_gen, z0[np .+ (1:nd)], p.nlp.idx.dyn_con, p.nlp.idx.stage_con, p.nlp.idx.gen_con)
    DTO.eval_obj_hess!(h0, p.nlp.idx.obj_hess, p.nlp.trajopt.obj, p.nlp.trajopt.x, p.nlp.trajopt.u, p.nlp.trajopt.w, σ)
    DTO.eval_hess_lag!(h0, p.nlp.idx.dyn_hess, p.nlp.trajopt.dyn, p.nlp.trajopt.x, p.nlp.trajopt.u, p.nlp.trajopt.w, p.nlp.trajopt.λ_dyn)
    DTO.eval_hess_lag!(h0, p.nlp.idx.stage_hess, p.nlp.trajopt.cons, p.nlp.trajopt.x, p.nlp.trajopt.u, p.nlp.trajopt.w, p.nlp.trajopt.λ_stage)

    sp_obj_hess = DTO.sparsity_hessian(obj, p.nlp.trajopt.x_dim, p.nlp.trajopt.u_dim)
    sp_dyn_hess = DTO.sparsity_hessian(dyn, p.nlp.trajopt.x_dim, p.nlp.trajopt.u_dim)
    sp_con_hess = DTO.sparsity_hessian(cons, p.nlp.trajopt.x_dim, p.nlp.trajopt.u_dim)
    sp_hess = collect([sp_obj_hess..., sp_dyn_hess..., sp_con_hess...]) 
    sp_key = sort(unique(sp_hess))

    idx_obj_hess = DTO.hessian_indices(obj, sp_key, p.nlp.trajopt.x_dim, p.nlp.trajopt.u_dim)
    idx_dyn_hess = DTO.hessian_indices(dyn, sp_key, p.nlp.trajopt.x_dim, p.nlp.trajopt.u_dim)
    idx_con_hess = DTO.hessian_indices(cons, sp_key, p.nlp.trajopt.x_dim, p.nlp.trajopt.u_dim)

    # indices
    @test sp_key[vcat(idx_obj_hess...)] == sp_obj_hess
    @test sp_key[vcat(idx_dyn_hess...)] == sp_dyn_hess
    @test sp_key[vcat(idx_con_hess...)] == sp_con_hess

    # Hessian
    h0_full = zeros(np, np)
    for (i, h) in enumerate(h0)
        h0_full[sp_key[i][1], sp_key[i][2]] = h
    end
    @test norm(h0_full - Lxx_func(z0)) < 1.0e-8
    @test norm(norm(h0 - Lxx_sp_func(z0))) < 1.0e-8

    h0 = zeros(nh)
    MOI.eval_hessian_lagrangian(p.nlp, h0, z0[1:np], σ, z0[np .+ (1:nd)])
    @test norm(norm(h0 - Lxx_sp_func(z0))) < 1.0e-8

    # a = z0[1:np]
    # b = z0[np .+ (1:nd)]
    # info = @benchmark MOI.eval_hessian_lagrangian($p, $h0, $a, $σ, $b)
end
