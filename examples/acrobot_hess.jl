using DirectTrajectoryOptimization 
using LinearAlgebra
using BenchmarkTools

# horizon 
T = 101 

# acrobot 
nx = 4 
nu = 1 
nw = 0 

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

dt = Dynamics(midpoint_implicit, nx, nx, nu, nw=nw, eval_hess=true)
dt.hess_cache
length(dt.sp_hess[2])
dyn = [dt for t = 1:T-1] 
model = DynamicsModel(dyn)

# initial state 
x1 = [0.0; 0.0; 0.0; 0.0] 

# goal state
xT = [0.0; π; 0.0; 0.0] 

# objective 
ft = (x, u) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
fT = (x, u) -> 0.1 * dot(x[3:4], x[3:4])
ct = Cost(ft, nx, nu, [t for t = 1:T-1], eval_hess=true)
cT = Cost(fT, nx, 0, [T], eval_hess=true)
obj = [ct, cT]

# constraints
x_init = Bound(nx, nu, [1], xl=x1, xu=x1)
x_goal = Bound(nx, 0, [T], xl=xT, xu=xT)
ct = (x, u) -> [-5.0 * ones(nu) - u; u - 5.0 * ones(nu)]
# cont = StageConstraint(ct, nx, nu, [t for t = 1:T-1], :inequality, eval_hess=true)
cont = StageConstraint()
cons = ConstraintSet([x_init, x_goal], [cont])

# hessian sparsity 
sp_obj_hess = sparsity_hessian(obj, model.dim.x, model.dim.u)
sp_dyn_hess = sparsity_hessian(dyn, model.dim.x, model.dim.u)
sp_con_hess = sparsity_hessian(cons.stage, model.dim.x, model.dim.u)
sp_hess = collect([sp_obj_hess..., sp_dyn_hess..., sp_con_hess...]) 
sp_key = sort(unique(sp_hess))
nh = length(sp_key)
h0 = zeros(nh)

idx_obj_hess = hessian_indices(obj, sp_key, model.dim.x, model.dim.u)
idx_dyn_hess = hessian_indices(dyn, sp_key, model.dim.x, model.dim.u)
idx_con_hess = hessian_indices(cons.stage, sp_key, model.dim.x, model.dim.u)

# problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, eval_hess=true)

# initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [0.01 * ones(nu) for t = 1:T-1]
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = x_interpolation[t]
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
initialize!(s, z0)

nh = length(s.p.sp_hess_lag)
h0 = zeros(nh)
λ = zeros(num_con(s.p.trajopt.model.dyn) + num_con(s.p.trajopt.con.stage))
@benchmark duals!($s.p.trajopt.λ_dyn, $s.p.trajopt.λ_stage, $λ, $s.p.idx.dyn_con, $s.p.idx.stage_con)

σ = 1.0
@benchmark eval_obj_hess!($h0, $s.p.idx.obj_hess, $s.p.trajopt.obj, $s.p.trajopt.x, $s.p.trajopt.u, $σ)
@benchmark eval_hess_lag!($h0, $s.p.idx.dyn_hess, $s.p.trajopt.model.dyn, $s.p.trajopt.x, $s.p.trajopt.u, $s.p.trajopt.w, $s.p.trajopt.λ_dyn)
@benchmark eval_hess_lag!($h0, $s.p.idx.stage_hess, $s.p.trajopt.con.stage, $s.p.trajopt.x, $s.p.trajopt.u, $s.p.trajopt.λ_stage)

# solve
@time solve!(s)
# @benchmark solve!($s)

# solution
@show trajopt.x[1]
@show trajopt.x[T]

using Plots
plot(hcat(trajopt.x...)')
plot(hcat(trajopt.u[1:end-1]..., trajopt.u[end-1])', linetype = :steppost)
