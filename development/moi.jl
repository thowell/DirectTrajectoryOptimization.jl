using InteractiveUtils

# test 
T = 26
nx = 2
nu = 1 
nw = 0
ft = (x, u) -> 1.0 * dot(x, x) + 0.1 * dot(u, u)
fT = (x, u) -> 1.0 * dot(x, x)
ct = Cost(ft, nx, nu, [t for t = 1:T-1])
cT = Cost(fT, nx, 0, [T])
obj = [ct, cT]

grad = zeros((T - 1) * (nx + nu) + nx)

function dynamics(z, u, w) 
    mass = 1.0 
    lc = 1.0 
    gravity = 9.81 
    damping = 0.1

    [z[2], (u[1] / ((mass * lc * lc)) - gravity * sin(z[1]) / lc - damping * z[2] / (mass * lc * lc))]
end

h = 0.1
function discrete_dynamics(y, x, u, w)
    y - (x + h * dynamics(y, u, w))
end

dt = Dynamics(discrete_dynamics, nx, nx, nu, nw);
dyn = [dt for t = 1:T-1] 
model = DynamicsModel(dyn)

nz = sum([t < T ? dyn[t].nx : dyn[t-1].ny for t = 1:T]) + sum([dyn[t].nu for t = 1:T-1])
nc = num_con(dyn) 
nj = num_jac(dyn)
z = rand(nz)
c = zeros(nc)
j = zeros(nj)
x = [zero(z[model.idx.x[t]]) for t = 1:T]
u = [[zero(z[model.idx.u[t]]) for t = 1:T-1]..., zeros(0)]
w = [zeros(nw) for t = 1:T-1]

con = StageConstraint()
cons = [con for t = 1:T]
cons isa StageConstraints
model isa DynamicsModel
obj isa Objective

trajopt = TrajectoryOptimizationProblem(obj, model, cons, x, u, w)

p = Problem(trajopt, nz, nc, 
    [-Inf * ones(nz), ones(nz)], [zeros(nc), zeros(nc)], 
    sparsity(dyn, model.dim.x, model.dim.u), [], false)

@code_warntype MOI.eval_objective(p, z)
@benchmark MOI.eval_objective($p, $z)
@benchmark MOI.eval_objective_gradient($p, $grad, $z)

@code_warntype MOI.eval_constraint(p, c, z)
@benchmark MOI.eval_constraint($p, $c, $z)

@code_warntype MOI.eval_constraint_jacobian(p, j, z)
@benchmark MOI.eval_constraint_jacobian($p, $j, $z)

sparsity_jacobian(p)
sparsity_hessian_lagrangian(p)

z0 = 0.1 * randn(p.num_var)
MOI.eval_constraint(p, c, z0)

function obje(z) 
    sum([ft(z[p.trajopt.model.idx.x[t]], z[p.trajopt.model.idx.u[t]]) for t = 1:T-1]) + fT(z[p.trajopt.model.idx.x[T]], zeros(0))
end
norm(obje(z0) - MOI.eval_objective(p, z0)) < 1.0e-8 
grad .= 0.0
MOI.eval_objective_gradient(p, grad, z0)
norm(ForwardDiff.gradient(obje, z0) - grad)

function dyn_con(z) 
    vcat([discrete_dynamics(z[p.trajopt.model.idx.x[t+1]], z[p.trajopt.model.idx.x[t]], z[p.trajopt.model.idx.u[t]], p.trajopt.w) for t = 1:T-1]...)
end
norm(c - dyn_con(z0)) < 1.0e-8 

dyn_con_jac = zeros(num_con(p.trajopt.model.dyn), num_xuy(p.trajopt.model.dyn))
MOI.eval_constraint_jacobian(p, j, z0)
sp = sparsity(p.trajopt.model.dyn, p.trajopt.model.dim.x, p.trajopt.model.dim.u)
for (i, v) in enumerate(sp) 
    dyn_con_jac[v...] = j[i] 
end
norm(ForwardDiff.jacobian(dyn_con, z0) - dyn_con_jac) < 1.0e-8

zl = -Inf * ones(p.num_var)
zu = Inf * ones(p.num_var)
cl = zeros(p.num_con) 
cu = zeros(p.num_con)

x1 = [0.0; 0.0]
xT = [π; 0.0] 
x_interp = linear_interpolation(x1, xT, T)
for (t, idx) in enumerate(p.trajopt.model.idx.x)
    z0[idx] = x_interp[t]
end

zl[p.trajopt.model.idx.x[1]] = x1 
zu[p.trajopt.model.idx.x[1]] = x1
zl[p.trajopt.model.idx.x[T]] = xT 
zu[p.trajopt.model.idx.x[T]] = xT

nlp_bounds = MOI.NLPBoundsPair.(cl, cu)
block_data = MOI.NLPBlockData(nlp_bounds, p, true)

solver = Ipopt.Optimizer()
solver.options["max_iter"] = 100
solver.options["tol"] = 1.0e-3
solver.options["constr_viol_tol"] = 1.0e-3
# solver.options["print_level"] = mapl
# solver.options["linear_solver"] = "ma57"

z = MOI.add_variables(solver, p.num_var)

for i = 1:p.num_var
    MOI.add_constraint(solver, z[i], MOI.LessThan(zu[i]))
    MOI.add_constraint(solver, z[i], MOI.GreaterThan(zl[i]))
    MOI.set(solver, MOI.VariablePrimalStart(), z[i], z0[i])
end

# Solve the problem
MOI.set(solver, MOI.NLPBlock(), block_data)
MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)

@time MOI.optimize!(solver)
