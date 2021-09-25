# dimensions
T = 26
nx = 2
nu = 1 
nw = 0

# objective
ft = (x, u) -> dot(x, x) + 1.0e-1 * dot(u, u)
fT = (x, u) -> 1.0 * dot(x, x)
ct = Cost(ft, nx, nu, [t for t = 1:T-1])
cT = Cost(fT, nx, 0, [T])
obj = [ct, cT]

# model
function dynamics(z, u, w) 
    mass = 1.0 
    lc = 0.5
    gravity = 9.81 
    damping = 0.1
    [z[2], (u[1] / (mass * lc * lc) - gravity * sin(z[1]) / lc - damping * z[2] / (mass * lc * lc))]
end

h = 0.1
function discrete_dynamics(y, x, u, w)
    y - (x + h * dynamics(y, u, w))
end

dt = Dynamics(discrete_dynamics, nx, nx, nu, nw=nw);
dyn = [dt for t = 1:T-1] 
model = DynamicsModel(dyn)

# constraints
x1 = [0.0; 0.0]
xT = [π; 0.0] 
x_init = Bound(nx, nu, [1], xl=x1, xu=x1)
x_goal = Bound(nx, 0, [T], xl=xT, xu=xT)
cons = ConstraintSet([x_init, x_goal], [StageConstraint()])

# problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt)

z = rand(s.p.num_var)
g = zeros(s.p.num_var)
c = zeros(s.p.num_con)
j = zeros(s.p.num_jac)

# MOI methods
@code_warntype MOI.eval_objective(s.p, z)
@benchmark MOI.eval_objective($s.p, $z)
@benchmark MOI.eval_objective_gradient($s.p, $g, $z)

@code_warntype MOI.eval_constraint(s.p, c, z)
@benchmark MOI.eval_constraint($s.p, $c, $z)

@code_warntype MOI.eval_constraint_jacobian(s.p, j, z)
@benchmark MOI.eval_constraint_jacobian($s.p, $j, $z)

# initialize
z0 = 0.0 * rand(s.p.num_var)
x_interp = linear_interpolation(x1, xT, T)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = x_interp[t]
end

initialize!(s, z0)
solve!(s)