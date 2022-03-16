# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization 
using LinearAlgebra
using Plots

# ## horizon 
T = 11 

# ## double integrator 
num_state = 2
nu = 1 
nw = 0 
nz = num_state + nu + num_state

# ## dynamics
function double_integrator(d, y, x, u, w)
    A = [1.0 1.0; 0.0 1.0] 
    B = [0.0; 1.0] 
    d .= y - (A * x + B * u[1])
end

# ## user-provided dynamics gradient
function double_integrator_grad(dz, y, x, u, w) 
    A = [1.0 1.0; 0.0 1.0] 
    B = [0.0; 1.0] 
    dz .= [-A -B I]
end

# ## fast methods
function double_integrator(y, x, u, w)
    A = [1.0 1.0; 0.0 1.0] 
    B = [0.0; 1.0] 
    y - (A * x + B * u[1])
end

function double_integrator_grad(y, x, u, w) 
    A = [1.0 1.0; 0.0 1.0] 
    B = [0.0; 1.0] 
    [-A -B I]
end

@variables y[1:num_state] x[1:num_state] u[1:nu] w[1:nw]

di = double_integrator(y, x, u, w) 
diz = double_integrator_grad(y, x, u, w) 
di_func = eval(Symbolics.build_function(di, y, x, u, w)[2])
diz_func = eval(Symbolics.build_function(diz, y, x, u, w)[2])

# ## model
dt = Dynamics(di_func, diz_func, num_state, num_state, nu)
dyn = [dt for t = 1:T-1] 
model = DynamicsModel(dyn)

# ## initialization
x1 = [0.0; 0.0] 
xT = [1.0; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.1 * dot(x, x) + 0.1 * dot(u, u)
oT = (x, u, w) -> 0.1 * dot(x, x)
ct = Cost(ot, num_state, nu, nw, [t for t = 1:T-1])
cT = Cost(oT, num_state, 0, nw, [T])
obj = [ct, cT]

# ## constraints
x_init = Bound(num_state, nu, [1], state_lower=x1, xu=x1)
x_goal = Bound(num_state, 0, [T], state_lower=xT, xu=xT)
cons = ConstraintSet([x_init, x_goal], [StageConstraint()])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, options=Options())

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [1.0 * randn(nu) for t = 1:T-1]
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = x_interpolation[t]
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
initialize!(s, z0)

# ## solve
@time solve!(s)

# ## solution
@show trajopt.x[1]
@show trajopt.x[T]

# ## state
plot(hcat(trajopt.x...)')

# ## control
plot(hcat(trajopt.u[1:end-1]..., trajopt.u[end-1])', linetype = :steppost)