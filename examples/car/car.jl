# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization 
using LinearAlgebra
using Plots

# ## horizon 
T = 51 

# ## car 
num_state = 3
nu = 2
num_parameter = 0 

function car(x, u, w)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function midpoint_implicit(y, x, u, w)
    h = 0.1 # timestep 
    y - (x + h * car(0.5 * (x + y), u, w))
end

# ## model
dt = Dynamics(midpoint_implicit, num_state, num_state, nu, num_parameter=num_parameter)
dyn = [dt for t = 1:T-1] 

# ## initialization
x1 = [0.0; 0.0; 0.0] 
xT = [1.0; 1.0; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.0 * dot(x - xT, x - xT) + 1.0 * dot(u, u)
oT = (x, u, w) -> 0.0 * dot(x - xT, x - xT)
ct = Cost(ot, num_state, nu, num_parameter)
cT = Cost(oT, num_state, 0, num_parameter)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
action_lower = -0.5 * ones(nu) 
action_upper = 0.5 * ones(nu)
bnd1 = Bound(num_state, nu, state_lower=x1, state_upper=x1, action_lower=action_lower, action_upper=action_upper)
bndt = Bound(num_state, nu, action_lower=action_lower, action_upper=action_upper)
bndT = Bound(num_state, 0, state_lower=xT, state_upper=xT)
bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

p_obs = [0.5; 0.5] 
r_obs = 0.1
function obs(x, u, w) 
    e = x[1:2] - p_obs
    return [r_obs^2.0 - dot(e, e)]
end

cont = Constraint(obs, num_state, nu, num_parameter, indices_inequality=collect(1:1))
conT = Constraint(obs, num_state, 0, num_parameter, indices_inequality=collect(1:1))
cons = [[cont for t = 1:T-1]..., conT]

# ## problem 
p = ProblemData(obj, dyn, cons, bounds, options=Options())

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [0.001 * randn(nu) for t = 1:T-1]

initialize_states!(p, x_interpolation)
initialize_controls!(p, u_guess)

# ## solve
solve!(p)

# ## solution
x_sol, u_sol = get_trajectory(p)

@show x_sol[1]
@show x_sol[T]

# ## state
plot(hcat(x_sol...)[1, :], hcat(x_sol...)[2, :], label = "", color = :orange, width=2.0)
pts = Plots.partialcircle(0.0, 2.0 * π, 100, r_obs)
cx, cy = Plots.unzip(pts)
plot!(Shape(cx .+ p_obs[1], cy .+ p_obs[2]), color = :black, label = "", linecolor = :black)

# ## control
plot(hcat(u_sol[1:end-1]..., u_sol[end-1])', linetype = :steppost)