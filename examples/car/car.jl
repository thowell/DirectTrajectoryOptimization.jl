# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization 
using LinearAlgebra
using Plots

# ## horizon 
T = 51 

# ## car 
nx = 3
nu = 2
nw = 0 

function car(x, u, w)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function midpoint_implicit(y, x, u, w)
    h = 0.1 # timestep 
    y - (x + h * car(0.5 * (x + y), u, w))
end

# ## model
dt = Dynamics(midpoint_implicit, nx, nx, nu, nw=nw)
dyn = [dt for t = 1:T-1] 
model = DynamicsModel(dyn)

# ## initialization
x1 = [0.0; 0.0; 0.0] 
xT = [1.0; 1.0; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.0 * dot(x - xT, x - xT) + 1.0 * dot(u, u)
oT = (x, u, w) -> 0.0 * dot(x - xT, x - xT)
ct = Cost(ot, nx, nu, nw, [t for t = 1:T-1])
cT = Cost(oT, nx, 0, nw, [T])
obj = [ct, cT]

# ## constraints
ul = -0.5 * ones(nu) 
uu = 0.5 * ones(nu)
bnd1 = Bound(nx, nu, [1], xl=x1, xu=x1, ul=ul, uu=uu)
bndt = Bound(nx, nu, [t for t = 2:T-1], ul=ul, uu=uu)
bndT = Bound(nx, 0, [T], xl=xT, xu=xT)

p_obs = [0.5; 0.5] 
r_obs = 0.1
function obs(x, u, w) 
    e = x[1:2] - p_obs
    return [r_obs^2.0 - dot(e, e)]
end
cont = StageConstraint(obs, nx, nu, nw, [t for t = 1:T-1], :inequality)
conT = StageConstraint(obs, nx, 0, nw, [T], :inequality)
cons = ConstraintSet([bnd1, bndt, bndT], [cont, conT])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, options=Options())

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [0.001 * randn(nu) for t = 1:T-1]
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
plot(hcat(trajopt.x...)[1, :], hcat(trajopt.x...)[2, :], label = "", color = :orange, width=2.0)
pts = Plots.partialcircle(0.0, 2.0 * π, 100, r_obs)
cx, cy = Plots.unzip(pts)
plot!(Shape(cx .+ p_obs[1], cy .+ p_obs[2]), color = :black, label = "", linecolor = :black)

# ## control
plot(hcat(trajopt.u[1:end-1]..., trajopt.u[end-1])', linetype = :steppost)