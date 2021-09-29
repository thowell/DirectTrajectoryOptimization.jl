using DirectTrajectoryOptimization 
using LinearAlgebra

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

dt = Dynamics(midpoint_implicit, nx, nx, nu, nw=nw);
dyn = [dt for t = 1:T-1] 
model = DynamicsModel(dyn)

# initial state 
x1 = [0.0; 0.0; 0.0; 0.0] 

# goal state
xT = [0.0; π; 0.0; 0.0] 

# interpolation 
x_interpolation = linear_interpolation(x1, xT, T)

# objective 
ft = (x, u) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
fT = (x, u) -> 0.1 * dot(x[3:4], x[3:4])
ct = Cost(ft, nx, nu, [t for t = 1:T-1])
cT = Cost(fT, nx, 0, [T])
obj = [ct, cT]

# constraints
x_init = Bound(nx, nu, [1], xl=x1, xu=x1)
x_goal = Bound(nx, 0, [T], xl=xT, xu=xT)
cons = ConstraintSet([x_init, x_goal], [StageConstraint()])

# problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt)

# initialize
z0 = 0.001 * randn(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = x_interpolation[t]
end
initialize!(s, z0)

# solve
solve!(s)

# solution
@show trajopt.x[1]
@show trajopt.x[T]

using Plots
plot(hcat(trajopt.x...)')
plot(hcat(trajopt.u[1:end-1]..., trajopt.u[end-1])', linetype = :steppost)