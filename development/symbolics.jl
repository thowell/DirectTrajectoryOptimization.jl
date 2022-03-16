# dimensions 
num_state = 2#12 
nu = 1#4
num_parameter = 0

# variables 
@variables y[1:num_state], x[1:num_state], u[1:nu], w[1:num_parameter]

y1 = ones(num_state) 
x1 = ones(num_state) 
u1 = ones(nu)
w1 = ones(num_parameter)

xu1 = [x1; u1] 
xuy1 = [x1; u1; y1]

# objective 

function objective(x, u)
    Q = Diagonal(ones(num_state)) 
    q = ones(num_state) 

    R = Diagonal(0.1 * ones(nu)) 
    r = zeros(nu)

    return log(transpose(x) * Q * x) + exp(dot(q, x)) + sin(transpose(u) * R * u)
end

obj = objective(x, u)
obj_x = Symbolics.gradient(obj, x)
obj_u = Symbolics.gradient(obj, u)
obj_xu = Symbolics.gradient(obj, [x; u])

obj_xx = Symbolics.hessian(obj, x)
obj_uu = Symbolics.hessian(obj, u)
obj_xuxu = Symbolics.hessian(obj, [x; u])

obj_xx_sp = Symbolics.sparsehessian(obj, x)
obj_uu_sp = Symbolics.sparsehessian(obj, u)
obj_xuxu_sp = Symbolics.sparsehessian(obj, [x; u])
typeof(obj_func[1])
obj_func = Symbolics.build_function([obj], x, u)
obj_x_func = Symbolics.build_function(obj_x, x, u)
obj_u_func = Symbolics.build_function(obj_u, x, u)
obj_xu_func = Symbolics.build_function(obj_xu, x, u)

obj_xx_func = Symbolics.build_function(obj_xx_sp.nzval, x, u)
obj_uu_func = Symbolics.build_function(obj_uu_sp.nzval, x, u)
obj_xuxu_func = Symbolics.build_function(obj_xuxu_sp.nzval, x, u)

f = eval(obj_func[1])
@benchmark f($x1, $u1)
@benchmark objective($x1, $u1)

fx = eval(obj_x_func[1])
@benchmark fx($x1, $u1)
@benchmark ForwardDiff.gradient(a -> objective(a, $u1), $x1)

fu = eval(obj_u_func[1])
@benchmark fu($x1, $u1)
@benchmark ForwardDiff.gradient(a -> objective($x1, a), $u1)

fxu = eval(obj_xu_func[1])
@benchmark fxu($x1, $u1)

fxx = eval(obj_xx_func[1])
@benchmark fxx($x1, $u1)

fuu = eval(obj_uu_func[1])
@benchmark fuu($x1, $u1)

fxuxu = eval(obj_xuxu_func[1])
@benchmark fxuxu($x1, $u1)

obj_1 = zeros(1)
obj_x1 = zeros(num_state) 
obj_u1 = zeros(nu)
obj_xu1 = zeros(num_state + nu)

obj_xx1 = zeros(length(obj_xx_sp.nzval))
obj_uu1 = zeros(length(obj_uu_sp.nzval))
obj_xuxu1 = zeros(length(obj_xuxu_sp.nzval))

f! = eval(obj_func[2])
@benchmark f!($obj_1, $x1, $u1) 

fx! = eval(obj_x_func[2])
@benchmark fx!($obj_x1, $x1, $u1)
@benchmark ForwardDiff.gradient!($obj_x1, a -> objective(a, $u1), $x1)

fu! = eval(obj_u_func[2])
@benchmark fu!($obj_u1, $x1, $u1)
@benchmark ForwardDiff.gradient!($obj_u1, a -> objective($x1, a), $u1)

fxu! = eval(obj_xu_func[2])
@benchmark fxu!($obj_xu1, $x1, $u1)

fxx! = eval(obj_xx_func[2])
@benchmark fxx!($obj_xx1, $x1, $u1)
# @benchmark ForwardDiff.hessian!($obj_xx1, a -> objective(a, $u1), $x1)

fuu! = eval(obj_uu_func[2])
@benchmark fuu!($obj_uu1, $x1, $u1)
# @benchmark ForwardDiff.hessian!($obj_uu1, a -> objective($x1, a), $u1)

fxuxu! = eval(obj_xuxu_func[2])
@benchmark fxuxu!($obj_xuxu1, $x1, $u1)


# discrete dynamics 
function quaternion_rotation_matrix(q)
	r, i, j, k  = q

	r11 = 1.0 - 2.0 * (j^2.0 + k^2.0)
	r12 = 2.0 * (i * j - k * r)
	r13 = 2.0 * (i * k + j * r)

	r21 = 2.0 * (i * j + k * r)
	r22 = 1.0 - 2.0 * (i^2.0 + k^2.0)
	r23 = 2.0 * (j * k - i * r)

	r31 = 2.0 * (i * k - j * r)
	r32 = 2.0 * (j * k + i * r)
	r33 = 1.0 - 2.0 * (i^2.0 + j^2.0)

	[r11 r12 r13;
     r21 r22 r23;
     r31 r32 r33]
end

function mrp_quaternion_map(mrp)
	n2 = mrp[1] * mrp[1] + mrp[2] * mrp[2] + mrp[3] * mrp[3]
    M = 2 / (1 + n2)
    return [(1 - n2) / (1 + n2), M * mrp[1], M * mrp[2], M * mrp[3]]
end

mrp_rotation_matrix(mrp) = quaternion_rotation_matrix(mrp_quaternion_map(mrp))

function dynamics(z, u, w) 
#     mass = 1.0 
#     lc = 1.0 
#     gravity = 9.81 
#     damping = 0.1

#     [z[2], (u[1] / ((mass * lc * lc)) - gravity * sin(z[1]) / lc - damping * z[2] / (mass * lc * lc))]
# end

    # states
    x = z[1:3]
    r = z[4:6]
    v = z[7:9]
    ω = z[10:12]

    # controls
    w1 = u[1]
    w2 = u[2]
    w3 = u[3]
    w4 = u[4]

    # forces
    kf = 0.1
    F1 = kf * w1
    F2 = kf * w2
    F3 = kf * w3
    F4 = kf * w4

    F = [0.0, 0.0, F1 + F2 + F3 + F4] # total rotor force in body frame

    # moments
    km = 0.01
    M1 = km * w1
    M2 = km * w2
    M3 = km * w3
    M4 = km * w4

    L = 0.25
    τ = [L * (F2 - F4), L * (F3 - F1), (M1 - M2 + M3 - M4)] # total rotor torque in body frame

    gravity = 9.81 
    mass = 1.0 
    inertia = Diagonal(ones(3)) 
    inertia_inv = Diagonal(ones(3))

    [v; 
     0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0*(ω' * r) * r); 
     [0.0, 0.0, -9.81] + (1.0 / mass) * mrp_rotation_matrix(r) * F; 
     inertia_inv * (τ - cross(ω, inertia * ω))]
end


h = 0.1
function discrete_dynamics(y, x, u, w)
    y - (x + h * dynamics(y, u, w))
end

dyn = discrete_dynamics(y, x, u, w)
dyn = simplify.(dyn)

# dense
dyn_y = Symbolics.jacobian(dyn, y)
dyn_x = Symbolics.jacobian(dyn, x)
dyn_u = Symbolics.jacobian(dyn, u)

dyn_xuy = Symbolics.jacobian(dyn, [x; u; y])

dyn_func = Symbolics.build_function(dyn, y, x, u, w)
dyn_y_func = Symbolics.build_function(dyn_y, y, x, u, w)
dyn_x_func = Symbolics.build_function(dyn_x, y, x, u, w)
dyn_u_func = Symbolics.build_function(dyn_u, y, x, u, w)
dyn_xuy_func = Symbolics.build_function(dyn_xuy, y, x, u, w)

# sparse 
dyn_y_sp = Symbolics.sparsejacobian(dyn, y)
dyn_x_sp = Symbolics.sparsejacobian(dyn, x)
dyn_u_sp = Symbolics.sparsejacobian(dyn, u)
dyn_xuy_sp = Symbolics.sparsejacobian(dyn, [x; u; y])

dyn_y_func_sp = Symbolics.build_function(dyn_y_sp.nzval, y, x, u, w)
dyn_x_func_sp = Symbolics.build_function(dyn_x_sp.nzval, y, x, u, w)
dyn_u_func_sp = Symbolics.build_function(dyn_u_sp.nzval, y, x, u, w)
dyn_xuy_func_sp = Symbolics.build_function(dyn_xuy_sp.nzval, y, x, u, w)

d = eval(dyn_func[1])
@benchmark d($y1, $x1, $u1, $w1)
@benchmark discrete_dynamics($y1, $x1, $u1, $w1)

dy = eval(dyn_y_func[1])
@benchmark dy($y1, $x1, $u1, $w1)
@benchmark ForwardDiff.jacobian(a -> discrete_dynamics(a, $x1, $u1, $w1), $y1)
dy_sp = eval(dyn_y_func_sp[1])
@benchmark dy_sp($y1, $x1, $u1, $w1)

dx = eval(dyn_x_func[1])
@benchmark dx($y1, $x1, $u1, $w1)
@benchmark ForwardDiff.jacobian(a -> discrete_dynamics($y1, a, $u1, $w1), $x1)
dx_sp = eval(dyn_x_func_sp[1])
@benchmark dx_sp($y1, $x1, $u1, $w1)

du = eval(dyn_u_func[1])
@benchmark du($y1, $x1, $u1, $w1)
@benchmark ForwardDiff.jacobian(a -> discrete_dynamics($y1, $x1, a, $w1), $u1)
du_sp = eval(dyn_u_func_sp[1])
@benchmark du_sp($y1, $x1, $u1, $w1)

dxuy = eval(dyn_xuy_func[1])
@benchmark dxuy($y1, $x1, $u1, $w1)
dxuy_sp = eval(dyn_xuy_func_sp[1])
@benchmark dxuy_sp($y1, $x1, $u1, $w1)

dyn_1 = zeros(num_state)
dyn_y1 = zeros(num_state, num_state)
dyn_x1 = zeros(num_state, num_state) 
dyn_u1 = zeros(num_state, nu)
dyn_xuy1 = zeros(num_state, num_state + num_state + nu)

dyn_y1_sp = zeros(length(dyn_y_sp.nzval))
dyn_x1_sp = zeros(length(dyn_x_sp.nzval)) 
dyn_u1_sp = zeros(length(dyn_u_sp.nzval))
dyn_xuy1_sp = zeros(length(dyn_xuy_sp.nzval))

d! = eval(dyn_func[2])
@benchmark d!($dyn_1, $y1, $x1, $u1, $w1)

dy! = eval(dyn_y_func[2])
@benchmark dy!($dyn_y1, $y1, $x1, $u1, $w1)
@benchmark ForwardDiff.jacobian!($dyn_y1, a -> discrete_dynamics(a, $x1, $u1, $w1), $y1)
dy_sp! = eval(dyn_y_func_sp[2])
@benchmark dy_sp!($dyn_y1_sp, $y1, $x1, $u1, $w1)

dx! = eval(dyn_x_func[2])
@benchmark dx!($dyn_x1, $y1, $x1, $u1, $w1)
@benchmark ForwardDiff.jacobian!($dyn_x1, a -> discrete_dynamics($y1, a, $u1, $w1), $x1)
dx_sp! = eval(dyn_x_func_sp[2])
@benchmark dx_sp!($dyn_x1_sp, $y1, $x1, $u1, $w1)

du! = eval(dyn_u_func[2])
@benchmark du!($dyn_u1, $y1, $x1, $u1, $w1)
@benchmark ForwardDiff.jacobian!($dyn_u1, a -> discrete_dynamics($y1, $x1, a, $w1), $u1)
du_sp! = eval(dyn_u_func_sp[2])
@benchmark du_sp!($dyn_u1_sp, $y1, $x1, $u1, $w1)

dxuy! = eval(dyn_xuy_func[2])
@benchmark dxuy!($dyn_xuy1, $y1, $x1, $u1, $w1)
dxuy_sp! = eval(dyn_xuy_func_sp[2])
@benchmark dxuy_sp!($dyn_xuy1_sp, $y1, $x1, $u1, $w1)
dxuy_sp!(dyn_xuy1_sp, y1, x1, u1, w1)
dyn_xuy1_sp

@variables λ[1:num_state]
dyn_λ = dot(λ, dyn)
dyn_λ_xuy = Symbolics.hessian(dyn_λ, [x; u; y])
dyn_λ_xuy_sp = Symbolics.sparsehessian(dyn_λ, [x; u; y])

dyn_λ_xuy_func = Symbolics.build_function(dyn_λ_xuy, y, x, u, w, λ)
dyn_λ_xuy_func_sp = Symbolics.build_function(dyn_λ_xuy_sp.nzval, y, x, u, w, λ)

λ1 = ones(num_state) 
dyn_λ_xuy1 = zeros(num_state + num_state + nu, num_state + num_state + nu)
dyn_λ_xuy1_sp = zeros(length(dyn_λ_xuy_sp.nzval))

dλ_xuy! = eval(dyn_λ_xuy_func[2])
@benchmark dλ_xuy!($dyn_λ_xuy1, $y1, $x1, $u1, $w1, $λ1)

dλ_xuy_sp! = eval(dyn_λ_xuy_func_sp[2])
@benchmark dλ_xuy_sp!($dyn_λ_xuy1_sp, $y1, $x1, $u1, $w1, $λ1)

# state constraint
function state_constraint(x) 
    x - ones(num_state)
end 

# control constraint
function control_constraint(u) 
    [u - ones(nu); ones(nu) - u]
end

# state control constraint
function state_control_constraint(x, u) 

end


# indices 
T = 10
state_dimensions = [num_state for t = 1:T]
action_dimensions = [nu for t = 1:T-1] 
num_parameter_dim = [num_parameter for t = 1:T-1]
state_action_dimensions = [[num_state + nu for t = 1:T-1]..., num_state]

idx_x = [collect((t > 1 ? sum(state_dimensions[1:t-1]) + sum(action_dimensions[1:t-1]) : 0) .+ (1:state_dimensions[t])) for t = 1:T] 
idx_u = [collect((t > 1 ? sum(state_dimensions[1:t-1]) + sum(action_dimensions[1:t-1]) : 0) + state_dimensions[t] .+ (1:action_dimensions[t])) for t = 1:T-1]
idx_xu = [collect((t > 1 ? sum(state_dimensions[1:t-1]) + sum(action_dimensions[1:t-1]) : 0) .+ (1:state_dimensions[t] + action_dimensions[t])) for t = 1:T-1] 
idx_xuy = [collect((t > 1 ? sum(state_dimensions[1:t-1]) + sum(action_dimensions[1:t-1]) : 0) .+ (1:state_dimensions[t] + action_dimensions[t] + state_dimensions[t+1])) for t = 1:T-1] 

typeof(idx_x)
num_variables = sum(state_dimensions) + sum(action_dimensions)
z1 = randn(num_variables)

# objective
obj_grad = zeros(num_variables)

xv = [z1[idx_x[t]] for t = 1:T]
uv = [[z1[idx_u[t]] for t = 1:T-1]..., zeros(0)]
wv = [zeros(num_parameter) for t = 1:T-1]

objv = [0.0]
gradv = [[obj_grad[idx_xu[t]] for t = 1:T-1]..., obj_grad[idx_x[T]]]

function eval_obj!(J, x, u, T) 
    for t = 1:T 
        f!(J, x[t], u[t])
    end
    return J 
end

eval_obj!(objv, xv, uv, T)
@benchmark eval_obj!($objv, $xv, $uv, $T)

function gradient!(grad, x, u, T)
    for t = 1:T
        fxu!(grad[t], x[t], u[t])
    end
end

gradient!(gradv, xv, uv, T)
@benchmark gradient!($gradv, $xv, $uv, $T)

# discrete dynamics 
nd_dim = state_dimensions[2:T]
num_constraint = sum(state_dimensions[2:T])
idx_dyn = [(t > 1 ? sum(state_dimensions[1 .+ (1:t-1)]) : 0) .+ (1:state_dimensions[t+1]) for t = 1:T-1]
con1 = zeros(num_constraint)
conv = [con1[idx_dyn[t]] for t = 1:T-1]

function constraints!(con, x, u, w, T)
    for t = 1:T-1
        d!(con[t], x[t+1], x[t], u[t], w[t])
    end
end

d!(conv[t], xv[t+1], xv[t], uv[t], wv[t])
constraints!(conv, xv, uv, wv, T)
@benchmark constraints!($conv, $xv, $uv, $wv, $T)

function constraints!(con, x, u, w, T)
    for t = 1:T-1
        d!(con[t], x[t+1], x[t], u[t], w[t])
    end
end

num_jacobian = length(dyn_xuy_sp.nzval) * (T - 1)
idx_jac = [(t - 1) * length(dyn_xuy_sp.nzval) .+ (1:length(dyn_xuy_sp.nzval)) for t = 1:T-1]
jac1 = zeros(num_jacobian)
jacv = [jac1[idx_jac[t]] for t = 1:T-1]

function jacobian!(jac, x, u, w, T)
    for t = 1:T-1
        dxuy_sp!(jac[t], x[t+1], x[t], u[t], w[t])
    end
end

dxuy_sp!(jacv[t], xv[t+1], xv[t], uv[t], wv[t])
jacobian!(jacv, xv, uv, wv, T)
@benchmark jacobian!($jacv, $xv, $uv, $wv, $T)

I_jac, J_jac, V_jac = findnz(dyn_xuy_sp)

function sparsity_jac(jac, nd_dim, state_action_dimensions, T;
    row_shift = 0, col_shift = 0)

    row = Int[]
    col = Int[]

    I, J, V = findnz(jac)

    for t = 1:T-1
        push!(row, (I .+ row_shift)...) 
        push!(col, (J .+ col_shift)...) 
        
        row_shift += nd_dim[t]
        col_shift += state_action_dimensions[t]
    end

    return collect(zip(row, col))
end

sparsity_jac(dyn_xuy_sp, nd_dim, state_action_dimensions, T)


