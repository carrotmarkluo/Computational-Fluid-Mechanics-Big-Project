import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

grid_data = np.load("grid1.npz")
X, Y = grid_data['X'], grid_data['Y']
size0 = X.shape
Mx, My = size0[0] - 1, size0[1] - 1
dt = 0.0005
dx = [1, 1]
gamma = 1.4
M = 0.4
alpha = 1.25 / 180 * np.pi
rho = np.ones(size0)
p = np.ones(size0)
a = np.sqrt(1.4)
u = a * M * np.cos(alpha) * np.ones(size0)
v = a * M * np.sin(alpha) * np.ones(size0)
state0 = np.stack((rho, u, v, p), axis=-1)
t0 = 0
tf = 2
timeplot = np.arange(t0, tf + 50 * dt, 50 * dt)

# # 检查初始流场
# plt.figure()
# plt.quiver(X, Y, u, v)
# plt.title("Initial Velocity Field")
# plt.show()

# Coordinate transformation
def Jacobi(X, Y):
    Max_i, Max_j = X.shape
    Mx = Max_i - 1
    My = Max_j - 1

    Xw = np.hstack((X[:, 1:My+1], X[:, My:My+1]))
    Xx = np.hstack((X[:, 0:1], X[:, 0:My]))
    Xa = np.vstack((X[Mx:Mx+1, :], X[:Mx, :]))
    Xd = np.vstack((X[1:Mx+1, :], X[1:2, :]))
    X_xi = (Xd - Xa) / 2
    X_eta = (Xw - Xx) / 2
    X_eta[:, [0, -1]] *= 2
    X_xixi = Xd - 2 * X + Xa
    X_etaeta = Xw - 2 * X + Xx
    X_etaeta[:, [0, -1]] = X_etaeta[:, [1, -2]]
    X_xieta = (X[1:Mx+1, 1:My+1] - X[Mx-1:Mx, 1:My+1] - X[1:Mx+1, 0:My] + X[Mx-1:Mx, 0:My]) / 4
    X_xieta = np.hstack((X_xieta[:, 0:1] * 2, X_xieta, X_xieta[:, -1:] * 2))

    Yw = np.hstack((Y[:, 1:My+1], Y[:, My:My+1]))
    Yx = np.hstack((Y[:, 0:1], Y[:, 0:My]))
    Ya = np.vstack((Y[Mx:Mx+1, :], Y[:Mx, :]))
    Yd = np.vstack((Y[1:Mx+1, :], Y[1:2, :]))
    Y_xi = (Yd - Ya) / 2
    Y_eta = (Yw - Yx) / 2
    Y_eta[:, [0, -1]] *= 2
    Y_xixi = Yd - 2 * Y + Ya
    Y_etaeta = Yw - 2 * Y + Yx
    Y_etaeta[:, [0, -1]] = Y_etaeta[:, [1, -2]]
    Y_xieta = (Y[1:Mx+1, 1:My+1] - Y[Mx-1:Mx, 1:My+1] - Y[1:Mx+1, 0:My] + Y[Mx-1:Mx, 0:My]) / 4
    Y_xieta = np.hstack((Y_xieta[:, 0:1] * 2, Y_xieta, Y_xieta[:, -1:] * 2))

    Jac = X_xi * Y_eta - X_eta * Y_xi
    return X_xi, X_eta, X_xixi, X_etaeta, X_xieta, Y_xi, Y_eta, Y_xixi, Y_etaeta, Y_xieta, Jac

def di(F, index):
    if index == '-1,0':
        return np.roll(F, 1, axis=0)
    elif index == '1,0':
        return np.roll(F, -1, axis=0)
    elif index == '0,1':
        F_ = np.roll(F, -1, axis=1)
        F_[:, -1, :, :] = 2 * F[:, -1, :, :] - F[:, -2, :, :]
        return F_
    elif index == '0,-1':
        F_ = np.roll(F, 1, axis=1)
        F_[:, 0, :, :] = 2 * F[:, 0, :, :] - F[:, 1, :, :]
        return F_
    elif index == '1/2,0':
        return 0.5 * (F + di(F, '1,0'))
    elif index == '-1/2,0':
        return 0.5 * (F + di(F, '-1,0'))
    elif index == '0,1/2':
        return 0.5 * (F + di(F, '0,1'))
    elif index == '0,-1/2':
        return 0.5 * (F + di(F, '0,-1'))
    elif index == '1/2,1/2':
        return 0.25 * (F + di(F, '1,0') + di(F, '0,1') + di(di(F, '0,1'), '1,0'))
    elif index == '1/2,-1/2':
        return 0.25 * (F + di(F, '1,0') + di(F, '0,-1') + di(di(F, '0,-1'), '1,0'))
    elif index == '-1/2,1/2':
        return 0.25 * (F + di(F, '-1,0') + di(F, '0,1') + di(di(F, '0,1'), '-1,0'))
    elif index == '-1/2,-1/2':
        return 0.25 * (F + di(F, '-1,0') + di(F, '0,-1') + di(di(F, '0,-1'), '-1,0'))

def edge(state, tau_x, tau_y, M, alpha):
    u = state[:, :, 1]
    v = state[:, :, 2]
    Mx = state.shape[0] - 1
    state[:, -1, 1] = u[:, -1] + 1 * (-u[:, -1] * tau_y + v[:, -1] * tau_x) * tau_y
    state[:, -1, 2] = v[:, -1] - 1 * (-u[:, -1] * tau_y + v[:, -1] * tau_x) * tau_x
    temp = np.sqrt(state[:, -1, 1]**2 + state[:, -1, 2]**2)
    temp[temp == 0] = 1
    state[:, -1, 1] = state[:, -1, 1] / temp * np.sqrt(u[:, -1]**2 + v[:, -1]**2)
    state[:, -1, 2] = state[:, -1, 2] / temp * np.sqrt(u[:, -1]**2 + v[:, -1]**2)
    state[[0, -1], -1, 1] = u[[0, -1], -1]
    state[[0, -1], -1, 2] = v[[0, -1], -1]
    state[Mx//4 + 1:Mx - Mx//4 + 1, 0, 0] = 1
    state[Mx//4 + 1:Mx - Mx//4 + 1, 0, 1] = np.sqrt(1.4) * M * np.cos(alpha)
    state[Mx//4 + 1:Mx - Mx//4 + 1, 0, 2] = np.sqrt(1.4) * M * np.sin(alpha)
    state[Mx//4 + 1:Mx - Mx//4 + 1, 0, 3] = 1
    state[-1, :, :] = state[0, :, :]

    # # Verify the boundary conditions 入口速度和密度
    # print("Boundary conditions at inlet:")
    # print("Density at inlet:", state0[Mx//4 + 1:Mx - Mx//4 + 1, 0, 0])
    # print("u at inlet:", state0[Mx//4 + 1:Mx - Mx//4 + 1, 0, 1])
    # print("v at inlet:", state0[Mx//4 + 1:Mx - Mx//4 + 1, 0, 2])
    # print("Pressure at inlet:", state0[Mx//4 + 1:Mx - Mx//4 + 1, 0, 3])
    return state

def streamline_fig(X, Y, U, V, limx, limy, k):
    Xp, Yp = np.meshgrid(np.linspace(limx[0], limx[1], 1001), np.linspace(limy[0], limy[1], 1001))
    Up = griddata((X.flatten(), Y.flatten()), U.flatten(), (Xp, Yp), method='cubic')
    Vp = griddata((X.flatten(), Y.flatten()), V.flatten(), (Xp, Yp), method='cubic')
    
    # Circular boundary
    mask = Xp**2 + Yp**2 >= 1**2
    Up *= mask
    Vp *= mask

    # # Function for the airfoil shape
    # def airfoil(x):
    #     return 0.594689181 * (0.298222773 * np.sqrt(x) - 0.127125232 * x - 0.357907906 * x**2 + 0.291984971 * x**3 - 0.105174606 * x**4) * (0 <= x) * (x <= 1)
    # airfoil_mask = np.logical_or(Yp >= airfoil(Xp + 0.5), Yp <= -airfoil(Xp + 0.5))
    # Up *= airfoil_mask
    # Vp *= airfoil_mask

    # Plotting
    kappa = 50
    Xp2, Yp2 = Xp[::kappa, ::kappa], Yp[::kappa, ::kappa]
    Up2, Vp2 = Up[::kappa, ::kappa], Vp[::kappa, ::kappa]
    if k != -1:
        plt.quiver(Xp2, Yp2, Up2, Vp2, k)
    plt.streamplot(Xp, Yp, Up, Vp, color='b', linewidth=1)
    plt.xlim(limx)
    plt.ylim(limy)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(X[:, -1], Y[:, -1])
    plt.show()

def calculate_U(state, gamma, Jac):
    U = state.copy()
    U[:, :, 1] = state[:, :, 0] * state[:, :, 1]
    U[:, :, 2] = state[:, :, 0] * state[:, :, 2]
    U[:, :, 3] = 0.5 * state[:, :, 0] * (state[:, :, 1]**2 + state[:, :, 2]**2) + 1 / (gamma - 1) * state[:, :, 3]
    U *= Jac[:, :, np.newaxis]
    return U

def calculate_state(U, gamma, Jac):
    if np.isnan(U).any():
        print('Warning: NaN detected')
    Jac = Jac[:, :, np.newaxis]
    U = U / Jac
    size0 = U.shape[:2]
    state = np.zeros((size0[0], size0[1], 4))
    state[:,:,0] = U[:,:,0]
    state[:,:,1] = U[:,:,1] / state[:,:,0]
    state[:,:,2] = U[:,:,2] / state[:,:,0]
    state[:,:,3] = (gamma - 1) * (U[:,:,3] - 0.5 * state[:,:,0] * (state[:,:,1]**2 + state[:,:,2]**2))
    state[:,:,0] = np.where(state[:,:,0] >= 0.1, state[:,:,0], np.where(state[:,:,0] < 0.001, 0.1, state[:,:,0]))
    state[:,:,3] = np.where(state[:,:,3] >= 0, state[:,:,3], 0)
    return state

# def calculate_F_G(state, gamma, Jac, xi_x, eta_x, xi_y, eta_y):
#     size0 = state.shape[:2]
#     F = np.zeros((size0[0], size0[1], 4))
#     G = np.zeros((size0[0], size0[1], 4))
    
#     F[:, :, 0] = state[:, :, 0] * state[:, :, 1]
#     F[:, :, 1] = state[:, :, 0] * state[:, :, 1]**2 + state[:, :, 4]
#     F[:, :, 2] = state[:, :, 0] * state[:, :, 1] * state[:, :, 3]
#     F[:, :, 3] = (0.5 * state[:, :, 0] * (state[:, :, 1]**2 + state[:, :, 3]**2) + 
#                   gamma / (gamma - 1) * state[:, :, 4]) * state[:, :, 1]
    
#     G[:, :, 0] = state[:, :, 0] * state[:, :, 3]
#     G[:, :, 1] = state[:, :, 0] * state[:, :, 1] * state[:, :, 3]
#     G[:, :, 2] = state[:, :, 0] * state[:, :, 3]**2 + state[:, :, 4]
#     G[:, :, 3] = (0.5 * state[:, :, 0] * (state[:, :, 1]**2 + state[:, :, 3]**2) + 
#                   gamma / (gamma - 1) * state[:, :, 4]) * state[:, :, 3]
    
#     F_h = Jac * (F * xi_x + G * xi_y)
#     G_h = Jac * (F * eta_x + G * eta_y)
    
#     return F_h, G_h

def calculate_F_G(state, gamma, Jac, xi_x, eta_x, xi_y, eta_y):
    size0 = state.shape[:2]
    F = np.zeros((size0[0], size0[1], 4))
    G = np.zeros((size0[0], size0[1], 4))
    
    # 计算压力（或者其他合适的变量），这里假设为 p
    p = (gamma - 1) * (state[:, :, 3] - 0.5 * state[:, :, 0] * (state[:, :, 1]**2 + state[:, :, 2]**2))
    
    F[:, :, 0] = state[:, :, 0] * state[:, :, 1]
    F[:, :, 1] = state[:, :, 0] * state[:, :, 1]**2 + p  # 替换 state[:, :, 4] 为 p
    F[:, :, 2] = state[:, :, 0] * state[:, :, 1] * state[:, :, 2]
    F[:, :, 3] = (0.5 * state[:, :, 0] * (state[:, :, 1]**2 + state[:, :, 2]**2) + 
                  gamma / (gamma - 1) * p) * state[:, :, 1]
    
    G[:, :, 0] = state[:, :, 0] * state[:, :, 2]
    G[:, :, 1] = state[:, :, 0] * state[:, :, 1] * state[:, :, 2]
    G[:, :, 2] = state[:, :, 0] * state[:, :, 2]**2 + p  # 替换 state[:, :, 4] 为 p
    G[:, :, 3] = (0.5 * state[:, :, 0] * (state[:, :, 1]**2 + state[:, :, 2]**2) + 
                  gamma / (gamma - 1) * p) * state[:, :, 2]
    
    # 扩展 xi_x, xi_y, eta_x, eta_y 的形状以匹配 F 和 G
    xi_x = xi_x[:, :, np.newaxis]
    xi_y = xi_y[:, :, np.newaxis]
    eta_x = eta_x[:, :, np.newaxis]
    eta_y = eta_y[:, :, np.newaxis]
    
    F_h = Jac * (F * xi_x + G * xi_y)
    G_h = Jac * (F * eta_x + G * eta_y)
    
    return F_h, G_h

def calculate_lambda(state, gamma, xi_x, eta_x, xi_y, eta_y):
    rho = state[:, :, 0]
    u = state[:, :, 1]
    v = state[:, :, 2]
    p = state[:, :, 3]
    a = np.sqrt(gamma * np.abs(p / rho))
    lambda_xi = np.abs(xi_x * u + xi_y * v) + a * np.sqrt(xi_x**2 + xi_y**2)
    lambda_eta = np.abs(eta_x * u + eta_y * v) + a * np.sqrt(eta_x**2 + eta_y**2)
    return lambda_xi, lambda_eta

def Rusanov_scheme(u, dx, dt, gamma, Jac, xi_x, eta_x, xi_y, eta_y):
    state = calculate_state(u, gamma, Jac)
    F, G = calculate_F_G(state, gamma, Jac, xi_x, eta_x, xi_y, eta_y)
    lambda_xi, lambda_eta = calculate_lambda(state, gamma, xi_x, eta_x, xi_y, eta_y)

    F_half = 0.5 * (di(F, '1,0') + F) - np.maximum(lambda_xi, di(lambda_xi, '1,0')) * 0.5 * (di(u, '1,0') - u)
    G_half = 0.5 * (di(G, '0,1') + G) - np.maximum(lambda_eta, di(lambda_eta, '0,1')) * 0.5 * (di(u, '0,1') - u)
    P_Delta = 1 / dx[0] * (F_half - di(F_half, '-1,0')) + 1 / dx[1] * (G_half - di(G_half, '0,-1'))

    return P_Delta

def Runge_Kutta4(x_scheme, u, dt, dx, *args):
    u1 = u - (dt / 4.0) * x_scheme(u, dx, dt, *args)
    u2 = u - (dt / 3.0) * x_scheme(u1, dx, dt, *args)
    u3 = u - (dt / 2.0) * x_scheme(u2, dx, dt, *args)
    u_next = u - dt * x_scheme(u3, dx, dt, *args)
    return u_next

# Main loop
X_xi, X_eta, X_xixi, X_etaeta, X_xieta, Y_xi, Y_eta, Y_xixi, Y_etaeta, Y_xieta, Jac = Jacobi(X, Y)
Jac = np.where(np.abs(Jac) > 1e-5, Jac, 1e-5)
xi_x = Y_eta / Jac
eta_x = -Y_xi / Jac
xi_y = -X_eta / Jac
eta_y = X_xi / Jac

dX = di(X[:, My], '1,0') - di(X[:, My], '-1,0')
dY = di(Y[:, My], '1,0') - di(Y[:, My], '-1,0')
tau_x = dX / np.sqrt(dX**2 + dY**2)
tau_y = dY / np.sqrt(dX**2 + dY**2)
state0 = edge(state0, tau_x, tau_y, M, alpha)
streamline_fig(X, Y, state0[:, :, 1], state0[:, :, 2], [-2, 2], [-2, 2], 0.5)

U_solution = calculate_U(state0, gamma, Jac)

for time in np.arange(t0 + dt, tf + dt, dt):
    U_solution = Runge_Kutta4(Rusanov_scheme, U_solution, dt, dx, gamma, Jac, xi_x, eta_x, xi_y, eta_y)
    state = edge(calculate_state(U_solution, gamma, Jac), tau_x, tau_y, M, alpha)
    U_solution = calculate_U(state, gamma, Jac)
    if time in timeplot:
        plt.figure(1)
        plt.contourf(X, Y, state[:, :, 0])
        plt.figure(2)
        U, V = state[:, :, 1], state[:, :, 2]
        streamline_fig(X, Y, U, V, [-2, 2], [-2, 2], 0.5)
        plt.draw()

U, V = state[:, :, 1], state[:, :, 2]
streamline_fig(X, Y, U, V, [-2, 2], [-2, 2], 1)
