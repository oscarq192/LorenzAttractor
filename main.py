import numpy as np
import pyvista as pv
import time

plotter = pv.Plotter()
plotter.set_background("black")

t0, tf, h = 0, 100, 0.01
sigma, rho, beta = 10, 28, 8/3
u0 = np.array([1.0, 1.0, 1.0])

def lorenz(t_n, u_n):
    x, y, z = u_n

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def rk4(t_n, u_n):
    k1 = lorenz(t_n, u_n)
    k2 = lorenz(t_n + h / 2, u_n + h * k1 / 2)
    k3 = lorenz(t_n + h / 2, u_n + h * k2 / 2)
    k4 = lorenz(t_n + h, u_n + h * k3)
    return u_n + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

N = int((tf - t0) / h)
t = np.linspace(t0, tf, N)
u = np.zeros((N, 3))
u[0] = u0
for i in range(1, N):
    u[i] = rk4(t[i-1], u[i-1])

n_points = u.shape[0]
lines = np.hstack([[n_points], np.arange(n_points)])
trajectory = pv.PolyData()
trajectory.points = u
trajectory.lines = lines
trajectory["time"] = np.arange(n_points)

tube = trajectory.tube(radius=0.05)
plotter.add_mesh(
    tube,
    scalars="time",
    cmap="plasma",
    smooth_shading=True,
    lighting=True,
    show_scalar_bar=False,
)

plotter.show()