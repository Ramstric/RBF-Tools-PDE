import numpy as np
import torch

from templates.custom_plotly import custom
import plotly.graph_objects as go

"""
def ode(i, j):
    # When u(x, 0) = 4sin(0.16x)
    if i == 0:
        return 0.

    # When u(0, t) = 0 and u(1, t) = sin
    elif j == 0:
        return 0.
    elif j == 30:
        return np.sin(2.1 * (i / 10))

    else:
        return 0.5
"""
def ode(i, j):
    # When u(x, 0) = 4sin(0.16x)
    if i == 0 and j >= 15:
        return np.sin(2.1 * (j / 10)+16)

    elif i == 0 and j < 15:
        return 0.

    # When u(0, t) = 0 and u(1, t) = sin
    elif j == 0 or j == 30:
        return 0.

    else:
        return 0.5


time = np.arange(0, 3.1, 0.1)
x = np.arange(0, 3.1, 0.1)
x, time = np.meshgrid(x, time, indexing="xy")
print(x.shape)
u = np.fromfunction(np.vectorize(ode), (x.shape[0], x.shape[1]))

indices = np.argwhere(u == 0.5)

inner_x = x[indices[:, 0], indices[:, 1]]
inner_time = time[indices[:, 0], indices[:, 1]]
inner_u = np.zeros(inner_x.shape)

indices = np.argwhere(u == 10)

dirichlet_x = x[indices[:, 0], indices[:, 1]]
dirichlet_time = time[indices[:, 0], indices[:, 1]]
dirichlet_u = np.zeros(dirichlet_x.shape)

indices = np.argwhere((u != 10) & (u != 0.5))

initial_x = x[indices[:, 0], indices[:, 1]]
initial_time = time[indices[:, 0], indices[:, 1]]
initial_u = u[indices[:, 0], indices[:, 1]]

boundary_x = np.concatenate((initial_x, dirichlet_x))
boundary_time = np.concatenate((initial_time, dirichlet_time))
boundary_u = np.concatenate((initial_u, dirichlet_u))

fig = go.Figure(data=[go.Scatter3d(
    z=inner_u,
    x=inner_x,
    y=inner_time,
    name="Inner points"
)])

fig.add_scatter3d(
    z=initial_u,
    x=initial_x,
    y=initial_time,
    name="Initial points"
)

fig.add_scatter3d(
    z=dirichlet_u,
    x=dirichlet_x,
    y=dirichlet_time,
    name="Dirichlet points"
)

fig.add_scatter3d(
    z=boundary_u,
    x=boundary_x,
    y=boundary_time,
    name="Boundary points"
)

# Save np arrays
np.save("./temp_geometry/boundary_x.npy", boundary_x)
np.save("./temp_geometry/boundary_time.npy", boundary_time)
np.save("./temp_geometry/boundary_u.npy", boundary_u)

np.save("./temp_geometry/inner_x.npy", inner_x)
np.save("./temp_geometry/inner_time.npy", inner_time)
np.save("./temp_geometry/inner_u.npy", inner_u)

print(torch.from_numpy(boundary_x))
print(torch.from_numpy(boundary_time))
print(torch.from_numpy(boundary_u))

print(torch.from_numpy(inner_x))
print(torch.from_numpy(inner_time))
print(torch.from_numpy(inner_u))

fig.update_layout(template=custom, font_size=12)

fig.show()
