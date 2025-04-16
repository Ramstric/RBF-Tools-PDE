import numpy as np
import torch

from templates.custom_plotly import custom
import plotly.graph_objects as go

from RBF import DifferentialEquationSolver as DESolver
from RBF import RadialBasisFunctions as rbf
import sys

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def derivative_operator(r, radius, variables=None):
    # For this code variables are [x, t] in that order
    # Therefore, variables[0] = x
    #            variables[1] = t
    return rbf.gaussian_xx(r, variables[0], radius) - rbf.gaussian_tt(r, variables[1], radius)
    #return rbf.multiquadric_xx(r, variables[1], radius) - rbf.multiquadric_tt(r, variables[0], radius)


# Smoothing parameter
#sigma = float(sys.argv[1])
sigma = 0.333
#sigma = 1.4

# Measured data. Load NP arrays and convert to torch tensors float64
boundary_x = np.load("./temp_geometry/boundary_x.npy")
boundary_time = np.load("./temp_geometry/boundary_time.npy")
boundary_u = np.load("./temp_geometry/boundary_u.npy")

inner_x = np.load("./temp_geometry/inner_x.npy")
inner_time = np.load("./temp_geometry/inner_time.npy")
inner_u = np.load("./temp_geometry/inner_u.npy")

boundary_x = torch.tensor(boundary_x, dtype=torch.float64, device=device)
boundary_time = torch.tensor(boundary_time, dtype=torch.float64, device=device)
boundary_u = torch.tensor(boundary_u, dtype=torch.float64, device=device)

inner_x = torch.tensor(inner_x, dtype=torch.float64, device=device)
inner_time = torch.tensor(inner_time, dtype=torch.float64, device=device)
inner_u = torch.tensor(inner_u, dtype=torch.float64, device=device)

u = torch.cat((boundary_u, inner_u))

DEInterpolator = DESolver.DifferentialInterpolator(boundary=[boundary_x, boundary_time], inner=[inner_x, inner_time],
                                                   f=u, radius=sigma, rbf_name="gaussian",
                                                   derivative_operator=derivative_operator)

#np.save("center_points.npy", DEInterpolator.center_points.numpy())
#np.save("weights_matrix.npy", DEInterpolator.weights_matrix.numpy())

# Interpolated data
time_interpol = torch.linspace(0, 3, 60, device=device)
x_interpol = torch.linspace(0, 3, 60, device=device)
x_interpol, time_interpol = torch.meshgrid(x_interpol, time_interpol, indexing="xy")

u_interpol = DEInterpolator.interpolate(x_interpol, time_interpol)

# Plotting
# Move the interpolated data to the CPU
time_interpol = time_interpol.cpu().numpy()
x_interpol = x_interpol.cpu().numpy()
u_interpol = u_interpol.cpu().numpy()

fig = go.Figure(data=[go.Surface(z=u_interpol, x=x_interpol, y=time_interpol, colorscale="Inferno")])

fig.update_layout(template=custom, font_size=12)

#fig.write_image("interpolated_wave_eq_" + str(sigma) + ".png")

fig.show()
