from project.gen_data import create_xt_grids, gen_solution
from project.post_processing import Plotter
from project.metric import *

import matplotlib.pyplot as plt

x_dim = 30
t_dim = 100

X, x, t = create_xt_grids(
    x_range= (0, 1),
    x_dim = x_dim,
    t_range= (0, 1),
    t_dim = t_dim)

y_true = gen_solution(X, x, t, "heat_1d_boundary_sin_exact", {
    "a_coeff": 0.4,
    "frequency": 1,
})

y_predict = gen_solution(X, x, t, "finite_difference_crank_nicolson", {
    "a_coeff": 0.4,
})

output_performance(y_true, y_predict)
plotter = Plotter(2,2)
plotter.plot_2d_y(X, y_true, "y_true")
plotter.plot_2d_y(X, y_predict, "y_predict")
errors = compute_errors(y_true, y_predict)
plotter.plot_2d_y(X, errors, "error plot")
plotter.plot_2d_colormesh(errors, x_dim, t_dim, "errors mesh")

plt.show()
