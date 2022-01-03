from dataclasses import dataclass

import numpy as np
import deepxde as dde

from heat import *
from project.post_processing import Plotter
from project.timer import Timer
from project.metric import *

@dataclass
class PinnParam:
    d_num_domain = 2540
    d_num_boundary = 80
    d_num_initial = 160
    nn_hidden_layer_size = 20
    nn_hidden_layer_num = 3

    def __str__(self) -> str:
        rs = "Pinn Param: \n"
        rs =  rs + "d_num_domain: {}\n".format(self.d_num_domain)
        rs =  rs + "d_num_boundary: {}\n".format(self.d_num_boundary)
        rs =  rs + "d_num_initial: {}\n".format(self.d_num_initial)
        rs =  rs + "nn_hidden_layer_size: {}\n".format(self.nn_hidden_layer_size)
        rs =  rs + "nn_hidden_layer_num: {}\n".format(self.nn_hidden_layer_num)
        return rs

def pde(x, y, a):
    """
    Expresses the PDE residual of the heat equation.
    """
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a*dy_xx

def create_model(heat_params: HeatConstParam, pinn_params: PinnParam):
    # Computational geometry:
    geom = dde.geometry.Interval(0, heat_params.L)
    timedomain = dde.geometry.TimeDomain(0, heat_params.t_end)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Initial and boundary conditions:
    bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.IC(
        geomtime, lambda x: np.sin(heat_params.n*np.pi*x[:, 0:1]/heat_params.L) , lambda _, on_initial: on_initial
        )

    l_pde = lambda x, y: pde(x, y, heat_params.a)

    geomtime_data = dde.data.TimePDE(
        geomtime, l_pde, [bc, ic],
        num_domain=pinn_params.d_num_domain,
        num_boundary=pinn_params.d_num_boundary,
        num_initial=pinn_params.d_num_initial,
        num_test=2540
    )

    nn_architecture = [2] + [pinn_params.nn_hidden_layer_size] * pinn_params.nn_hidden_layer_num + [1]
    net = dde.nn.FNN(
        nn_architecture,
        "tanh",
        "Glorot normal")

    model = dde.Model(geomtime_data, net)

    return geomtime_data, model

def train_model(model):
    # Build and train the model:
    ti = Timer()
    ti.start()
    model.compile("adam", lr=1e-3)
    model.train(epochs=20000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    ti.stop()
    print("Training time: "+ ti.str_elapsed_time())

    return losshistory, train_state

def plot_train_data(geomtime_data):
    X_bc_train = geomtime_data.bc_points()
    print("train_bc_points shape", X_bc_train.shape)

    plotter = Plotter(2, 2)
    plotter.plot_2d(X_bc_train, "train_bc_points")

    # Train points give random values
    X_train_points = geomtime_data.train_points()
    print("train_points shape", X_train_points.shape)
    plotter.plot_2d(X_train_points, "train_points")

    # Train next batch resamples at every epoch because of
    # PDEResidualResampler, and resample_train_points()
    X_train_next_batch, _, _ = geomtime_data.train_next_batch()
    print("train_next_batch shape", X_train_next_batch.shape)
    plotter.plot_2d(X_train_next_batch, "train_next_batch")

    geomtime_data.resample_train_points()
    X_train_next_batch, _, _ = geomtime_data.train_next_batch()
    plotter.plot_2d(X_train_next_batch, "resampled_train_next_batch")

def predict_and_output_report(
    model,
    heat_params: HeatConstParam,
    pinn_params: PinnParam = None,
    timer_repeat_times = 25,
    losshistory = None,
    train_state = None):
    plotter = Plotter(4, 2)

    plot_analytical_and_numerical(plotter, heat_params)

    # III. Physics Informed Neural Network test solution``
    function_name = "NN prediction"
    ti = Timer(timer_repeat_times)
    y_pred = ti.time_average(lambda : model.predict(heat_params.X_test))
    y_pred = y_pred.reshape(-1, 1)
    print(function_name + ": "+ ti.str_average())
    plotter.plot_2d_y(heat_params.X_test, y_pred, function_name)
    plotter.plot_2d_colormesh(
        y_pred,
        heat_params.test_x_dim,
        heat_params.test_t_dim)

    # IV. Performance metrics
    numerical_analytical_errors = compute_errors(heat_params.y_analytical, heat_params.y_numerical)
    plotter.plot_2d_colormesh(
        numerical_analytical_errors,
        heat_params.test_x_dim,
        heat_params.test_t_dim,
        "Numerical vs Analytical errors")
    pinn_analytical_errors = compute_errors(heat_params.y_analytical, y_pred)
    plotter.plot_2d_colormesh(
        pinn_analytical_errors,
        heat_params.test_x_dim,
        heat_params.test_t_dim,
        "NN vs Analytical errors")

    if losshistory and train_state:
        dde.saveplot(losshistory, train_state, issave=False, isplot=True)
        print("IGNORE ^^^ PDE test data SOLUTION PLOT!!! ^^^")

    print("--- Numerical vs Analytical Report ---")
    output_performance(heat_params.y_analytical, heat_params.y_numerical)

    # f = model.predict(heat_params.X_test, operator=pde)
    print("--- NN vs Analytical Report ---")
    output_performance(heat_params.y_analytical, y_pred)
    # print("Mean residual:", np.mean(np.absolute(f)))
    # print("L2 relative error:", dde.metrics.l2_relative_error(heat_params.y_analytical, y_pred))
    # np.savetxt("test.dat", np.hstack((X_test, y_analytical, y_pred)))

    print(heat_params)
    if pinn_params != None:
        print(pinn_params)
