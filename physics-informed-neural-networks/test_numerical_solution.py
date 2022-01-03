from project.gen_data import gen_data
from project.post_processing import Plotter

import matplotlib.pyplot as plt
import numpy as np


def test_scheme(scheme_name, data_dict):
    X, y = gen_data(scheme_name, data_dict)
    Plotter(1,1).plot_2d_y(X, y)
    plt.show()

def test_numerical_convergence():
    params_1 = {
    "x_range": (0, 1),
    "t_range": (0, 0.05),
    "x_dim": 10,
    "t_dim": 20,
    "a_coeff": 1,
    "frequency": 1,
    }

    params_2 = {
    "x_range": (0, 1),
    "t_range": (0, 0.05),
    "x_dim": 20,
    "t_dim": 20,
    "a_coeff": 1,
    "frequency": 1,
    }

    params_2_test = {
    "x_range": (0, 1),
    "t_range": (0, 0.05),
    "x_dim": 19,
    "t_dim": 20,
    "a_coeff": 1,
    "frequency": 1,
    }

    params_3 = {
    "x_range": (0, 1),
    "t_range": (0, 0.05),
    "x_dim": 40,
    "t_dim": 20,
    "a_coeff": 1,
    "frequency": 1,
    }

    params_3_test = {
    "x_range": (0, 1),
    "t_range": (0, 0.05),
    "x_dim": 39,
    "t_dim": 20,
    "a_coeff": 1,
    "frequency": 1,
    }

    params_4 = {
    "x_range": (0, 1),
    "t_range": (0, 0.05),
    "x_dim": 80,
    "t_dim": 20,
    "a_coeff": 1,
    "frequency": 1,
    }

    params_4_test = {
    "x_range": (0, 1),
    "t_range": (0, 0.05),
    "x_dim": 79,
    "t_dim": 20,
    "a_coeff": 1,
    "frequency": 1,}

    params_5_test = {
    "x_range": (0, 1),
    "t_range": (0, 0.05),
    "x_dim": 159,
    "t_dim": 20,
    "a_coeff": 1,
    "frequency": 1,}

    X1, y1 = gen_data("finite_difference_crank_nicolson", params_1)
    X2, y2 = gen_data("finite_difference_crank_nicolson", params_2)
    X3, y3 = gen_data("finite_difference_crank_nicolson", params_3)
    X4, y4 = gen_data("finite_difference_crank_nicolson", params_4)

    X2, y2_comp = gen_data("finite_difference_crank_nicolson", params_2_test)
    X3, y3_comp = gen_data("finite_difference_crank_nicolson", params_3_test)
    X4, y4_comp = gen_data("finite_difference_crank_nicolson", params_4_test)
    X5, y5_comp = gen_data("finite_difference_crank_nicolson", params_5_test)
    y1 = y1.reshape(20, 10)
    y2 = y2.reshape(20, 20)
    y3 = y3.reshape(20, 40)
    y4 = y4.reshape(20, 80)

    y2_comp = y2_comp.reshape(20,19)
    y3_comp = y3_comp.reshape(20, 39)
    y4_comp = y4_comp.reshape(20, 79)
    y5_comp = y5_comp.reshape(20, 159)
    e1 = np.max(np.abs(y1 - y2_comp[:,0::2]))
    e2 = np.max(np.abs(y2 - y3_comp[:,0::2]))
    e3 = np.max(np.abs(y3 - y4_comp[:,0::2]))
    e4 = np.max(np.abs(y4 - y5_comp[:,0::2]))

    r1 = np.log(e2/e1)/np.log(2)
    r2 = np.log(e3/e2)/np.log(2)
    r3 = np.log(e4/e3)/np.log(2)

    # test_scheme("finite_difference_crank_nicolson", params)


test_scheme("finite_difference_crank_nicolson", {
    "x_range": (0, 1),
    "t_range": (0, 1),
    "x_dim": 30,
    "t_dim": 100,
    "a_coeff": 0.4,
    "frequency": 1,
})

test_scheme("finite_difference_euler", {
    "x_range": (0, 1),
    "t_range": (0, 1),
    "x_dim": 7,
    "t_dim": 100,
    "a_coeff": 0.4,
    "frequency": 1,
})

test_numerical_convergence()