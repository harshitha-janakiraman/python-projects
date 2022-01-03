from dataclasses import dataclass
from project.gen_data import create_xt_grids, gen_solution
from project.post_processing import Plotter
from project.timer import Timer

@dataclass
class HeatConstParam:
    a = 0.4 # Thermal diffusivity
    L = 1 # Length of the bar, Assum x starts at 0
    n = 1 # Frequency of the sinusoidal initial conditions
    t_end = 1 # Assume t_start  = 0
    test_x_dim = 60
    test_t_dim = 200
    analytical_function_name = "heat_1d_boundary_sin_exact"
    numerical_function_name = "finite_difference_crank_nicolson"
    X_test = None
    y_analytical = None
    y_numerical = None

    def __str__(self) -> str:
        rs = "Heat Const Param: \n"
        rs =  rs + "a: {}\n".format(self.a)
        rs =  rs + "L: {}\n".format(self.L)
        rs =  rs + "n: {}\n".format(self.n)
        rs =  rs + "t_end: {}\n".format(self.t_end)
        rs =  rs + "test_x_dim: {}\n".format(self.test_x_dim)
        rs =  rs + "test_t_dim: {}\n".format(self.test_t_dim)
        rs =  rs + "analytical_function_name: " + self.analytical_function_name + "\n"
        rs =  rs + "numerical_function_name: " + self.numerical_function_name + "\n"
        return rs

def solve_analytical_and_numerical(heat_params:HeatConstParam, timer_repeat_times = 25):
    heat_params.X_test, x, t = create_xt_grids(
        x_range= (0, heat_params.L),
        x_dim = heat_params.test_x_dim,
        t_range= (0, heat_params.t_end),
        t_dim = heat_params.test_t_dim)

    ti = Timer(timer_repeat_times)
    heat_params.y_analytical = ti.time_average(lambda: gen_solution(
        heat_params.X_test, x, t,
        heat_params.analytical_function_name,
        {
            "a_coeff": heat_params.a,
            "frequency": heat_params.n,
        }
    ))
    print(heat_params.analytical_function_name + ": "+ ti.str_average())

    heat_params.y_numerical = ti.time_average(lambda : gen_solution(
        heat_params.X_test, x, t,
        heat_params.numerical_function_name,
        {
            "a_coeff": heat_params.a,
        }))
    print(heat_params.numerical_function_name + ": "+ ti.str_average())

def plot_analytical_and_numerical(plotter: Plotter, heat_params: HeatConstParam):
    # I. Analytical test solution
        plotter.plot_2d_y(
            heat_params.X_test,
            heat_params.y_analytical,
            heat_params.analytical_function_name)
        plotter.plot_2d_colormesh(
            heat_params.y_analytical,
            heat_params.test_x_dim,
            heat_params.test_t_dim)

    # II. Numerical test solution
        plotter.plot_2d_y(
            heat_params.X_test,
            heat_params.y_numerical,
            heat_params.numerical_function_name)
        plotter.plot_2d_colormesh(
            heat_params.y_numerical,
            heat_params.test_x_dim,
            heat_params.test_t_dim)
