import numpy as np

def gen_data(func_name = "", dict_args = {}):
    """
    Returns the grid X, and solution y, by n data points
    X.shape (n, 2)
    y.shape (n, 1)

    Parameters
    ----------
    func_name : str
    """

    if func_name == "":
        return None, None
    X, x, t = create_xt_grids(
        dict_args["x_range"],
        dict_args["x_dim"],
        dict_args["t_range"],
        dict_args["t_dim"])
    y = gen_solution(X, x, t,
        func_name,
        dict_args)

    return X, y

def create_xt_grids(x_range, x_dim, t_range, t_dim):
    # Bounds of 'x' and 't':
    t_min, t_max = t_range
    x_min, x_max = x_range

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)

    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T

    return X, x, t

def gen_solution(X, x, t, func_name = "", dict_args = {}):
    if func_name == "":
        return None, None

    return globals()[func_name](X, x, t, dict_args)

def finite_difference_crank_nicolson(X, x, t, dict_args):
    '''Return the numerical solution using a Crank-Nicolson scheme for a given x and t
    Refer to
    https://georg.io/2013/12/03/Crank_Nicolson
    For an example of how to implement this.

    '''
    a = dict_args["a_coeff"]

    h = (x[1] - x[0])[0]
    k = (t[1] - t[0])[0]

    num_space_points = len(x)-2
    sigma = a*k / (2 * h**2)
    d = np.empty(num_space_points);
    d.fill(2*sigma);
    # d[0] = sigma; d[-1] = sigma
    subd = np.empty(num_space_points - 1); subd.fill(-sigma)
    supd = np.empty(num_space_points - 1); supd.fill(-sigma)

    D_matrix = np.diag(d) + np.diag(subd, -1) + np.diag(supd, 1)
    A_matrix = np.eye(num_space_points) + D_matrix
    B_matrix = np.eye(num_space_points) - D_matrix

    def compute_next_time_step(prev_result, A_matrix, B_matrix):
        u = np.concatenate((
            [0],
            np.linalg.solve(A_matrix, np.dot(B_matrix, prev_result[1:-1])),
            [0]))
        return u

    solution = [np.sin(np.pi * x).squeeze()] # populate with initial condition
    for i in range(1, len(t)):
        solution.append(compute_next_time_step(solution[i-1], A_matrix, B_matrix))
    return np.array(solution).reshape(-1, 1)

def finite_difference_euler(X, x, t, dict_args):
    '''Return the numerical solution using a forward euler scheme for a given x and t'''

    h = x[1] - x[0]
    k = t[1] - t[0]
    num_space_points = len(x)
    # create finite difference matrix, assuming boundary conditions are 0
    d = np.empty(num_space_points); d.fill(-2)
    subd = np.empty(num_space_points - 1); subd.fill(1)
    supd = np.empty(num_space_points - 1); supd.fill(1)

    matrix = np.eye(num_space_points) + k / h**2 * ( np.diag(d) + np.diag(subd, -1) + np.diag(supd, 1)); matrix

    def compute_next_time_step(prev_result, matrix):
        return matrix.dot(prev_result)

    solution = [np.sin(np.pi * x)] # populate with initial condition
    for i in range(1, len(t)):
        solution.append(compute_next_time_step(solution[i-1], matrix))
    return np.array(solution).reshape(-1, 1)

def heat_1d_boundary_sin_exact(X, x, t, dict_args):
    """
    Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    func_name : str
    """
    a = dict_args["a_coeff"]
    n = dict_args["frequency"]

    def heat_eq_exact_solution(x, t, a, L, n):
        """
        Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """
        return np.exp(-(n**2*np.pi**2*a*t)/(L**2))*np.sin(n*np.pi*x/L)

    # Obtain the value of the exact solution for each generated point:
    L = x.max() - x.min()
    xx, tt = np.meshgrid(x, t)
    return heat_eq_exact_solution(xx.T, tt.T, a, L, n).T.flatten()[:,None]
