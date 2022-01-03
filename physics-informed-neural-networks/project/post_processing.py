import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
class Plotter:
    def __init__(self, rows, cols):
        self.fig = plt.figure()
        standard_size_in = 7
        self.standard_marker_size = 1
        self.fig.set_figheight(standard_size_in * rows)
        self.fig.set_figwidth(standard_size_in * cols)
        self.rows = rows
        self.cols = cols
        self.idx = 1

    def plot_2d_y(self, X, y, label = ""):
        ax = self.fig.add_subplot(self.rows, self.cols, self.idx, projection='3d')
        self.idx = self.idx + 1
        ax.scatter(X[:, 0], X[:, 1], y, s = self.standard_marker_size)
        ax.title.set_text(label)

    def plot_2d(self, X, label = ""):
        ax = self.fig.add_subplot(self.rows, self.cols, self.idx, projection='3d')
        self.idx = self.idx + 1
        ax.scatter(X[:, 0], X[:, 1], s = self.standard_marker_size)
        ax.title.set_text(label)

    def plot_2d_colormesh(self,y , x_dim, t_dim, label = ""):
        ax = self.fig.add_subplot(self.rows, self.cols, self.idx)
        c = ax.pcolormesh(y.flatten().reshape(t_dim, x_dim))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(c, cax=cax, orientation='vertical')
        ax.title.set_text(label)
        self.idx = self.idx + 1
