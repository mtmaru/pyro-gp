import torch
import pyro
from pyro.nn import PyroModule, PyroParam, pyro_method
import pyro.distributions as dist
from pyro import poutine

from models import VSGP

class Identity(PyroModule):
    def __init__(self):
        super().__init__()

    def forward(self, f, y = None):
        # f.shape = (num_output_dim, num_data_points)
        # y.shape = (num_output_dim, num_data_points)

        # (num_output_dim, num_data_points)
        y = pyro.deterministic(self._pyro_get_fullname("y"), f, f.dim())

        return y

class Exp(PyroModule):
    def __init__(self, num_output_dim):
        super().__init__()

        # (num_output_dim, 1)
        # 後で (num_output_dim, num_data_points) にブロードキャストされる
        self.mu = PyroParam(torch.zeros(num_output_dim, 1))

    def forward(self, f, y = None):
        # f.shape = (num_output_dim, num_data_points)
        # y.shape = (num_output_dim, num_data_points)

        # (num_output_dim, num_data_points)
        y = pyro.deterministic(self._pyro_get_fullname("y"), (f + self.mu).exp(), f.dim())

        return y

class VSHGPRegression(PyroModule):
    def __init__(self, Z, num_output_dim, kernel_f, kernel_r, whiten, jitter = 1e-6):
        super().__init__()

        self.gp_f = VSGP(Z, num_output_dim, kernel_f, Identity(), whiten, jitter)
        self.gp_r = VSGP(Z, num_output_dim, kernel_r, Exp(num_output_dim), whiten, jitter)

    @pyro_method
    def model(self, X, y = None, num_data = None):
        # X.shape = (num_data_points, num_input_dim)
        # y.shape = (num_data_points, num_output_dim)

        # (num_output_dim, num_data_points)
        f = self.gp_f.model(X, y, num_data)
        # (num_output_dim, num_data_points)
        r = self.gp_r.model(X, y, num_data)

        num_data = X.shape[0] if num_data is None else num_data
        with poutine.scale(scale = num_data / X.shape[0]):
            # (num_output_dim, num_data_points)
            y = y.T if y is not None else y
            y = pyro.sample(self._pyro_get_fullname("y"), dist.Normal(loc = f, scale = r).to_event(f.dim()), obs = y)

        return y

    @pyro_method
    def guide(self, X, y = None, num_data = None):
        # X.shape = (num_data_points, num_input_dim)
        # y.shape = (num_data_points, num_output_dim)

        self.gp_f.guide(X, y, num_data)
        self.gp_r.guide(X, y, num_data)

    def forward(self, X, num_samples):
        # X.shape = (num_data_points, num_input_dim)

        # (num_data_points, num_output_dim)
        _, f = self.gp_f(X, num_samples)
        # (num_data_points, num_output_dim)
        _, r = self.gp_r(X, num_samples)

        return f, r
