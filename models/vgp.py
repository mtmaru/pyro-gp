import torch
import pyro
from pyro.nn import PyroModule, PyroParam, pyro_method
import pyro.distributions as dist
from torch.distributions import constraints

class VGP(PyroModule):
    def __init__(self, X, y, kernel, likelihood, whiten, jitter = 1e-6):
        super().__init__()

        # (num_data_points, num_input_dim)
        self.X = X
        # (num_output_dim, num_data_points)
        self.y = y.T

        self.num_output_dim = self.y.shape[0]
        self.kernel = kernel
        self.likelihood = likelihood
        self.whiten = whiten
        self.jitter = jitter

        # (num_output_dim, num_data_points)
        self.f_loc = PyroParam(torch.zeros(self.num_output_dim, self.X.shape[0]))
        # (num_output_dim, num_data_points, num_data_points)
        self.f_cov = PyroParam(
            torch.eye(self.X.shape[0]).repeat([self.num_output_dim, 1, 1]),
            constraints.positive_definite
        )

    @pyro_method
    def model(self):
        # (num_output_dim, num_data_points, num_data_points)
        Kff = self.kernel(self.X, self.X)
        Kff = Kff.repeat([self.num_output_dim, 1, 1])
        Kff = Kff + torch.eye(self.X.shape[0]).repeat([self.num_output_dim, 1, 1]) * self.jitter

        if self.whiten:
            # (num_output_dim, num_data_points)
            f_loc = torch.zeros(self.num_output_dim, self.X.shape[0])
            # (num_output_dim, num_data_points, num_data_points)
            f_cov = torch.eye(self.X.shape[0]).repeat([self.num_output_dim, 1, 1])
            # (num_output_dim, num_data_points)
            f = pyro.sample(self._pyro_get_fullname("f"), dist.MultivariateNormal(
                loc = f_loc,
                covariance_matrix = f_cov
            ).to_event(f_loc.dim() - 1))

            # (num_output_dim, num_data_points, num_data_points)
            Lff = torch.linalg.cholesky(Kff)
            # (num_output_dim, 1, num_data_points)
            #   * (num_output_dim, num_data_points, num_data_points)
            # = (num_output_dim, 1, num_data_points)
            f = f.unsqueeze(1) @ Lff.transpose(1, 2)
            # (num_output_dim, num_data_points)
            f = f.squeeze(1)
        else:
            # (num_output_dim, num_data_points)
            f_loc = torch.zeros(self.num_output_dim, self.X.shape[0])
            # (num_output_dim, num_data_points, num_data_points)
            f_cov = Kff
            # (num_output_dim, num_data_points)
            f = pyro.sample(self._pyro_get_fullname("f"), dist.MultivariateNormal(
                loc = f_loc,
                covariance_matrix = f_cov
            ).to_event(f_loc.dim() - 1))

        # (num_output_dim, num_data_points)
        y = self.likelihood(f, self.y)

    @pyro_method
    def guide(self):
        # (num_output_dim, num_data_points)
        f = pyro.sample(self._pyro_get_fullname("f"), dist.MultivariateNormal(
            loc = self.f_loc,
            covariance_matrix = self.f_cov
        ).to_event(self.f_loc.dim() - 1))

    def forward(self, Xnew, num_samples):
        # Xnew.shape = (num_newdata_points, num_input_dim)

        with pyro.plate("new", size = num_samples):
            # (num_output_dim, num_data_points, num_data_points)
            Kff = self.kernel(self.X, self.X)
            Kff = Kff.repeat([self.num_output_dim, 1, 1])
            Kff = Kff + torch.eye(self.X.shape[0]).repeat([self.num_output_dim, 1, 1]) * self.jitter
            # (num_output_dim, num_data_points, num_newdata_points)
            Kfg = self.kernel(self.X, Xnew)
            Kfg = Kfg.repeat([self.num_output_dim, 1, 1])
            # (num_output_dim, num_newdata_points, num_newdata_points)
            Kgg = self.kernel(Xnew, Xnew)
            Kgg = Kgg.repeat([self.num_output_dim, 1, 1])

            if self.whiten:
                # (num_output_dim, num_data_points, num_data_points)
                Lff = torch.linalg.cholesky(Kff)
                # (num_output_dim, 1, num_data_points)
                #   * (num_output_dim, num_data_points, num_data_points)
                # = (num_output_dim, 1, num_data_points)
                f_loc = self.f_loc.unsqueeze(1) @ Lff.transpose(1, 2)
                # (num_output_dim, num_data_points)
                f_loc = f_loc.squeeze(1)
                # (num_output_dim, num_data_points, num_data_points)
                #   * (num_output_dim, num_data_points, num_data_points)
                #   * (num_output_dim, num_data_points, num_data_points)
                # = (num_output_dim, num_data_points, num_data_points)
                f_cov = Lff @ self.f_cov @ Lff.transpose(1, 2)
            else:
                # (num_output_dim, num_data_points)
                f_loc = self.f_loc
                # (num_output_dim, num_data_points, num_data_points)
                f_cov = self.f_cov

            # (num_output_dim, 1, num_data_points)
            #   * (num_output_dim, num_data_points, num_data_points)
            #   * (num_output_dim, num_data_points, num_newdata_points)
            # = (num_output_dim, 1, num_newdata_points)
            g_loc = f_loc.unsqueeze(1) @ Kff.inverse() @ Kfg
            # (num_output_dim, num_newdata_points)
            g_loc = g_loc.squeeze(1)
            # (num_output_dim, num_newdata_points, num_data_points)
            #   * (num_output_dim, num_data_points, num_data_points)
            #   * (num_output_dim, num_data_points, num_newdata_points)
            # = (num_output_dim, num_newdata_points, num_newdata_points)
            g_cov = Kgg - Kfg.transpose(1, 2) @ (Kff + f_cov).inverse() @ Kfg
            # (num_output_dim, num_newdata_points)
            g_var = g_cov.diagonal(0, 2)
            # (num_output_dim, num_newdata_points)
            g = pyro.sample(self._pyro_get_fullname("g"), dist.Normal(
                loc = g_loc,
                scale = g_var.sqrt()
            ).to_event(g_loc.dim()))

            # (num_output_dim, num_newdata_points)
            ynew = self.likelihood(g)

        return g.transpose(-1, -2), ynew.transpose(-1, -2)
