import torch
import pyro
from pyro.nn import PyroModule, PyroParam, pyro_method
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.infer import Predictive

class SVSGP(PyroModule):
    def __init__(self, Z, num_output_dim, kernel, likelihood, whiten, subsample_size, jitter = 1e-6):
        super().__init__()

        self.num_output_dim = num_output_dim
        self.kernel = kernel
        self.likelihood = likelihood
        self.whiten = whiten
        self.subsample_size = subsample_size
        self.jitter = jitter

        # (num_inducing_points, num_input_dim)
        self.Z = PyroParam(Z.clone())
        # (num_output_dim, num_inducing_points)
        self.u_loc = PyroParam(torch.zeros(self.num_output_dim, self.Z.shape[0]))
        # (num_output_dim, num_inducing_points, num_inducing_points)
        self.u_cov = PyroParam(
            torch.eye(self.Z.shape[0]).repeat([self.num_output_dim, 1, 1]),
            constraints.positive_definite
        )

    @pyro_method
    def model(self, X, y = None):
        # X.shape = (num_data_points, num_input_dim)
        # y.shape = (num_data_points, num_output_dim)

        # (num_output_dim, num_data_points)
        y = y.T if y is not None else y

        # (num_output_dim, num_inducing_points, num_inducing_points)
        Kuu = self.kernel(self.Z, self.Z)
        Kuu = Kuu.repeat([self.num_output_dim, 1, 1])
        Kuu = Kuu + torch.eye(self.Z.shape[0]).repeat([self.num_output_dim, 1, 1]) * self.jitter

        if self.whiten:
            # (num_output_dim, num_inducing_points)
            u_loc = torch.zeros(self.num_output_dim, self.Z.shape[0])
            # (num_output_dim, num_inducing_points, num_inducing_points)
            u_cov = torch.eye(self.Z.shape[0]).repeat([self.num_output_dim, 1, 1])
            # (num_output_dim, num_inducing_points)
            u = pyro.sample("u", dist.MultivariateNormal(
                loc = u_loc,
                covariance_matrix = u_cov
            ).to_event(u_loc.dim() - 1))

            # (num_output_dim, num_inducing_points, num_inducing_points)
            Luu = torch.linalg.cholesky(Kuu)
            # (num_output_dim, 1, num_inducing_points)
            #   * (num_output_dim, num_inducing_points, num_inducing_points)
            # = (num_output_dim, 1, num_inducing_points)
            u = u.unsqueeze(1) @ Luu.transpose(1, 2)
            # (num_output_dim, num_inducing_points)
            u = u.squeeze(1)
        else:
            # (num_output_dim, num_inducing_points)
            u_loc = torch.zeros(self.num_output_dim, self.Z.shape[0])
            # (num_output_dim, num_inducing_points, num_inducing_points)
            u_cov = Kuu
            # (num_output_dim, num_inducing_points)
            u = pyro.sample("u", dist.MultivariateNormal(
                loc = u_loc,
                covariance_matrix = u_cov
            ).to_event(u_loc.dim() - 1))

        if y is not None:
            with pyro.plate("obs", size = X.shape[0], subsample_size = self.subsample_size) as index:
                # (subsample_size, num_input_dim)
                X = X.index_select(0, index)
                # (num_output_dim, subsample_size)
                y = y.index_select(1, index)
                # (num_output_dim, num_data_points)
                f = self.sample_f(X, u, Kuu)
                # (num_output_dim, num_data_points)
                y = self.likelihood(f, y)
        else:
            # (num_output_dim, num_data_points)
            f = self.sample_f(X, u, Kuu)
            # (num_output_dim, num_data_points)
            y = self.likelihood(f, y)

    @pyro_method
    def guide(self, X, y = None):
        # X.shape = (num_data_points, num_input_dim)
        # y.shape = (num_data_points, num_output_dim)

        # (num_output_dim, num_inducing_points, num_inducing_points)
        Kuu = self.kernel(self.Z, self.Z)
        Kuu = Kuu.repeat([self.num_output_dim, 1, 1])
        Kuu = Kuu + torch.eye(self.Z.shape[0]).repeat([self.num_output_dim, 1, 1]) * self.jitter

        # (num_output_dim, num_inducing_points)
        u = pyro.sample("u", dist.MultivariateNormal(
            loc = self.u_loc,
            covariance_matrix = self.u_cov
        ).to_event(self.u_loc.dim() - 1))

        if self.whiten:
            # (num_output_dim, num_inducing_points, num_inducing_points)
            Luu = torch.linalg.cholesky(Kuu)
            # (num_output_dim, 1, num_inducing_points)
            #   * (num_output_dim, num_inducing_points, num_inducing_points)
            # = (num_output_dim, 1, num_inducing_points)
            u = u.unsqueeze(1) @ Luu.transpose(1, 2)
            # (num_output_dim, num_inducing_points)
            u = u.squeeze(1)

        if y is not None:
            with pyro.plate("obs", size = X.shape[0], subsample_size = self.subsample_size) as index:
                # (subsample_size, num_input_dim)
                X = X.index_select(0, index)
                # (num_output_dim, num_data_points)
                f = self.sample_f(X, u, Kuu)
        else:
            # (num_output_dim, num_data_points)
            f = self.sample_f(X, u, Kuu)

    def sample_f(self, X, u, Kuu):
        # X.shape = (num_data_points, num_input_dim)
        # u.shape = (num_output_dim, num_inducing_points)

        # (num_output_dim, num_inducing_points, num_inducing_points)
        Kuu_inv = Kuu.inverse()
        # (num_output_dim, num_inducing_points, num_data_points)
        Kuf = self.kernel(self.Z, X)
        Kuf = Kuf.repeat([self.num_output_dim, 1, 1])
        # (num_output_dim, num_data_points, num_data_points)
        Kff = self.kernel(X, X)
        Kff = Kff.repeat([self.num_output_dim, 1, 1])

        # (num_output_dim, 1, num_inducing_points)
        #   * (num_output_dim, num_inducing_points, num_inducing_points)
        #   * (num_output_dim, num_inducing_points, num_data_points)
        # = (num_output_dim, 1, num_data_points)
        f_loc = u.unsqueeze(1) @ Kuu_inv @ Kuf
        # (num_output_dim, num_data_points)
        f_loc = f_loc.squeeze(1)
        # (num_output_dim, num_data_points, num_inducing_points)
        #   * (num_output_dim, num_inducing_points, num_inducing_points)
        #   * (num_output_dim, num_inducing_points, num_data_points)
        # = (num_output_dim, num_data_points, num_data_points)
        f_cov = Kff - Kuf.transpose(1, 2) @ Kuu_inv @ Kuf
        # (num_output_dim, num_data_points)
        f_var = f_cov.diagonal(0, 2)
        # (num_output_dim, num_data_points)
        f = pyro.sample("f", dist.Normal(
            loc = f_loc,
            scale = f_var.sqrt()
        ).to_event(f_loc.dim()))

        return f

    def forward(self, X, num_samples):
        # X.shape = (num_data_points, num_input_dim)

        # samples["f"].shape = (num_output_dim, num_data_points)
        # samples["y"].shape = (num_output_dim, num_data_points)
        predictive = Predictive(
            model = self.model,
            guide = self.guide,
            num_samples = num_samples,
            return_sites = ["f", "y"]
        )
        samples = predictive.get_samples(X)

        # (num_data_points, num_output_dim), (num_data_points, num_output_dim)
        return samples["f"].transpose(-1, -2), samples["y"].transpose(-1, -2)
