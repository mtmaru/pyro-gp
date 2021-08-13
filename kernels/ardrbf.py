import torch
from pyro.nn import PyroModule, PyroParam
from torch.distributions import constraints

class ARDRBF(PyroModule):
    def __init__(self, num_input_dim):
        super().__init__()

        self.num_input_dim = num_input_dim
        self.scale = PyroParam(torch.ones(1, 1, num_input_dim), constraints.positive)

    def forward(self, l, r):
        # (num_data_l_points, num_input_dim) -> (num_data_l_points, 1, num_input_dim)
        l = l.unsqueeze(1)
        scaled_l = l / self.scale
        # (num_data_r_points, num_input_dim) -> (1, num_data_r_points, num_input_dim)
        r = r.unsqueeze(0)
        scaled_r = r / self.scale
        # (num_data_l_points, num_data_r_points, num_input_dim)
        d2 = (scaled_l - scaled_r) ** 2
        # (num_data_l_points, num_data_r_points)
        d2 = d2.sum(axis = 2)
        # (num_data_l_points, num_data_r_points)
        k = torch.exp(-0.5 * d2)

        return k
