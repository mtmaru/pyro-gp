import pyro
from pyro.nn import PyroModule
import pyro.distributions as dist

class Binary(PyroModule):
    def __init__(self):
        super().__init__()

    def forward(self, f, y = None):
        # f.shape = (num_output_dim, num_data_points)
        # y.shape = (num_output_dim, num_data_points)

        # (num_output_dim, num_newdata_points)
        y = pyro.sample(self._pyro_get_fullname("y"), dist.Bernoulli(logits = f).to_event(f.dim()), obs = y)

        return y
