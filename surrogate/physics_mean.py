import torch
import gpytorch
from typing import Optional

class PhysicsFlightMean(gpytorch.means.Mean):
    """
    A custom prior mean that uses the ODE trajectory prediction as the base,
    allowing the GP to learn a linear scaling and bias correction on top of it.
    
    Formula: mu(x) = scale * x_ode + bias
    """
    def __init__(self, ode_feature_index: int, init_scale: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        self.ode_feature_index = ode_feature_index
        
        self.register_parameter(name="scale", parameter=torch.nn.Parameter(torch.tensor(init_scale, dtype=torch.float64)))
        self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.tensor(init_bias, dtype=torch.float64)))

    def forward(self, x: torch.Tensor):
        # x is the normalized feature tensor
        # We need the UNNORMALIZED ODE prediction, but x is normalized.
        # However, the target Y is also standardized by BoTorch's Standardize transform.
        # So we just learn the mapping from the normalized ODE feature to the standardized target!
        
        ode_val = x[..., self.ode_feature_index]
        return self.scale * ode_val + self.bias

class ConstantPhysicsMean(gpytorch.means.Mean):
    """
    Fallback for targets like launch_spin_rate which don't have a direct ODE output.
    Uses a standard constant mean.
    """
    def __init__(self):
        super().__init__()
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64)))

    def forward(self, x: torch.Tensor):
        return self.constant.expand(x.shape[:-1])
