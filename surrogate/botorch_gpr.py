import torch
import gpytorch
import logging
import warnings
import numpy as np
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel, LinearKernel
from gpytorch.priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood

from .physics_mean import PhysicsFlightMean, ConstantPhysicsMean

logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float64)

def _recover_prior_support_raw_parameters(module: torch.nn.Module, min_raw: float = -12.0) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "raw_constant" in name or "mean_module" in name:
                continue
            if ".raw_" not in name and not name.startswith("raw_"):
                continue
            param.clamp_(min=min_raw)

def _fit_single_mll_with_restarts(sub_mll: ExactMarginalLogLikelihood, num_restarts: int = 5) -> None:
    """
    Fits a single ExactMarginalLogLikelihood using multi-start L-BFGS-B with Adam fallback.
    Retains the model state dict achieving the lowest negative marginal log likelihood.
    """
    best_loss = float('inf')
    best_state_dict = None
    
    # Save baseline initial state
    base_state = {k: v.clone() for k, v in sub_mll.state_dict().items()}
    
    for restart in range(max(1, num_restarts)):
        sub_mll.train()
        sub_mll.model.train()
        if hasattr(sub_mll, 'likelihood'):
            sub_mll.likelihood.train()
            
        if restart > 0:
            # Re-seed with smooth perturbation around initial parameter space
            with torch.no_grad():
                for name, param in sub_mll.named_parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * 0.15
                        param.add_(noise)
                        
        with gpytorch.settings.lazily_evaluate_kernels(False), gpytorch.settings.cholesky_jitter(1e-3):
            try:
                fit_gpytorch_mll(
                    sub_mll,
                    optimizer_kwargs={
                        "options": {
                            "maxiter": 1000,
                            "ftol": 1e-9,
                            "gtol": 1e-6
                        }
                    }
                )
            except Exception as e:
                # Robust Adam fallback if L-BFGS-B terminates abnormally on this restart
                sub_mll.train()
                sub_mll.model.train()
                if hasattr(sub_mll, 'likelihood'):
                    sub_mll.likelihood.train()
                    
                optimizer = torch.optim.Adam(sub_mll.parameters(), lr=0.03)
                for _ in range(120):
                    optimizer.zero_grad()
                    output = sub_mll.model(*sub_mll.model.train_inputs)
                    try:
                        loss = -sub_mll(output, sub_mll.model.train_targets)
                    except ValueError:
                        _recover_prior_support_raw_parameters(sub_mll.model)
                        _recover_prior_support_raw_parameters(sub_mll.likelihood)
                        output = sub_mll.model(*sub_mll.model.train_inputs)
                        loss = -sub_mll(output, sub_mll.model.train_targets)
                        
                    loss.sum().backward()
                    optimizer.step()

        # Evaluate MLL loss
        sub_mll.eval()
        with torch.no_grad():
            output = sub_mll.model(*sub_mll.model.train_inputs)
            try:
                curr_loss = -sub_mll(output, sub_mll.model.train_targets).item()
            except Exception:
                curr_loss = float('inf')
                
        if curr_loss < best_loss and not np.isnan(curr_loss):
            best_loss = curr_loss
            best_state_dict = {k: v.clone() for k, v in sub_mll.state_dict().items()}
            
    if best_state_dict is not None:
        sub_mll.load_state_dict(best_state_dict)
    sub_mll.eval()

def fit_mll(mll: gpytorch.mlls.MarginalLogLikelihood, num_restarts: int = 5) -> None:
    """
    Fits MLL across all models using multi-start L-BFGS-B with Adam fallbacks.
    For ModelListGP / SumMarginalLogLikelihood, optimizes each sub-model independently.
    """
    if isinstance(mll, SumMarginalLogLikelihood):
        for idx, sub_mll in enumerate(mll.mlls):
            _fit_single_mll_with_restarts(sub_mll, num_restarts=num_restarts)
    else:
        _fit_single_mll_with_restarts(mll, num_restarts=num_restarts)

def build_independent_gps(train_X, train_Y, feature_cols, target_cols):
    """
    Builds an ensemble of independent SingleTaskGPs with:
    - Custom PhysicsFlightMean priors per target channel
    - Additive Composite Kernel (Linear ballistic carrier + Matérn-5/2 local aerodynamic perturbation)
    - Informative Gamma priors regularizing lengthscales and outputscales
    """
    models = []
    
    # Map targets to their corresponding ODE prior feature indices
    target_to_ode = {
        'launch_spin_rate': 'ode_cl0',
        'apex_t': 'ode_apex_t', 'apex_x': 'ode_apex_x', 'apex_y': 'ode_apex_y', 'apex_z': 'ode_apex_z',
        'landing_t': 'ode_landing_t', 'landing_x': 'ode_landing_x', 'landing_y': 'ode_landing_y', 'landing_z': 'ode_landing_z'
    }
    
    dim = train_X.shape[1]
    
    for i, target in enumerate(target_cols):
        y_col = train_Y[:, i:i+1]
        
        # Physics-informed mean prior
        if target in target_to_ode:
            ode_idx = feature_cols.index(target_to_ode[target])
            mean_module = PhysicsFlightMean(ode_feature_index=ode_idx)
        else:
            mean_module = ConstantPhysicsMean()
            
        # Composite Kernel: Linear (global kinematics) + Matern-5/2 (aerodynamic corrections)
        lin_kernel = ScaleKernel(
            LinearKernel(),
            outputscale_prior=GammaPrior(2.0, 0.5)
        )
        matern_kernel = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=dim,
                lengthscale_prior=GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=GammaPrior(2.0, 0.5)
        )
        covar_module = lin_kernel + matern_kernel
        
        model = SingleTaskGP(
            train_X, y_col,
            mean_module=mean_module,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1)
        )
        models.append(model)
        
    model_list = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    return model_list, mll

def build_kronecker_gp(train_X, train_Y):
    model = KroneckerMultiTaskGP(
        train_X, train_Y,
        outcome_transform=Standardize(m=train_Y.shape[1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll

def predict(model, test_X):
    """Returns posterior predictive mean and variance."""
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(test_X)
        return posterior.mean, posterior.variance
