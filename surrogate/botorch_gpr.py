import torch
import gpytorch
import logging
import warnings
from botorch.models import KroneckerMultiTaskGP, SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel
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

def fit_mll(mll: gpytorch.mlls.MarginalLogLikelihood) -> None:
    """Fit MLL using L-BFGS-B with Adam fallback."""
    mll.train()
    if hasattr(mll, 'model'):
        mll.model.train()
    if hasattr(mll, 'likelihood'):
        mll.likelihood.train()
        
    with gpytorch.settings.lazily_evaluate_kernels(False), gpytorch.settings.cholesky_jitter(1e-3):
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            logger.warning(f"[Model] fit_gpytorch_mll failed: {e}. Falling back to Adam optimizer.")
            model = mll.model if hasattr(mll, 'model') else mll.models[0]
            
            # Re-assert train mode, as fit_gpytorch_mll can leave the model in eval mode on failure
            mll.train()
            model.train()
            if hasattr(mll, 'likelihood'):
                mll.likelihood.train()
            
            optimizer = torch.optim.Adam(mll.parameters(), lr=0.05)
            
            for _ in range(150):
                optimizer.zero_grad()
                if isinstance(mll, SumMarginalLogLikelihood):
                    output = model(*model.train_inputs)
                    loss = -mll(output, model.train_targets)
                else:
                    output = model(*model.train_inputs) # type: ignore
                    try:
                        loss = -mll(output, model.train_targets) # type: ignore
                    except ValueError:
                        _recover_prior_support_raw_parameters(model)
                        _recover_prior_support_raw_parameters(mll.likelihood)
                        output = model(*model.train_inputs) # type: ignore
                        loss = -mll(output, model.train_targets) # type: ignore
                        
                loss.sum().backward()
                optimizer.step()

    if hasattr(mll, 'model'):
        mll.model.eval()
    if hasattr(mll, 'likelihood'):
        mll.likelihood.eval()

def build_independent_gps(train_X, train_Y, feature_cols, target_cols):
    """
    Builds an ensemble of independent SingleTaskGPs, one per target.
    This is extremely stable and handles N=491 instantly.
    """
    models = []
    
    # Map targets to their corresponding ODE prior feature indices
    target_to_ode = {
        'apex_t': 'ode_apex_t', 'apex_x': 'ode_apex_x', 'apex_y': 'ode_apex_y', 'apex_z': 'ode_apex_z',
        'landing_t': 'ode_landing_t', 'landing_x': 'ode_landing_x', 'landing_y': 'ode_landing_y', 'landing_z': 'ode_landing_z'
    }
    
    for i, target in enumerate(target_cols):
        y_col = train_Y[:, i:i+1]
        
        # Decide on mean module
        if target in target_to_ode:
            ode_idx = feature_cols.index(target_to_ode[target])
            mean_module = PhysicsFlightMean(ode_feature_index=ode_idx)
        else:
            mean_module = ConstantPhysicsMean()
            
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_X.shape[1]))
        
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
    """
    Builds a KroneckerMultiTaskGP over all targets jointly.
    Note: BoTorch KroneckerMTGP uses a single mean function for all tasks by default,
    which limits our ability to use per-task physics priors easily.
    We will stick to independent GPs (ModelListGP) which gives the same predictions 
    but allows custom physics means per target output.
    """
    model = KroneckerMultiTaskGP(
        train_X, train_Y,
        outcome_transform=Standardize(m=train_Y.shape[1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll

def predict(model, test_X):
    """Returns posterior mean and variance."""
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(test_X)
        return posterior.mean, posterior.variance
