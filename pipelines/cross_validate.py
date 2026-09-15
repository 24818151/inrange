import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
import time

# Ensure repository root is on sys.path for direct imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from surrogate.feature_engineering import (
    extract_raw_features, 
    generate_physics_priors, 
    build_feature_tensor,
    build_cascaded_stage2_tensor
)
from surrogate.botorch_gpr import build_independent_gps, fit_mll, predict

def run_cv(num_restarts: int = 20):
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    train_path = ROOT_DIR / 'train.csv'
    train_df = pd.read_csv(train_path)
    
    print("Extracting features and generating physics priors with session wind (un-throttled ODE)...")
    t_prior_start = time.time()
    feat_df = extract_raw_features(train_df)
    priors_df = generate_physics_priors(train_df)
    print(f"Physics prior extraction completed in {time.time() - t_prior_start:.2f}s")
    
    stage1_targets = ['launch_spin_rate', 'apex_t', 'apex_x', 'apex_y', 'apex_z']
    stage2_targets = ['landing_t', 'landing_x', 'landing_y', 'landing_z']
    all_targets = stage1_targets + stage2_targets
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros((len(train_df), len(all_targets)))
    
    fold = 1
    for train_idx, val_idx in kf.split(train_df):
        print(f"\n==================== Fold {fold} / 5 ====================")
        train_feat, val_feat = feat_df.iloc[train_idx], feat_df.iloc[val_idx]
        train_prior, val_prior = priors_df.iloc[train_idx], priors_df.iloc[val_idx]
        
        # ----------------------------------------------------
        # Stage 1: Predict Spin Rate and Apex State (t, x, y, z)
        # ----------------------------------------------------
        print(f"[Fold {fold}] Stage 1: Fitting Spin & Apex GPs ({num_restarts} restarts)...")
        X_train_s1, scaler_s1 = build_feature_tensor(train_feat, train_prior)
        X_val_s1, _ = build_feature_tensor(val_feat, val_prior, scaler_s1)
        
        Y_train_s1 = torch.tensor(train_df.iloc[train_idx][stage1_targets].values, dtype=torch.float64)
        
        model_s1, mll_s1 = build_independent_gps(X_train_s1, Y_train_s1, scaler_s1['cols'], stage1_targets)
        
        t0 = time.time()
        fit_mll(mll_s1, num_restarts=num_restarts)
        print(f"[Fold {fold}] Stage 1 fit completed in {time.time()-t0:.2f}s")
        
        pred_train_s1, _ = predict(model_s1, X_train_s1)
        pred_val_s1, _ = predict(model_s1, X_val_s1)
        all_preds[val_idx, :len(stage1_targets)] = pred_val_s1.numpy()
        
        # ----------------------------------------------------
        # Stage 2: Cascaded Landing GPs conditioned on predicted Apex
        # ----------------------------------------------------
        print(f"[Fold {fold}] Stage 2: Fitting Landing GPs conditioned on predicted Apex ({num_restarts} restarts)...")
        X_train_s2, scaler_s2, cols_s2 = build_cascaded_stage2_tensor(
            X_train_s1, pred_train_s1.numpy(), scaler_s1
        )
        X_val_s2, _, _ = build_cascaded_stage2_tensor(
            X_val_s1, pred_val_s1.numpy(), scaler_s1, scaler_s2
        )
        
        Y_train_s2 = torch.tensor(train_df.iloc[train_idx][stage2_targets].values, dtype=torch.float64)
        
        model_s2, mll_s2 = build_independent_gps(X_train_s2, Y_train_s2, cols_s2, stage2_targets)
        
        t1 = time.time()
        fit_mll(mll_s2, num_restarts=num_restarts)
        print(f"[Fold {fold}] Stage 2 fit completed in {time.time()-t1:.2f}s")
        
        pred_val_s2, _ = predict(model_s2, X_val_s2)
        all_preds[val_idx, len(stage1_targets):] = pred_val_s2.numpy()
        
        fold += 1
        
    print("\n==================== Cross Validation Results ====================")
    y_true = train_df[all_targets].values
    rmse = np.sqrt(np.mean((y_true - all_preds)**2, axis=0))
    mae = np.mean(np.abs(y_true - all_preds), axis=0)
    
    res_df = pd.DataFrame({'Target': all_targets, 'RMSE': rmse, 'MAE': mae})
    print(res_df.to_string(index=False))
    
    std_y = np.std(y_true, axis=0)
    scaled_rmse = rmse / std_y
    print(f"\nApproximate Composite Score (Mean Scaled RMSE): {np.mean(scaled_rmse):.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross Validation for the Cascaded GPR.")
    parser.add_argument('--restarts', type=int, default=20, help="Number of restarts for the L-BFGS-B GPR multi-start optimization (default: 20).")
    args = parser.parse_args()
    
    run_cv(num_restarts=args.restarts)
