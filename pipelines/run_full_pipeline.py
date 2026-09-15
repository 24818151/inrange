import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
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

def generate_submission(num_restarts: int = 20):
    # Set seeds for deterministic reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    train_path = ROOT_DIR / 'train.csv'
    test_path = ROOT_DIR / 'test.csv'
    sample_sub_path = ROOT_DIR / 'sample_submission.csv'
    out_path = ROOT_DIR / 'submission.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)
    
    print("Extracting features (Train/Test)...")
    train_feat = extract_raw_features(train_df)
    test_feat = extract_raw_features(test_df)
    
    print("Estimating Session Wind and Generating physics priors (Un-throttled ODE, 3-start search)...")
    t_priors = time.time()
    train_prior, test_prior = generate_physics_priors(train_df, test_df)
    print(f"Physics priors generated in {time.time() - t_priors:.2f}s")
    
    stage1_targets = ['launch_spin_rate', 'apex_t', 'apex_x', 'apex_y', 'apex_z']
    stage2_targets = ['landing_t', 'landing_x', 'landing_y', 'landing_z']
    
    # -------------------------------------------------------------------------
    # Stage 1: Train Spin Rate and Apex State (t, x, y, z) on Full Train Set
    # -------------------------------------------------------------------------
    print(f"\n[Stage 1] Fitting Spin & Apex GPs on full train set ({num_restarts} restarts)...")
    X_train_s1, scaler_s1 = build_feature_tensor(train_feat, train_prior)
    X_test_s1, _ = build_feature_tensor(test_feat, test_prior, scaler_s1)
    
    Y_train_s1 = torch.tensor(train_df[stage1_targets].values, dtype=torch.float64)
    
    model_s1, mll_s1 = build_independent_gps(X_train_s1, Y_train_s1, scaler_s1['cols'], stage1_targets)
    
    t0 = time.time()
    fit_mll(mll_s1, num_restarts=num_restarts)
    print(f"[Stage 1] Completed in {time.time() - t0:.2f}s")
    
    print("[Stage 1] Predicting Apex & Spin on Train and Test sets...")
    pred_train_s1, _ = predict(model_s1, X_train_s1)
    pred_test_s1, _ = predict(model_s1, X_test_s1)
    
    # -------------------------------------------------------------------------
    # Stage 2: Cascaded Landing GPs conditioned on predicted Apex State
    # -------------------------------------------------------------------------
    print(f"\n[Stage 2] Constructing Cascaded Features and Fitting Landing GPs ({num_restarts} restarts)...")
    X_train_s2, scaler_s2, cols_s2 = build_cascaded_stage2_tensor(
        X_train_s1, pred_train_s1.numpy(), scaler_s1
    )
    X_test_s2, _, _ = build_cascaded_stage2_tensor(
        X_test_s1, pred_test_s1.numpy(), scaler_s1, scaler_s2
    )
    
    Y_train_s2 = torch.tensor(train_df[stage2_targets].values, dtype=torch.float64)
    
    model_s2, mll_s2 = build_independent_gps(X_train_s2, Y_train_s2, cols_s2, stage2_targets)
    
    t1 = time.time()
    fit_mll(mll_s2, num_restarts=num_restarts)
    print(f"[Stage 2] Completed in {time.time() - t1:.2f}s")
    
    print("[Stage 2] Predicting Landing state on Test set...")
    pred_test_s2, _ = predict(model_s2, X_test_s2)
    
    # -------------------------------------------------------------------------
    # Assemble Final Submission
    # -------------------------------------------------------------------------
    print("\nAssembling submission.csv...")
    submission = sample_sub.copy()
    
    test_s1_np = pred_test_s1.numpy()
    test_s2_np = pred_test_s2.numpy()
    
    for i, col in enumerate(stage1_targets):
        submission[col] = test_s1_np[:, i]
    for i, col in enumerate(stage2_targets):
        submission[col] = test_s2_np[:, i]
        
    submission.to_csv(out_path, index=False)
    print(f"Success! Saved to {out_path}")

    # Physical assertion checks
    assert submission.shape == sample_sub.shape
    assert list(submission.columns) == list(sample_sub.columns)
    assert submission.isnull().sum().sum() == 0
    assert (submission['apex_t'] < submission['landing_t']).all(), "Physical violation: apex_t >= landing_t"
    assert (submission['launch_spin_rate'] > 0).all(), "Physical violation: non-positive spin"
    print("All physical integrity and schema checks passed 100%.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the full 2-Stage Cascaded GPR pipeline.")
    parser.add_argument('--restarts', type=int, default=20, help="Number of restarts for the L-BFGS-B GPR multi-start optimization (default: 20).")
    args = parser.parse_args()
    
    generate_submission(num_restarts=args.restarts)
