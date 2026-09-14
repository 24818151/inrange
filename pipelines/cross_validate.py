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

from surrogate.feature_engineering import extract_raw_features, generate_physics_priors, build_feature_tensor
from surrogate.botorch_gpr import build_independent_gps, fit_mll, predict

def run_cv():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    train_path = ROOT_DIR / 'train.csv'
    train_df = pd.read_csv(train_path)
    
    print("Extracting features and generating physics priors with session wind (takes ~30s)...")
    feat_df = extract_raw_features(train_df)
    priors_df = generate_physics_priors(train_df)
    
    target_cols = ['launch_spin_rate', 'apex_t', 'apex_x', 'apex_y', 'apex_z', 
                   'landing_t', 'landing_x', 'landing_y', 'landing_z']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros((len(train_df), len(target_cols)))
    
    fold = 1
    for train_idx, val_idx in kf.split(train_df):
        print(f"\n--- Fold {fold} ---")
        train_feat, val_feat = feat_df.iloc[train_idx], feat_df.iloc[val_idx]
        train_prior, val_prior = priors_df.iloc[train_idx], priors_df.iloc[val_idx]
        
        train_y = train_df.iloc[train_idx][target_cols].values
        
        X_train, scaler = build_feature_tensor(train_feat, train_prior)
        X_val, _ = build_feature_tensor(val_feat, val_prior, scaler)
        Y_train = torch.tensor(train_y, dtype=torch.float64)
        
        model, mll = build_independent_gps(X_train, Y_train, scaler['cols'], target_cols)
        
        t0 = time.time()
        fit_mll(mll)
        print(f"Fit completed in {time.time()-t0:.2f}s")
        
        mean, _ = predict(model, X_val)
        all_preds[val_idx] = mean.numpy()
        fold += 1
        
    print("\n--- Cross Validation Results ---")
    y_true = train_df[target_cols].values
    rmse = np.sqrt(np.mean((y_true - all_preds)**2, axis=0))
    mae = np.mean(np.abs(y_true - all_preds), axis=0)
    
    res_df = pd.DataFrame({'Target': target_cols, 'RMSE': rmse, 'MAE': mae})
    print(res_df.to_string(index=False))
    
    std_y = np.std(y_true, axis=0)
    scaled_rmse = rmse / std_y
    print(f"\nApproximate Composite Score (Mean Scaled RMSE): {np.mean(scaled_rmse):.4f}")

if __name__ == '__main__':
    run_cv()
