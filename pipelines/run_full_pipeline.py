import pandas as pd
import numpy as np
import torch
import time
import sys
sys.path.insert(0, r'C:\GitHub\Mode-Matching\Python')

from inrange.surrogate.feature_engineering import extract_raw_features, generate_physics_priors, build_feature_tensor
from inrange.surrogate.botorch_gpr import build_independent_gps, fit_mll, predict

def generate_submission():
    print("Loading data...")
    train_df = pd.read_csv(r'C:\GitHub\Mode-Matching\Python\inrange\train.csv')
    test_df = pd.read_csv(r'C:\GitHub\Mode-Matching\Python\inrange\test.csv')
    sample_sub = pd.read_csv(r'C:\GitHub\Mode-Matching\Python\inrange\sample_submission.csv')
    
    print("Extracting features (Train/Test)...")
    train_feat = extract_raw_features(train_df)
    test_feat = extract_raw_features(test_df)
    
    print("Estimating Session Wind and Generating physics priors (Pass 1 & Pass 2, takes ~45s)...")
    train_prior, test_prior = generate_physics_priors(train_df, test_df)
    
    target_cols = ['launch_spin_rate', 'apex_t', 'apex_x', 'apex_y', 'apex_z', 
                   'landing_t', 'landing_x', 'landing_y', 'landing_z']
                   
    train_y = train_df[target_cols].values
    
    X_train, scaler = build_feature_tensor(train_feat, train_prior)
    X_test, _ = build_feature_tensor(test_feat, test_prior, scaler)
    Y_train = torch.tensor(train_y, dtype=torch.float64)
    
    print("Building and fitting Independent GPs on FULL train set...")
    model, mll = build_independent_gps(X_train, Y_train, scaler['cols'], target_cols)
    
    t0 = time.time()
    fit_mll(mll)
    t1 = time.time()
    print(f"Fit completed in {t1-t0:.2f}s")
    
    print("Predicting on test set...")
    mean, var = predict(model, X_test)
    preds = mean.numpy()
    
    print("Writing submission.csv...")
    submission = sample_sub.copy()
    for i, col in enumerate(target_cols):
        submission[col] = preds[:, i]
        
    out_path = r'C:\GitHub\Mode-Matching\Python\inrange\submission.csv'
    submission.to_csv(out_path, index=False)
    print(f"Success! Saved to {out_path}")

if __name__ == '__main__':
    generate_submission()
