# ⛳ From QRFH Antennas to Golf Trajectories: A Physics-Informed GPR Approach

This repository contains my solution for the **Inrange Student Competition** (Kaggle). The goal is to predict the hidden parameters of a golf shot—`launch_spin_rate`, `apex`, and `landing` coordinates—given only the launch conditions and the first 4 downrange radar checkpoints (truncated at 60 metres).

Leveraging the exact same Physics-Informed Machine Learning (PIML) philosophy I use for electromagnetic surrogate modelling of Quad-Ridge Flared Horn (QRFH) antennas, I replaced my computational electromagnetic (CEM) solver with an aerodynamic ODE solver and bridged the two domains.

## 🏗️ Architecture

The solution uses a hybrid **Cascaded Gaussian Process Regression (GPR)** approach to achieve highly accurate, physically constrained extrapolations of the golf ball's trajectory:

1. **Phase 1: Deterministic 3D Aerodynamic ODE (`physics/`)**
   - Models velocity-dependent drag, spin-decaying Magnus lift, and ambient horizontal session wind.
   - Fits the aerodynamic coefficients ($C_d$, $C_{l0}$) using a multi-start L-BFGS-B inverse solver across three distinct aerodynamic regimes (driver, mid-iron, wedge) on the observed 60m radar checkpoints.
   - Includes full post-impact **Bounce & Roll** mechanics for a holistic terminal rest simulation.

2. **Phase 2: Cascaded Surrogate Model (`surrogate/`)**
   - Extracts the ODE's forward-integrated trajectory as a `PhysicsFlightMean` structural prior.
   - **Stage 1 GPs:** Predict the exact launch spin and 4D apex coordinates.
   - **Stage 2 GPs:** Predict the landing coordinates, *conditioned directly on the Stage 1 predicted apex*. This mathematically converts a severe extrapolation problem into a short-range descent interpolation.

## 📊 Final Performance (5-Fold CV)
* **Composite Scaled RMSE:** 0.1630
* **Landing Placement Error:** ~2.79m (Lateral MAE), ~3.14m (Depth MAE)
* **Physical Integrity:** An automated audit of the final `submission.csv` reveals **0% violations** of physical invariants (e.g., apex strictly precedes landing, spin is positive).

## 🎮 Interactive Physics Dashboard (`app.py`)
To visualise the underlying mechanics, this repository includes a lightweight Streamlit dashboard that renders any dataset shot in interactive 3D, tracing the launch, radar checkpoints, 60m boundary net, apex, and simulated post-impact bounce & roll. 

*Note: To remain lightweight for web hosting, this dashboard deliberately bypasses the heavy PyTorch/GPR machine learning dependencies. It runs **only** the Phase 1 deterministic aerodynamic ODE inverse solver. This intentionally isolates and demonstrates the raw strength of the structural physics prior before the Cascaded GPR corrects the final residuals.*

To launch the 3D physics simulator locally:
```bash
streamlit run app.py
```

## 📂 Repository Structure

* `app.py` - Streamlit interactive physics dashboard with 3D Plotly animations.
* `requirements.txt` - Python dependencies for the cloud environment and ML pipeline.
* `physics/` - The core ODE physics engine, trajectory integrators, and wind estimators.
* `surrogate/` - GPyTorch models, custom Mean modules, and feature engineering.
* `pipelines/` - Execution scripts for 5-Fold Cross Validation and final Kaggle submission generation.
* `analysis/` - Trajectory visualisation, EDA tools, and figure exporter (`export_figures.py`).
* `assets/` - Publication-ready high-DPI figures for the competition writeup report.
* `LaTeX/` - The compiled academic report detailing the mathematics and covariance kernels.

## 🚀 Getting Started

Ensure you have Python 3.11+ installed.
```bash
# Clone the repository
git clone https://github.com/[Your-Username]/inrange.git
cd inrange

# Install dependencies
pip install -r requirements.txt
```

### Generating the Kaggle Submission
To run the full two-stage Cascaded GPR pipeline:
```bash
python pipelines/run_full_pipeline.py
```
This will output a `submission.csv` ready for upload. To evaluate the model locally, run `python pipelines/cross_validate.py`.
