# ⛳ Inrange Golf: Physics-Informed GPR

This repository contains the solution for the **Inrange Student Competition** (Kaggle). The goal is to predict the hidden parameters of a golf shot—`launch_spin_rate`, `apex`, and `landing` coordinates—given only the launch conditions and the first 4 downrange radar checkpoints (up to 60 meters).

🔗 **[Live Interactive Demo](https://your-streamlit-url.app)** *(Replace with your Streamlit Cloud URL)*

## 🧠 Architecture

The solution uses a hybrid **Physics-Informed Gaussian Process (GPR)** approach to achieve highly accurate extrapolations of the golf ball's trajectory:

1. **Custom 3D Aerodynamic ODE (`physics/`)**
   - Models velocity-dependent drag, spin-decaying Magnus lift, and ambient horizontal wind.
   - Fits the aerodynamic coefficients ($C_d$, $C_{l0}$) and ambient wind vector per session using an L-BFGS-B inverse solver on the observed 60m radar checkpoints.
   - Includes full post-impact **Bounce & Roll** mechanics for accurate terminal distance simulation.

2. **BoTorch / GPyTorch Surrogate (`surrogate/`)**
   - Extracts the ODE's forward-integrated trajectory as a `PhysicsFlightMean` prior.
   - Trains an ensemble of Exact Gaussian Processes to learn the residual discrepancies between the idealized physics and the true targets.

## 📂 Repository Structure

* `app.py` - Streamlit interactive physics dashboard with 3D Plotly animations.
* `requirements.txt` - Python dependencies for the cloud environment.
* `physics/` - The core ODE physics engine, trajectory integrators, and wind estimators.
* `surrogate/` - GPyTorch models, custom Mean modules, and feature engineering.
* `pipelines/` - Execution scripts for 5-Fold Cross Validation and final Kaggle submission generation.
* `analysis/` - Trajectory visualization and EDA tools.

## 🚀 Getting Started

### Local Setup
Ensure you have Python 3.11+ installed.
```bash
# Clone the repository
git clone https://github.com/your-username/inrange-golf-solution.git
cd inrange-golf-solution

# Install dependencies
pip install -r requirements.txt
```

### Generating the Kaggle Submission
To run the full two-pass physics engine and fit the Gaussian Processes:
```bash
python pipelines/run_full_pipeline.py
```
This will output a `submission.csv` ready for upload. To evaluate the model locally, run `python pipelines/cross_validate.py`.

### Running the Interactive Demo
To launch the 3D physics simulator locally:
```bash
streamlit run app.py
```
