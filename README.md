# DT-MPC

Implementation of differentiable tube-based MPC controller based on the 2024 Robotics: Science and Systems paper *Differentiable Robust Model Predictive Control*.

## Initial repository structure

- `/dynamics`: base plant dynamics and safety-embedded dynamics wrappers
- `/solvers`: optimal control solver interfaces
- `/learning`: DOC engine and DT-MPC training loop interfaces
- `/scripts`: runnable entry points
- `/tests`: smoke tests for the initial package structure

## Dependencies

- `numpy==2.4.4`

## Setup
Before running the script, setup the repo using:

cd DT-MPC
pip install -e .

Need to add the conda environment as an exception to Windows Security > Virus & thread protection > Exceptions because of JAX.
## test the basic dubin scene
python .\tests\test_dubin_trajectory.py

## test the embedded state scene
python .\tests\test_embedded_state.py

## run DT MPC

python tests/test_dt_mpc_learning.py          # FD check + forest run + plots

python scripts/run_training.py                 # compact small-course demo
