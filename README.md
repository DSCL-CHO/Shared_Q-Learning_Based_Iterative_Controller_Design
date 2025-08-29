# 🤖 Shared Q-Learning<br>Based Iterative Controller Design
# 1️⃣ Method
This repository integrates ***multi-agent shared Q-Learning*** for grid-based path planning with an offline, ***iterative PD gain tuning*** scheme for accurate path tracking by multiple Franka Emika Panda manipulators in ***PyBullet***.
Agents periodically share the best-performing Q-table to accelerate learning; on the control side, PD gains (Kp, Kd) are tuned offline using RMSE as the performance metric. The framework decouples learning (planning) from control (tracking), which eliminates online optimization during control runtime and simplifies deployment.
****
reference
# 2️⃣ Getting Started
## Required Packages
- Python 3.10.12
- 
  ✅ numpy

  ✅ matplotlib

  ✅ pybullet 3.2.7
## 🔧 Installation
```bash
git clone <your-repo-url>.git
cd <your-repo-name>
```
## 🔧 Install Python dependencies (e.g., via pip)
#### Quick
```bash
pip install --upgrade pip
pip install numpy pybullet pybullet_data matplotlib
```
#### Recommended (virtual environment)
```bash
python3 -m venv .venv
source .venv/bin/activate      

pip install --upgrade pip
pip install numpy pybullet pybullet_data matplotlib
```
# 3️⃣ Repository Structure
- SQL-IPDC/
  - common/
      - \_\_init\_\_.py
      - gridworld_render.py
      - gridworld.py
      - utils.py
  - q_learning/
      - \_\_init\_\_.py
      - train_q_learning.py
      - extract_path.py
  - kp_then_kd/
      - 01_tune_kp.py
      - 02_tune_kd.py
      - coord_to_xyz.py
      - panda_env.py
  - kd_then_kp/
      - 01_tune_kd.py
      - 02_tune_kp.py
      - coord_to_xyz.py
      - panda_env.py
  - README.md
****
# 🤖 MPC Panda
# 1️⃣ Method
The proposed project implements joint-space **model predictive control (MPC)** for the 7-DoF Franka Emika Panda manipulator. The plant is modeled in continuous time, and a discrete-time prediction model is obtained by integrating the dynamics with a **fourth-order Runge–Kutta (RK4)**. At each sampling instant, the controller solves a finite-horizon optimal control problem to compute joint torques that track a prescribed sequence of reference joint angles while satisfying joint-position and joint-velocity constraints (with optional torque limits). The optimization is formulated in ***CasADi*** and solved with ***IPOPT***. Simulations demonstrate accurate tracking of both constant and time-varying references under these constraints.
****
reference
# 2️⃣ Getting Started
## Required Packages
- Python 3.10.12
  
  ✅ matplotlib
  
  ✅ numpy

  ✅ CasADi 3.6.3 (https://web.casadi.org/get/)

## Installation
```bash
git clone https://github.com.~~
```
## Install Python dependencies (e.g., via pip)
### Quick
```bash
pip install numpy matplotlib "casadi==3.6.3"

# run
python panda/main_panda.py
```
### Recommended (virtual environment)
```bash
python3 -m venv .venv
source .venv/bin/activate     
pip install --upgrade pip
pip install numpy matplotlib "casadi==3.6.3"

# run
python panda/main_panda.py
```
# 3️⃣ Repository Structure
- MPC_Panda/
  - MPC/
      - main.py
      - plant.py
      - get.M.py
      - get.C.py
      - get.G.py   
  - README.md
   
