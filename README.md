<a id="table-of-contents"></a>

## 📚 Table of Contents
- [🤖SQL-IPDC (Shared Q-Learning + Iterative PD)](#sql-ipdc)
  - [1️⃣ Method](#sql-method)
  - [2️⃣ Getting Started](#sql-getting-started)
  - [3️⃣ Repository Structure](#sql-structure)
  - [4️⃣ Instructions](#sql-instructions)
- [🤖MPC](#mpc-panda)
  - [1️⃣ Method](#mpc-method)
  - [2️⃣ Getting Started](#mpc-getting-started)
  - [3️⃣ Repository Structure](#mpc-structure)
  - [4️⃣ Instructions](#mpc-instructions)

----

# 🤖 Shared Q-Learning<br>Based Iterative Controller Design
# 1️⃣ Method
This repository integrates ***multi-agent shared Q-Learning*** for grid-based path planning with an offline, ***iterative PD gain tuning*** scheme for accurate path tracking by multiple Franka Emika Panda manipulators in ***PyBullet***.
Agents periodically share the best-performing Q-table to accelerate learning; on the control side, PD gains (Kp, Kd) are tuned offline using RMSE as the performance metric. The framework decouples learning (planning) from control (tracking), which eliminates online optimization during control runtime and simplifies deployment.
****
reference
# 2️⃣ Getting Started
## Required Packages
- Python 3.10.12
- Python packages:
  
  ✅ numpy

  ✅ matplotlib

  ✅ pybullet 3.2.7
 
## 🔧 Installation
```bash
git clone https://github.com/DSCL-CHO/Shared_Q-Learning_Based_Iterative_Controller_Design.git
cd Shared_Q-Learning_Based_Iterative_Controller_Design/SQL-IPDC
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
# 4️⃣ Instructions
Below is a step-by-step guide to each folder and how to run the corresponding scripts.
## 4.1 common/
**`gridworld.py`**  — Defines the reward function.
## 4.2 q_learning/
**`train_q_learning.py`**  — Runs ***shared Q-learning*** across multiple agents.

Produces/plots each agent’s cumulative reward curves (and logs).
```bash
cd q_learning
python train_q_learning.py
```
**`extract_path.py `** — Extracts the optimal path from the learned Q-table and prints/exports the result.

```bash
cd q_learning
python extract_path.py
```
## 4.3 kp_then_kd/
**`01_tune_kp.py`** — Fix ***Kd*** and ***tune Kp first*** based on tracking error metrics (e.g., RMSE).
**`02_tune_kd.py`** — Using the best Kp, ***tune Kd*** to further reduce the error.

```bash
cd kp_then_kd
python 01_tune_kp.py
python 02_tune_kd.py
```
## 4.4 kd_then_kp/
**`01_tune_kd.py`** — Fix ***Kp*** and ***tune Kd first*** based on tracking error metrics (e.g., RMSE).
**`02_tune_kp.py`** — Using the best Kd, ***tune Kp*** to further reduce the error.

```bash
cd kd_then_kp/
python 01_tune_kd.py
python 02_tune_kp.py
```
****
# 🤖 MPC Panda
# 1️⃣ Method
The proposed project implements joint-space **model predictive control (MPC)** for the 7-DoF Franka Emika Panda manipulator. The plant is modeled in continuous time, and a discrete-time prediction model is obtained by integrating the dynamics with a **fourth-order Runge–Kutta (RK4)**. At each sampling instant, the controller solves a finite-horizon optimal control problem to compute joint torques that track a prescribed sequence of reference joint angles while satisfying joint-position and joint-velocity constraints (with optional torque limits). The optimization is formulated in ***CasADi*** and solved with ***IPOPT***. Simulations demonstrate accurate tracking of both constant and time-varying references under these constraints.
****
reference
# 2️⃣ Getting Started
## Required Packages
- Python 3.10.12
- Python packages:

  ✅ matplotlib
  
  ✅ numpy

  ✅ CasADi 3.6.3 (https://web.casadi.org/get/)

## 🔧 Installation
```bash
git clone https://github.com/DSCL-CHO/Shared_Q-Learning_Based_Iterative_Controller_Design.git
cd Shared_Q-Learning_Based_Iterative_Controller_Design/MPC
```
## 🔧 Install Python dependencies (e.g., via pip)
### Quick
```bash
pip install numpy matplotlib "casadi==3.6.3"
```
### Recommended (virtual environment)
```bash
python3 -m venv .venv
source .venv/bin/activate     
pip install --upgrade pip
pip install numpy matplotlib "casadi==3.6.3"
```
# 3️⃣ Repository Structure
- MPC_Panda/
  - MPC/
      - main.py
      - plant.py
      - get_M.py
      - get_C.py
      - get_G.py   
- README.md
# 4️⃣ Instructions
Below is a step-by-step guide to each folder and how to run the corresponding scripts.

**`main.py`**  — Run MPC simulation.
- ⏱️ `sim_T` — total simulation time
- ⏳ `sim_dt` — simulation step
- ⏲️ `mpc_dt` — MPC control step
- 🔭 `N_horizon` — prediction horizon

```bash
# Run
python main_panda.py
```
