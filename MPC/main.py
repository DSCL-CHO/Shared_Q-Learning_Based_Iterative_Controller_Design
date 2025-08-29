
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib


import casadi as ca
from plant import make_dynamics        # 연속계 f(x,u) (토크기반 1-DOF 쓰는 경우)
from get_G import G_mtx
from get_M import M_mtx
from get_C import C_mtx
import math
import time

def make_rk4(f, dt):
    """
    f(x,u)를 이용한 1스텝 RK4 이산화 함수 F(x,u)=x_{k+1}
    반환: CasADi Function F(x,u)->x_next
    """
    x  = ca.MX.sym('x', 14)
    u  = ca.MX.sym('u', 7)
    u  = ca.reshape(u, (7,1))

    k1 = f(x,               u)
    k2 = f(x + 0.5*dt*k1,   u)
    k3 = f(x + 0.5*dt*k2,   u)
    k4 = f(x + dt*k3,       u)
    x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    F = ca.Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])
    return F

class TorqueNMPC:
    """
    Direct multiple shooting NMPC:
      - 상태: x = [q; dq] \in R^{14}
      - 입력: u = tau \in R^{7}
      - 제약: X_{k+1} = F_mpc(X_k, U_k) (연속계 f를 RK4로 적분)
    """
    def __init__(self, n, dt, N,
                 Qq=None, Qd=None, R=None,
                 tau_max=None):
        self.n  = n
        self.dt = float(dt)
        self.N  = int(N)

        # 가중치
        Qq = 10*np.ones(n) if Qq is None else np.array(Qq, dtype=float) # 조인트 7개 cost 
        Qd = 10*np.ones(n) if Qd is None else np.array(Qd, dtype=float)
        R  =  1*np.ones(n) if R  is None else np.array(R, dtype=float)
        self.Qq = ca.diag(ca.DM(Qq))
        self.Qd = ca.diag(ca.DM(Qd))
        self.R  = ca.diag(ca.DM(R))

        self.tau_max = None if tau_max is None else ca.DM(tau_max).reshape((n,1))

        # MPC에서 쓸 RK4(저주기)
        self.F = make_rk4(f, self.dt).expand()     # ★ 추가/교체

        opti = ca.Opti() # casadi option 
        X = opti.variable(2*n, N+1)   # 14 x (N+1)
        U = opti.variable(n,   N)     #  7 x N
        
        X0   = opti.parameter(2*n)    # 초기상태
        Xref = opti.parameter(2*n)    # 참조 [q_ref; qd_ref]

        # 초기조건
        opti.subject_to(X[:, 0] == X0) # constraint 
        # opti.subject_to(X[:, 1] == 2*X0)
        # opti.subject_to(X[0~13, 0] <= 1.57)

        # 비용 & 다단 제약
        cost = 0
        for k in range(N): # termination at k = N-1 
            xk = X[:, k]
            uk = U[:, k]
            xk1 = X[:, k+1]

            xk1_pred = self.F(xk, uk)
            opti.subject_to(xk1 == xk1_pred)

            # if self.tau_max is not None:
            #     opti.subject_to(-self.tau_max <= uk)
            #     opti.subject_to( uk <= self.tau_max)

            qk  = xk[0:n]
            qdk = xk[n:2*n]
            qr  = Xref[0:n]
            qdr = Xref[n:2*n]
            e_q  = qr - qk
            e_qd = qdr - qdk

            cost += ca.mtimes([e_q.T,  self.Qq, e_q]) \
                  + ca.mtimes([e_qd.T, self.Qd, e_qd]) \
                  + ca.mtimes([uk.T,   self.R,  uk]) # cost is positive 
            # cost += ca.mtimes([e_q.T,  self.Qq, e_q]) \
            #       + ca.mtimes([uk.T,   self.R,  uk]) # cost is positive 

        # 종단 비용 terminal cost 
        e_qN  = X[0:n,      -1] - Xref[0:n]
        e_qdN = X[n:2*n,    -1] - Xref[n:2*n]
        cost += ca.mtimes([e_qN.T,  self.Qq, e_qN]) \
              + ca.mtimes([e_qdN.T, self.Qd, e_qdN])
        # cost += ca.mtimes([e_qN.T,  self.Qq, e_qN])

        opti.minimize(cost)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

        self.opti = opti
        self.X, self.U = X, U
        self.X0, self.Xref = X0, Xref

        # warm start
        self._u_guess = None
        self._x_guess = None

    def solve(self, x0, xref):
        x0   = ca.DM(x0).reshape((2*self.n, 1))
        xref = ca.DM(xref).reshape((2*self.n, 1))

        self.opti.set_value(self.X0,   x0) # TorqueNMPC의 _init_ 에서 X0를 Parameter로 정의 했지만, x0로 setting 함
        self.opti.set_value(self.Xref, xref)

        if self._x_guess is not None:
            self.opti.set_initial(self.X, self._x_guess)
        else:
            self.opti.set_initial(self.X, ca.repmat(x0, 1, self.N+1))
        if self._u_guess is not None:
            self.opti.set_initial(self.U, self._u_guess)
        else:
            self.opti.set_initial(self.U, ca.DM.zeros(self.n, self.N))

        sol = self.opti.solve() # opti casadi 
        Uopt = np.array(sol.value(self.U))   # (n, N)
        Xopt = np.array(sol.value(self.X))   # (2n, N+1)

        # warm-start shift
        self._u_guess = np.hstack([Uopt[:,1:], Uopt[:,-1:]])
        self._x_guess = np.hstack([Xopt[:,1:], Xopt[:,-1:]])

        tau0 = Uopt[:, 0]    # 첫 스텝 토크
        # print(tau0)
        # dd
        return tau0
# 시뮬레이터/컨트롤러 주기

n = 7
sim_T   = 60
sim_dt  = 0.001
mpc_dt  = 0.01
tspan   = np.arange(0.0, sim_T + sim_dt, sim_dt)
N_horizon = 10

#________________________________________________

# 목표 (joint-space)
q_ref  = np.array([0, -np.pi/4, 0, -3 * np.pi/4, 0, np.pi/2, np.pi/4], dtype=float) # home configuration
qd_ref = np.zeros(n, dtype=float)

#________________________________________________
f      = make_dynamics(linear_solver='symbolicqr').expand()          # ★ 추가/교체
F_sim  = make_rk4(f, sim_dt).expand()      # ★ 추가/교체

mpc = TorqueNMPC(
    n=n, dt=mpc_dt, N=N_horizon,
    Qq= 10*np.ones(n),
    Qd= 10*np.ones(n),
    R = 1*np.ones(n),
    tau_max=100.0*np.ones(n)  # 필요시 조정 또는 None
)

# 초기 상태
x = np.zeros(2*n)  # [q0(7)=0, dq0(7)=0]
u_hold = np.zeros(n)

q_hist   = []
tau_hist = []
mpc_stride = int(round(mpc_dt/sim_dt))

def F_step_np(x_np, u_np):
    """
    x_np: (14,) or (14,1) numpy
    u_np: (7,)  or (7,1)  numpy
    return: (14,) numpy (다음 상태)
    """
    x_col = np.asarray(x_np, dtype=float).reshape((2*n, 1))  # (14,1)
    u_col = np.asarray(u_np, dtype=float).reshape((n, 1))    # (7,1)
    x_next_dm = F_sim(x_col, u_col)      # CasADi가 내부에서 DM로 변환
    return x_next_dm.full().reshape(-1)  # (14,) numpy
errors = []# 📏

mpc_times = []

for k in range(len(tspan)-1):
    xref = np.hstack([q_ref, qd_ref])
    if k % mpc_stride == 0:
        start = time.time()             ## mpc 기준
        u_hold = mpc.solve(x, xref)   # 연속 적분기 기반 NMPC 해 (토크)
        end = time.time()
        mpc_times.append(end - start)
        print(f"{end - start:.5f} ")
    # print(x[0:n]-q_ref)
    # 1ms 시뮬레이션 스텝 (CasADi RK4)
    x = F_step_np(x, u_hold)

# 📏
    q_now = x[0:n]
    e = q_now - q_ref
    err_norm = np.linalg.norm(e)          # 7D 오차 → 스칼라
    errors.append(err_norm)
#  📏
    q_hist.append(x[0:n].copy())
    tau_hist.append(u_hold.copy())

rmse = np.sqrt(np.mean(np.array(errors) ** 2))#  📏
print("Final RMSE (overall robot error):", rmse)#  📏


q_hist   = np.array(q_hist)
tau_hist = np.array(tau_hist)
# t = tspan[:-1]
t = tspan[1:]

# ================================================================================ 

# [MOD REPLACE] 모든 조인트(7개) 플롯: 4x2 서브플롯으로 배치
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
axes = axes.flatten()

for j in range(n):
    ax = axes[j]
    ax.plot(t, q_hist[:, j], label=f"q{j+1}")
    ax.axhline(q_ref[j], linestyle='--', label="q_ref")
    ax.set_title(f"Joint {j+1}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("q [rad]")
    ax.legend()
    ax.set_ylim(-3.14, 3.14) 
# 남는 서브플롯(8번째)을 깔끔히 제거
for k in range(n, len(axes)):
    fig.delaxes(axes[k])

fig.suptitle("Joint tracking with NMPC (torque input)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])

fig, axes1 = plt.subplots(4, 2, figsize=(12, 10))
axes1 = axes1.flatten()
for j in range(n):
    ax = axes1[j]
    ax.plot(t, tau_hist[:, j], label=f"q{j+1}")
    # ax.axhline(q_ref[j], linestyle='--', label="q_ref")
    ax.set_title(f"Torque {j+1}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tau [Nm]")
    ax.legend()

# 남는 서브플롯(8번째)을 깔끔히 제거
for k in range(n, len(axes1)):
    fig.delaxes(axes1[k])

fig.suptitle("Torque input", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

np.savez_compressed(
    "panda_mpc_log.npz",
    t=t,                 # (T,)
    q_hist=q_hist,       # (T, 7)
    q_ref=q_ref,         # (7,)
    mpc_times=mpc_times,
    tau_hist=tau_hist    # (T, 7)
)

