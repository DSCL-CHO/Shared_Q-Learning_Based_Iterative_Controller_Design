import casadi as ca
import numpy as np
from get_G import G_mtx
from get_M import M_mtx
from get_C import C_mtx

 
# arr_q = np.array(q[i])
def to_mx(A):
    if isinstance(A, (ca.MX, ca.SX, ca.DM)):
        return A
    A = np.array(A, dtype=object)
    if A.ndim == 0:
        return ca.MX(A.item())
    elif A.ndim == 1:
        return ca.vertcat(*[A[i] for i in range(A.shape[0])])
    elif A.ndim == 2:
        rows = [ca.hcat([A[i, j] for j in range(A.shape[1])])
                for i in range(A.shape[0])]
        return ca.vertcat(*rows)
    else:
        raise ValueError("to_mx: 지원하지 않는 차원입니다.")
    

def make_dynamics(linear_solver='symbolicqr'):
    n = 7
    q  = ca.MX.sym('q',  n)
    dq = ca.MX.sym('dq', n)
    u  = ca.MX.sym('u',  n)
    x  = ca.vertcat(q, dq)

    M = to_mx(M_mtx([q[i]  for i in range(n)]))
    C = to_mx(C_mtx([q[i]  for i in range(n)], [dq[i] for i in range(n)])).reshape((n,1))
    G = to_mx(G_mtx([q[i]  for i in range(n)])).reshape((n,1))
    tau = ca.reshape(u, (n,1))

    # 수치 안정화(대칭화 + 미소 정규화)
    M = 0.5*(M + M.T) + 1e-9*ca.DM.eye(n) # M: Symetric 

    # rhs = tau - C - G 
    # rhs = tau - C
    rhs = tau
    # ★ 핵심: SX에서도 expand 가능한 linsol 지정
    # ddq = tau 
    
    # ddq = (M**-1)*tau
    ddq = ca.solve(M, rhs, linear_solver)

    xdot = ca.vertcat(dq, ddq)
    return ca.Function('f', [x, u], [xdot], ['x','u'], ['xdot'])
 


