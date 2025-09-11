import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from kp_then_kd.panda_env import MultiPandaEnv
from common.gridworld import GridWorld
from kp_then_kd.coord_to_xyz import coord_to_xyz

# setting parameter
robot_n = 9                # N
init_range = (100,500)     # [Kmin, Kmax]
kx_ratio = 0.4             # 𝛗
conv_thresh = 0.01         # ϵ
max_iter = 10              # L

# import result of best_kp
best_kp_path = "saved_models/best_kp_value.npy"

if os.path.exists(best_kp_path):
    kp = int(np.load(best_kp_path))
else:
    raise FileNotFoundError("best_kp_value.npy 파일이 없습니다. optimize_kp.py에서 먼저 Kp를 최적화하세요.")
print("✅best_kp_path✅=",kp)
x_offsets = [i * 0.8 for i in range(robot_n)]
path = np.load("saved_models/best_path.npy", allow_pickle=True)

def run_simulation(kd_list, iteration):
    env = MultiPandaEnv(num_robots=robot_n, gui=False,
                        x_offsets=x_offsets, 
                        kp_list=[kp]*robot_n, 
                        kd_list=kd_list)
    gw = GridWorld()
    # env.render_grid_overlay(x_offset=0, grid_shape=gw.shape,
    #                         reward_map=gw.reward_map,
    #                         wall_states=gw.wall_states,
    #                         goal_state=gw.goal_state)

    for coord in path:
        pos_list = [coord_to_xyz(coord, origin=(x, -0.1, 0.6)) for x in x_offsets]
        env.move_all(pos_list, steps=2000)
    # plot
    # for i, error_list in enumerate(env.errors):
    #     plt.plot(error_list, label=f"Robot {i+1} (Kd={kd_list[i]})")
    # for i in range(robot_n):
    #      plt.plot(env.errors[i], label=f"Robot  {i+1} (Kd={kd_list[i]})")

    # plt.title(f"Iteration {iteration+1} - EE Position Error")
    # plt.xlabel("Step")
    # plt.ylabel("L2 Error")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # RMSE 계산
    rmse_list = []
    for err in env.errors:
        rmse = np.sqrt(np.mean(np.array(err) ** 2))
        rmse_list.append(rmse)
 
    env.disconnect()
    return rmse_list

# random Kd
kd_list = sorted(np.random.choice(np.arange(init_range[0], init_range[1]), size=robot_n, replace=False))
last_best_rmse = float('inf')

for iteration in range(max_iter):
    print(f"\n🔁 Iteration {iteration+1} | Kd candidates: {kd_list}")
    rmses = run_simulation(kd_list, iteration)

    # for i, (kd_val, rmse) in enumerate(zip(kd_list, rmses)):
        # print(f"  Robot {i+1}: Kd={kd_val} → RMSE={rmse:.4f}")
    for i in range(robot_n):
        print(f"로봇 {i+1}: Kd = {kd_list[i]} → 평균 오차 = {rmses[i]:.4f}")

    best_idx = int(np.argmin(rmses))
    best_kd = kd_list[best_idx]
    best_rmse = rmses[best_idx]

    print(f"\n✅ Best Kd = {best_kd} (RMSE={best_rmse:.4f})")

    if abs(last_best_rmse - best_rmse) < conv_thresh:
        print("\n🎯 수렴 조건 만족 → 최적 Kd 탐색 종료!")
        break
    last_best_rmse = best_rmse

    kx = best_kd * kx_ratio
    kd_list = [int(best_kd + (i - robot_n//2) * (kx / (robot_n//2))) for i in range(robot_n)]

    # input("\n▶ 다음 반복으로 넘어가려면 Enter...")

print(f"\n🏁 최종 선택된 Kd = {best_kd} (RMSE={best_rmse:.4f})")

