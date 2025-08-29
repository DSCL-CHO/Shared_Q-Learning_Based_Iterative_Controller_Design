import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
import numpy as np
from common.gridworld import GridWorld

env = GridWorld()
# 학습된 Q-table 불러오기
q_table = np.load("saved_models/best_q_table.npy", allow_pickle=True).item()

def extract_best_path(env, q_table):
    path = []
    state = env.reset()     # start_state
    path.append(state)      # 시작상태를 경로에 추가

    for _ in range(100):
        # q_table 은 이미 한 상태에서 네방향마다 Q값을 저장한 상태
        qs = [q_table.get((state, a), -1e9) for a in env.actions()]
        best_action = int(np.argmax(qs))
        next_state, _, done = env.step(best_action)
        path.append(next_state)

        if done:
            break
        state = next_state
    return path

if __name__ == "__main__":

    path = extract_best_path(env, q_table)

    print("\n[최적 경로 추출 완료]")
    print(" → ".join([str(pos) for pos in path]))

    # 저장
    np.save("saved_models/best_path.npy", path)