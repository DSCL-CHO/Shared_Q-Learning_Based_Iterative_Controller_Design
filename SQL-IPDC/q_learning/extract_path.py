import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
import numpy as np
from common.gridworld import GridWorld

env = GridWorld()
# import learned Q-table
q_table = np.load("saved_models/best_q_table.npy", allow_pickle=True).item()
def extract_best_path(env, q_table):
    path = []
    state = env.reset()    
    path.append(state)      
    for _ in range(100):
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
    np.save("saved_models/best_path.npy", path)
