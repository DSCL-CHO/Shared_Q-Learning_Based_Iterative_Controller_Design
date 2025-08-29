import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from collections import defaultdict
from copy import deepcopy
from common.gridworld import GridWorld
from common.utils import greedy_probs
from common.gridworld_render import Renderer

class QLearningAgent:
    def __init__(self, epsilon=0.1):
        self.alpha = 0.8
        self.gamma = 0.9
        self.epsilon = epsilon
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0.0)

    def get_action(self, state):
        probs = list(self.b[state].values())
        actions = list(self.b[state].keys())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:            
            next_q_max = 0
        else:             
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]   
            next_q_max = max(next_qs)  

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)

    def rebuild_policies(self):   # shared 
        for state in self.pi:
            self.pi[state] = greedy_probs(self.Q, state, 0)
            self.b[state] = greedy_probs(self.Q, state, self.epsilon)

def run_episode(agent, env):
    state = env.reset()
    total_reward = 0

    # method_1#
    # while True:
    #     action = agent.get_action(state)
    #     next_state, reward, done = env.step(action)
    #     agent.update(state, action, reward, next_state, done)
    #     total_reward += reward
    #     if done:
    #         break
    #     state = next_state
    # return total_reward
            
    for t in range(100):  # 최대 100스텝 제한
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward

def evaluate_q_table(env, q_table):  #(env, q_table):받은 q_table 을 이용해서 출발->목표 도달까지 받을 총 보상을 계산
    state = env.reset()
    total_reward = 0
    for _ in range(100):
        qs = [q_table.get((state, a), -1e9) for a in range(4)]  # 한 state 에서 모든 상태의 q를 리스트로 만듦(4개들어감)
        action = int(np.argmax(qs))
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == "__main__":
    agent_count = 5

    total_episodes = 500
    share_every = 500

    os.makedirs("saved_models", exist_ok=True)  #  폴더가 없으면 생성 (학습된 Q-table 저장용)

    # 10개의 Q-learning 에이전트
    # 각 에이전트마다 독립 환경
    agents = []
    envs = []
    reward_histories = [[] for _ in range(agent_count)] #📉
    for i in range(agent_count):
        agents.append(QLearningAgent(epsilon=0.1)) 
        envs.append(GridWorld()) 
        # agents = [A₁, A₂, A₃, ..., A₁₀]
        # envs   = [E₁, E₂, E₃, ..., E₁₀]
        # A₁은 E₁에서 학습 ...각자 독립적으로 행동하지만, 주기적으로 q-table 공유
    eval_x_shared = []#$#$#$
    eval_y_shared = []#$#$#$#


    for episode in range(total_episodes):
        for i in range(agent_count):
            run_episode(agents[i], envs[i])  # Q 학습
            reward = run_episode(agents[i], envs[i])  # #📉 각 에이전트 보상 반환
            reward_histories[i].append(reward)  #📉

        if (episode + 1) % share_every == 0:
            
            print(f"\n📤 공유 시점: {episode + 1} 에피소드 이후")

            # 각 에이전트의 Q-table을 평가
            scores = []
            for i in range(agent_count):
                score = evaluate_q_table(GridWorld(), agents[i].Q)
                scores.append(score)

            best_agent_index = int(np.argmax(scores))  #  가장 잘 수행한 Q-table 선택
            best_q_table = deepcopy(agents[best_agent_index].Q)

            for i in range(agent_count):
                if i != best_agent_index:
                    agents[i].Q = deepcopy(best_q_table)
                    agents[i].rebuild_policies()
            eval_x_shared.append(episode + 1)#$#$#$#
            eval_y_shared.append(scores[best_agent_index])#$#$#$#$
            print(f"➡️ 에이전트 #{best_agent_index + 1}의 Q-table을 공유합니다.")

    # 최종 best Q-table 저장
final_scores = []
for i in range(len(agents)):
    env = GridWorld()               # 평가용 새 환경
    agent = agents[i]              # i번째 에이전트
    score = evaluate_q_table(env, agent.Q)  # Q-table 평가
    final_scores.append(score)
    # print("🎯 에이전트", i + 1, "평가 보상:", round(score, 2))
    
best_final_index = int(np.argmax(final_scores))
np.save("saved_models/best_q_table.npy", dict(agents[best_final_index].Q))
print(f"\n✅ 최종 Q-table 저장 완료 (에이전트 #{best_final_index + 1})")

# 최종 best_q_table 불러오기
best_q_table = np.load("saved_models/best_q_table.npy", allow_pickle=True).item()
env = GridWorld()
renderer = Renderer(env.reward_map, env.goal_state, env.wall_states)
renderer.render_q(best_q_table)# Q-table 시각화

#📉
import matplotlib.pyplot as plt
# 전체 에이전트 보상 그래프 그리기
plt.figure(figsize=(10, 6))
for i, rewards in enumerate(reward_histories):
    plt.plot(rewards, label=f"Agent {i+1}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Rewards for All Agents")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#_____________________-
np.save("saved_models/eval_x_shared.npy", np.array(eval_x_shared))
np.save("saved_models/eval_y_shared.npy", np.array(eval_y_shared))
