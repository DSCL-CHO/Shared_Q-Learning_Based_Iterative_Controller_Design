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

    # method_1
    # while True:
    #     action = agent.get_action(state)
    #     next_state, reward, done = env.step(action)
    #     agent.update(state, action, reward, next_state, done)
    #     total_reward += reward
    #     if done:
    #         break
    #     state = next_state
    # return total_reward

    # method_2 
    # maximum 100 steps limit
    for t in range(100):  
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward
# calculate accumulated rewards
def evaluate_q_table(env, q_table):  
    state = env.reset()
    total_reward = 0
    for _ in range(100):
        qs = [q_table.get((state, a), -1e9) for a in range(4)] 
        action = int(np.argmax(qs))
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == "__main__":
    # Setting Parameters
    agent_count = 5
    total_episodes = 500
    share_every = 500
    
    os.makedirs("saved_models", exist_ok=True)  

    # independent enviroment for each agent
    agents = []
    envs = []
    reward_histories = [[] for _ in range(agent_count)] #📉
    for i in range(agent_count):
        agents.append(QLearningAgent(epsilon=0.1)) 
        envs.append(GridWorld()) 
    eval_x_shared = []
    eval_y_shared = []
    for episode in range(total_episodes):
        for i in range(agent_count):
            run_episode(agents[i], envs[i])  
            reward = run_episode(agents[i], envs[i])  
            reward_histories[i].append(reward)  

        if (episode + 1) % share_every == 0:
            print(f"\n📤 공유 시점: {episode + 1} 에피소드 이후")

            scores = []
            for i in range(agent_count):
                score = evaluate_q_table(GridWorld(), agents[i].Q)
                scores.append(score)
                
            # decide best Q-table
            best_agent_index = int(np.argmax(scores)) 
            best_q_table = deepcopy(agents[best_agent_index].Q)

            for i in range(agent_count):
                if i != best_agent_index:
                    agents[i].Q = deepcopy(best_q_table)
                    agents[i].rebuild_policies()
            eval_x_shared.append(episode + 1)
            eval_y_shared.append(scores[best_agent_index])
            print(f"➡️ 에이전트 #{best_agent_index + 1}의 Q-table을 공유합니다.")

# save final best Q-table
final_scores = []
for i in range(len(agents)):
    env = GridWorld()               
    agent = agents[i]              
    score = evaluate_q_table(env, agent.Q)  
    final_scores.append(score)
    # print("🎯 에이전트", i + 1, "평가 보상:", round(score, 2))
best_final_index = int(np.argmax(final_scores))
np.save("saved_models/best_q_table.npy", dict(agents[best_final_index].Q))
print(f"\n✅ 최종 Q-table 저장 완료 (에이전트 #{best_final_index + 1})")

# load final best Q-table
best_q_table = np.load("saved_models/best_q_table.npy", allow_pickle=True).item()
env = GridWorld()
renderer = Renderer(env.reward_map, env.goal_state, env.wall_states)
renderer.render_q(best_q_table)

# plot all agnet's accumulated rewards
import matplotlib.pyplot as plt
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

np.save("saved_models/eval_x_shared.npy", np.array(eval_x_shared))
np.save("saved_models/eval_y_shared.npy", np.array(eval_y_shared))
