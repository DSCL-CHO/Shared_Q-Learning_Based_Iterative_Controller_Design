import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
import numpy as np
import common.gridworld_render as render_helper

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_map = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
        }
# reward_map
        self.reward_map = np.array([
            [None,  0, 0,   1000],
            [0,  0,  0,  0],
            [0, 0,  0, None],
            [0,   0,  0, -1.0],
            [20, 0  ,  50, -1.0],
            [0,   30,  0, -1.0]   
                    
        ])
# state
        self.goal_state = (0, 3)            # goal
        self.wall_states = [(0, 0),(2,3)]   # walls
        self.start_state = (5, 0)           # start
        self.agent_state = self.start_state # reset position

        self.one_time_rewards = {(3,0),(4,0),(4,2),(5,1)}  
        self.visited_bonus_states = set()          
    @property
    def height(self):
        return len(self.reward_map) 

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape
    
    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                if (h, w) not in self.wall_states:
                    yield (h, w)
  
    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state in self.wall_states: 
            next_state = state
        return next_state  
    
    def reward(self, state, action, next_state):
        if next_state in self.one_time_rewards:
            if next_state not in self.visited_bonus_states:
                self.visited_bonus_states.add(next_state)
                return self.reward_map[next_state]
            else:
                return 0 
        else:
            return self.reward_map[next_state]      
        

    def reset(self):
        self.agent_state = self.start_state
        self.visited_bonus_states = set() 
        return self.agent_state
        
    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        # reward = self.reward_map[next_state] if self.reward_map[next_state] is not None else 0
        reward = self.reward(state, action, next_state) 
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,self.wall_states)

        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state, self.wall_states)
        renderer.render_q(q, print_value)
