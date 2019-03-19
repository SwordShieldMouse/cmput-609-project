from includes import *

class SCGrid():
    def __init__(self):
        self.reward = -1
        self.actions = [-1, 1] # convenience used to index actions
        self.action_dim = 2
        self.obs_dim = 1
        self.obs = [1] # all states look the same in this env, so only return one number
        self.states = [0, 1, 2, 3]
        self.curr_state = 0

    def step(self, action):
        # if in second state, reverse moves
        if self.curr_state == 1:
            action = 1 - action
            self.curr_state += self.actions[action]
        elif (self.curr_state + self.actions[action]) >= 0 and (self.curr_state + self.actions[action]) <= 3:
            self.curr_state += self.actions[action]
        if self.curr_state == 3:
            done = True
        else:
            done = False
        info = None
        return self.obs, self.reward, done, info

    def reset(self):
        self.curr_state = 0
        return self.obs
