import mkl
mkl.set_num_threads(1)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np

import logging
import time
import gym

def state_to_chanels(state):
    return Variable(torch.Tensor([np.moveaxis(np.array(state[1])[:, : ,:3], -1, 0)]))


class Model(nn.Module):
    def __init__(self, rng_state):
        super().__init__()
        
        # TODO: padding?
        self.conv1 = nn.Conv2d(3, 16, (3, 6), 2)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), 1)
        self.dense = nn.Linear(4*16, 128)
        self.out = nn.Linear(128, 12)
        
        
        self.rng_state = rng_state
        torch.manual_seed(rng_state)
            
        self.evolve_states = []
            
        self.add_tensors = {}
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal(tensor)
            else:
                tensor.data.zero_()
                        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(1, -1)
        x = F.relu(self.dense(x))
        x = F.relu(self.out(x))
        return x
    
    def evolve(self, sigma, rng_state):
        torch.manual_seed(rng_state)
        self.evolve_states.append((sigma, rng_state))
            
        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)
            
    def compress(self):
        return CompressedModel(self.rng_state, self.evolve_states)

def uncompress_model(model):    
    start_rng, other_rng = model.start_rng, model.other_rng
    m = Model(start_rng)
    for sigma, rng in other_rng:
        m.evolve(sigma, rng)
    return m

def random_state():
    return random.randint(0, 2**31-1)

class CompressedModel:
    def __init__(self, start_rng=None, other_rng=None):
        self.start_rng = start_rng if start_rng is not None else random_state()
        self.other_rng = other_rng if other_rng is not None else []
        
    def evolve(self, sigma, rng_state=None):
        self.other_rng.append((sigma, rng_state if rng_state is not None else random_state()))

def evaluate_model(env, model, logger, max_eval=20000):
#     env = gym.make(env)
    env.set_scramble(20, 20, True)
    model = uncompress_model(model)
    initial_state = env.reset()
    cur_state = state_to_chanels(initial_state)
    total_reward = 0
    net_reward = 0

    total_frames = 0
    model.eval()
    for _ in range(max_eval):
        total_frames += 4
        values = model(cur_state)[0]
        net_reward += values.max()/values.sum()
        action = values.argmax()
        new_state, reward, is_done, _ = env.step(action.item())
        if is_done:
            break
        cur_state = state_to_chanels(new_state)
      

    net_reward += reward        
    fitness = (net_reward/(env.get_step_count() + 1)) + (1/env.get_step_count())
    return fitness if reward != 1 else fitness.item(), total_frames, reward == 1


def printMe(logger, message):
    logger.debug(message)
    print(message)
