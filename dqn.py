# the dqn to play 2048

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np

import matplotlib.pyplot as plt
import math
import random
from collections import namedtuple, deque
from itertools import count
import webdriver2048 as env
import os
import csv

# reproducibility
# torch.manual_seed(1)
# random.seed(1)

# HYPERPARAMETERS
memory_capacity = 10000
batch_size = 1024
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
n_hidden = 256
tau = 0.005
lr = 1e-4
ending_penalty = -1
wall_penalty = -10
alpha = 0.7 # alpha * scoreReward + (1  - alpha) * sumReward
beta = 1.0 # part of Boltzmann distribution
multiplier = 1.0

# ENVIRONMENT/TRAINING SETTINGS
n_actions = 4
n_observations = 16 # 2048 uses a 4x4 matrix
state = None
num_episodes = 50
tiles = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Cuda is available: {torch.cuda.is_available()}')
# METRICS
# starting w/ value 0 so I can set plot xlim left = 1 and have the episodes start at 1
scores = [0]
times = [0]
action_counts = [0] * 4
action_labels = ["up", "down", "left", "right"]
losses = [0]
epsilon_thresholds = [0]
steps_done = [0]
rewards = [0]
allMetrics = [scores, times, action_counts, losses, epsilon_thresholds, steps_done, rewards]
plt.ion()






Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_actions, n_observations, n_hidden):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_hidden)
        self.layer4 = nn.Linear(n_hidden, n_actions)
        # initialization recommended by paper in notes
        # kaiming init kept b/c i'm using relu. use xavier for tanh
    def forward(self, x):
        x = F.relu(self.layer1(x)) # paper in notes suggests using tanh instead
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x) # interesting, no relu on last layer
        return x

# start adding to device stuff
policy_net = DQN(n_actions, n_observations, n_hidden).to(device)
target_net = DQN(n_actions, n_observations, n_hidden).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr = lr, amsgrad = True)
memory = ReplayMemory(memory_capacity)



def select_action(state, i, terminated, random_duration, load_ep = 0):
    global steps_done
    # i starts at 0
    if isinstance(steps_done, int):
        steps_done = [steps_done]
    # + load_ep so in a situation where we load episode 1900 but only want
    # random for an additional 50 episodes, we're random until episode 1950
    if i >= random_duration + load_ep:
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * steps_done[0] / eps_decay) # magic equation
        if terminated:
            epsilon_thresholds[0] = eps_threshold
        steps_done[0] += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # Boltzmann (beta * softmax?) distribution code from https://datascience.stackexchange.com/questions/61262/agent-always-takes-a-same-action-in-dqn-reinforcement-learning
                q_values = policy_net(state)
                q_values = q_values - torch.max(q_values)
                action_probs = torch.exp(beta * q_values) / torch.sum(torch.exp(beta * q_values), dim = 1, keepdims = True)
                action_probs = action_probs.to('cpu')
                action_probs = action_probs.numpy()
                action_probs = action_probs / action_probs.sum(axis = 1, keepdims = True)
                return np.random.choice(a = n_actions, p = action_probs[0])
                # return policy_net(state).max(1).indices.view(1, 1)
    
    return torch.tensor([[env.sample()]], dtype = torch.long, device = device)

def plot_metric(metric, metric_name, fig_num = 1, show_result = False):
    plt.figure(fig_num)
    metric_t = torch.tensor(metric, dtype = torch.float32)
    if show_result:
        plt.title(f'Result {metric_name}')
    else:
        plt.clf()
        plt.title(f'Training {metric_name}')
    if metric_name == "Rewards" or metric_name == "Reward":
        plt.xlabel("Action #")
    else:
        plt.xlabel('Episode')
    plt.ylabel(metric_name)
    plt.plot(metric_t.numpy())
    if len(metric_t) >= 100:
        means = metric_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.xlim(left = 1)
    plt.pause(0.01)

def plot_action_frequencies(fig_num):
    plt.figure(fig_num)
    action_counts_t = torch.tensor(action_counts, dtype= torch.float32)
    action_frequencies = action_counts_t / action_counts_t.sum()
    plt.bar(action_labels, action_frequencies, color = ['blue', 'green', 'red', 'purple'])

    plt.title('Action Frequencies')
    plt.xlabel('Actions')
    plt.ylabel('Frequency')
    plt.pause(0.01)

def save_metrics(iteration, folder = "metrics"):
    folder = os.path.join(folder, f"data_iteration_{iteration}")
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, metric in enumerate(allMetrics):
        filename = os.path.join(folder, f'metric_{i + 1}.csv')
        # if is action counts
        if i == 2:
            with open(filename, mode = 'w', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow(['Index', 'Value'])
                for index, value in enumerate(metric):
                    writer.writerow([index, value])
        else:
            with open(filename, mode = 'a', newline = '') as file:
                writer = csv.writer(file)

                if isinstance(metric, dict):
                    if file.tell() == 0:
                        writer.writerow(['Key', 'Value'])
                    for key, value in metric.items():
                        writer.writerow([key, value])
                elif isinstance(metric, (list, tuple)):
                    if file.tell() == 0:
                        writer.writerow(['Index', 'Value'])
                    for index, value in enumerate(metric):
                        writer.writerow([index, value])
                else:
                    if file.tell() == 0:
                        writer.writerow(['Value'])
                    writer.writerow([metric])
        
# only useful when displaying the graphs?
def load_metrics(iteration, folder = "metrics"):
    folder = os.path.join(folder, f'data_iteration_{iteration}')
    global scores, times, action_counts, losses, epsilon_thresholds, steps_done, rewards
    for i, _ in enumerate(allMetrics):
        filename = os.path.join(folder, f'metric_{i + 1}.csv')
        with open(filename, mode = 'r', newline = '') as file:
            reader = csv.reader(file)
            header = next(reader)

            if header == ['Key', 'Value']:
                metric = {float(rows[0]): float(rows[1]) for rows in reader}
            elif header == ['Index', 'Value']:
                metric = [float(rows[1]) for rows in reader]
            else:
                # can only be header == ['Value'] based on how things are saved
                metric = float(next(reader)[0])
        allMetrics[i] = metric
    scores, times, action_counts, losses, epsilon_thresholds, steps_done, rewards = allMetrics

def load_counts_and_steps(iteration, folder = "metrics"):
    global action_counts, steps_done
    folder = os.path.join(folder, f"data_iteration_{iteration}")
    counts_path = os.path.join(folder, 'metric_3.csv')
    steps_path = os.path.join(folder, 'metric_6.csv')

    with open(counts_path, 'r', newline = '') as file:
        reader = csv.reader(file)
        header = next(reader)
        action_counts = [int(rows[1]) for rows in reader]
    with open(steps_path, 'r', newline = '') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            steps_done = int(row[0])
    

def save_weights(episode, iteration, folder = "model_weights"):
    folder = os.path.join(folder, f'iter_{iteration}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    policy_path = os.path.join(folder, f'policy_episode_{episode}')
    target_path = os.path.join(folder, f'target_episode_{episode}')
    memory_path = os.path.join(folder, f'replay_memory_{episode}')

    torch.save(memory, memory_path)
    torch.save(policy_net.state_dict(), policy_path)
    torch.save(target_net.state_dict(), target_path)

def load_weights(episode, iteration, folder = "model_weights"):
    global memory
    folder = os.path.join(folder, f'iter_{iteration}')
    policy_path = os.path.join(folder, f'policy_episode_{episode}')
    target_path = os.path.join(folder, f'target_episode_{episode}') 
    memory_path = os.path.join(folder, f'replay_memory_{episode}')

    memory = torch.load(memory_path)
    policy_net.load_state_dict(torch.load(policy_path))
    target_net.load_state_dict(torch.load(target_path))


def optimize_model(terminated):
    global losses
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype = torch.bool, device = device)

    non_final_next_states = torch.cat([s.to(device) for s in batch.next_state if s is not None])
    
    state_batch = torch.cat([s.to(device) for s in batch.state])
    action_batch = torch.cat([a.to(device) for a in batch.action])
    reward_batch = torch.cat([r.to(device) for r in batch.reward])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device = device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    if terminated:
        losses[0] = loss.item()
    optimizer.zero_grad()
    loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(
        iteration, 
        num_episodes = num_episodes, 
        reward_type = 'score', # score, score_bug, sum, score_sum, merge
        random_duration = 50,
        track_reward = False,
        bad_end = False,
        avoid_wall = False,
        load = False, 
        load_it = None, 
        load_ep = 0
        ):
    
    global action_counts
    if load:
        #global scores, times, action_counts, losses, epsilon_thresholds, steps_done, rewards, multiplier
        # kind of makes metrics as a hyperparameter obsolete since loading it like this eliminates the ability to choose metrics
        load_weights(load_ep, load_it)
        #load_metrics(load_it)
        load_counts_and_steps(iteration)
    else: 
        # save 0s as the first values in order to help with plotting
        save_metrics(iteration)
        
    
    
    terminated = False
    for i in range(load_ep, load_ep + num_episodes):
        if i == 0 or i == load_ep:
            isOver = False
        else:
            isOver = True
        state = env.reset(isOver)
        state = torch.tensor(state, dtype = torch.float32, device = device)
        state = state.view(-1) # shape (16)
        state = state.unsqueeze(0) # shape (1, 16)
        start_time = time.time()
        
        prevScore = 0
        prevSum = 0
        prevObs = torch.tensor(0, device = device)
        prevTileCounts = {8: 0, 16: 0, 32: 0, 64: 0, 128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}
        tileCounts = {8: 0, 16: 0, 32: 0, 64: 0, 128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}

        # reset rewards so the values append properly when being saved to a file
        rewards = []
        for t in count():
            interTileCounts = {8: 0, 16: 0, 32: 0, 64: 0, 128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}
            action = select_action(state, i, terminated, random_duration, load_ep)
            action_counts[action] += 1
            obs, currentReward, terminated = env.step(action)
            if not torch.is_tensor(action):
                action = torch.tensor([[action]], dtype = torch.long, device = device)
            obs = torch.tensor(obs, dtype = torch.float32, device = device)
            obs = obs.view(-1) # shape (16)
            if reward_type == 'score':
                reward = currentReward - prevScore
                prevScore = currentReward
            elif reward_type == 'score_bug':
                reward = currentReward
            elif reward_type == 'sum':
                currentReward = obs.sum().item()
                reward = currentReward - prevSum
                prevSum = currentReward
            elif reward_type == 'score_sum':
                # previous bug: score was implemented right, while sum
                # was still wrong
                scoreReward = currentReward - prevScore
                sumReward = obs.sum().item() - prevSum
                prevScore = currentReward
                prevSum = obs.sum().item()
                reward = alpha * scoreReward + (1 - alpha) * sumReward
            
            
            # reward for the value of the merged tile, once
            elif reward_type == 'merge':
                reward = 0
                multiplier = 1.0
                if not torch.equal(obs, prevObs):
                    prevObs = obs
                    reward = 1
                    for tile in obs:
                        if tile in tiles:
                            interTileCounts[int(tile.item())] += 1
                    for k, v in interTileCounts.items():
                        tileCounts[k] = v - prevTileCounts[k]
                    
                    prevTileCounts = interTileCounts
                    
                    for tile in tiles:
                        if tileCounts[tile] > 0:
                            reward += tileCounts[tile] * tile * multiplier
                        multiplier += 0.25
            
            if avoid_wall:
                if reward == 0: # smashing into wall
                    reward = wall_penalty
                elif reward_type == 'merge':
                    reward -= 1 # so racking up 2s does not generate reward
            
            

            if terminated:
                next_state = None
                if bad_end:
                    reward = ending_penalty
            else:
                next_state = obs.clone().detach().to(device).unsqueeze(0) # shape (1, 16)
        

            if track_reward:
                rewards.append(reward)
                # print(reward)

            reward = torch.tensor([reward], device = device)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model(terminated)

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
                target_net.load_state_dict(target_net_state_dict)
            
            if terminated:
                end_time = time.time()
                times[0] = int(end_time - start_time)
                prevScore = scores[0]
                scores[0] = env.get_score()
                allMetrics[2] = action_counts
                allMetrics[6] = rewards
                save_metrics(iteration)
                if ((i + 1) % 50 == 0) or (scores[0] - prevScore >= 2000):
                    save_weights(i + 1, iteration)
                print(f'Episode {i + 1}/{num_episodes + load_ep} Completed: {scores[-1]}, {times[-1]}s, {losses[-1]:.2f} loss, {epsilon_thresholds[-1]:.2f} eps thresh')
                print(f'{(float(i + 1) / (num_episodes + load_ep) * 100):.2f}% Complete')
                break


