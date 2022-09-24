import torch
import torch.nn.functional as F
from environment import CartPoleEnvManager
from epsilonGreedy import EpsilonGreedyStrategy
from agent import Agent
from replayMemory import ReplayMemory
from QNetwork import DQN
from Q_Value import QValues
import torch.optim as optim
from itertools import count
from utils import Experience, extract_tensors


if __name__ == '__main__':
    batch_size = 256
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10
    memory_size = 50000
    lr = 0.001
    num_episodes = 100

    device = torch.device("cpu")
    em = CartPoleEnvManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    episode_durations = []
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()

        for timestep in count():
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            next_state = em.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards
                
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    em.close()