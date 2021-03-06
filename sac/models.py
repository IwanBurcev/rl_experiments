import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ModelsOverseer():
    def __init__(self, state_dim, hidden_dim, action_dim, gamma=0.99, lr=1e-4):
        self._value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        self._target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        self._soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self._soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self._policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self._target_value_net.parameters(), self._value_net.parameters()):
            target_param.data.copy_(param.data)

        self.gamma = gamma

        self.value_optimizer = optim.Adam(self._value_net.parameters(), lr=lr)
        self.soft_q_optimizer1 = optim.Adam(self._soft_q_net1.parameters(), lr=lr)
        self.soft_q_optimizer2 = optim.Adam(self._soft_q_net2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self._policy_net.parameters(), lr=lr)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

    @property
    def policy_net(self):
        return self._policy_net

    def update(self, replay_buffer, batch_size, soft_tau=1e-2, use_priors=False, eps=1.0):

        if use_priors:
            (state, action, reward, next_state, done), importance, indicies = replay_buffer.sample(batch_size)
        else:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self._soft_q_net1(state, action)
        predicted_q_value2 = self._soft_q_net2(state, action)
        predicted_value = self._value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self._policy_net.evaluate(state)

        # Training Q Function
        target_value = self._target_value_net(next_state)
        target_q_value = reward + (1 - done) * self.gamma * target_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Value Function
        predicted_new_q_value = torch.min(self._soft_q_net1(state, new_action), self._soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value - log_prob
        errors = predicted_value - target_value_func.detach()
        if use_priors:
            value_loss = torch.mean(torch.pow(errors * (torch.from_numpy(importance).to(device) ** (1 - eps)), 2))
        else:
            value_loss = torch.mean(torch.pow(errors, 2))

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Training Policy Function
        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self._target_value_net.parameters(), self._value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        if use_priors:
            return errors, indicies

        # print(
        #     f"Q1 loss: {q_value_loss1}, Q2 loss: {q_value_loss2}, Value loss: {value_loss}, Policy loss: {policy_loss}")


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super().__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - \
                   torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()
        return action[0]
