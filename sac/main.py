import gym
import torch

from sac.models import ModelsOverseer
from sac.prioritized_replay_buffer import PrioretizedReplayBuffer
from sac.replay_buffer import ReplayBuffer
from sac.eval import validate
from sac.utils import NormalizedActions, plot

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def make_env(env_name):
    env = NormalizedActions(gym.make(env_name))
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    return env, action_dim, state_dim


if __name__ == '__main__':
    env_name = "Pendulum-v0"
    env, action_dim, state_dim = make_env(env_name)
    hidden_dim = 128
    use_priors = True
    replay_buffer_size = 100000
    if use_priors:
        replay_buffer = PrioretizedReplayBuffer(replay_buffer_size)
    else:
        replay_buffer = ReplayBuffer(replay_buffer_size)

    models = ModelsOverseer(state_dim, hidden_dim, action_dim, gamma=0.99, lr=1e-3)
    policy_net = models.policy_net

    ################---Train---################
    max_frames = 40000
    max_steps = 500
    frame_idx = 0
    rewards = []
    batch_size = 512
    gamma = 0.99
    soft_tau = 1e-2
    eps = 1.0

    while frame_idx < max_frames:
        # for frame_idx in tqdm(range(max_frames), total=max_frames):
        state = env.reset()
        episode_reward = 0
        if frame_idx % 1000 == 0:
            print(frame_idx, '/', max_frames)
        # print(replay_buffer)

        for step in range(max_steps):
            if frame_idx >= 8000:
                action = policy_net.get_action(state).detach().numpy()
                next_state, reward, episode_end, _ = env.step(action)
            else:
                action = env.action_space.sample()
                next_state, reward, episode_end, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, episode_end)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if len(replay_buffer) > batch_size:
                if use_priors:
                    errors, indicies = models.update(replay_buffer, batch_size, use_priors=use_priors, eps=eps)
                    replay_buffer.set_priorities(indicies, errors)
                else:
                    models.update(replay_buffer, batch_size)

            # if frame_idx >= max_frames:
            #    plot(frame_idx, rewards)

            if episode_end:
                eps = max(0.1, eps * 0.99)
                break
        rewards.append(episode_reward)

    validate(env_name, policy_net, 'tmp.gif', eval_steps=50000)
