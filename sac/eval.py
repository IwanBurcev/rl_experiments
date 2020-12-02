import gym

from sac.utils import NormalizedActions
from sac.visualize import display_frames_as_gif


def validate(env_name, policy, outfile, eval_steps=50000):
    env = NormalizedActions(gym.make(env_name))

    state = env.reset()
    frames = []
    for t in range(eval_steps):
        # Render into buffer.
        frames.append(env.render(mode='rgb_array'))
        action = policy.get_action(state)
        state, reward, done, info = env.step(action.detach().numpy())
        if done:
            break
    env.close()
    display_frames_as_gif(frames, outfile)
