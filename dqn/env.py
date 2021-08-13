import gym

class Environment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def reset(self):
        return self.env.reset()

    def render(self, mode = 'human'):
        self.env.render(mode)

    def step(self, action):
        return self.env.step(action)

    def get_action_dim(self):
        return self.env.action_space.n

    def get_obs_dim(self):
        return self.env.observation_space.shape

if __name__ == '__main__':
    env = Environment('CartPole-v0')
    env.reset()
    # while 1:
    #     env.render()
    #     obs, reward,done,info = env.step(env.env.action_space.sample())
    #     print(reward)
    print(env.get_action_dim())
    print(env.get_obs_dim())
