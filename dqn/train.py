import torch
import numpy as np

from dqn.algorithm import DQNAlgorithm, DeepModel
from dqn.agent import Agent
from dqn.env import Environment
from dqn.utils import ReplyMemory


#运行一个episode  收集数据
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step +=1
        action = agent.sample(obs.reshape(1,-1).astype(np.float32))#(bsz,)

        next_obs, reward, done, info = env.step(action[0])
        total_reward += reward
        rpm.append((obs, action, next_obs, reward, done))
        if done:
            break
        obs = next_obs
    return total_reward


#评估五次求平均
def evaluate(env, agent, render=True):
    eval_reward_list = []
    for i in range(5):
        eval_reward = 0
        obs = env.reset()
        while True:
            action = agent.predicate(obs.reshape(1,-1).astype(np.float32))
            obs, reward, done, _ = env.step(action[0])
            eval_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward_list.append(eval_reward)
    return np.mean(eval_reward_list)

def train_batch(env, agent, rpm, bsz):
    obs, a, next_obs, reward, terminal = rpm.sample(bsz)#(bsz, 4) (bsz, 1) (bsz, 4) (bsz,) (bsz,)
    loss = agent.learn(obs, a, next_obs, reward, terminal)
    return loss
def train(env_name):
    #init environment
    env = Environment(env_name)

    #super parameter
    action_dim = env.get_action_dim()
    obs_dim = env.get_obs_dim()[0]
    e_greedy = 0.1
    e_greedy_decrement = 1e-6
    update_target_steps = 100
    bsz = 64

    #rpm
    MAX_MEMORY_SIZE = 256
    rpm = ReplyMemory(MAX_MEMORY_SIZE)

    #init agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deep_model = DeepModel(obs_dim, action_dim)
    alg = DQNAlgorithm(deep_model, action_dim)
    agent = Agent(alg, action_dim, device, e_greedy, e_greedy_decrement, update_target_steps)

    max_eps = 2000
    current_eps = 0
    while len(rpm) < MAX_MEMORY_SIZE:
        run_episode(env,agent,rpm)

    while current_eps<=max_eps:
        batch_loss = train_batch(env,agent,rpm,bsz)
        reward = run_episode(env,agent,rpm)
        print('current eps:{} loss:{}'.format(current_eps, batch_loss))
        if current_eps%200 ==0:
            reward = evaluate(env,agent)
        print(reward)

        current_eps+=1
    # for i in range(MAX_MEMORY_SIZE):
    #     agent.sample(obs)
    #
    # while current_eps<=max_eps:
    #
    #
    #     current_eps +=1

if __name__ == '__main__':
    train('CartPole-v0')



