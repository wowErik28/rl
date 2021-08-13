import numpy as np
import torch
'''
主要工作为 1.数据采样 2.数据预先处理
fit_once 写在algorithm里面，agent.learn()包括数据预处理以及sync_target_model
'''
class Agent:
    def __init__(self, alg, act_dim, device,e_greedy, e_greedy_decrement, update_target_steps):
        self.alg = alg
        self.act_dim = act_dim
        self.device = device
        self.e_greedy = e_greedy
        self.e_greedy_decrement = e_greedy_decrement
        self.update_target_steps = update_target_steps
        self.global_steps = 0

    def sample(self, obs):
        '''
        通过采样策略 输出action
        :param obs: numpy (bsz, obs_dim)
        :return: action numpy(bsz, 1)
        '''
        bsz, _ = obs.shape
        rand = np.random.rand(1)
        a = None
        if rand<=self.e_greedy:
            a = np.random.randint(0, self.act_dim, size=(bsz,))
        else:
            obs = torch.tensor(obs, device=self.device)
            a = self.alg.predicate(obs, self.device)#numpy (bsz,)

        #先广泛采样
        self.e_greedy = max(0.01, self.e_greedy - self.e_greedy_decrement)
        return a

    def predicate(self, obs):
        '''
        确定性策略 输出action
        :param obs: numpy (bsz, obs_dim)
        :return: action numpy(bsz,)
        '''
        obs = torch.tensor(obs, device=self.device)
        a = self.alg.predicate(obs, self.device)  # numpy (bsz,)
        return a

    def learn(self, obs, a, next_obs, reward, terminal):
        '''
        :param obs: numpy(bsz,obs_dim)
        :param a:   numpy(bsz, 1)
        :param next_obs: numpy(bsz,obs_dim)
        :param reward: numpy(bsz,)
        :param terminal: numpy(bsz,)   float
        :return: batch_loss numpy(1,)
        '''
        #先判断是否同步target
        if self.global_steps%self.update_target_steps == 0:
            self.alg.sync_target_model()
        self.global_steps +=1

        #将数据放入cuda 以及预处理
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, device=self.device).long()
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        terminal = torch.tensor(terminal, dtype=torch.float32, device=self.device)

        #然后训练
        loss = self.alg.learn(obs, a, next_obs, reward, terminal, self.device) #(1,)
        return loss
