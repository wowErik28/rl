from collections import deque
import random
import numpy as np

class ReplyMemory:
    def __init__(self, max_size):
        self.deque = deque(maxlen=max_size)

    def append(self, exp):
        '''
        :param exp: (obs, a, next_obs, reward, terminal)
        :return: None
        '''
        self.deque.append(exp)

    def sample(self, bsz):
        '''
        :param bsz: int
        :return: obs, a, next_obs, reward, terminal  all in array (bsz,...)
        '''
        if bsz > len(self):
            bsz = len(self)

        mini_batch = random.sample(self.deque, bsz) # [(,,,,),(,,,,),...]
        obs_list , a_list, next_obs_list, reward_list, terminal_list = [],[],[],[],[]
        for exp in mini_batch:
            obs, a, next_obs, reward, terminal = exp
            obs_list.append(obs)
            a_list.append(a)
            next_obs_list.append(next_obs)
            reward_list.append(reward)
            terminal_list.append(terminal)
        return np.array(obs_list, dtype=np.float32), np.array(a_list, dtype=np.int32),\
               np.array(next_obs_list, dtype=np.float32), np.array(reward_list, dtype=np.float32).reshape(-1),\
               np.array(terminal_list, dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.deque)

if __name__ == '__main__':
    rpm = ReplyMemory(100)
    for i in range(10):
        obs, a, next_obs, reward, terminal = np.random.randn(4), np.random.randint(0,2,size=(1,)), \
                                             np.random.randn(4), np.random.randn(1), np.random.randint(0,2,size=(1,))
        rpm.append((obs, a, next_obs, reward, terminal))
    a,b,c,d,e = rpm.sample(100)
    print(a.shape, b.shape, c.shape, d.shape, e.shape)