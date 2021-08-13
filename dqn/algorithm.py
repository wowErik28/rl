import torch
import torch.nn as nn
import torch.optim as optim

import copy

class DeepModel(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim_list=[64, 128]):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_list[0])
        self.fc2 = nn.Linear(hidden_dim_list[0], hidden_dim_list[1])
        self.fc3 = nn.Linear(hidden_dim_list[1], action_dim)

    def forward(self, obs):
        y = torch.relu(self.fc1(obs))
        y = torch.relu(self.fc2(y))
        y = self.fc3(y)

        #y.shape = (bsz,action_dim)
        return y

class DQNAlgorithm:
    def __init__(self, model, action_dim, gamma = 0.9, lr = 0.001, optmizer_name = 'Adam',
                 loss_fuc_name = 'MSELoss'):
        self.model = model
        self.target_model = copy.deepcopy(self.model)

        assert isinstance(action_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr

        self.optimizer = getattr(optim,optmizer_name)(self.model.parameters(), lr)
        self.loss_func = getattr(nn, loss_fuc_name)()
    def predicate(self, obs, device = None):
        self.model.eval().to(device)
        return torch.max(self.model(obs), dim=1)[1].cpu().numpy()

    def learn(self, obs, a, next_obs, reward, terminal, device = None):
        '''
         training the model once
         default obs, a, next_obs, reward, terminal are all in gpu
        :param obs: tensor(bsz, obs_dim)           float
        :param a: tensor(bsz, 1)                   long
        :param next_obs: tensor(bsz, obs_dim)      float
        :param reward: tensor(bsz)                 float
        :param terminal: tensor(bsz)               float
        :return: loss numpy
        '''
        self.model.train()
        self.model.to(device)
        y = self.model(obs) #y (bsz, action_dim) Q_value

        #1.one_hot a[0,1,1,0,1]
        a_onehot = torch.zeros_like(y, dtype=torch.float32, device=device).scatter(1,a,1)

        q = torch.sum(y*a_onehot, dim=1)# q (bsz,)
        #2.compute traget
        self.target_model.eval()
        self.target_model.to(device)
        q_max,_ = torch.max(self.target_model(next_obs), dim=1) #(bsz,)
        target = reward + self.gamma*(1-terminal)*q_max

        loss = self.loss_func(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def sync_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


if __name__ == '__main__':
    bsz = 2
    obs_dim = 4
    action_dim = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deepmodel =  DeepModel(4,2).to(device)
    # obs = torch.randn((3,4)).to(device)
    # action = deepmodel(obs)
    # print(action)
    alg = DQNAlgorithm(deepmodel, action_dim)
    # alg.sync_target_model()
    # print(alg.target_model.state_dict())
    # print(alg.model.state_dict())

    obs = torch.randn(bsz,obs_dim).to(device)
    a = torch.tensor([[0],[1]]).long().to(device)
    next_obs = torch.randn(bsz,obs_dim).to(device)
    reward = torch.tensor([0.,1.]).to(device)
    terminal = torch.tensor([0.,1.]).to(device)

    loss = alg.learn(obs,a,next_obs,reward,terminal,device)
    print(loss)
    pred = alg.predicate(obs)
    print(pred)




