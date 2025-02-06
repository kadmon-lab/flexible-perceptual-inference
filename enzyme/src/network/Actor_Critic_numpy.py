import torch; import torch.nn as nn; 
from enzyme.src.network.forward_backward_numpy import forward_backward
# from enzyme.src.network.rnn import LSTMCell as gate_returning_pytorch_NET;
from torch.nn import LSTM 

class Actor_Critic(forward_backward):
    def __init__(self, **params):
        super(Actor_Critic, self).__init__()
        self.__dict__.update(params)
        self.construct_net()        
        self.curr_episode = 0
        self.optim = torch.optim.RMSprop(self.parameters(), self.lr, self.alpha, self.eps, self.weight_decay, self.momentum, self.centered)
        self.MSE = nn.MSELoss()
        self.CEL = nn.CrossEntropyLoss()
        self.reset()
        self.to(self.device)
        self = torch.nn.DataParallel(self, device_ids=list(range(torch.cuda.device_count())))

    def construct_net(self):             
        self.main_mem = 0
        self.soft = nn.Softmax(dim = -1)
        self.to_value = nn.Linear(self.hid_dim, 1)
        self.true_ctx = torch.zeros(1, device = self.device).float()       
        self.to_action = nn.Linear( self.hid_dim  + self.pavlovian_bias * self.inp_dim , self.act_dim)
        self.actor, self.critic = [LSTM(self.inp_dim, self.hid_dim, device = self.device) for _ in range(2)]    
        self.actor_input, self.critic_input = [torch.empty(1,1,self.inp_dim, device = self.device) for _ in range(2)]
        
        self.actor_streams, self.critic_streams =  [tuple([torch.zeros(1, 1, self.hid_dim, requires_grad = True, device = self.device) for _ in range(2)]) for _ in range(2)]
        self.state_net_streams, self.ctx_net_streams = [tuple([torch.zeros(1, 1, self.hid_dim, requires_grad = True, device = self.device) for _ in range(2)]) for _ in range(2)]
        self.bayes_list =  ['bayes', 'bayes_optim', 'factorized', 'factorized_optim']
        self.optim_list =  ['bayes_optim', 'factorized_optim']

        self.open = torch.ones(1, self.hid_dim, requires_grad = False, device = self.device)
        self.closed = torch.zeros(1, self.hid_dim, requires_grad = False, device = self.device)
        self.values = torch.FloatTensor(30000).to(self.device)
        self.action_probs = torch.FloatTensor(30000).to(self.device)

    def store_outcome(self):
        self.values[self.step] = self.value
        self.action_probs[self.step] = self.action_prob[self.action]

    def reset(self, only_grad = False):
        self.actor_streams = (self.actor_streams[0].detach(), self.actor_streams[1].detach())
        self.critic_streams = (self.critic_streams[0].detach(), self.critic_streams[1].detach())            
        self.action_probs = self.action_probs.detach()
        self.values = self.values.detach()
        self.curr_episode += 1
        self.step = 0

    def save_simple(self):
            for agent in [self.actor, self.critic]:
                agent.state_dict["igates"]["weight"] = agent.weight_ih_l0
                del(agent.weight_ih_l0)
                agent.state_dict["igates"]["bias"] = agent.bias_ih_l0
                del(agent.bias_ih_l0)
                
                agent.state_dict["hgates"]["weight"] = agent.weight_hh_l0
                del(agent.weight_hh_l0)
                agent.state_dict["hgates"]["bias"] = agent.bias_hh_l0
                del(agent.bias_hh_l0)
  