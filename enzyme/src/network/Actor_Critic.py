import torch; import torch.nn as nn; from enzyme.src.network.forward_backward import forward_backward
from torch.nn.quantizable.modules import LSTMCell as quantum_pytorch_NET;
from enzyme.src.network.rnn import LSTMCell as gate_returning_pytorch_NET;
from enzyme.src.network.handmade import NET as handmade_NET
from enzyme.src.mouse_task.bayesian_agent import bayesian_independent as bayes

class Actor_Critic(forward_backward):
    def __init__(self, **params):
        super(Actor_Critic, self).__init__()
        self.__dict__.update(params)
        self.construct_net()        
        self.curr_episode = 0
        self.optim = torch.optim.RMSprop(self.parameters(), self.lr, self.alpha, self.eps, self.weight_decay, self.momentum, self.centered)
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma = self.lr_decay)
        self.MSE = nn.MSELoss()
        self.CEL = nn.CrossEntropyLoss()
        self.reset()
        self.open = torch.ones(1, self.hid_dim, requires_grad = False, device = self.device)
        self.closed = torch.zeros(1, self.hid_dim, requires_grad = False, device = self.device)
        self.to(self.device)
        self = torch.nn.DataParallel(self, device_ids=list(range(torch.cuda.device_count())))

    def construct_net(self):             
        NET = handmade_NET if self.handmade else quantum_pytorch_NET if self.use_vanilla_torch else gate_returning_pytorch_NET
        self.consec_r = 0
        self.consec_a = 0
        self.main_mem = 0
        self.soft = nn.Softmax(dim = -1)
        self.to_value = nn.Linear(self.hid_dim, 1)
        self.true_ctx = torch.zeros(1, device = self.device).float()       
        self.to_action = nn.Linear( self.hid_dim  + self.pavlovian_bias * self.inp_dim , self.act_dim)
        extra_net_kwargs = dict()  if self.use_vanilla_torch else dict(lesion = self.lesion, override = self.mode) 
        self.actor, self.critic = [NET(self.inp_dim, self.hid_dim, device = self.device, **extra_net_kwargs) for _ in range(2)]    
        
        self.actor_streams, self.critic_streams =  [tuple([torch.zeros(1, 1, self.hid_dim, requires_grad = True, device = self.device) for _ in range(2)]) for _ in range(2)]
        self.state_net_streams, self.ctx_net_streams = [tuple([torch.zeros(1, 1, self.hid_dim, requires_grad = True, device = self.device) for _ in range(2)]) for _ in range(2)]
        self.bayes_list =  ['bayes', 'bayes_optim', 'factorized', 'factorized_optim']
        self.optim_list =  ['bayes_optim', 'factorized_optim']
        if self.use_vanilla_torch:
            self.actor_streams, self.critic_streams =  [tuple([torch.zeros(1, self.hid_dim, requires_grad = True, device = self.device) for _ in range(2)]) for _ in range(2)]

        if self.subnets != None: 
            if self.subnets in self.bayes_list:
                self.factorize = (self.subnets == 'factorized') or (self.subnets == 'factorized_optim')                
                self.bayes = bayes(device = self.device, factorize = self.factorize)
                if self.subnets not in self.optim_list:
                    self.subnets_to_action = nn.Linear(2 + self.pavlovian_bias * self.inp_dim, self.act_dim)
                    self.subnets_to_value = nn.Linear(2 + self.pavlovian_bias * self.inp_dim, 1)                    
            else:
                self.to_ctx = nn.Linear(self.hid_dim, 1)
                self.to_state = nn.Linear(self.hid_dim, 2)                                      # we predict the one hot encoded state for cross-entropy loss
                self.state_net = NET(self.inp_dim, self.hid_dim, lesion = self.lesion, override = self.mode, device = self.device)     
                self.ctx_net = NET(self.inp_dim, self.hid_dim, lesion = self.lesion, override = self.mode, device = self.device) 

        if self.expansion != None:
            self.to_expansion = nn.Linear(self.hid_dim, self.expansion)
            self.from_expansion = nn.Linear(self.expansion, self.hid_dim)
              
    def store_outcome(self, reward, state, ctx, only_grad = False):
        if not  only_grad:
            self.Q_values = torch.cat((self.Q_values, self.Q_value[:, None].detach()), -1)     # add Q values to memory
            self.gates = torch.cat((self.gates, self.gate[:, :, None].detach()), -1)           # add actor LSTM gates to memory
            self.LTMs = torch.cat((self.LTMs, self.LTM.squeeze(0).T.detach()), -1)             # add actor LSTM output to memory
            self.outputs = torch.cat((self.outputs, self.output.squeeze(0).T.detach()), -1)    # add actor LSTM output to memory
            self.inputs = torch.cat((self.inputs, self.input.squeeze(0).T), -1)                              
            self.pred_ctxs = torch.cat((self.pred_ctxs, self.pred_ctx[:,None]), -1)
        self.actions = torch.cat((self.actions, self.action))                              # add action taken to memory
        self.rewards = torch.cat((self.rewards, reward))                                   # add experienced reward to memory
        self.values = torch.cat((self.values, self.value))                                 # add predicted value to memory
        self.action_probs = torch.cat((self.action_probs, self.action_prob[:, None]), -1)  # add probability of actions to memory

        if self.subnets != None:
            self.pred_states = torch.cat((self.pred_states, self.pred_state[:, None]), -1)
            self.pred_ctxs = torch.cat((self.pred_ctxs, self.pred_ctx[:,None]), -1)
            self.true_states = torch.cat((self.true_states, torch.tensor([[1-state], [state]], device = self.device)), -1)
            self.true_ctxs = torch.cat((self.true_ctxs, torch.tensor([[ctx]], device = self.device)),-1)            
        self.true_ctx = ctx

    def reset(self, only_grad = False):
        if (only_grad or self.skip_storage) and (self.curr_episode > 0):
            self.actor_streams = (self.actor_streams[0].detach(), self.actor_streams[1].detach())
            self.critic_streams = (self.critic_streams[0].detach(), self.critic_streams[1].detach())            
            self.action_probs = self.action_probs.detach()
            self.values = self.values.detach()
        else: 
            self.actor_streams = (self.actor_streams[0].detach(), self.actor_streams[1].detach())
            self.critic_streams = (self.critic_streams[0].detach(), self.critic_streams[1].detach())
            self.inputs, self.actions, self.action_probs, self.rewards, self.values, self.Q_values, self.gates, self.outputs, self.LTMs =  [torch.tensor([], device = self.device) for _ in range(9)]        
            self.pred_states, self.pred_ctxs, self.true_states, self.true_ctxs =  [torch.tensor([], device = self.device) for _ in range(4)]
            self.pred_ctx = torch.zeros(1, device = self.device)        
            if self.subnets != None:
                self.state_net_streams = (self.state_net_streams[0].detach(), self.state_net_streams[1].detach())
                self.ctx_net_streams = (self.ctx_net_streams[0].detach(), self.ctx_net_streams[1].detach())
                self.output = self.actor_streams[0]        
                self.LTM = self.actor_streams[1]
        self.curr_episode += 1