import torch.nn as nn; from torch.distributions.categorical import Categorical; import torch;  from sklearn.decomposition import PCA; 
import torch.nn.functional as F; import pylab as plt; import numpy as np

class forward_backward(nn.Module):
    def __init__(self, **params):
        super(forward_backward, self).__init__()
        self.__dict__.update(params)
          
    def forward(self, net_input):  
        net_input = net_input[None, None, :]                                                  # LSTM inputs and outputs are of shape: sequence length, batch size, input dim
        self.gate = torch.zeros(4,1, device=self.device)
        if self.use_vanilla_torch:
            net_input = net_input.squeeze(0)
            
        self.input = actor_input = critic_input = net_input
        self.get_actor(actor_input)                                                             # run forward Actor
        if self.subnets == None:
            self.default_action()
        elif self.subnets in self.optim_list:
            self.action, self.action_prob, self.Q_value = self.bayes.flow_action()  
        else:
            self.subnet_action()

        self.get_critic(critic_input)                                                            # run forward Critic
        return self.action, self.action_prob, self.value
          
    def get_actor(self, actor_input):
        if self.subnets == None:
            self.actor_standard(actor_input)
        elif self.subnets != None:
            if self.subnets not in self.bayes_list:
                self.actor_subnets(actor_input)
            if self.subnets in self.bayes_list:  
                self.get_bayes_actor(actor_input)
                        
    def actor_standard(self, actor_input):
         """ option to train only readout """
         if self.train_recurrent and self.train:
             actor_return = self.actor(actor_input, self.actor_streams)
         else:
             with torch.no_grad(): 
                 actor_return = self.actor(actor_input, self.actor_streams) 
         if self.use_vanilla_torch:
             (S,L) = actor_return
             self.actor_output = S
         else:
             self.actor_output, (S,L), G = actor_return
             if self.get_dynamics_of == 'actor':
                 self.gate = G
         
         """ extra expansion layer """ 
         if self.expansion != None: 
             self.actor_output = self.from_expansion(torch.tanh(self.to_expansion(self.actor_output)))
         if self.pavlovian_bias:
             self.actor_output = torch.cat((self.actor_output, self.input), -1).float()        
         self.actor_streams = (S, L)                                                             # long and short term memory streams
         self.LTM = L.detach()[None, ...] if self.use_vanilla_torch else L.detach()
         self.output = S.detach()[None, ...] if self.use_vanilla_torch else S.detach() 

    def actor_subnets(self, actor_input):
        if self.train_recurrent and self.train:
            state_recur, (state_S, state_L), self.gate = self.state_net(actor_input, self.state_net_streams)
            ctx_recur, (ctx_S, ctx_L), _ = self.ctx_net(actor_input, self.ctx_net_streams)
        else:
            with torch.no_grad():
                state_recur, (state_S, state_L), self.gate = self.state_net(actor_input, self.state_net_streams)
                ctx_recur, (ctx_S, ctx_L), _ = self.ctx_net(actor_input, self.ctx_net_streams)

        self.pred_state = self.soft(self.to_state(state_recur)).view(-1)         
        self.pred_ctx = torch.sigmoid(self.to_ctx(ctx_recur)).view(-1)                     
        self.ctx_net_streams = (ctx_S, ctx_L)                                                            
        self.state_net_streams = (state_S, state_L)                                                            
        network_LLR = torch.log((self.pred_state.detach()[0, None] + eps) / (self.pred_state.detach()[1, None] + eps))            
        self.state_ctx = torch.cat((network_LLR, self.pred_ctx.detach()), -1)  
        
        self.LTM = state_L.detach()  
        self.output = state_S.detach()        

    def get_bayes_actor(self, actor_input):
        if self.subnets in self.optim_list:
            self.actor_standard(actor_input)
        
        # bayes expects inputs as one_hot 
        GO  = actor_input[0,0,1].cpu().numpy().astype(int)
        ACT = actor_input[0,0,3].cpu().numpy().astype(int)
        REW = actor_input[0,0,4].cpu().numpy().astype(int)
        p_state, ctx = self.bayes.bayes_forward(GO, ACT, REW)
        self.postprocess_bayes_forward(p_state, ctx)
        if self.pavlovian_bias:
            self.state_ctx = torch.cat((self.bayes_LLR, self.pred_ctx, actor_input[0,0]), -1).float()
        else:
            self.state_ctx = torch.cat((self.bayes_LLR, self.pred_ctx), -1).float()
        self.pred_state = torch.tensor([1-p_state, p_state], device = self.device).float() 

    def postprocess_bayes_forward(self, p_state, ctx, kill_context = False):
        if torch.is_tensor(p_state):
            self.pred_ctx = ctx[None] * (1-kill_context)
            self.bayes_LLR = p_state[None]
        else:
            self.pred_ctx = torch.tensor([ctx], device = self.device).float() * (1-kill_context)
            self.bayes_LLR = torch.tensor([p_state], device = self.device).float() 


    def default_action(self):
        self.Q_value = self.to_action(self.actor_output).view(-1)                                     # output to Q values                  
        self.action_prob = self.soft(self.Q_value).view(-1)                                           # Q values to action probabilities         
        if self.mechanistic is not None:
            self.mechanistic_action()

        if self.argmax:
            self.action = torch.tensor([torch.argmax(self.Q_value)]).to(self.device)
        else:
            self.action =  Categorical(self.action_prob).sample().view(-1).detach()                  # sample action                   
                
    def subnet_action(self):
        self.Q_value = self.subnets_to_action(self.state_ctx).view(-1)
        self.action_prob = self.soft(self.Q_value).view(-1)                                      # Q values to action probabilities            
        if self.argmax:
             self.action = torch.tensor([torch.argmax(self.Q_value)]).to(self.device)
        else:
             self.action =  Categorical(self.action_prob).sample().view(-1).detach()                  # sample action     

    def mechanistic_action(self):
        timescale = 0.6
        inp = self.input[0, 0, 1]
        self.main_mem = self.main_mem * timescale + inp * (1-timescale)
        if self.mechanistic == "no_ctx":
            bias = 7.3
            scaling = 6
            ctx_scaling = 0
        if self.mechanistic == "ctx":
            bias = 9.3
            scaling = 10
            ctx_scaling = 3

        DV = scaling*self.main_mem - bias - self.true_ctx*ctx_scaling
        A = 1/(1+torch.exp(-DV))
        self.Q_value = torch.tensor([1-DV/2, DV/2], device = self.device)
        self.action_prob = torch.tensor([1-A, A], device = self.device)
             
    
    def get_critic(self, critic_input):        
        if (self.subnets in self.bayes_list) and (self.subnets not in self.optim_list):  
            self.value = self.subnets_to_value(self.state_ctx)
        else:
            if self.train_recurrent and self.train:
                critic_return  = self.critic(critic_input, self.critic_streams)
            else:
                with torch.no_grad():
                    critic_return = self.critic(critic_input, self.critic_streams)
    
            if self.use_vanilla_torch:
                (S, L) = critic_return
                critic_output = S
            else:
                critic_output, (S,L), G = critic_return
                if self.get_dynamics_of == 'critic':
                    self.gate = G
                    
            if self.expansion != None: 
                critic_output = self.from_expansion(torch.tanh(self.to_expansion(critic_output)))
            self.value = self.to_value(critic_output).view(-1)                                       #  output to value    
            self.critic_streams = (S, L)                                                             # long and short term memory streams  

    def backwards(self, R = 0, critic_loss = 0, policy_loss = 0, entropy_loss = 0, episode_ratio = 0, eps = 1e-10):
        if self.decrease_entropy != False:                                                                             # if linearly decreasing entropy throughout training
            if self.decrease_entropy == True:
                self.B_ent = 1 - episode_ratio                                                                             # entropy goes from 1 to 0 
            else:
                self.B_ent = self.decrease_entropy*self.B_ent                                                              # entropy goes from 1 to 0 
        log_p_act = torch.clip(self.action_probs, min = eps).log()

        self.rewards = (self.cost_of_action + 1)*self.rewards   
        act_inds = torch.where(self.actions*(1-torch.roll(self.actions, 1)))[0]
        if len(act_inds) > 0:
            self.rewards[act_inds] = self.rewards[act_inds] - self.cost_of_action
                        
        self.rewards[-1] = self.values[-1].detach()          
        for i, r in reversed(list(enumerate(self.rewards))):                                                           # iterate through rewards from most to least recent 
            act_i = self.actions[i].long()                                                                             # index of action taken             
            R = r + self.discount*R                                                                                    # sum of discounted future rewards
            
            RPE = R - self.values[i]           
            critic_loss = critic_loss +  RPE**2                                                                         # value loss is MSE between predicted value and sum of discounted future reward
            log_likelihood = log_p_act[act_i, i]       
            entropy_loss = entropy_loss + self.action_probs[act_i, i] * log_likelihood              # entropy loss is maximizing entropy
            policy_loss = policy_loss - log_likelihood *  RPE.detach()                               # policy loss minimizes the negative LLR(action) * reward prediction error                                                                 # value loss is MSE between predicted value and sum of discounted future reward
        self.update(critic_loss, entropy_loss, policy_loss)
        
    def update(self, critic_loss, entropy_loss, policy_loss):
        loss = (policy_loss + self.B_val*critic_loss + self.B_ent*entropy_loss).sum()
        if self.subnets != None and self.subnets not in self.bayes_list:
            loss = loss + self.get_subnet_loss()            
        self.optim.zero_grad()
        
        loss.backward()       
        self.optim.step()
        self.schedule.step()
        
    def get_subnet_loss(self):
        state_loss = self.CEL(self.pred_states, self.true_states)/self.true_states.shape[1]
        ctx_loss = self.MSE(self.pred_ctxs, self.true_ctxs)
        return state_loss + ctx_loss
        
    def empty_layers(self):
        return self.closed.unsqueeze(0).detach(), self.open.detach(), self.open.detach(), self.open.detach(), self.open.detach()