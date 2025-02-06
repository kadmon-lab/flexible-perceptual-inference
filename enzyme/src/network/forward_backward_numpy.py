import torch.nn as nn; from torch.distributions.categorical import Categorical; import torch;  from sklearn.decomposition import PCA; 
import torch.nn.functional as F; import pylab as plt; import numpy as np

class forward_backward(nn.Module):
    def __init__(self, **params):
        super(forward_backward, self).__init__()
        self.__dict__.update(params)
          
    def forward(self, O, A, R, C, give_ctx):
        self.actor_input[0, 0, 0] = 1-O
        self.actor_input[0, 0, 1] = O
        self.actor_input[0, 0, 2] = 1-A
        self.actor_input[0, 0, 3] = A
        self.actor_input[0, 0, 4] = R
        
        self.critic_input[0, 0, 0] = 1-O
        self.critic_input[0, 0, 1] = O
        self.critic_input[0, 0, 2] = 1-A
        self.critic_input[0, 0, 3] = A
        self.critic_input[0, 0, 4] = R
        if give_ctx:
            self.actor_input[0, 0, 5] = C
            self.critic_input[0, 0, 5] = C

        self.actor_standard(self.actor_input)
        self.get_critic(self.critic_input)                                                            # run forward Critic
        self.step += 1
        return self.action, self.action_prob, self.value

    def actor_standard(self, actor_input):
         """ option to train only readout """
         if self.train_recurrent and self.train:
             actor_return = self.actor(actor_input, self.actor_streams)
         else:
             with torch.no_grad(): 
                 actor_return = self.actor(actor_input, self.actor_streams) 
         # actor_output, (S,L), G = actor_return
         actor_output, (S,L) = actor_return
         self.actor_streams = (S,L)
         self.Q_value = self.to_action(actor_output).view(-1)                                     # output to Q values                  
         self.action_prob = self.soft(self.Q_value).view(-1)                                           # Q values to action probabilities         
         self.action =  Categorical(self.action_prob).sample().view(-1).detach()                  # sample action                   
    
    def get_critic(self, critic_input):        
        if self.train_recurrent and self.train:
            critic_return  = self.critic(critic_input, self.critic_streams)
        else:
            with torch.no_grad():
                critic_return = self.critic(critic_input, self.critic_streams)
        # critic_output, (S,L), G = critic_return      
        critic_output, (S,L) = critic_return      
        self.value = self.to_value(critic_output).view(-1)                                       #  output to value    
        self.critic_streams = (S, L)                                                             # long and short term memory streams  

    def backwards(self, rewards, R = 0, critic_loss = 0, policy_loss = 0, entropy_loss = 0, eps = 1e-10):                                                      # entropy goes from 1 to 0 
        log_p_act = torch.clip(self.action_probs[:self.step+1], min = eps).log()
        rewards[self.step] = self.values[self.step].detach()   
        RPE = torch.zeros(self.step+1).to(self.device)    
        for i in range(self.step):
            R = rewards[self.step - i] + self.discount*R                                                                                    # sum of discounted future rewards
            RPE[i] = R - self.values[self.step - i]

        critic_loss = (RPE**2).sum()                                                                     # value loss is MSE between predicted value and sum of discounted future reward
        policy_loss =  -(log_p_act * RPE.detach()).sum()                         # policy loss minimizes the negative LLR(action) * reward prediction error                                                                 # value loss is MSE between predicted value and sum of discounted future reward
        self.update(critic_loss, entropy_loss, policy_loss)
        
    def update(self, critic_loss, entropy_loss, policy_loss):
        loss = (policy_loss + self.B_val*critic_loss + self.B_ent*entropy_loss).sum()      
        self.optim.zero_grad()        
        loss.backward()       
        self.optim.step()
