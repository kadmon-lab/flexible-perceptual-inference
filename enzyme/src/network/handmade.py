import torch; from torch import nn; import math
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class NET(nn.Module):
    def __init__(self, input_dim, hidden_dim, lesion = None, override = None, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.override = override
        self.lesion = lesion
        self.device = device
        self.construction()
        self.to(device)

    def construction(self):             
        self.soft = nn.Softmax(dim = -1)
        self.igates = torch.nn.Linear(1,1).to(self.device)
        self.hgates = torch.nn.Linear(1,1).to(self.device)
        # mem_len = 1
        # mem_len = 15
        # mem_len = 20
        mem_len = 20
        # mem_len = 50
        print(mem_len * self.input_dim)
        
        if self.override == "RNN":
            self.igates = torch.nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
            self.hgates = torch.nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
            self.init_weights()

        if self.override == "LSTM":
            self.igates = torch.nn.Linear(self.input_dim, 4 * self.hidden_dim).to(self.device)
            self.hgates = torch.nn.Linear(self.hidden_dim, 4 * self.hidden_dim).to(self.device)
            self.init_weights()
            
        if self.override == "GRU":
            self.igates = torch.nn.Linear(self.input_dim, 2 * self.hidden_dim).to(self.device)
            self.hgates = torch.nn.Linear(self.hidden_dim, 2 * self.hidden_dim).to(self.device)
            self.icgate = torch.nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
            self.hcgate = torch.nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
            self.icgate.bias.requires_grad = False 
            self.hcgate.bias.requires_grad = False 
            self.icgate.weight.requires_grad = False 
            self.hcgate.weight.requires_grad = False 
            self.init_weights(detach = True)
            
        if self.override == "exp_decay":
            max_decay = .6
            self.ctx_mask = torch.zeros(self.hidden_dim).to(self.device)
            self.ctx_mask[-1] = 1 
            self.decay_range = torch.linspace(0, max_decay, self.hidden_dim).to(self.device)
            self.init_weights(detach = True)

        if self.override == "exp_decay_learnable":
            self.decay_range = torch.nn.Parameter(torch.rand(self.hidden_dim, device = self.device))
            self.init_weights(detach = True)

        if self.override == "exp_decay_interaction":
            self.decay_range = torch.nn.Parameter(torch.rand(self.hidden_dim, device = self.device))
            self.inp_to_trace = torch.nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
            self.state_modulation = torch.nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
            self.init_weights(detach = True)
            
        if self.override == "dynamic_decay":
          self.inp_to_trace = torch.nn.Linear(self.input_dim, 1).to(self.device)
          self.inp_to_decay = torch.nn.Linear(self.input_dim, 1).to(self.device)
          self.init_weights(detach = True)
            
        if self.override == 'FF_mem': 
            self.mem_2_out =  torch.nn.Linear(self.input_dim*mem_len, self.hidden_dim).to(self.device)
            self.memory =  torch.zeros(self.input_dim, mem_len).to(self.device)
            self.init_weights(detach = True)
                
        if self.override == 'FF_mem_simple': 
            self.mem_2_out =  torch.nn.Linear(self.input_dim*mem_len, self.hidden_dim).to(self.device)
            self.memory =  torch.zeros(self.input_dim, mem_len).to(self.device)
            self.init_weights(detach = True)
    
        if self.override == 'FF_mem_gated': 
            self.mem_2_linear =  torch.nn.Linear(self.input_dim*mem_len, self.hidden_dim).to(self.device)
            self.mem_2_gate =  torch.nn.Linear(self.input_dim*mem_len, self.hidden_dim).to(self.device)
            self.memory =  torch.zeros(self.input_dim, mem_len).to(self.device)
            self.init_weights(detach = True)

        if self.override == 'FF_int':
            self.prev_inp, self.inp_int = [torch.zeros(1, self.input_dim, requires_grad = False).to(self.device) for _ in range(2)]
            self.int_2_FF =  torch.nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
            self.init_weights(detach = True)

        if self.override == 'LSTM_mem': 
            self.mem_2_LSTM =  torch.nn.Linear(self.input_dim*mem_len, self.hidden_dim).to(self.device)
            self.igates = torch.nn.Linear(self.hidden_dim, 4 * self.hidden_dim).to(self.device)
            self.hgates = torch.nn.Linear(self.hidden_dim, 4 * self.hidden_dim).to(self.device)
            self.memory =  torch.zeros(self.input_dim, mem_len).to(self.device)
            self.init_weights(detach = True)

        self.open = torch.ones(1, self.hidden_dim, requires_grad = False, device = self.device)
        self.closed = torch.zeros(1, self.hidden_dim, requires_grad = False, device = self.device)
            
    def forward(self, inp, recur):
        inp, STM, LTM = [x.squeeze(0) for x in [inp, recur[0], recur[1]]]       
        """ inp is size 1 x inp dim """
        
        if self.override == "RNN":
            out = torch.tanh((inp @ self.weight_ih_l0.T) + (STM @ self.weight_hh_l0.T) + self.bias_ih_l0 + self.bias_hh_l0)            
            LTM = STM = out.unsqueeze(0)
            f = i = c = o = out.detach()

        if self.override == "LSTM": 
            STM = (1 - int('STM' in self.lesion)) * STM
            gates = inp @ self.weight_ih_l0.T + STM @ self.weight_hh_l0.T + self.bias_ih_l0 + self.bias_hh_l0
            f, i, c, o = gates.chunk(4, 1)
            f = self.closed if 'LTM' in self.lesion else self.open if 'FORGET' in self.lesion else torch.sigmoid(f) 
            i = self.open if 'INPUT' in self.lesion else torch.sigmoid(i)
            o = self.open if 'OUTPUT' in self.lesion else torch.sigmoid(o) 

            c = torch.tanh(c)
            LTM = i * c + LTM * f  
            STM = torch.tanh(LTM) * o
            LTM = LTM.unsqueeze(0)
            STM = STM.unsqueeze(0)
            
        if self.override == "GRU": 
            gates = inp @ self.weight_ih_l0.T + LTM @ self.weight_hh_l0.T + self.bias_ih_l0 
            f, o = gates.chunk(2, 1)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            i = 1 - o
            STM = LTM * f
            
            c = torch.tanh(self.icgate(inp) + self.hcgate(STM))
            out = i * c + LTM * o           
            LTM = STM = out.unsqueeze(0)
            f = i = c = o = out.detach()

        if self.override == "exp_decay":
            STM = STM * self.decay_range[None, :] + (1-self.decay_range) * inp[0, 1] 
            STM = STM * (1 - self.ctx_mask) + self.ctx_mask * inp[0, -1]
            LTM, f, i, c, o = self.empty_layers()
            STM = STM.unsqueeze(0)

        if self.override == "exp_decay_learnable":
            d = torch.sigmoid(self.decay_range[None, :])
            STM =  STM * d +  (1-d) * inp[0, 1]
            LTM, f, i, c, o = self.empty_layers()
            STM = STM.unsqueeze(0)


        if self.override == "exp_decay_interaction":
            d = torch.sigmoid(self.decay_range[None, :])
            state_d = torch.sigmoid(self.state_modulation(STM))
            inp = torch.sigmoid(self.inp_to_trace(inp))
            STM = STM * d * state_d + (1-d) * inp            
            STM = STM.unsqueeze(0)
            LTM, f, i, c, o = self.empty_layers()
 
        if self.override == "dynamic_decay":
            d = torch.sigmoid(self.inp_to_decay(inp))
            STM = STM * d  + (1-d) * self.inp_to_trace(inp)            
            STM = STM.unsqueeze(0)
            LTM, f, i, c, o = self.empty_layers()           
            

        if self.override == 'FF_mem':
            self.memory = torch.roll(self.memory, -1, 1)
            self.memory[:, -1] = inp[0, 0]     
            mem = self.memory.flatten()[None, :]                   
            STM = torch.tanh(self.mem_2_out(mem))[None,:,:]
            # STM = torch.relu(self.mem_2_out(mem))[None,:,:]
            LTM, f, i, c, o = self.empty_layers()
            
        if self.override == 'FF_mem_simple':
            self.memory = torch.roll(self.memory, -1, 1)
            self.memory[:, -1] = inp[0, :]     
            mem = self.memory.flatten()[None, :]                   
            STM = self.mem_2_out(mem)[None,:,:]               
            LTM, f, i, c, o = self.empty_layers()

        if self.override == 'FF_mem_gated':
            LTM, f, i, c, o = self.empty_layers()
            self.memory = torch.roll(self.memory, -1, 1)
            self.memory[:, -1] = inp[0, :]     
            mem = self.memory.flatten()[None, :]                   
            c = torch.tanh(self.mem_2_linear(mem))
            f = torch.sigmoid(self.mem_2_gate(mem))
            STM = c*f
            STM = STM[None,:,:]
            
        if self.override == 'FF_int':
            pos_neg_inp = ((inp[:2] - .5).float()*2)
            same_inp = (pos_neg_inp == self.prev_inp).bool()
            self.inp_int = self.inp_int * same_inp + pos_neg_inp  
            inp = torch.cat((inp[2:], self.inp_int))
            STM = torch.relu(self.int_2_FF(inp))[None, :, :]
            LTM, f, i, c, o = self.empty_layers()
            self.prev_inp = pos_neg_inp
                       
        if self.override == 'LSTM_mem':
            self.memory = torch.roll(self.memory, -1, 1)
            self.memory[:, -1] = inp[0, :]     
            mem = self.memory.flatten()[None, :]            
            inp = torch.tanh(self.mem_2_LSTM(mem))
            gates = inp @ self.weight_ih_l0.T + STM @ self.weight_hh_l0.T + self.bias_ih_l0 + self.bias_hh_l0
            f, i, c, o = gates.chunk(4, 1)
            f =  torch.sigmoid(f) 
            i =  torch.sigmoid(i)
            o = torch.sigmoid(o) 
            c = torch.tanh(c)
            LTM = i * c + LTM * f  
            STM = torch.tanh(LTM) * o
            LTM = LTM.unsqueeze(0)
            STM = STM.unsqueeze(0)

        return STM, (STM, LTM), torch.cat((f, i, c, o), 0)
    
    def empty_layers(self):
        return self.closed.unsqueeze(0).detach(), self.open.detach(), self.open.detach(), self.open.detach(), self.open.detach()
    
    def init_weights(self, detach = False):
        if detach: 
            self.igates.weight.requires_grad = False 
            self.hgates.weight.requires_grad = False 
            self.igates.bias.requires_grad = False
            self.hgates.bias.requires_grad = False 
        self.weight_ih_l0 = self.igates.weight
        self.weight_hh_l0 = self.hgates.weight
        self.bias_ih_l0 = self.igates.bias
        self.bias_hh_l0 = self.hgates.bias
        # self.weight_ih_l0 = torch.nn.Parameter((2*torch.rand(4*self.hidden_dim, self.input_dim, device = self.device)-1)/(math.sqrt(self.hidden_dim)))
        # self.weight_hh_l0 = torch.nn.Parameter((2*torch.rand(4*self.hidden_dim, self.hidden_dim, device = self.device)-1)/(math.sqrt(self.hidden_dim)))
        # self.bias_ih_l0 = torch.nn.Parameter((2*torch.rand(4*self.hidden_dim, device = self.device)-1)/(math.sqrt(self.hidden_dim)))
        # self.bias_hh_l0 = torch.nn.Parameter((2*torch.rand(4*self.hidden_dim, device = self.device)-1)/(math.sqrt(self.hidden_dim)))
