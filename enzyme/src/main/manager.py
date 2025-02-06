import numpy as np; from enzyme.src.mouse_task.data_structures import mouse_data_struct; import torch; import copy

class manager():
    def __init__(self, simulation, simulation_params, agent):
        self.__dict__.update(simulation_params)
        self.agent = agent
        self.sim = simulation(**simulation_params);                          
        if self.sim.sim_ID == 1:
          self.sim1_log()
        if self.sim.sim_ID == 2:
          self.sim2_log()
        if self.sim.sim_ID == 3:
          self.sim3_log()

    def sim1_log(self): 
        self.normalized_rewards, self.action_ratio, self.reward_ratio, self.held_out_log = [ np.zeros(self.episodes) for _ in range(4) ]

    def sim2_log(self):
        self.episode_lr_estim = np.zeros((self.episodes, self.sim.RW_steps)) 
        self.episode_input = np.zeros((self.episodes, self.agent.inp_dim, self.sim.num_trials)) 
        self.episode_temp, self.episode_lr = [np.zeros((self.episodes, self.sim.num_trials - 1 )) for _ in range(2)]
        self.episode_Qs, self.episode_action_probs = [np.zeros((self.episodes, self.agent.act_dim, self.sim.num_trials)) for _ in range(2)]     
        self.episode_volatility, self.episode_RRP, self.episode_actions = [np.zeros((self.episodes, self.sim.num_trials)) for _ in range(3)]
        self.episode_i_gate, self.episode_f_gate, self.episode_c_gate, self.episode_o_gate, self.episode_output = [np.zeros((self.episodes, self.agent.hid_dim, self.sim.num_trials)) for _ in range(5)]

            
    def sim3_log(self):
        if not self.skip_storage:
            self.last_PGO = -1
            self.ones = torch.ones(1,1).to(bool)
            self.trials = np.arange(self.sim.num_trials, dtype = object)        
            self.manager_fields = ["episode", "trial", "last_PGO", "lick_prob", "value", "Qs", "net_input", "net_output", "LTM", "f_gate",  "i_gate", "c_gate", "o_gate","episode_inp_W", "episode_recur_W"]
            self.lick_prob, self.value, self.Qs, self.net_input, self.net_output, self.LTM,self.f_gate, self.i_gate, self.c_gate, self.o_gate  = [np.empty(self.sim.num_trials, dtype = object) for _ in range(10)]
            self.data = mouse_data_struct(self.manager_fields + self.sim.fields, self.episodes * self.sim.num_trials, self.sim.num_trials)

    def log_trials(self):
      if self.sim.sim_ID == 1:
        action_ratio, reward_ratio = self.sim.get_episode_log()                                              # get episode left and right actions from simulation
        self.action_ratio[self.e] = action_ratio                                                             # get episode action ratio
        self.reward_ratio[self.e] = reward_ratio                                                             # get episode reward ratio
        self.held_out_log[self.e] = self.sim.held_out
        self.normalized_rewards[self.e] = sum(self.sim.reward_log/self.sim.reward_normalize)                 # store outcome normalized to max possible
    
      if self.sim.sim_ID == 2 and self.sim.training == False:
        if self.sim.mod != None: 
          lr_estim=lr_actual= 0 
          temp= np.zeros(self.sim.num_trials)
        else:
          lr_estim, lr_actual, temp = self.sim.get_episode_log(self)
        self.episode_temp[self.e, :] = temp[:-1]                                                           # all except last just to have same shape as others
        self.episode_lr_estim[self.e, :] = lr_estim
        self.episode_lr[self.e, :] = lr_actual
        self.get_sim_2_network_data()
          
      if self.sim.sim_ID == 3:
        if not self.skip_storage:
            self.get_sim_3_network_data() 
            self.data.push_multiple(self.manager_fields, self.manager_data_log)
            self.data.push_multiple(self.sim.fields, self.sim.get_episode_log())
            self.data.increment()
            self.last_PGO = self.sim.PGO
                  
    def get_from_agent(self):
        self.LTMs = self.agent.LTMs
        self.values = self.agent.values
        self.Q_values = self.agent.Q_values
        self.gates = self.agent.gates
        self.inputs = self.agent.inputs
        self.outputs = self.agent.outputs
        self.actions = self.agent.actions.detach()
        self.action_probs = self.agent.action_probs.detach()
        self.episode_inp_W = self.agent.state_dict()['actor.igates.weight'].detach().clone().cpu().numpy()
        self.episode_recur_W = self.agent.state_dict()['actor.hgates.weight'].detach().clone().cpu().numpy()
           
    def get_sim_2_network_data(self):
        self.get_from_agent()
        #episode, neuron, trial 
        self.episode_action_probs[self.e, :, :] = self.action_probs.cpu()
        self.episode_f_gate[self.e, :, :] = self.gates[0, :].cpu()
        self.episode_i_gate[self.e, :, :] = self.gates[1, :].cpu() 
        self.episode_c_gate[self.e, :, :] = self.gates[2, :].cpu()
        self.episode_o_gate[self.e, :, :] = self.gates[3, :].cpu()
        self.episode_actions[self.e, :] = self.actions.cpu()
        self.episode_output[self.e, :] = self.outputs.cpu()
        self.episode_Qs[self.e, :, :] = self.Q_values.cpu()
        self.episode_input[self.e, :] = self.inputs.cpu()
        self.episode_volatility[self.e,:] = self.sim.v
        self.episode_RRP[self.e,:] = self.sim.r

    def get_sim_3_network_data(self, last_end = 0):
        if not self.skip_storage:
            self.get_from_agent() 
            ends = np.cumsum(self.sim.end_times+1)
            for t, next_end in enumerate(ends):
                s = int(last_end)
                e = int(next_end) 
                o = self.ones.repeat(1, e-s)
                self.get_net_data(s, e, t, o)
                last_end = next_end
    
            self.manager_data_log = \
                [self.e, self.trials, self.last_PGO, self.lick_prob, self.value, self.Qs, self.net_input, self.net_output,\
                self.LTM, self.f_gate, self.i_gate, self.c_gate, self.o_gate, self.episode_inp_W, self.episode_recur_W]      
            
    def get_net_data(self, s, e, t, o):        
        self.Qs[t] = self.Q_values[:, s : e]
        self.net_input[t] = self.inputs[:, s : e] 
        self.value[t] = self.values[s : e].detach()
        self.lick_prob[t] = self.action_probs[1, s : e].detach()
        self.net_output[t] = self.outputs[:, s : e]  if self.store_tensors  else o     # COMMENT HERE TO VIEW TRAINING PC EVOLUTION
        self.f_gate[t] = self.gates[0, :, s : e] if self.store_tensors  else o
        self.i_gate[t] = self.gates[1, :, s : e] if self.store_tensors  else o
        self.c_gate[t] = self.gates[2, :, s : e] if self.store_tensors  else o
        self.o_gate[t] = self.gates[3, :, s : e] if self.store_tensors  else o
        self.LTM[t] = self.LTMs[:, s : e] if self.store_tensors  else o
