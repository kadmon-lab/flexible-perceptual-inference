import numpy as np;  from enzyme.src.mouse_task.mouse_plotting import mouse_plotting; import torch; import matplotlib.cm as cm;         

class mouse_task_(mouse_plotting):
    def __init__(self, **params):
        super(mouse_task_, self).__init__()
        self.__dict__.update(params)
        self.reset(0)
        
    def reset(self, e):
        """
        Resets the world with a new theta.
        """
        self.e = e
        if e == 0:
            self.total_T = 0 
        self.inaction, self.curr_trial, self.sim_ID, self.temp_resolution = [-10, -1, 3, 100]
        self.fields = ["backbone", "safe_backbone", "stim", "rew", "act_time", "stim_dur", "stim_end", "W4L_end", "end_times", "PGO", "safe_PGO", "gos", "nogos", \
                      "last_nogo", "last_pot_nogo", "wait_from_last", "wait_from_last_pot", "plant_inds", "plant_PGO", "plant_ID"]
        # self.plant_options = int(type(self.plant_type[0]) != list or self.plant_type == 'random') or np.shape(self.plant_type)[0] 
        self.plant_options = int(self.plant_type is None or type(self.plant_type) == str or type(self.plant_type) == list) or np.shape(self.plant_type)[0] 
        # SINGLE PLANTED TRIALS MUST BE LISTS, MULTIPLE PLANTED TRIAL OPTIONS MUST BE ARRAYS        
        self.rew, self.stim_dur, self.stim_end, self.W4L_end, self.end_times, self.prem_end = [np.zeros(self.num_trials) for _ in range(6)]
        self.plant_PGO, self.plant_ID, self.act_time, self.wait_from_last = [self.inaction * np.ones(self.num_trials) for _ in range(4)]
        self.stim, self.pot_stim = [np.empty(self.num_trials, dtype = object) for _ in range(2)]
        self.PGO_N = len(self.PGO_range)
        self.prep_trials()

    def prep_trials(self):
        """ probabilistic safe """
        self.safe_PGO = 1#float(np.random.choice([0,1]))
        """ probabilistic safe """
        self.trial_dur = self.exp_max + self.ITI_mean + self.ITI_PM + self.W4L + self.prem_dur  
        self.backbone = np.ones((self.num_trials, self.trial_dur))      
        self.safe_backbone = np.zeros((self.num_trials, self.trial_dur))      
        self.PGO_ind = np.random.randint(self.PGO_N)
        self.PGO = self.PGO_range[self.PGO_ind]
        self.ITI_PGO = self.PGO 
        self.W4L_window = np.arange(self.W4L)

        self.stim_durs = np.clip(np.random.exponential(self.exp_mean, size = self.num_trials), a_min = self.exp_min, a_max = self.exp_max).astype(int)
        self.ITI_durs = np.random.uniform(self.ITI_mean - self.ITI_PM, self.ITI_mean + self.ITI_PM, size = self.num_trials).astype(int)
        self.plant_inds = np.random.choice([0, 1], p = [1 - self.plant_prob, self.plant_prob], size = self.num_trials)


        self.t_from_nogo = 0 
        self.reset_trial()

    def reset_trial(self):
        self.curr_trial += 1 
        self.make_beep_train()
        if self.plant_inds[self.curr_trial] == 1:
            self.overwrite_with_plant()
        if self.start_NOGO:
            self.curr_stim = self.curr_stim.copy()
            self.curr_stim[0] = 0
        if self.end_NOGO:
            self.curr_ITI = self.curr_ITI.copy()
            self.curr_ITI[-1] = 0

        self.stim_ind, self.W4L_ind, self.ITI_ind, self.prem_ind, self.backbone_ind = [0 for _ in range(5)]
        self.cont, self.ongoing_stim, self.ongoing_W4L, self.ongoing_ITI, self.acted = [True, True, False, False, False]

    def make_beep_train(self, P = None, stim = None, ITI = None, W4L = None, prem = None):
        P = self.PGO if P == None else P
        self.curr_stim = np.random.binomial(1, P, self.stim_durs[self.curr_trial]) if stim is None else stim
        self.curr_ITI = np.random.binomial(1, P, self.ITI_durs[self.curr_trial]) if ITI is None else ITI 
        self.curr_W4L = np.random.choice(a = [0, 1], p=[1 - self.safe_PGO, self.safe_PGO], size=self.W4L) if W4L is None else W4L
        self.curr_prem = np.random.binomial(1, P, self.prem_dur) if prem is None else prem 


    def overwrite_with_plant(self):
        if self.plant_options == 1 and self.plant_type == 'random':
            plant_PGO_ind = np.random.randint(self.PGO_N)
            self.plant_PGO[self.curr_trial] = self.PGO_range[plant_PGO_ind]
            self.make_beep_train(P = self.plant_PGO[self.curr_trial])
        else:
            if self.plant_options == 1:            
                train = np.array(self.plant_type)
            else:
                idx = np.random.choice(self.plant_options)
                self.plant_ID[self.curr_trial] = idx
                train = np.array(self.plant_type[idx])
            self.make_beep_train(stim = train, ITI = train)
            self.stim_durs[self.curr_trial] = len(self.curr_stim)        
            self.ITI_durs[self.curr_trial] = len(self.curr_ITI)

    def get_observation(self):
        self.ongoing_stim = (self.acted == False) and (self.stim_ind < self.stim_durs[self.curr_trial])
        self.ongoing_W4L = not self.ongoing_stim and (self.acted == False) and (self.W4L_ind < self.W4L_window[-1])
        self.ongoing_prem = not self.ongoing_stim and not self.ongoing_W4L and (self.W4L_ind == 0) and self.acted and (self.prem_ind < self.prem_dur)
        self.ongoing_ITI = not self.ongoing_stim and not self.ongoing_W4L and not self.ongoing_prem and (self.ITI_ind < self.ITI_durs[self.curr_trial]-1)
        observation = self.ongoing()
        self.get_state_context()
        return observation, self.true_state, self.true_ctx
    
    def ongoing(self):
        if np.any(self.theta_traj != None):
            self.handle_theta_traj()
        
        if self.ongoing_stim:
            self.backbone[self.curr_trial, self.backbone_ind] = self.curr_stim[self.stim_ind]
            self.stim_ind += 1 
        elif self.ongoing_W4L:
            self.backbone[self.curr_trial, self.backbone_ind] = self.curr_W4L[self.W4L_ind]            
            self.safe_backbone[self.curr_trial, self.backbone_ind] = 1
            self.W4L_ind += 1 
        elif self.ongoing_prem:
            self.backbone[self.curr_trial, self.backbone_ind] = self.curr_prem[self.prem_ind]
            self.prem_ind += 1 
        elif self.ongoing_ITI:
            self.backbone[self.curr_trial, self.backbone_ind] = self.curr_ITI[self.ITI_ind]
            self.ITI_ind += 1 
        else:
            self.backbone[self.curr_trial, self.backbone_ind] = self.curr_ITI[self.ITI_ind]
            self.end_trial()

        self.t_from_nogo = self.backbone[self.curr_trial, self.backbone_ind] * (self.t_from_nogo + 1)
        if self.inp_is_consec:
            return self.t_from_nogo             # remember to set inp to "raw"            
        return self.backbone[self.curr_trial, self.backbone_ind]


    def get_state_context(self):
        self.true_state = int(self.ongoing_W4L) 
        self.true_ctx = self.PGO

    def continue_trial(self, s):
        if self.cont == False:
          if self.curr_trial < self.num_trials - 1:
              self.reset_trial()
          return False
        self.total_T += 1 
        return True
  
    def get_reward(self, observation, action):
        r = self.handle_action(action)
        self.backbone_ind += 1
        reward = torch.tensor(r)
        return reward[None].float().to(self.device)

    def handle_action(self, action, r = 0):
        if action:
            if self.acted == False:
                self.acted = True
                self.act_time[self.curr_trial] = int(self.backbone_ind)
                if self.ongoing_W4L:
                    r = self.rew[self.curr_trial] = 1 
                elif self.ongoing_stim:
                    r = self.rew[self.curr_trial] = self.neg_rew
                self.wait_from_last[self.curr_trial] = self.t_from_nogo
            self.t_from_nogo = 0
        return r     
    
    def handle_theta_traj(self):
        P = self.theta_traj[self.total_T % len(self.theta_traj)]
        x = np.random.binomial(1, P, 1)
        if self.ongoing_stim:
            self.curr_stim[self.stim_ind] = x 
        elif self.ongoing_ITI: 
            self.curr_ITI[self.ITI_ind] = x

    def end_trial(self):
        self.cont = False
        self.pot_stim[self.curr_trial] = self.curr_stim
        self.stim_end[self.curr_trial] = self.stim_ind 
        self.stim_dur[self.curr_trial] = self.stim_durs[self.curr_trial]
        self.W4L_end[self.curr_trial] = self.stim_ind + self.W4L_ind
        self.stim[self.curr_trial] = self.backbone[self.curr_trial, :self.stim_ind]        
        self.prem_end[self.curr_trial] = self.stim_ind + self.prem_ind
        self.end_times[self.curr_trial] = self.stim_ind + self.W4L_ind + self.prem_ind + self.ITI_ind

        
    def get_episode_log(self):
        backbone, safe_backbone, gos, nogos, last_nogo, last_pot_nogo, consec_gos, wait_from_last, wait_from_last_pot, plant_PGOs, plant_IDs =\
            [np.empty(self.num_trials, dtype = object) for _ in range(11)]
        for t in range(self.num_trials):
            safe_backbone[t] = self.safe_backbone[t, :int(self.end_times[t]+1)]
            backbone[t] = self.backbone[t, :int(self.end_times[t]+1)]
            nogos[t] = np.where(self.stim[t] == 0)[0]
            gos[t] = np.where(self.stim[t] == 1)[0]

            pot_nogos = np.where(self.pot_stim[t] == 0)[0]
            last_nogo[t] = None if len(nogos[t]) == 0 else nogos[t][-1]
            last_pot_nogo[t] =  None if len(pot_nogos) == 0 else pot_nogos[-1]
            plant_PGOs[t] = None if self.plant_prob == 0 else self.plant_PGO[t]
            plant_IDs[t] = None if self.plant_prob == 0 else self.plant_ID[t]
            
            last =  -1 if last_nogo[t] == None else int(last_nogo[t])
            last_pot =  -1 if last_pot_nogo[t] == None else int(last_pot_nogo[t])
            wait_from_last_pot[t] =  self.end_times[t] if self.act_time[t] == self.inaction else self.act_time[t] - last_pot 
                                                      
        log = [backbone, safe_backbone, self.stim, self.rew, self.act_time, self.stim_dur, self.stim_end, self.W4L_end, self.end_times,\
               self.PGO, self.safe_PGO, gos, nogos, last_nogo, last_pot_nogo,  self.wait_from_last, wait_from_last_pot, self.plant_inds, plant_PGOs, plant_IDs]
        return log