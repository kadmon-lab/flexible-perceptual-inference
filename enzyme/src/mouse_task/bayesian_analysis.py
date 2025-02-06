import numpy as np
from tqdm import tqdm; from enzyme.src.mouse_task.trajectory_processing import trajectory_processing;  from sklearn.linear_model import LinearRegression; import time; import pylab  as plt

class bayesian_analysis(trajectory_processing):
    def __init__(self, **params):
        self.__dict__.update(params)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Bayesian Flow 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def get_bayes_flow(self, factorize = False, skip = False, ITI_approx = True, ctx_approx = True):
        self.ctx_approx = ctx_approx
        self.ITI_approx = ITI_approx
        self.factorize = factorize
        self.init_bayes_flow()
        self.init_distributional_vars()
        if not skip:
            self.infer_theta()
            
        if self.factorize:
            self.factorized_theta_structured = self.raw_to_structured(self.bayes_theta_backbone)
            self.factorized_belief_structured = self.raw_to_structured(self.bayes_belief_backbone)
        else:
            self.flow_theta_structured = self.raw_to_structured(self.bayes_theta_backbone)
            self.flow_belief_structured = self.raw_to_structured(self.bayes_belief_backbone)
            self.flow_thresh_structured = self.raw_to_structured(self.bayes_thresh_backbone)
            self.flow_belief_RMS_structured = self.raw_to_structured(self.bayes_belief_RMS_backbone)
            self.flow_thresh_RMS_structured = self.raw_to_structured(self.bayes_thresh_RMS_backbone)


    def init_bayes_flow(self):
        self.bayes_theta_backbone, self.bayes_belief_backbone, self.bayes_thresh_backbone, self.bayes_belief_RMS_backbone, self.bayes_thresh_RMS_backbone = [.5*np.ones(self.num_steps) for _ in range(5)]
        self.step_inputs = self.to(self.flatten_trajectory(self.data.net_input), 'np')
        self.belief_dist = np.zeros((3, self.num_steps))
        if self.step_inputs.shape[0] > 3:
            self.stim_traj = self.step_inputs[:2]
            self.act_traj = self.step_inputs[3]
            self.rew_traj = self.step_inputs[4]
        else:    
            # if input type is "raw" input 
            self.stim_traj = self.step_inputs[0]
            self.act_traj = self.step_inputs[1]
            self.rew_traj = self.step_inputs[2]

    def get_bayes_mu_var(self, eps = 1e-12):
        p_state = self.joint_dist.sum(-1)
        p_theta = self.joint_dist.sum(0)[:, None]
        p_theta = p_theta/p_theta.sum()
       
        p_safe_and_theta = self.joint_dist[1, :, None]
        var = p_safe_and_theta * (p_safe_and_theta/p_theta - p_state[1, None, None])**2        
        self.bayes_belief_RMS_backbone[self.i] = np.sqrt(var.sum())

        weighted_p_safe = (self.total_ana_rews*p_theta).sum(0)
        rew_rate = (self.total_ana_rews/(self.total_ana_acts + self.analytical_ITI_mean))
        self.weighted_rew_rate = (p_theta*rew_rate).sum(0)
        opt_threshold = np.argmax(self.weighted_rew_rate)
        self.bayes_thresh_backbone[self.i] = weighted_p_safe[opt_threshold]
        
        var = p_theta * (self.total_opt_thresh[:,None] - opt_threshold)**2 
        self.bayes_thresh_RMS_backbone[self.i] = np.sqrt(var.sum())
        
        self.theta_est =  (self.bayes_range[None, :] * self.joint_dist).sum()   
        self.belief_est = -np.log(1-np.clip(p_state[1], a_min = eps, a_max = 1-eps))

    # MICRO
    def init_distributional_vars(self):
        ITI_mean = self.analytical_ITI_mean
        ITI_PM = self.ITI_PM
        if self.ITI_approx: 
            ITI_PM = 0
            ITI_mean = 1
            print(f"bayes ITI_mean = {ITI_mean} ITI PM = {ITI_PM}, ITI_approx = {self.ITI_approx}, ctx_approx = {self.ctx_approx}")

        self.delta_max = 2 * ITI_PM        
        self.micro_N = 2 + ITI_mean + ITI_PM 
        self.min_leave_time = ITI_mean - ITI_PM - 1
        self.max_leave_time = ITI_mean + ITI_PM
        self.joint_dist = np.ones((self.micro_N, self.bayes_resolution))                                                                                   # 0 = unsafe, 1 = safe, 2:28 = ITI 
        self.T_r = np.eye(self.micro_N)
        self.T_r[0,0] = 0
        self.T_r[1,1] = 0
        self.T_r[2, 0] = 1
        self.T_r[2, 1] = 1
        
        binary = np.array([0,1])
        self.P_X__s_theta = np.zeros((self.micro_N, self.bayes_resolution, 2))                                                                                
        self.P_X__s_theta[:,:,:] = binary[None,:]*self.bayes_range[:,None] + (1-binary)[None:]*(1-self.bayes_range[:,None])       
        self.P_X__s_theta[1] = binary   
        self.P_X__s_theta = self.P_X__s_theta/self.P_X__s_theta.sum(-1, keepdims=True)
        """ factorizing """ 
        self.P_X__s_only = self.P_X__s_theta.mean(1)                                                                                 # dims: states x NOGO/GO 
        self.P_x__active_theta_only = self.P_X__s_theta.mean(0)                                                                             # dims: theta x NOGO/GO 
                  
        self.T_s = np.zeros((self.micro_N, self.micro_N))
        self.T_s[0, 0] = 1 - 1/self.exp_mean
        self.T_s[0, -1] = 1
    
        self.T_s[1, 0] = 1/self.exp_mean
        self.T_s[1, 1] = 1

        self.ITI_leave_prob = np.ones(self.max_leave_time)
        for ITI_time in range(self.micro_N-3):        # i is state we transition to 
            if ITI_time < self.min_leave_time:
                self.ITI_leave_prob[ITI_time] = 0
            else:
                self.ITI_leave_prob[ITI_time] = 1/(self.max_leave_time - ITI_time)
            
            self.T_s[ITI_time+3, ITI_time+2] = 1 - self.ITI_leave_prob[ITI_time]
            self.T_s[0, ITI_time+2] = self.ITI_leave_prob[ITI_time]

        self.diag_inds = np.diag_indices(self.bayes_resolution)
        self.switch_prob =  1/(self.trial_ends.mean() * self.num_trials)

        self.T_theta = np.zeros((self.bayes_resolution, self.bayes_resolution))
        self.T_theta[:] = self.switch_prob / (self.bayes_resolution - 1)
        self.T_theta[self.diag_inds] = 1 - self.switch_prob
        self.T_theta = self.T_theta/self.T_theta.sum(0, keepdims=True)
              
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    infer theta given observations and computed likelihood 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def infer_theta(self):
        for self.i in tqdm(range(self.num_steps), desc=f"(factorize = {self.factorize})", disable=False):
            self.take_step_forward()
            self.get_theta_distributional()              
            
    def take_step_forward(self):
        self.curr_stim = int(self.stim_traj[1, self.i])
        self.curr_act = int(self.i - 1 in self.cum_act_times) 
        self.curr_rew = int(self.rew_traj[self.i])

    def get_theta_distributional(self, reset_every_action = True):       
        if self.curr_act:
            if reset_every_action or self.exp_ITI or (self.curr_rew or (self.joint_dist.sum(-1)[2] < np.random.rand())):
                self.joint_dist = self.T_r @ (self.r_mask(self.curr_rew) * self.joint_dist)
                self.joint_dist = self.joint_dist/self.joint_dist.sum()     
                self.T_since_action = 0
            
        self.update_joint()   
        self.get_bayes_mu_var()
       
        self.bayes_theta_backbone[self.i] = self.theta_est
        self.bayes_belief_backbone[self.i] = self.belief_est
        if np.isnan(self.theta_est):
            print("THETA NAN OCCURED")
        if np.isnan(self.belief_est):
            print("STATE NAN OCCURED")

    # MICRO
    def update_joint(self, eps = 1e-12):
        if self.factorize: 
            """ EXPERIMENTAL alternative factorizing method """            
            # p = (1 - self.bayes_range) # self.joint_dist.sum(0)
            # self.sample_theta_ind = np.random.choice(self.bayes_range_inds, p = p/p.sum())
            # PX = self.P_X__s_theta[:, self.sample_theta_ind, self.curr_stim, None] * self.P_x__active_theta_only[None, :,self.curr_stim]
            """ EXPERIMENTAL alternative factorizing method """            
            PX = self.P_X__s_only[:,self.curr_stim, None] * self.P_x__active_theta_only[None, :,self.curr_stim]   
        else:
            PX = self.P_X__s_theta[:,:,self.curr_stim]            

        if self.ctx_approx:
            self.joint_dist = (self.T_s @ self.joint_dist)
            self.joint_dist = (self.T_theta @ self.joint_dist.T).T
        else:    
            self.get_ctx_T()
            
        self.joint_dist = PX * self.joint_dist 
        self.joint_dist = np.clip(self.joint_dist, a_min = eps, a_max = None)  
        self.joint_dist = self.joint_dist/self.joint_dist.sum()    

    def r_mask(self, r):
        return np.array([[1-r, r] + [1-r]*self.max_leave_time]).T 

    def get_ctx_T(self):
        delta = 1/self.trials_per_block 
        self.T_theta[:] = delta / (self.bayes_resolution - 1)
        self.T_theta[self.diag_inds] = 1 - delta
        self.T_theta = self.T_theta/self.T_theta.sum(0, keepdims=True)   
        
        through_theta = self.T_s[0, 2:] @ (self.T_theta @ self.joint_dist[2:].T).T
        skip_theta = self.T_s[0, 2:] @ self.joint_dist[2:]

        self.joint_dist = self.T_s @ self.joint_dist 
        self.joint_dist[0] = self.joint_dist[0] + through_theta - skip_theta
    

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Bayesian span memory durations  
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    def bayes_init(self):
        self.acting = 0                                                                                                                                                        # network action 
        self.a = 1/self.exp_mean                                                                                                                                                     # p(safe)
        self.a_0 = 1 - self.a                                                                                                                                             # 1 - p(safe)
        # useful vectors 
        self.padding = np.ones(self.max_mem)                                                                                                                              # padding used as stimuli until step is larger than memory length  
        self.discount_forget = np.linspace(.85, .9, self.max_mem)[:, None]**(self.mem_range[None, :]-1)                                                                     # vector of every possible discounted past forgetting rate 
        self.bayes_reset()                                                                                                                                                # initialize new storage vectors

    def bayes_reset(self):
        # storage matrix initializations 
        self.Psafe_log, self.bayes_PGO_log, self.mem_log, self.model_free_log, self.weight_log, self.dist_log =  [np.empty(self.data_len, dtype = object) for _ in range(6)]             # Dims: [ number of trials ] = [ Trials ]
        self.curr_Psafe, self.curr_Punsafe, self.curr_PGO, self.curr_model_free, self.curr_mem = [np.ones((self.max_mem, self.trial_dur)) for _ in range(5)]                             # Dims: [ Free parameter X max trial duration] = [ Mem , Max ]
        self.curr_weight = np.ones((self.max_mem, self.trial_dur))
        self.curr_dist = np.ones((self.bayes_resolution, self.trial_dur)) 
        self.curr_n, self.running_stim = [np.zeros((self.max_mem, 1)) for _ in range(2)]                                                                                                 # Dims: [ Free parameter ] = [ Mem ]       
        self.bayes_dist = np.ones((self.max_mem, self.bayes_resolution))                                                                                                                 # Dims: [ Free parameter X bayes dist resolution] = [ Mem , Bayes ]
        self.weights = np.ones((self.max_mem, self.max_mem))                                                                                                                             # Dims: [ Free parameter X past m stimuli ] = [ Mem , Mem ]
        self.bayes_step = self.known = self.unsafe = self.loaded_n = 0                                                                                                                   # initilalizations of several variables
        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Trigger function
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    def get_bayes(self):
        self.bayes_init()
        for self.curr_trial in tqdm(range(self.data_len), desc="bayes loop"):                                                                                                                      # for each trial
            self.end = self.trial_ends[self.curr_trial] + 1 
            self.bayes_loop()                                                                                                                                             # run bayesian algorithm on trial 
            self.bayes_log()                                                                                                                                              # store bayes results

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    bayes processing
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    def bayes_log(self):
        self.model_free_log[self.curr_trial] = self.curr_model_free[:, :self.end].copy()                                                                                  # record model free window avg stim
        self.Psafe_log[self.curr_trial] = self.curr_Psafe[:, :self.end].copy()                                                                                            # record SAFE state belief 
        self.bayes_PGO_log[self.curr_trial] = self.curr_PGO[:, 1:self.end+1].copy()                                                                                       # record PGO estimate
        self.mem_log[self.curr_trial] = self.curr_mem[:,:self.end].copy()
        
        """ note we only store largest memory weights and dist due to memory constraints"""
        self.weight_log[self.curr_trial] = self.curr_weight[:,:self.end].copy()
        self.dist_log[self.curr_trial] = self.curr_dist[:,:self.end].copy()
        
        self.curr_PGO[:, 0] = self.curr_PGO[:,self.end].copy()                                                                                                            # start next trial estimate with curr trial's end estimate
        if self.curr_trial % int(self.data_len/10) == 0:                                                                                                                  
            print(str(int(100 * self.curr_trial/self.data_len)) + "%")                                                                                                    # print percentage complete
                
    def bayes_loop(self):
        for self.curr_step in range(self.end):                                                                                                                            # for each step in current trial
            self.Inference_step()                                                                                                                                         # perform inference of state beliefs 
            self.Estimation_step()                                                                                                                                        # perform estimation of parameters
            self.bayes_step += 1 
            self.unsafe += 1 
            self.known = np.clip(self.known + 1, a_min = None, a_max = self.max_mem - 1)                                                                                  # clamp the number of known steps by the maximum free param
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    bayes calculations
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """ Inference step """
    def Inference_step(self):
        self.update_n()                                                                                                                                                   # update number of consecutive gos
        self.update_b_c()                                                                                                                                                 # update inference variables
        self.update_psafe()                                                                                                                                               # update safe state probability 
        
    def update_n(self):
        self.stim_1 = self.backbone_flat[self.bayes_step]                                                                                                                 # get if curr stim is GO 
        self.curr_n = np.minimum(self.known, self.curr_n)                                                                                                                 # cap consec GOs by last known
        self.curr_n = (1-self.acting)*self.stim_1*(self.curr_n + 1)                                                                                                       # increment if curr GO
        self.curr_n = np.minimum(self.mem_range[:, None], self.curr_n)                                                                                                    # cap consec GOs by free param (usually mem size)
        
    def update_b_c(self):     
        self.b = self.a_0*self.curr_PGO[:, self.curr_step, None].copy()                                                                                                   # use curr estimate of PGO 
        self.b_n = (self.b**self.curr_n)                                                                                                                                  # b^n
        self.b_0 = 1 - self.b                                                                                                                                             # 1 - b
        self.b_n_0 = 1 - self.b_n                                                                                                                                         # 1 - b^n 
        self.b_sum_to_n_min_1 = self.b_n_0 / self.b_0                                                                                                                     # sum over i from 0 -> n-1 of b^i = (1-b^n) / (1-b)
        n_min_k = np.maximum(0, self.curr_n - np.arange(self.max_mem))                                                                                                    # create vector of n, n-1, n-2 .... 0 
        self.b_k_min_n = self.b**(n_min_k) - self.b**self.curr_n                                                                                                          # subtract b^n from sum through all possible consec GOs 
        self.c = self.a_0*self.b_0                                                                                                                                        # (1-a) * (1-b)
    
    def update_psafe(self):
        self.curr_Punsafe[:, self.curr_step] = (self.b_n/(1 - self.c * self.b_sum_to_n_min_1)).squeeze()                                                                  # calculate psafe = b^n / ( 1 - c * (1 - b^n)/(1-b))
        self.curr_Psafe[:, self.curr_step] = (1 - self.curr_Punsafe[:, self.curr_step])                                                                                   # get Psafe as a function of n and current PGO estimate  
        
    """ Estimation step """
    def Estimation_step(self):
        stim =  self.backbone_flat[self.bayes_step - self.max_mem+1 : self.bayes_step+1] if self.bayes_step > self.max_mem else self.padding                              # get window of stim (or padding for begining of process)
        self.last_m_stim = np.flip(stim)                                                                                                                                  # reverse order to get [new ... old] for matrix mulitiplication convinience 
        self.check_for_action()                                                                                                                                           # check if action occured
        self.update_PGO()                                                                                                                                                 # Handle probabilistic weighting and PGO update 
        if not self.stim_1 or self.acting:                                                                                                                                     # If NOGO or action 
            self.known = 0                                                                                                                                                # all states are known, reset known index to 0
            
    def check_for_action(self):
        self.acting = self.curr_step == self.act_times[self.curr_trial]                                                                                                        # Check if action was taken at curr step 
        self.R = self.rews[self.curr_trial] == 1                                                                                                                          # Check if reward was recieved at curr trial 
            
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    bayes specifics
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
        
    def update_PGO(self):       
        """ 
        bayes types:
            standard =              simple memory window with weights for each stim in memory, for sufficient statistic
            soft =                  expert bayesian updating only after NOGO, for full posterior distribution 
            discount =              standard + decaying vector multiplied with the weights, for gradual forgetting
            hybrid =                weighted likelihood of each stim in memory, for full posterior distribution
            dynamic =              soft forgetting with dynamic discounting based on the estimated uncertainty
        
        """
        if self.bayes_type == 'standard':
            PGO_update, PGO_prior = self.standard_estimate()
        if self.bayes_type == 'soft': 
            PGO_update, PGO_prior = self.soft_estimate()
        if self.bayes_type == 'discount':
            PGO_update, PGO_prior = self.discount_estimate()
        if self.bayes_type == 'hybrid':
            PGO_update, PGO_prior = self.hybrid_estimate()
            
        self.update_remaining_currs(PGO_update, PGO_prior)
        if np.any(np.isnan(self.curr_PGO[:, self.curr_step])):                                                 # if anything fucked up print it out 
            print("nan present")
            
    def update_remaining_currs(self, PGO_update, PGO_prior):
        self.curr_model_free[:, self.curr_step] = np.cumsum(self.last_m_stim)/self.mem_range                   # model free estimate is just the average stim for each window length 
        self.curr_PGO[:, self.curr_step + 1] =  PGO_update + PGO_prior                                         # Update curr estimate (weighted average) of PGO 
        self.curr_mem[:, self.curr_step] = self.last_m_stim
            
    def standard_estimate(self):
        self.update_weights()                                                                                  # Update weights         
        stim_weights, prior_weights = self.normalize_weights(self.weights)                                     # normalize weights 
        PGO_update = stim_weights @ self.last_m_stim                                                           # weighted avg of last m stim: [Mem , Mem ] X [ Mem ] = [ Mem ], is PGO update for each value of free paramter (e.g. memory duration)
        PGO_prior = prior_weights * self.curr_PGO[:, self.curr_step]                                           # remainder of weight put into current prior
 
        self.curr_weight[:, self.curr_step] = stim_weights[-1]                                              # log stim weights (with masking)
        return PGO_update, PGO_prior            
    
    def soft_estimate(self, eps = 1e-5):
        self.update =  1 if self.curr_n[-1] > 0 else (self.bayes_range**self.loaded_n)*(1 - self.bayes_range)  # if NOGO, likelihoods are equal to sigma ^ n * ( 1 -sigma ) , else each likelihood is equal (assumed safe)
        self.loaded_n = self.curr_n[-1]                                                                        # prepare next step's n (since curr_n will be 0 if NOGO)
        self.bayes_dist = self.norm_by_sum( self.bayes_dist * self.update + eps, dim = -1)                     # normalize the posterior with an epsilon decay to uniform distributio
        PGO_update =  (self.bayes_dist * self.bayes_range).sum(-1)                                             # Update to the curr estimate of PGO (prior weight is 0 because posterior already used prior)

        self.curr_dist[:, :, self.curr_step] = self.bayes_dist                                                 # log bayes dist
        return PGO_update, 0
    
    def discount_estimate(self):
        self.update_weights()                                                                                  # Update weights   
        W = self.weights * self.discount_forget                                                                # gradual forgetting
        norm = self.discount_forget.sum(-1, keepdims= True)                                                    # get norm for weights 
        stim_weights =  W/norm                                                                                 # normalize weights
        prior_weights = (1-stim_weights.sum(-1))                                                               # get weight of prior 
        PGO_update = stim_weights @ self.last_m_stim                                                           # weighted avg of last m stim: [Mem , Mem ] X [ Mem ] = [ Mem ], is PGO update for each value of free paramter (e.g. memory duration)
        PGO_prior = prior_weights * self.curr_PGO[:, self.curr_step]                                           # remainder of weight put into current prior 
        return PGO_update, PGO_prior
    
    def hybrid_estimate(self, eps = 1e-5):                                                                     # hybrid has both memory (either hard cutoff or discounted forgetting) and distributions 
        self.update_weights()                                                                                  # Update weights   
        """"""""""""""""""""" 2 options to choose from """""""""""""""""""""
        F = np.tri(self.max_mem)                                                                               # hard cutoff forgetting
        # F = self.discount_forget                                                                               # gradual forgetting  
        """"""""""""""""""""" 2 options to choose from """""""""""""""""""""

        Z = (1-F)[:,:,None]                                                                                    # zero fill
        W = (self.weights * F )[:,:,None] 
        S = self.last_m_stim[:, None]                                                                          # get last M stim
        B = self.bayes_range[None, :]                                                                          # get every possible PGO 
        likelihood = (S*B + (1-S)*(1-B))[None,:,:]                                                             # the likelihood of each individual stim is PGO if GO or (1 - PGO) if NOGO = GO * PGO + ( 1 - GO ) * ( 1 - PGO )
                
        """ standard bayes """
        L = (likelihood * W + (1-W)*S) * F[:,:,None] + Z
        update = self.norm_by_sum(L.prod(1), dim = -1)                                                         # likelihood: multiply likelihood of each stim, by weight of each state
        """ exponential weighting """
        # update = self.norm_by_sum( ( likelihood ** W ).prod(1), dim = -1)                                    # likelihood: multiply likelihood of each stim, to the power of the weight, e.g. weight = 0 means all PGO are equally probable        

        "not sure what is going on here"
        """ ML estimate """ 
        # self.bayes_dist = self.norm_by_sum(self.bayes_range[None,:] * update + eps, dim = -1)                # get posterior distribution
        """ MAP estimate """ 
        # self.bayes_dist = self.norm_by_sum(self.bayes_dist * update + eps, dim = -1)                         # get posterior distribution

        self.bayes_dist = update                                                                               # get posterior distribution
        
        PGO_update =  (self.bayes_dist * self.bayes_range[None, :]).sum(-1)                                    # current estimate of PGO for all memory durations

        self.curr_dist[:, self.curr_step] = self.bayes_dist[-1]                                                 # log bayes dist
        self.curr_weight[:, self.curr_step] = W.squeeze(2)[-1]                                                  # log stim weights 
        return PGO_update, 0
    
    def update_weights(self):
        self.weights = np.roll(self.weights, 1)                                                     # Shift weights to preserve weights after last known
        self.weights[:, 0] = 0                                                                      # Set first weight to 0 as default
        num = self.b_k_min_n                                                                        # Action + reward case numerator 
        denom = self.b_n_0                                                                          # Action + reward case denominator 
        if not self.acting:                                                                         # If no action
            b_0_b_n = (self.b_0*self.b_n )                                                          # (1-b) * b^n 
            num = b_0_b_n + self.a*num                                                              # No action case numerator 
            denom = b_0_b_n + self.a*denom                                                          # No action case denominator 
        W = num/(1e-12+denom)
        if not self.stim_1 or (self.acting and not self.R):                                         # If NOGO or action and no reward
            self.unsafe = 0                                                                         # Current state is unsafe, reset unsafe index to 0 
        W[:, self.unsafe:] = 1                                                                      # Update weights from last known unsafe to be 1 
        self.weights[:, :self.known + 1] =  W[:, :self.known + 1]                                   # update unsafe probabilities up to where already known 
                
    def normalize_weights(self, weights):
        weights = weights * np.tri(self.max_mem)                                                    # Set upper triangle to zero to implement memory lengths 
        prior_weights = (1 - weights)* np.tri(self.max_mem)                                         # prior weight comes from any weights less than 1
        prior_weights = (prior_weights).sum(-1)[:,None]                                             # get total prior weight
        denom = self.mem_range[:, None]                                                             # Normalize weights to sum to 1 
        return weights/denom, (prior_weights/denom).squeeze()  
    
    def norm_by_sum(self, A, dim):
        return A / A.sum(dim, keepdims = True)   
