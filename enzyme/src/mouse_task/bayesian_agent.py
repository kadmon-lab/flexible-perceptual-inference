# import numpy as np; from tqdm import tqdm; import torch
        
# class bayesian_independent():
#     def __init__(self, device, factorize = False, ctx_approx = True, ITI_approx = True):
#         self.trials_per_block = 20
#         self.steps_per_trial = 25 
#         self.ctx_approx = ctx_approx
#         self.ITI_mean = 14
#         self.ITI_PM = 10
#         self.exp_mean = 10
#         ITI_PM = 10
#         ITI_mean = 14
#         print(f"bayes ITI_mean = {ITI_mean} ITI PM = {ITI_PM}, ITI_approx = {ITI_approx}, ctx_approx = {ctx_approx}, trials_per_block = {self.trials_per_block}")
        
#         self.factorize = factorize
#         self.device = device

#         self.trial_dur = 100
#         self.bayes_resolution = 100
#         self.delta_max = 2 * ITI_PM
#         self.micro_N = 2 + ITI_mean + ITI_PM 
#         self.min_leave_time = ITI_mean - ITI_PM - 1 
#         self.max_leave_time = ITI_mean + ITI_PM

#         self.micro_N__no_ITI = 3
#         self.min_leave_time__no_ITI = 0
#         self.max_leave_time__no_ITI = 1

#         self.action_tensor = torch.tensor([0, 1]).to(device)
#         self.bayes_range = np.linspace(0, 1, self.bayes_resolution) 
#         self.bayes_range_inds = np.arange(self.bayes_resolution)
#         self.x_axis = np.arange(self.trial_dur)
#         self.get_total_analytical_optim()
#         self.init_distributional_vars()
        
#     def init_distributional_vars(self):
#         self.joint_dist = np.ones((self.micro_N, self.bayes_resolution))                   # 0 = unsafe, 1 = safe, 2:28 = ITI 
#         self.joint_dist = self.joint_dist/self.joint_dist.sum()

#         self.joint_dist__no_ITI = np.ones((self.micro_N__no_ITI, self.bayes_resolution))
#         self.joint_dist__no_ITI = self.joint_dist__no_ITI/self.joint_dist__no_ITI.sum()

#         self.T_r = np.eye(self.micro_N)
#         self.T_r[0,0] = 0
#         self.T_r[1,1] = 0
#         self.T_r[2, 0] = 1
#         self.T_r[2, 1] = 1

#         self.T_r__no_ITI = self.T_r[:3,:3]
        
#         binary = np.array([0,1])
#         self.P_X__s_theta = np.zeros((self.micro_N, self.bayes_resolution, 2))                                                                                
#         self.P_X__s_theta[:,:,:] = binary[None,:]*self.bayes_range[:,None] + (1-binary)[None:]*(1-self.bayes_range[:,None])       
#         self.P_X__s_theta[1] = binary   
#         self.P_X__s_theta = self.P_X__s_theta/self.P_X__s_theta.sum(-1, keepdims=True)

#         self.P_X__s_theta__no_ITI = self.P_X__s_theta[:3]
        
#         """ factorizing """ 
#         self.P_X__s_only = self.P_X__s_theta.mean(1)                                                                                 # dims: states x NOGO/GO 
#         self.P_x__active_theta_only = self.P_X__s_theta.mean(0)                                                                             # dims: theta x NOGO/GO 
    
#         self.T_s = np.zeros((self.micro_N, self.micro_N))
#         self.T_s[0, 0] = 1 - 1/self.exp_mean
#         self.T_s[0, -1] = 1
#         self.T_s[1, 0] = 1/self.exp_mean
#         self.T_s[1, 1] = 1

#         self.T_s__no_ITI = np.zeros((self.micro_N__no_ITI, self.micro_N__no_ITI))
#         self.T_s__no_ITI[0, 0] = 1 - 1/self.exp_mean
#         self.T_s__no_ITI[0, -1] = 1
#         self.T_s__no_ITI[1, 0] = 1/self.exp_mean
#         self.T_s__no_ITI[1, 1] = 1

#         self.ITI_leave_prob = np.ones(self.max_leave_time)
#         for ITI_time in range(self.micro_N-3):        # i is state we transition to 
#             if ITI_time < self.min_leave_time:
#                 self.ITI_leave_prob[ITI_time] = 0
#             else:
#                 self.ITI_leave_prob[ITI_time] = 1/(self.max_leave_time - ITI_time)
#             self.T_s[ITI_time+3, ITI_time+2] = 1 - self.ITI_leave_prob[ITI_time]
#             self.T_s[0, ITI_time+2] = self.ITI_leave_prob[ITI_time]


#         self.ITI_leave_prob__no_ITI = np.ones(self.max_leave_time__no_ITI)
#         for ITI_time in range(self.micro_N__no_ITI-3):        # i is state we transition to 
#             if ITI_time < self.min_leave_time__no_ITI:
#                 self.ITI_leave_prob__no_ITI[ITI_time] = 0
#             else:
#                 self.ITI_leave_prob__no_ITI[ITI_time] = 1/(self.max_leave_time__no_ITI - ITI_time)
#             self.T_s__no_ITI[ITI_time+3, ITI_time+2] = 1 - self.ITI_leave_prob__no_ITI[ITI_time]
#             self.T_s__no_ITI[0, ITI_time+2] = self.ITI_leave_prob__no_ITI[ITI_time]

#         self.diag_inds = np.diag_indices(self.bayes_resolution)
#         self.switch_prob =  1/(self.trials_per_block * self.steps_per_trial)
#         self.T_theta = np.zeros((self.bayes_resolution, self.bayes_resolution))
#         self.T_theta[:] = self.switch_prob / (self.bayes_resolution - 1)
#         self.T_theta[self.diag_inds] = 1 - self.switch_prob
#         self.T_theta = self.T_theta/self.T_theta.sum(0, keepdims=True)
    
#     def bayes_forward(self, stim = None, act = None, rew = None):
#         self.curr_stim = stim 
#         self.curr_act = act 
#         self.curr_rew = rew                
#         if self.curr_act:
#             self.joint_dist = self.T_r @ (self.r_mask(self.curr_rew) * self.joint_dist)
#             self.joint_dist = self.joint_dist/self.joint_dist.sum()     
            
#         self.update_joint()   
#         p_state, belief_est, theta_est = self.to_estimates()
#         if np.isnan(theta_est):
#             print("THETA NAN OCCURED")
#         if np.isnan(belief_est):
#             print("STATE NAN OCCURED")
#         return p_state[1], theta_est
        
#     def to_estimates(self, eps = 1e-12):
#         self.p_state = self.joint_dist.sum(-1)
#         self.theta_est = (self.bayes_range[None, :] * self.joint_dist).sum()
#         self.state_est = -np.log(1-np.clip(self.p_state[1], a_min = eps, a_max = 1-eps))
#         return self.p_state, self.state_est, self.theta_est

#     def update_joint(self, eps = 1e-12):
#         if self.factorize: 
#             """ EXPERIMENTAL alternative factorizing method """            
#             # p = (1 - self.bayes_range) # self.joint_dist.sum(0)
#             # self.sample_theta_ind = np.random.choice(self.bayes_range_inds, p = p/p.sum())
#             # PX = self.P_X__s_theta[:, self.sample_theta_ind, self.curr_stim, None] * self.P_x__active_theta_only[None, :,self.curr_stim]
#             """ EXPERIMENTAL alternative factorizing method """            
#             PX = self.P_X__s_only[:,self.curr_stim, None] * self.P_x__active_theta_only[None, :,self.curr_stim]   
#         else:
#             PX = self.P_X__s_theta[:,:,self.curr_stim]            

#         if self.ctx_approx:
#             self.joint_dist = (self.T_s @ self.joint_dist)
#             self.joint_dist = (self.T_theta @ self.joint_dist.T).T
#         else:    
#             self.get_ctx_T()
            
#         self.joint_dist = PX * self.joint_dist 
#         self.joint_dist = self.joint_dist/self.joint_dist.sum()    
#     def r_mask(self, r):
#         return np.array([[1-r, r] + [1-r]*self.max_leave_time]).T 

#     def get_ctx_T(self):
#         delta = 1/self.trials_per_block 
#         self.T_theta[:] = delta / (self.bayes_resolution - 1)
#         self.T_theta[self.diag_inds] = 1 - delta
#         self.T_theta = self.T_theta/self.T_theta.sum(0, keepdims=True)   
        
#         T_s =  self.T_s[0, 2:]
#         ITI = self.joint_dist[2:]
#         correction = (T_s @ (self.T_theta @ ITI.T).T) - (T_s @ ITI)
#         self.joint_dist = self.T_s @ self.joint_dist 
#         self.joint_dist[0] = self.joint_dist[0] + correction

#     def flow_action(self, eps = 1e-12):
#         opt_LLH, opt_belief = self.theta_dist_to_optimal_thresh()
#         p_state = np.clip(self.joint_dist.sum(-1), a_min = eps, a_max = 1 - eps)
#         curr_LLH = -np.log(1-p_state[1])
#         Q = curr_LLH - opt_LLH
#         Q =  torch.from_numpy(np.array([-Q, Q])).to(self.device).float().squeeze(-1)
#         action = curr_LLH >= opt_LLH 
#         if action:
#             return self.action_tensor[1, None], self.action_tensor, Q  
#         return  self.action_tensor[0, None], 1 - self.action_tensor, Q

#     def theta_dist_to_optimal_thresh(self, eps = 1e-12):
#         denom = self.total_ana_acts + self.ITI_mean
#         p_theta = self.joint_dist.sum(0)[:,None]       
#         p_theta = p_theta/p_theta.sum()
#         weighted_p_safe = (self.total_ana_rews*p_theta).sum(0)
#         rew_rate = self.total_ana_rews/denom
#         self.weighted_rew_rate = (p_theta*rew_rate).sum(0)
#         self.opt_threshold = np.argmax(self.weighted_rew_rate) 
#         opt_belief = weighted_p_safe[self.opt_threshold]
#         opt_belief_log = -np.log(1-opt_belief)
#         """ EXPERIMENT """
#         """
#         p_theta_not_ITI =  self.joint_dist[:2]
#         p_not_ITI = p_theta_not_ITI.sum()
#         if p_not_ITI > eps:
#             p_theta__not_ITI = p_theta_not_ITI.sum(0)[:,None]       
#             p_theta__not_ITI = p_theta__not_ITI/p_theta__not_ITI.sum()

#             p_safe__not_ITI = (p_theta__not_ITI * self.total_ana_rews).sum(0)
#             self.weighted_rew_rate = (p_theta__not_ITI * self.total_ana_rews/denom).sum(0)
#             self.opt_threshold = np.argmax(self.weighted_rew_rate) 
#             opt_belief = p_not_ITI * p_safe__not_ITI[self.opt_threshold]
#         opt_belief_log = -np.log(1-opt_belief)
#         """
#         """ EXPERIMENT """
#         return opt_belief_log, opt_belief
 

#     """ get optimal belief analytically"""             
#     def get_total_analytical_optim(self):
#         self.total_ana_acts, self.total_ana_rews = [np.zeros((self.bayes_resolution, self.trial_dur)) for _ in range(2)]
#         for self.p_i, self.curr_PGO in enumerate(self.bayes_range):
#             self.total_ana_acts[self.p_i], self.total_ana_rews[self.p_i] = self.ana_of_consec_GO()
#         mean_belief = self.total_ana_rews.mean(0)
#         mean_rew_rate = (self.total_ana_rews/(self.total_ana_acts + self.ITI_mean)).mean(0)
#         self.opt_avg_thresh = mean_belief[np.argmax(mean_rew_rate)]

#     def ana_of_consec_GO(self):
#         action_times, reward_prob = [np.zeros(self.trial_dur) for _ in range(2)]
#         a = 1/self.exp_mean 
#         b = (1-a) * self.curr_PGO 
#         c = (1-a) * (1-self.curr_PGO)
#         for j, x in enumerate(self.x_axis):           
#             num_loop = sum([(b**k)*(x*a + c*(1+k)) for k in range(x)])
#             denom_loop = sum([ c * (b**k) for k in range(x)])            
#             num = x*(b**x) + num_loop
#             denom = 1-denom_loop
#             action_times[j] = num/denom 
#             reward_prob[j] = 1 - (b**x)/denom
#         return action_times, reward_prob
    
""" OG """
import numpy as np; from tqdm import tqdm; import torch
        
class bayesian_independent():
    def __init__(self, device, factorize = False, ctx_approx = True, ITI_approx = True):
        self.trials_per_block = 20
        self.steps_per_trial = 25 
        self.ctx_approx = ctx_approx
        self.ITI_mean = 14 #15 
        self.ITI_PM = 10
        self.exp_mean = 10
        ITI_mean = self.ITI_mean
        ITI_PM = self.ITI_PM 
        if ITI_approx: 
            ITI_PM = 0
            ITI_mean = 1
        print(f"bayes ITI_mean = {ITI_mean} ITI PM = {ITI_PM}, ITI_approx = {ITI_approx}, ctx_approx = {ctx_approx}, trials_per_block = {self.trials_per_block}")
        
        self.factorize = factorize
        self.device = device

        self.trial_dur = 100
        self.bayes_resolution = 100
        self.delta_max = 2 * ITI_PM
        self.micro_N = 2 + ITI_mean + ITI_PM 
        self.min_leave_time = ITI_mean - ITI_PM - 1 
        self.max_leave_time = ITI_mean + ITI_PM

        self.action_tensor = torch.tensor([0, 1]).to(device)
        self.bayes_range = np.linspace(0, 1, self.bayes_resolution) 
        self.bayes_range_inds = np.arange(self.bayes_resolution)
        self.x_axis = np.arange(self.trial_dur)
        self.get_total_analytical_optim()
        self.init_distributional_vars()
        
    def init_distributional_vars(self):
        self.joint_dist = np.ones((self.micro_N, self.bayes_resolution))                   # 0 = unsafe, 1 = safe, 2:28 = ITI 
        self.joint_dist = self.joint_dist/self.joint_dist.sum()
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
        self.switch_prob =  1/(self.trials_per_block * self.steps_per_trial)
        self.T_theta = np.zeros((self.bayes_resolution, self.bayes_resolution))
        self.T_theta[:] = self.switch_prob / (self.bayes_resolution - 1)
        self.T_theta[self.diag_inds] = 1 - self.switch_prob
        self.T_theta = self.T_theta/self.T_theta.sum(0, keepdims=True)
    
    def bayes_forward(self, stim = None, act = None, rew = None):
        self.curr_stim = stim 
        self.curr_act = act 
        self.curr_rew = rew                
        if self.curr_act:
            self.joint_dist = self.T_r @ (self.r_mask(self.curr_rew) * self.joint_dist)
            self.joint_dist = self.joint_dist/self.joint_dist.sum()     
            
        self.update_joint()   
        p_state, belief_est, theta_est = self.to_estimates()
        if np.isnan(theta_est):
            print("THETA NAN OCCURED")
        if np.isnan(belief_est):
            print("STATE NAN OCCURED")
        return p_state[1], theta_est
        
    def to_estimates(self, eps = 1e-12):
        self.p_state = self.joint_dist.sum(-1)
        self.theta_est = (self.bayes_range[None, :] * self.joint_dist).sum()
        self.state_est = -np.log(1-np.clip(self.p_state[1], a_min = eps, a_max = 1-eps))
        return self.p_state, self.state_est, self.theta_est

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
        self.joint_dist = self.joint_dist/self.joint_dist.sum()    
    def r_mask(self, r):
        return np.array([[1-r, r] + [1-r]*self.max_leave_time]).T 

    def get_ctx_T(self):
        delta = 1/self.trials_per_block 
        self.T_theta[:] = delta / (self.bayes_resolution - 1)
        self.T_theta[self.diag_inds] = 1 - delta
        self.T_theta = self.T_theta/self.T_theta.sum(0, keepdims=True)   
        
        T_s =  self.T_s[0, 2:]
        ITI = self.joint_dist[2:]
        correction = (T_s @ (self.T_theta @ ITI.T).T) - (T_s @ ITI)
        self.joint_dist = self.T_s @ self.joint_dist 
        self.joint_dist[0] = self.joint_dist[0] + correction

    def flow_action(self, eps = 1e-12):
        opt_LLH, opt_belief = self.theta_dist_to_optimal_thresh()
        p_state = np.clip(self.joint_dist.sum(-1), a_min = eps, a_max = 1 - eps)
        curr_LLH = -np.log(1-p_state[1])
        Q = curr_LLH - opt_LLH
        Q =  torch.from_numpy(np.array([-Q, Q])).to(self.device).float().squeeze(-1)
        action = curr_LLH >= opt_LLH 
        if action:
            return self.action_tensor[1, None], self.action_tensor, Q  
        return  self.action_tensor[0, None], 1 - self.action_tensor, Q

    def theta_dist_to_optimal_thresh(self, eps = 1e-12):
        denom = self.total_ana_acts + self.ITI_mean
        p_theta = self.joint_dist.sum(0)[:,None]       
        p_theta = p_theta/p_theta.sum()
        weighted_p_safe = (self.total_ana_rews*p_theta).sum(0)
        rew_rate = self.total_ana_rews/denom
        self.weighted_rew_rate = (p_theta*rew_rate).sum(0)
        self.opt_threshold = np.argmax(self.weighted_rew_rate) 
        opt_belief = weighted_p_safe[self.opt_threshold]
        opt_belief_log = -np.log(1-opt_belief)
        return opt_belief_log, opt_belief
 

    """ get optimal belief analytically"""             
    def get_total_analytical_optim(self):
        self.total_ana_acts, self.total_ana_rews = [np.zeros((self.bayes_resolution, self.trial_dur)) for _ in range(2)]
        for self.p_i, self.curr_PGO in enumerate(self.bayes_range):
            self.total_ana_acts[self.p_i], self.total_ana_rews[self.p_i] = self.ana_of_consec_GO()
        mean_belief = self.total_ana_rews.mean(0)
        mean_rew_rate = (self.total_ana_rews/(self.total_ana_acts + self.ITI_mean)).mean(0)
        self.opt_avg_thresh = mean_belief[np.argmax(mean_rew_rate)]

    def ana_of_consec_GO(self):
        action_times, reward_prob = [np.zeros(self.trial_dur) for _ in range(2)]
        a = 1/self.exp_mean 
        b = (1-a) * self.curr_PGO 
        c = (1-a) * (1-self.curr_PGO)
        for j, x in enumerate(self.x_axis):           
            num_loop = sum([(b**k)*(x*a + c*(1+k)) for k in range(x)])
            denom_loop = sum([ c * (b**k) for k in range(x)])            
            num = x*(b**x) + num_loop
            denom = 1-denom_loop
            action_times[j] = num/denom 
            reward_prob[j] = 1 - (b**x)/denom
        return action_times, reward_prob
