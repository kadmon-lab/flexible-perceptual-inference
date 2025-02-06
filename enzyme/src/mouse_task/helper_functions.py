import numpy as np; import torch; import scipy.linalg as lin; import pylab as plt; from sklearn.linear_model import LinearRegression; from tqdm import tqdm
from numpy import dot; from numpy.linalg import norm;  import matplotlib.cm as cm; from scipy.stats import spearmanr as spr; from sklearn.metrics import r2_score
class helper_functions():
    def __init__(self, **params):
        self.__dict__.update(params)
       
    """ analytical analysis helpers"""
    def get_cosin_similarity(self, a, b, mean_subtract = False):
        if mean_subtract:
            a = a - a.mean()
            b = b - b.mean()
        return dot(a, b)/(norm(a)*norm(b))


    """ splitting data """ 
    
    def get_indices(self, col = "block", From = 0, Til = 1e9, action_above = None,  action_below = None, stim_above = None,  stim_below = None, planted = None, plant_PGO = None,  plant_ID = None, prev_PGO = None, curr_PGO = None,\
        PGO_in_list = [], plant_PGO_in_list = [], plant_ID_in_list = [], avg_stim_in_list =[], rew = None, cum_rew = False, needs_acted = False, flatten = False, prep_mus = True, eps_init = 1, eps_final = None,\
        cross_validation = "testing", til_action = False, from_action = False, from_nog = False, within_trial = True, align_on = "onset", run_traj = False, full_prep = False):
        self.postprocess_vars( col, align_on, til_action, from_action, from_nog, within_trial, flatten, prep_mus, run_traj, full_prep)
        """
        Generates an array of indices trial_inds that pick out trials.
        trial: trial number from last context switch
        col (color): the variable to split on
        """
        cond_A = (self.trial >= From) * (self.trial <= Til) 
        cond_B = 1 if prev_PGO is None else self.last_PGOs == prev_PGO 
        cond_C = 1 if curr_PGO is None else self.PGOs == curr_PGO 
        cond_D = 1 if planted is None else self.plant_inds == int(planted)
        cond_E = 1 if plant_PGO is None else self.plant_PGOs == plant_PGO 
        cond_F = 1 if needs_acted is False else self.all_times != self.inaction 
        cond_G = 1 if stim_above is None else self.stim_durs > stim_above 
        cond_H = 1 if stim_below is None else self.stim_durs < stim_below 
        cond_I = 1 if rew is None else self.rews == rew 
        cond_K = (self.eps >= eps_init) * (self.eps <= (eps_final or self.eps[-1]) )
        cond_L = 1 if plant_ID is None else self.plant_IDs == plant_ID 
        cond_M = 1 if cross_validation == None else self.eps < self.eps[-1]/2 if cross_validation == "training" else self.eps >= self.eps[-1]/2
        cond_N = 1 if len(PGO_in_list) == 0 else np.isin(self.PGOs, PGO_in_list)
        cond_O = 1 if len(plant_PGO_in_list) == 0 else np.isin(self.plant_PGOs, plant_PGO_in_list)
        cond_P = 1 if len(avg_stim_in_list) == 0 else np.isin(self.avg_stim, avg_stim_in_list)
        cond_Q = 1 if action_above is None else self.all_times > action_above 
        cond_R = 1 if action_below is None else self.all_times < action_below 
        cond_S = 1 if len(plant_ID_in_list) == 0 else np.isin(self.plant_IDs, plant_ID_in_list)
        cond_T = 1 if cum_rew is False else self.cum_rew_per_block == self.trial + 1 

        # Boolean array that picks trials
        self.trial_inds = np.where( cond_A * cond_B * cond_C * cond_D * cond_E * cond_F * cond_E * cond_G\
            * cond_H  * cond_I * cond_K * cond_L * cond_M * cond_N * cond_O * cond_P * cond_Q * cond_R * cond_S * cond_T)[0]
        self.postprocess_indices()
        if self.run_traj:
            self.run_trajectory(False)
        
    def postprocess_vars(self, col, align_on, til_action, from_action, from_nog, within_trial, flatten, prep_mus, run_traj, full_prep):
        self.within_trial = within_trial
        self.from_action = from_action
        self.til_action = til_action
        self.from_nog = from_nog
        self.align_on = align_on 
        self.col = col
        if full_prep: 
            self.prep_mus = self.run_traj = self.flatten = True 
        else:
            self.prep_mus = prep_mus 
            self.run_traj = run_traj
            self.flatten = flatten 

    def postprocess_indices(self):        
        self.ends = self.trial_ends[self.trial_inds] + 1        
        self.durs = self.stim_durs[self.trial_inds]
        self.pots = self.last_pot[self.trial_inds]
        self.nogs = self.last_nog[self.trial_inds] + 1
        self.acts = self.act_times[self.trial_inds]

        self.cum_ends = np.cumsum(self.ends)
        self.starts = np.insert(self.ends, 0, 0)
        self.cum_starts = np.cumsum(self.starts)       
        self.cum_acts = (self.acts + self.cum_starts[:-1]).astype(int)
        self.cum_nogs = (self.nogs + self.cum_starts[:-1]).astype(int)
        
        self.proj_trials = self.trial[self.trial_inds]
        self.trial_percent = 1 if len(self.trial_inds) == 0 else self.proj_trials/(self.proj_trials.max()+1)    
        
        self.full_inds = np.hstack([np.arange(s, e) for s, e in zip(self.cum_starts, self.cum_ends)]).astype(int)
        self.time_in_trial = np.hstack([np.arange(s, e) for s, e in zip(np.zeros_like(self.ends), self.ends)]).astype(int)
        self.time_til_act = np.hstack([np.arange(s, e) - a for s, e, a in zip(np.zeros_like(self.ends), self.ends, self.acts)]).astype(int)

        # self.full_til_act_inds = np.hstack([np.arange(s, e) for s, e in zip(self.cum_starts, self.cum_acts)]).astype(int)
        self.full_til_act_inds = np.hstack([np.arange(s, e+1) for s, e in zip(self.cum_starts, self.cum_acts)]).astype(int)

        self.full_from_act_inds = np.hstack([np.arange(s+2, e) for s, e in zip(self.cum_acts , self.cum_ends)]).astype(int)
        self.pre_act, self.post_act = [np.zeros(len(self.full_inds)) for _ in range(2)]
        self.pre_act[self.full_til_act_inds] = 1
        self.post_act[self.full_from_act_inds] = 1
        self.full_ind_num = len(self.full_inds)
        
        if self.flatten == True:
            self.flattening()
        self.get_split()
        
    def get_split(self):
        if self.col == 'all':
            self.split_cond = np.zeros(len(self.trial_inds))
        else: 
            diff = 0 if None in self.plant_PGOs else np.around((self.plant_PGOs - self.PGOs).astype(float), 2)
            data = [self.PGOs, self.plant_PGOs, self.plant_IDs, self.last_PGOs, diff, self.rews,  self.binary_stim]
            names = ['block', 'plant_PGO', 'plant_ID', 'prev_PGO', 'diff', 'rew', 'binary_stim']
            self.split_cond = self.get_data_of(data, names, select = self.col)[self.trial_inds]  
        self.split_trial_cols = self.Cmap.to_rgba(self.split_cond.astype(float))
        self.split = np.unique(self.split_cond)
        self.split_num = len(self.split)       
        if self.prep_mus:
            self.preprocess_mus()
        
                                                   
    def get_split_inds(self, split_after = False):                                                                                  # for self.split_i, self.split_curr in enumerate(self.split):
        self.half = int(self.max_traj_steps/2)
        if split_after:
            self.split_inds = np.arange(len(self.split_cond))                                                                       # for if you want the trajectory without splitting
        else:
            self.split_inds = np.where((self.split_cond == self.split_curr))[0]                                                     # split inds is all trials in condition
       
        self.split_starts = self.cum_starts[self.split_inds]
        self.split_ends = self.cum_ends[self.split_inds]
        # self.split_starts = self.split_starts 
        if self.til_action:                                                                                                         # for only using data until action
            self.split_ends = self.cum_acts[self.split_inds] + 2*(self.align_on == 'action')
        self.full_split_ind_pairs = np.array([np.arange(s, e) for s, e in zip(self.split_starts, self.split_ends)], dtype = object)
        self.full_split_inds_flat = np.hstack(self.full_split_ind_pairs)                                                            # all inds of split conditions
        self.get_split_alignment()
        
    def get_split_alignment(self):
        self.split_acts = self.acts[self.split_inds]
        self.aligned_acts = (self.acts[self.split_inds] - self.half).astype(int)
        self.aligned_durs = (self.durs[self.split_inds] - self.half).astype(int)
        self.aligned_pot = (self.pots[self.split_inds] - self.half).astype(int)
        self.aligned_nog = (self.nogs[self.split_inds] - self.half).astype(int)
        data = [0, self.aligned_acts, self.aligned_durs, self.aligned_pot, self.aligned_nog]
        self.offset = self.get_data_of(data, ['onset', 'action', 'W4L', 'last_pot', 'last_nog'], select = self.align_on)
        self.label_offset = int(self.max_traj_steps/2) if self.align_on != 'onset' else 0
        self.traj_xlabel = f"Time(s) from {self.align_on}"
        self.aligned_starts = self.split_starts + self.offset 

    """ trajectory processing """

    def get_trajectory_step_inds(self):
        running_inds = self.aligned_starts + self.step 
        if self.within_trial:
            post_start = running_inds >= self.split_starts 
            pre_end = running_inds < self.split_ends - 1
            post_action = (running_inds >= self.cum_acts[self.split_inds]) if self.from_action else 1            
            post_nog = (running_inds >= self.cum_nogs[self.split_inds]) if self.from_nog else 1            
            conds = np.where(post_start * post_action * post_nog * pre_end)[0]        
            self.step_inds = running_inds[conds]                                                                                        # Specific step data for condition           
            self.step_split_cond = self.split_cond[conds]                                                                               # Each step in step inds' split conditions
        else:
            self.step_inds = running_inds 
            self.step_split_cond = running_inds

        self.emp_act_prob[self.split_i, self.step] = sum(self.split_acts == (self.step + self.offset))
       
    def get_trajectory_diffs(self):
        self.GO_traj_diffs = np.zeros((self.PC_dim, self.split_num, self.max_traj_steps))
        self.NOGO_traj_diffs = np.zeros((self.PC_dim, self.split_num, self.max_traj_steps))
        for self.split_i, self.split_curr in enumerate(self.split):
            self.get_split_inds()
            for self.step in range(self.max_traj_steps):
                self.get_trajectory_step_inds()
                GO_step_inds = np.where(self.input_flat[1, self.step_inds])[0]
                NOGO_step_inds = np.where(self.input_flat[0, self.step_inds])[0]
                self.PC_mus[:, self.split_i, self.step] = self.PC_flat[:, self.step_inds].mean(-1) 
                self.PC_prev_mus[:, self.split_i, self.step] = self.PC_prev[:, self.step_inds].mean(-1) 
                self.PC_traj_diffs[:, self.split_i, self.step] = self.PC_diff[:, self.step_inds].mean(-1)
                diffs = self.PC_prev[:, self.step_inds+1]-self.PC_prev[:, self.step_inds]
                self.GO_traj_diffs[:, self.split_i, self.step] = diffs[:, GO_step_inds].mean(-1)
                self.NOGO_traj_diffs[:, self.split_i, self.step] = diffs[:, NOGO_step_inds].mean(-1)

                
    def get_GO_NOGO_inds(self):
        act_cond = self.pre_act if self.til_action else self.post_act if self.from_action else 1 
        self.GO_inds = np.where(self.input_flat[1] * act_cond )[0]
        self.NOGO_inds = np.where(self.input_flat[0] * act_cond )[0]     
        self.REW_inds = np.where(self.input_flat[-1] * act_cond )[0]     
        self.ACTION_inds = self.full_inds if (not self.til_action and not self.from_action) else np.where(act_cond)[0]

    def flattening(self):
  
        # self.theta_dist_flat = np.vstack(np.hstack(self.theta_dist_structured[self.trial_inds])).T.astype(np.float64)       
        # self.theta_dist__safe_flat = np.vstack(np.hstack(self.theta_dist__safe_structured[self.trial_inds])).T.astype(np.float64)
        # self.theta_dist__unsafe_flat = np.vstack(np.hstack(self.theta_dist__unsafe_structured[self.trial_inds])).T.astype(np.float64)
        self.MF_theta_flat =  np.vstack(self.flatten_trajectory(self.MF_theta_structured[self.trial_inds])).T.astype(np.float64)       
        self.flow_theta_flat = self.flatten_trajectory(self.flow_theta_structured[self.trial_inds]).astype(np.float64)
        self.flow_belief_flat = self.flatten_trajectory(self.flow_belief_structured[self.trial_inds]).astype(np.float64)
        self.flow_thresh_flat = self.flatten_trajectory(self.flow_thresh_structured[self.trial_inds]).astype(np.float64)
        self.flow_belief_RMS_flat = self.flatten_trajectory(self.flow_belief_RMS_structured[self.trial_inds]).astype(np.float64)
        self.flow_thresh_RMS_flat = self.flatten_trajectory(self.flow_thresh_RMS_structured[self.trial_inds]).astype(np.float64)
        self.factorized_theta_flat = self.flatten_trajectory(self.factorized_theta_structured[self.trial_inds]).astype(np.float64)
        self.factorized_belief_flat = self.flatten_trajectory(self.factorized_belief_structured[self.trial_inds]).astype(np.float64)
        self.mechanistic_DV_flat = self.flatten_trajectory(self.mechanistic_DV_structured[self.trial_inds])

        self.lick_prob_flat = self.flatten_trajectory(self.lick_prob[self.trial_inds])
        self.value_flat = self.flatten_trajectory(self.value[self.trial_inds])
        self.Q_flat = self.flatten_trajectory(self.Q_values[self.trial_inds])
        self.QDIFF_flat = self.Q_flat[1] - self.Q_flat[0]
        self.output_flat = self.flatten_trajectory(self.net_output[self.trial_inds]) 
        self.input_flat = self.flatten_trajectory(self.net_input[self.trial_inds])
        self.consec_flat = self.flatten_trajectory(self.consec_stim[self.trial_inds])
        self.PGO_flat = self.flatten_trajectory(self.step_PGO[self.trial_inds])
        self.PSAFE_flat = self.flatten_trajectory(self.safe_backbone[self.trial_inds])
        self.plant_PGO_flat = self.flatten_trajectory(self.step_plant_PGO[self.trial_inds])
        
        self.last_nogo_flat = self.flatten_trajectory(self.step_last_nogo[self.trial_inds])

        self.LTM_flat = self.flatten_trajectory(self.LTM[self.trial_inds])
        self.I_gate_flat = self.flatten_trajectory(self.i_gate[self.trial_inds])
        self.F_gate_flat = self.flatten_trajectory(self.f_gate[self.trial_inds])
        self.C_gate_flat = self.flatten_trajectory(self.c_gate[self.trial_inds])
        self.O_gate_flat = self.flatten_trajectory(self.o_gate[self.trial_inds])
        
        self.PC_flat = self.proj.transform(self.output_flat.T).T        
        self.pot_RT_flat = self.flatten_trajectory(self.step_pot_RT[self.trial_inds])
        self.RT_flat = self.flatten_trajectory(self.step_RT[self.trial_inds])
        self.trial_flat = self.flatten_trajectory(self.step_trial[self.trial_inds])
        self.get_GO_NOGO_inds()
        self.get_reg_vec()        
        self.get_net_decodings()
        self.postprocess_bayes_flat()
        self.handle_cog_map()


    def get_reg_vec(self, data_noise = 0):    
        if self.regress_from == 'PCA':
            singles = np.concatenate( ([PC[None,:] for PC in self.PC_flat]), 0)
            # inter_interactions = np.concatenate( ([self.PC_flat[i, None,:]*self.PC_flat[(i+1)%self.PC_dim, None,:] for i in range(self.PC_dim)]), 0)
            # within_interactions = np.concatenate( ([self.PC_flat[i, None,:]*self.PC_flat[i, None,:] for i in range(self.PC_dim)]), 0)
            # signed_interactions_1 = np.concatenate( ([self.PC_flat[i, None,:]*np.abs(self.PC_flat[(i+1)%self.PC_dim, None,:]) for i in range(self.PC_dim)]), 0)
            # signed_interactions_2 = np.concatenate( ([np.abs(self.PC_flat[i, None,:])*self.PC_flat[(i+1)%self.PC_dim, None,:] for i in range(self.PC_dim)]), 0)
            # self.pred_from = inter_interactions
            # self.pred_from = np.concatenate((signed_interactions_1, signed_interactions_2), 0 )
            # self.pred_from = np.concatenate((inter_interactions, signed_interactions_1, signed_interactions_2, within_interactions), 0 )
            # self.pred_from = np.concatenate((singles, inter_interactions, within_interactions), 0 )
            # self.pred_from = np.concatenate((singles, inter_interactions), 0 )
            self.pred_from = singles        
        
        if self.regress_from == "PC1":
            self.pred_from  = self.PC_flat[0,None,:]
        if self.regress_from == "PC2":
            self.pred_from  = self.PC_flat[1,None,:]
        if self.regress_from == "PC3":
            self.pred_from  = self.PC_flat[2,None,:]
        if self.regress_from == "PC12":
            self.pred_from  = self.PC_flat[:2]
        if self.regress_from == 'STM':
            self.pred_from = self.output_flat
        if self.regress_from == 'LTM':
            self.pred_from = self.LTM_flat
        if self.regress_from == 'Q':
            self.pred_from = self.Q_flat[None,:]
        if self.regress_from == 'QDIFF':
            self.pred_from = self.QDIFF_flat[None,:]

        self.pred_from_N = self.pred_from.shape[0]
        
        if data_noise > 0:
            std = np.std(self.pred_from, -1)[:, None]
            gauss = np.random.randn(self.pred_from_N, self.pred_from.shape[-1]) * std
            self.pred_from = self.pred_from + data_noise * gauss

    def postprocess_bayes_flat(self):
        flow_inds = np.argmin(np.abs(self.PGO_range[:, None] - self.flow_theta_flat), axis=0)
        self.flow_opt_thresh_flat = self.opt_belief[flow_inds]
        self.flow_dist_from_opt_flat = self.log_to_prob(self.flow_belief_flat) -  self.flow_opt_thresh_flat
            
        factorized_inds = np.argmin(np.abs(self.PGO_range[:, None] - self.factorized_theta_flat), axis=0)
        self.factorized_opt_thresh_flat = self.opt_belief[factorized_inds]
        self.factorized_dist_from_opt_flat = self.log_to_prob(self.factorized_belief_flat) -  self.factorized_opt_thresh_flat

        self.PGO_inds = np.searchsorted(self.PGO_range,self.PGO_flat) 
        self.opt_thresh_flat = self.opt_belief[self.PGO_inds]     
          
    def handle_cog_map(self):
        self.PC_Xlab = "PC1 projection" 
        self.PC_Ylab = "PC2 projection" 
        self.PC_Zlab = "PC3 projection" 
               
        if self.cog_map == 'regression':
            self.PC_flat[0, :] =  self.net_belief_flat
            self.PC_Xlab = r"$-\log(1 - \hat s^{net})$" 
            self.PC_flat[1, :] = self.net_theta_flat
            self.PC_Ylab = r"$\hat\theta^{net}$" 
        
        if self.cog_map == 'theory':
            self.PC_flat[0, :] = self.flow_belief_flat.astype('float32')       
            self.PC_Xlab = r"$-\log(1 - \hat s^{theory})$" 
            self.PC_flat[1, :] =  self.flow_theta_flat  
            self.PC_Ylab = r"$\hat\theta$" 
            
        if self.cog_map == 'naive':
            self.PC_flat[0, :] = self.factorized_belief_flat.astype('float32')       
            self.PC_flat[1, :] =  self.factorized_theta_flat  
            self.PC_Xlab = r"$-\log(1 - \hat s^{theory})$" 
            self.PC_Ylab = r"$\hat\theta$"  
            
        if self.cog_map == 'policy' or self.cog_map == 'theory_policy':
            # L = np.clip(self.lick_prob_flat, a_min = 1e-5, a_max = 1-(1e-5))
            # self.PC_flat[0, :] = np.log(L) - np.log(1-L) 
            # self.PC_flat[1, :] = self.RT_flat/self.RT_flat.mean() 

            self.PC_flat[0, :] =  self.net_belief_flat
            self.PC_flat[1, :] = self.PGO_reg.predict(self.pred_from.T)
            self.PC_Ylab = 'predicted upcoming waiting time (normalized by max)'
            self.PC_Xlab = 'log( P(action) / (1-P(action) ) )'
                           
        self.PC_labs = [self.PC_Xlab, self.PC_Ylab, self.PC_Zlab]
        self.make_prev_diff()

    def make_prev_diff(self):
        self.PC_diff = np.diff(self.PC_flat, prepend=0)
        self.PC_prev = np.concatenate((np.zeros((self.PC_dim, 1)), self.PC_flat[:, :-1]), -1)
        self.net_belief_diff = np.diff(self.net_belief_flat, prepend=0)
        self.net_belief_prev = np.concatenate((np.zeros(1), self.net_belief_flat[:-1]), -1)
        self.net_theta_prev = np.concatenate((np.zeros(1), self.net_theta_flat[:-1]), -1)
        self.net_theta_diff = np.diff(self.net_theta_flat, prepend=0)
        self.flow_belief_diff = np.diff(self.flow_belief_flat, prepend=0)
        self.flow_belief_prev = np.concatenate((np.zeros(1), self.flow_belief_flat[:-1]), -1)
        self.flow_theta_diff = np.diff(self.flow_theta_flat, prepend=0)
        self.flow_theta_prev = np.concatenate((np.zeros(1), self.flow_theta_flat[:-1]), -1)
        self.factorized_belief_diff = np.diff(self.factorized_belief_flat, prepend=0)
        self.factorized_belief_prev = np.concatenate((np.zeros(1), self.factorized_belief_flat[:-1]), -1)
        self.factorized_theta_diff = np.diff(self.factorized_theta_flat, prepend=0)
        self.factorized_theta_prev = np.concatenate((np.zeros(1), self.factorized_theta_flat[:-1]), -1)
            
    def raw_to_structured(self, X, time_dim = 1, running_ind = 0):
        structured_array = np.empty(self.data_len, dtype = object) 
        for curr_trial in range(self.data_len):
            end = self.trial_ends[curr_trial] + 1 
            trial_data = np.empty(end, dtype = object)
            
            for curr_step in range(end):
                trial_data[curr_step] = X[running_ind] if time_dim == 1 else X[:, running_ind] if time_dim == 2 else X[:, :, running_ind] if time_dim == 3 else X[:, :, :, running_ind]               
                running_ind = running_ind + 1            
            structured_array[curr_trial] = trial_data
        return structured_array 
                
    def raw_to_function(self, func):
        self.i = 0
        for self.curr_trial in range(self.data_len):                           # for each trial
            self.PGO = self.PGOs[self.curr_trial]
            self.act = self.act_times[self.curr_trial]
            self.end = self.trial_ends[self.curr_trial] + 1        
            for self.curr_step in range(self.end):                             # for each step in current trial
                self.stim = int(self.backbone_flat[self.i])
                self.acting = int(self.curr_step == self.act)
                self.post_act = int(self.curr_step > self.act)
                func()
                self.i += 1        

    def project(self, vector_a, vector_b): # project b onto a
        # Normalize vectors
        norm_a = np.linalg.norm(vector_a, axis = 0)[:, None]
        norm_b = np.linalg.norm(vector_b, axis = 0)
        unit_vector_a = vector_a / norm_a
        unit_vector_b = vector_b / norm_b
        projection = np.dot(unit_vector_a, unit_vector_b) * unit_vector_b
        return projection
    
    """ preprocessing and postprocessing """     

    def preprocess_mus(self):           
        self.output_mus, self.LTM_mus, self.I_gate_mus, self.F_gate_mus, self.C_gate_mus, self.O_gate_mus = [np.zeros((self.hid_dim, self.split_num, self.max_traj_steps)) for _ in range(6)] 
        self.PC_mus, self.PC_traj_diffs, self.PC_prev_mus, self.PC_PGO_R2_mus = [np.zeros((self.PC_dim, self.split_num, self.max_traj_steps)) for _ in range(4)]
        self.flow_theta_vars, self.flow_belief_vars, self.factorized_theta_vars, self.factorized_belief_vars, self.net_theta_vars, self.net_belief_vars, \
        self.net_theta_mus, self.net_belief_mus, self.net_dist_from_opt_mus, self.net_DV_est_mus, self.net_dist_from_opt_vars, self.flow_opt_thresh_mus,\
        self.factorized_dist_from_opt_mus, self.flow_dist_from_opt_mus, self.flow_dist_from_opt_vars, self.flow_belief_RMS_mus,\
        self.emp_act_prob, self.GO_mus, self.PGO_mus, self.PSAFE_mus, self.lick_prob_mus, self.value_mus, self.QDIFF_mus, self.N_mus, \
        self.flow_theta_mus, self.flow_belief_mus, self.factorized_theta_mus, self.factorized_belief_mus, self.mechanistic_DV_mus \
            = [np.zeros((self.split_num, self.max_traj_steps)) for _ in range(29)]
        self.pred_from_mus = np.zeros((self.pred_from_N, self.split_num, self.max_traj_steps))
        self.act_col = np.zeros((self.split_num, self.max_traj_steps, 4))
        self.split_leg = [f"{self.col} {i}" for i in self.split]        
        self.split_cols = self.Cmap.to_rgba(np.linspace(0, 1, self.split_num))
            
        
    def norm_W(self, W):
        return W/(lin.norm(W, axis = 1)[:, None])

    def flatten_trajectory(self, trajectory):
        return np.hstack([self.to(T, 'np') for T in trajectory])
    
    def trial_var_to_step_var(self, var):
        return var*(1 + 0*(self.data.backbone))

    def to(self, X, to):
        # if X == None:
        #     return X
        if type(X) == list:
            return [self.to(x, to) for x in X]
        if to == 'np' and type(X) == torch.Tensor:
            return X.detach().cpu().numpy() 
        if to == 'tensor' and type(X) != torch.Tensor:
            return torch.from_numpy(X).to(self.device).float()                                  
        return X
    
    def get_data_of(self, data, names, select, tensor = False):                                 # Returns of the data (e.g. plant PGOs or block PGOs) selected
        if tensor == True: 
            return self.to(self.get_data_of(self.to(data, 'np'), names, select), 'tensor')
        names = np.array(names, dtype = str)
        return np.array(data[int(np.where(select == names)[0])])

    def correlate(self, X, Y):                                                                  # Dim 1 = variables, Dim 2 = Samples
        X = X - X.mean(-1, keepdims = True)
        Y = Y - Y.mean(-1, keepdims = True)    
        X_var = (X**2).sum(-1, keepdims = True)
        Y_var = (Y**2).sum(-1, keepdims = True)
        var = X_var @ Y_var.T
        if type(var) != float: 
            var = var.astype(float)
        return (X @ Y.T) / np.sqrt(var)

    def R2(self, x, y):
        if len(x.shape) == 1:
            x = x[:,None]
        if len(y.shape) == 1:
            y = y[:,None]
        m = LinearRegression().fit(x,y)
        return m.score(x,y)
    
    def round_to_nearest(self, X, Y):
        rounded_X = np.zeros_like(X)
        for i, x in enumerate(X):
            nearest_idx = np.abs(Y - x).argmin()
            rounded_X[i] = Y[nearest_idx]
        return rounded_X
     
    def MI(self, X, Y, x_res = 1, y_res = 2, eps = 1e-20, disable = False):
        x_round = X.round(x_res)
        y_round = Y.round(y_res)
        x_unique = np.unique(x_round)
        y_unique = np.unique(y_round)
        joint_dist = np.zeros((len(x_unique), len(y_unique)))
        
        for x_i, x in enumerate(tqdm(x_unique, desc = 'calculating joint', disable = disable)):
            for y_i, y in enumerate(y_unique):
                inds = np.where((x_round == x)*(y_round == y))
                joint_dist[x_i, y_i] = len(inds[0])

        joint_dist = joint_dist/joint_dist.sum() 
        x_dist = joint_dist.sum(1)[:, None]
        y_dist = joint_dist.sum(0)[None, :]       
        MI = joint_dist * np.log( joint_dist / (y_dist * x_dist + eps) + eps)
        non_zero = np.where(joint_dist.flatten() > eps)[0]
        MI = MI.flatten()[non_zero].sum()
        return MI 
        
    def norm_by_sum(self, A, dim):
        return A / A.sum(dim, keepdims = True)
    
    def where(self, X, Y = None):
        if Y is None:
            return np.where(X)[0] 
        return np.where(X == Y)[0]
    
    def percent_complete(self, i, total):
        if (i % int(total/10)) == 0:
            print(str(int(100 * i/total)) + "%")                                                                                                    
            
    """ LSTM Dynamics helpers"""
    def get_null_norms(self, W, null):
            return self.to((W @ null).norm(dim=0), 'np')

    def sort_dynamic(self, W):
       return np.take_along_axis(np.take_along_axis(self.to(W, 'np'), self.sort_inds_0, 0), self.sort_inds_1, 1)
   
    def get_split_from_mu(self, mus):
        return [self.to(mu[:, self.split_i, :].T, 'tensor') for mu in mus]

    def get_mean_across_dim(self, X, dim):
        return [self.to(x.mean(dim)[None, :], 'tensor') for x in X]

    def get_layer(self, x, b, W, act):
        return act(self.clip(x + b) @ W.T)
 
    def clip(self, tensor):
        return torch.clip(tensor, -1, 1)

    def log_to_prob(self, x):
        return 1 - np.exp(-x.astype(float))
    
    def log_to_LLR(self, x):
        x = self.log_to_prob(x)
        return self.prob_to_LLR(x)
    
    def prob_to_LLR(self, x, eps = 1e-4):
        x = np.clip(x, a_min = eps, a_max = 1-eps)
        return np.log(x/(1-x))

    def LLR_to_prob(self, x):
        y = np.exp(x)
        return y/(y+1)

    def prob_to_log(self, x):
        return -np.log(1-x.astype(float))


    """ mouse plotting helpers"""
    
    def smooth(self, X, bin_size, axis = 0, rescale = False, keep_shape = False):
        kernel = np.ones(bin_size) / bin_size
        mode = 'same' if keep_shape else 'full'
        smoothed_arr =  np.apply_along_axis(lambda m: np.convolve(m, kernel, mode=mode), axis=axis, arr= X)
        if rescale: 
            smoothed_arr -= smoothed_arr.min()
            smoothed_arr *= 1/smoothed_arr.max()
            smoothed_arr = smoothed_arr*(X-X.min()).max() + X.min()
        return smoothed_arr
    
    def get_survival(self, x):
        dim = len(x.shape)
        x = np.insert(x, 0, np.zeros(1), axis = -1)
        from_one = x[:, 1:] if dim == 2 else x[:, :, 1:]
        til_one = x[:, :-1] if dim == 2 else x[:, :, :-1]
        return from_one * np.cumprod(1 - til_one, axis = -1)

    def get_hazard(self, data):
        if data.shape[0] == 1:
            return data/(np.clip(1-data.cumsum()+data[0], a_min = 1e-5, a_max = None))
        return [p/(np.clip(1-p.cumsum()+p[0], a_min = 1e-5, a_max = None)) for p in data]
        
    def get_action_color(self):
        self.act_cols = np.empty(len(self.all_times), dtype = object)     
        for i, a in enumerate(self.all_times):
            self.act_cols[i] = 'g' if self.rews[i] > 0 else 'r' if a > self.inaction else 'orange'
            
    def get_PGO_standards(self):
        self.PGO_LEG = [f"PGO = {p}" for p in self.PGO_range]
        self.PGO_ALPHA = [.5]*self.PGO_N
        self.PGO_COLOR = self.Cmap.to_rgba(np.linspace(0, 1, self.PGO_N))
        self.PGO_PLOT = ["line"]*self.PGO_N
        self.use_alpha_mus = False

    def make_standard_plot(self, ys ,xlab, plots, alphs, cols, xax = None, ylab = None, xticks = None, title = None, xlim = None, ylim = None, hlines = None, err = None, leg = None, traj = False, save_SVG = False, fig_x = 10, fig_y = 5, show=True):
        if not hasattr(self, 'save_plot'):
            self.save_plot = False
            self.font_dict = {}

        if self.show:
            if traj: 
                self.init_traj_plot()
            else:
                from enzyme import FIGPATH, TEXPATH
                # force reload
                import enzyme
                from importlib import reload
                try:    
                    reload(enzyme)
                except:
                    pass

                # TEXTWIDTH, init_mpl = enzyme.TEXTWIDTH, enzyme.init_mpl
                TEXTWIDTH = enzyme.TEXTWIDTH
                from enzyme.src.helper import save_plot

                # plt = init_mpl(usetex=False)
                fig = plt.figure(figsize=(fig_x, fig_y)) if not self.save_plot else plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH / 3)) 
                                
        for i, (y, p, c, a) in enumerate(zip(ys, plots, cols, alphs)):
            c = c if c is not None else f"C{i}"
            x = np.arange(len(y)) if xax is None else xax[i] 
            if p == 'line':
                plt.plot(x, y, color = c, alpha = a, linewidth = 3) 
            if p == 'scatter':
                plt.plot(x, y, '-o', color = c, alpha = a)
            if p == 'bar':
                plt.bar(i, y, color = c, alpha = a)
            if hlines is not None:
                plt.axhline(hlines[i], color = c, alpha = a, linestyle = '-.') 
            if err is not None:
                plt.fill_between(x, y-err[i], y+err[i], color = c, alpha = a/2)
                
        if xticks is not None:            plt.xticks(np.arange(len(xticks)), xticks)
        if title is not None:            plt.title(title,  fontdict=self.font_dict)
        if ylim is not None:             plt.ylim([ylim[0], ylim[1]])
        if xlim is not None:             plt.xlim([xlim[0], xlim[1]])
        
        if leg is not None:
            if leg == 'heat':
                self.make_color_bar()
            # else:
            #     plt.legend(leg[:i+1] if hlines is None else leg, loc = 'upper right')
        if ylab is not None:
            plt.ylabel(ylab,  fontdict=self.font_dict)
        plt.xlabel(xlab,  fontdict=self.font_dict); 
        if save_SVG or self.save_plot:
            plt.savefig(self.save_path / f"{title}.svg")
            plt.savefig(self.save_path / f"{title}.png")
            # print(f"saved to: " / self.save_path / f"{title}.png")
        if show:
            plt.show()

    def set_cog_map_labels(self, plt, subplot = False):
        if subplot: 
            plt.set_xlabel(f"{self.PC_Xlab}")
            plt.set_ylabel(f"{self.PC_Ylab}")
        else:
            plt.xlabel(f"{self.PC_Xlab}")
            plt.ylabel(f"{self.PC_Ylab}")
            