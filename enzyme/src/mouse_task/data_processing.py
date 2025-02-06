import numpy as np; import pylab as plt
from tqdm import tqdm; from scipy.optimize import minimize as optimize
from enzyme.colors import cmap; from enzyme.src.mouse_task.analytical_analysis import analytical_analysis; import matplotlib.cm as cm; from numpy.linalg import norm

class data_processing(analytical_analysis):
    def __init__(self, **params):
        self.__dict__.update(params)

    def plot_all_episodes(self, manager):
        self.preprocess_data(manager, skip_bayes_factorized = True, skip_bayes_full = True)
        self.plotting()
        
    def preprocess_data(self, manager, skip_bayes_factorized = True, skip_bayes_full = True):
        manager.data.postprocess_specials()
        self.preprocess_shallow_data(manager.data)
        self.preprocess_standard_data()
        print("getting analytical reward rate")
        self.get_total_analytical_optim()
        if self.training == False:
            self.preprocess_deep_data(manager)
            print("initiating bayesian analyes")  
            self.get_bayes_flow(factorize = True, skip = skip_bayes_factorized)
            self.get_bayes_flow(factorize = False, skip = skip_bayes_full)
        
        print("getting behavior from block switch")
        self.get_from_switch()
        print("smoothing time-series data")
        self.convolve_data()
        self.get_network_analytics()
        print("getting numerical and analytical behavior")
        self.get_network_optimality()        
        
    def preprocess_shallow_data(self, data = None):        
        if data != None:
            self.data = data
        self.analytical_ITI_mean = self.ITI_mean - 1 
        self.eps = self.data.episode
        self.held_out = self.eps.mean()                                 # hold out half of episodes for testing 
        self.episodes = self.eps[-1] + 1
        self.all_times = self.data.act_time
        self.trial_ends = self.data.end_times.astype(int)
        self.data_len = len(self.all_times)                             # total number of trials
        self.data_range = np.arange(self.data_len)                
        self.num_trials = np.max(self.data.trial.astype(int) + 1)       # trials per block
        self.rews = self.data.rew
        self.PGOs = self.data.PGO
        self.plant_PGOs = self.data.plant_PGO

        self.plant_inds = self.data.plant_inds
        self.plant_IDs = self.data.plant_ID
        self.PGO_range = np.unique(self.PGOs)
        self.PGO_N = len(self.PGO_range)
        self.wait_from_last_pot = self.data.wait_from_last_pot
        self.wait_from_last = self.data.wait_from_last
        self.RT_max = self.wait_from_last_pot.max()
        self.RT_mean = self.wait_from_last_pot.mean()
        self.RT_med = np.median(self.wait_from_last_pot.astype(int))
        self.acted = self.all_times != self.inaction
        self.act_inds = np.where(self.acted)[0]
        self.stim_durs = self.data.stim_dur
        self.last_PGOs = self.data.last_PGO
        self.trial = self.data.trial.astype(int)
        self.act_times = np.array([a if a != self.inaction else e for a, e in zip(self.all_times, self.trial_ends)])
        self.cum_end_times = np.cumsum(self.trial_ends) + np.arange(self.data_len)
        self.cum_start_times = np.cumsum(np.insert(self.trial_ends + 1, 0, 0)[:-1])
        self.cum_act_times = (self.act_times + self.cum_start_times).astype(int)
        self.switch = (self.trial == self.trial[-1]).astype(int) - (self.trial==0).astype(int)
        self.last_pot = np.array([last_pot if last_pot is not None else dur for dur, last_pot in zip(self.stim_durs, self.data.last_pot_nogo)])
        self.last_nog = np.array([last_nogo-1 if last_nogo is not None else 0 for last_nogo in self.data.last_nogo])
        self.Cmap = cm.ScalarMappable(cmap=cmap, norm = cm.colors.Normalize(0, 1))
        self.callable_Cmap = cmap
        self.get_PGO_standards()
        self.get_action_color()
        self.inp_W = np.stack(self.data.episode_inp_W)
        self.recur_W = np.stack(self.data.episode_recur_W)
        self.R_norm = norm(self.recur_W, axis = (1,2))
        self.I_norm = norm(self.inp_W, axis = (1,2))
        self.R_DW, self.I_DW = [np.zeros(len(self.recur_W)) for _ in range(2)]
        for i in range(len(self.recur_W)-1):
            self.R_DW[i] = (abs(self.recur_W[i] - self.recur_W[i+1])).mean()
            self.I_DW[i] = (abs(self.inp_W[i] - self.inp_W[i+1])).mean()
        
    def preprocess_standard_data(self):
        self.backbone_flat = np.hstack(self.data.backbone.flatten())
        self.mechanistic_DV_structured = self.raw_to_structured(0 * self.backbone_flat)

        self.num_steps = self.backbone_flat.shape[0]                    # total number of steps
        self.safe_PGOs = self.data.safe_PGO
        self.safe_backbone = self.data.safe_backbone
        self.similarity =np.round(1-np.abs(self.PGOs - self.safe_PGOs).astype(float), decimals=2)
        self.step_plant_PGO = ( [p or 0 for p in self.plant_PGOs])*(1 + 0*(self.data.backbone))
        if np.any(self.theta_traj != None ):
            self.theta_traj_N = len(self.theta_traj)
            self.step_PGO = np.array([])
            while len(self.step_PGO) < self.num_steps:
                self.step_PGO = np.concatenate((self.step_PGO, self.theta_traj), 0)
            self.step_PGO = self.step_PGO[:self.num_steps]
            self.step_PGO = self.raw_to_structured(self.step_PGO)
        else:
            self.step_PGO = self.trial_var_to_step_var(self.PGOs)
        self.step_pot_RT = self.trial_var_to_step_var(self.wait_from_last_pot)
        self.step_RT = self.trial_var_to_step_var(self.wait_from_last)
        self.step_trial = self.trial_var_to_step_var(self.trial)
        self.step_last_nogo = self.trial_var_to_step_var(self.last_nog)
            
        c = 0
        self.cum_rew_per_block = np.zeros(self.data_len)
        for i, r in enumerate(self.rews):
            c = (self.trial[i]>0)*c + r
            self.cum_rew_per_block[i] = c
            
        self.PGO_backbone =  np.hstack(self.step_PGO)                                            # important since flat will be indexed
        self.step_episode = self.trial_var_to_step_var(self.eps)       
        self.all_inds = np.arange(self.num_steps)
        self.safe_backbone_flat = np.hstack(self.data.safe_backbone.flatten())
        self.trial_steps = np.hstack([np.arange(s, e) for s, e in zip(self.trial_ends*0, self.trial_ends + 1)]).astype(int)
        self.trial_pre_act = np.hstack([np.arange(e) <= a for a, e in zip(self.act_times, self.trial_ends + 1)]).astype(int)
        self.stim = self.data.stim      
        self.binary_stim = np.array([(s.mean() > .5)  for s in self.stim])  
        self.avg_stim = np.array([s.mean() for s in self.stim]).round(1)
        
        self.preprocess_dists()
        self.preprocess_misc()

    def preprocess_misc(self):
        self.PC_dim = 3
        self.fig_W = 10
        self.fig_H = 5
        self.font_dict = {'fontsize': 10, 'fontweight': 'medium'}
        self.bayes_style = '--'
        self.max_mem = 20
        self.mem_range = 1 + np.arange(self.max_mem)     # vector going from 1 to free parameter length (e.g. mem length)
        self.bayes_resolution = 100                      # resolution of bayesian distributions
        self.bayes_range = np.linspace(0, 1, self.bayes_resolution) 
        self.bayes_range_inds = np.arange(0, self.bayes_resolution, 1)
        
        self.xax_range = 45        
        self.thresh_N = 45  
        self.bin_size = .2
        self.thresh = 1 + np.arange(self.thresh_N) 
        self.x_axis = 1 + np.arange(self.trial_dur)
        self.cog_map = None # [ 'regression', 'theory', None ]
        self.factorize = False

        self.bayes_type = 'hybrid' # ['standard', 'soft', 'hybrid', 'discount', 'dynamic']
        self.get_PCA_angles()

        self.act_xax = np.arange(len(self.act_inds))
        self.num_trial_xax = np.arange(self.num_trials-1)
        self.wait_from_switch_all = np.zeros((self.PGO_N, self.num_trials), dtype=object)  # last axis is over repititions
        self.wait_from_switch = np.zeros((self.PGO_N, self.num_trials))
        
        self.PGO_corr, self.last_PGO_corr = [np.zeros(self.num_trials) for _ in range(2)]
        self.rate_mu, self.rew_mu, self.wait_mu, self.act_mu = [np.zeros(self.PGO_N) for _ in range(4)]
        
    def preprocess_deep_data(self, manager):            
        self.net = manager.agent.state_dict()
        self.lick_prob = self.data.lick_prob # network output, not emperical
        self.value = self.data.value
        self.net_input = self.data.net_input #NOGO / GO/ NO LICK / LICK/ REW
        self.net_output = self.data.net_output
        self.Q_values = self.data.Qs
        self.i_gate = self.data.i_gate
        self.f_gate = self.data.f_gate
        self.c_gate = self.data.c_gate
        self.o_gate = self.data.o_gate
        self.LTM = self.data.LTM
        self.gate_names =['INPUT gate', 'FORGET gate', 'CELL gate', 'OUTPUT gate']
        self.activity_names = self.gate_names  + ['LTM', 'net output'] 
        
        self.step_inputs = self.to(self.flatten_trajectory(self.net_input), 'np')
        self.get_consec_stim()
        self.get_window_avg()
        self.make_last_m_stim_matrix()
        
    def get_consec_stim(self,  curr_N = 0, prev_stim = 0):
        if self.step_inputs.shape[0] > 3:
            S = self.step_inputs[1]
            A = self.step_inputs[3]
        else: # if input type is "raw" input 
            S = self.step_inputs[0]
            A = self.step_inputs[1]
        consec = np.zeros(self.num_steps)
        for i in range(self.num_steps):
            stim = S[i] 
            if (A[i] == 0) and (stim == prev_stim or i == 0):                                               # action counts as nogo
            # if  (stim == prev_stim or i == 0):                                                            # action does not count as nogo
                curr_N = curr_N + int(stim == 1) - int(stim == 0)
            else:
                curr_N = int(stim == 1) - int(stim == 0)                                                    # reset to +/- 1
            consec[i] = curr_N 
            prev_stim = stim
        self.consec_stim = self.raw_to_structured(consec)
        
    def get_window_avg(self):
        self.MF_num = 50
        window_log = np.zeros((self.MF_num, self.num_steps))
        for i in range(self.MF_num): 
            window_log[i, :] = self.smooth(self.step_inputs[1], 2*(i + 1))[i:-(i+1)]
        self.MF_theta_structured = self.raw_to_structured(window_log, time_dim = 2)


    def make_last_m_stim_matrix(self, m = 20):
        S = self.step_inputs[1]
        L= len(S)
        self.last_m = np.zeros((m, L))
        for i in range(L):
            if i > m:
                self.last_m[:, i] = S[i-m+1 : i+1]
                
    def get_best_exp_theta_est(self):
        # self.get_indices(cross_validation = None, planted = False, til_action = True, eps_init = 0, flatten = True)
        ms = [3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        self.exp_theta_r2 = np.zeros(len(ms))
        for m_i, m in enumerate(ms):
            self.make_last_m_stim_matrix(m)
            self.exp_weights = np.flip(-np.arange(self.last_m.shape[0]))
            self.res = optimize(self.exp_est_theta, x0 = [.5, .5])
            self.exp_est_theta(self.res.x)
            self.exp_theta_r2[m_i] = self.R2(self.weighted_sum, self.PGO_flat)
        plt.axhline(self.R2(self.net_theta_flat, self.PGO_flat), label = "theta decoding from STM")
        plt.plot(ms, self.exp_theta_r2, label = "theta decoding from exponential window", c = 'C1')
        plt.xlabel("exponential window")
        plt.ylabel("explained variance")
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
        
    def exp_est_theta(self, x):
        weights = np.exp(x[0]*(self.exp_weights-x[1]))
        self.weighted_sum = (weights @ self.last_m)
        return 1/self.R2(self.weighted_sum, self.PGO_backbone)
 
    def get_PCA_angles(self):
        self.default_angles = [ (90, -90),(0, -90), (0, 0)]
        self.morphological = [(-134, 156)]
        self.contextual = [(-161, -144)]
        self.action_centric =  [(-165, 90)] 
        self.diff_A = [(29,9)] 
        self.diff_B = [(-15, -30)]
        self.super_angles = []
        for angle in (self.default_angles, self.morphological, self.contextual, self.action_centric, self.diff_A, self.diff_B ):
            for a in angle:
                self.super_angles.append(a)
        self.angles3D = self.super_angles#[self.default_angles[0], self.diff_A[0], self.diff_B[0], self.contextual[0]] #default_angles

    def convolve_data(self):
        self.episode_xax = np.arange(self.episodes) 
        trials_per_bin = self.num_trials
        self.episode_rews, self.episode_PGOs, self.episode_action_times, self.episode_wait_from_last, self.episode_rew_rate = [np.zeros((self.PGO_N, self.episodes)) for _ in range(5)]
        for p_i, p in enumerate (self.PGO_range):
            PGO_inds = np.where( self.PGOs == p )[0] 
            s = 0 
            R = self.rews[PGO_inds]
            A = self.act_times[PGO_inds]
            S = self.wait_from_last[PGO_inds]
            for i in range(self.episodes):
                e = s + trials_per_bin
                self.episode_rews[p_i, i] = np.mean(R[s : e])
                self.episode_action_times[p_i, i] = np.mean(A[s : e])
                self.episode_wait_from_last[p_i, i] = np.mean(S[s : e])
                self.episode_rew_rate[p_i, i] = self.episode_rews[p_i, i] / (self.episode_action_times[p_i, i] + self.ITI_mean)
                s += self.num_trials
                
    def preprocess_dists(self):     
        # self.get_indices(col = 'block', flatten = False, prep_mus = False, cross_validation=None, needs_acted = True)
        self.get_indices(col = 'block', flatten = False, prep_mus = False, cross_validation=None, needs_acted = False)
        self.wait_PDF, self.wait_PDF_avg, self.wait_PDF_var, self.dist_x_lim, self.wait_xax = self.to_dist(self.wait_from_last)
        self.wait_pot_PDF, _, _, _, self.wait_pot_xax = self.to_dist(self.wait_from_last_pot)
        self.wait_hazard = self.get_hazard(self.wait_PDF)
        self.wait_pot_hazard = self.get_hazard(self.wait_pot_PDF)

    def to_dist(self, waiting_data):
        PDF, avg, var, xax, xlim = [np.zeros(self.split_num, dtype = object) for _ in range(5)]
        for self.split_i, self.split_curr in enumerate(self.split):
            self.split_inds = np.where((self.split_cond == self.split_curr))[0]
            PGO_waits = waiting_data[self.trial_inds][self.split_inds]
            # pot_RTs = np.arange(np.min(PGO_waits), np.max(PGO_waits)+1)
            pot_RTs = np.unique(PGO_waits)
            y = [len(np.where(PGO_waits == RT)[0]) for RT in pot_RTs]
            PDF[self.split_i] = np.array(y)/np.sum(y)
            avg[self.split_i] = np.mean(PGO_waits)
            var[self.split_i] = np.var(PGO_waits)
            xax[self.split_i] = pot_RTs
            xlim[self.split_i] = len(y)
        return PDF, avg, var, xlim, xax

    def get_from_switch(self):
        for t in range(self.num_trials):
            for j, p in enumerate(self.PGO_range):
                self.get_indices(From = t, Til = t, curr_PGO = p, planted = False, prep_mus = False,  cross_validation = None)
                self.wait_from_switch_all[j, t] = self.wait_from_last[self.trial_inds-1]      
                self.wait_from_switch[j, t] = self.wait_from_last[self.trial_inds-1].mean() 
            self.get_curr_prev_PGO_corr(t)

    def get_curr_prev_PGO_corr(self, t):
        self.get_indices(From = t, Til = t, planted = False, prep_mus = False)
        target = self.wait_from_last[self.trial_inds-1]
        self.PGO_corr[t] = self.correlate(X = self.PGOs[self.trial_inds], Y = target)
        self.last_PGO_corr[t] = self.correlate(X = self.last_PGOs[self.trial_inds], Y = target)
        
    def get_network_analytics(self):
        self.rate_std = np.zeros_like(self.rate_mu)
        self.wait_std = np.zeros_like(self.wait_mu)
        for j, p in enumerate(self.PGO_range):
            self.get_indices(curr_PGO = p, planted = False, prep_mus = False, needs_acted = True, cross_validation = None)
            # self.get_indices(curr_PGO = p, planted = False, prep_mus = False, needs_acted = False, cross_validation = None)
            rews = self.rews[self.trial_inds]                      
            acts = self.all_times[self.trial_inds]
            ends = self.trial_ends[self.trial_inds] # + 1
            waits = self.wait_from_last[self.trial_inds] 
            self.rate_mu[j] = rews.mean() / ends.mean()
            self.rew_mu[j] = rews.mean()
            self.wait_mu[j] = waits.mean() 
            self.wait_std[j] = waits.std() 
            self.act_mu[j] = acts.mean()
          
    def get_network_optimality(self):
        self.ana_similarity = self.get_cosin_similarity(self.ana_max_thresh, self.ana_max_thresh) 
        self.net_similarity = self.get_cosin_similarity(self.wait_mu, self.ana_max_thresh) 
        self.fixed_similarity = self.get_cosin_similarity(self.fixed_max_thresh, self.ana_max_thresh)     

