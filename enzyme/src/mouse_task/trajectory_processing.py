from enzyme.src.mouse_task.flow_fields import flow_fields; from scipy.signal import correlate as cross; import numpy as np; from sklearn.decomposition import PCA;
import sklearn; from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor; from tqdm import tqdm
from scipy.optimize import minimize
# from sklearn.manifold import Isomap; from scipy.sparse import csr_matrix, lil_matrix; import umap; 
# from sklearn.mixture import  GaussianMixture
 
"""Class for performing the PCA basis construction and projection"""
class trajectory_processing(flow_fields):
    def __init__(self, **params):
        self.__dict__.update(params)
        
    def run_PCA(self):
        self.get_indices(planted = False, flatten = False, prep_mus = False, cross_validation = "training")
        self.get_minimal_flat()
        self.get_PCA_basis()
        self.run_regressions()
        self.PC_flat = self.proj.transform(self.output_flat.T).T
        self.get_LSTM_weights()
        # self.get_theta_error()
        # self.proj_PCs_onto_gates()
        # self.get_corrs()
        # self.plot_gate_eigen()
        # self.plot_corrs()   
            
    def get_corrs(self):
        self.inp_cross = np.zeros((3, self.PC_dim, self.max_traj_steps))
        self.auto_corrs = np.zeros((3, self.max_traj_steps))
        inputs = self.input_flat[[1,3,4], :] #GO/LICK/REW
        PCs = self.proj.transform(self.output_flat.T).T
        self.inp_corrs = self.correlate(X = inputs, Y = PCs)
        for self.dim in range(self.PC_dim):
            self.get_PC_cross(inputs, PCs)

    def get_PC_cross(self, inputs, PCs):
        Y = PCs[self.dim, :]
        Y = (Y - Y.mean())/Y.var()
        keep = int(len(Y))
        offset = int(self.max_traj_steps/2)
        for inp_dim in range(inputs.shape[0]):   
            X = inputs[inp_dim,:] 
            X = (X - X.mean())/X.var()
            inp = cross(X, Y)[keep - offset : keep + offset]
            auto = cross(Y, Y)[keep - offset : keep + offset]
            self.inp_cross[inp_dim, self.dim, :] = inp/max(abs(inp))
            self.auto_corrs[self.dim, :] = auto/max(abs(auto))
                
    def run_trajectory(self, plot = True):
        self.get_trajectory_mus()
        if plot:
            self.plot_trajectories()
            self.plot_on_PCA_basis()        

    def get_PCA_basis(self):
            # self.theta_dist_flat = np.vstack(np.hstack(self.theta_dist_structured[self.trial_inds])).T
            # self.belief_dist_flat = np.vstack(np.hstack(self.belief_dist_structured[self.trial_inds])).T

            # PGO_ind = np.searchsorted(self.PGO_range,self.PGO_flat) 
            # thresh_ind = np.clip(self.consec_flat, a_min = 0, a_max = self.thresh_N).astype(int)
            
            data = [np.random.randn(self.hid_dim, self.hid_dim), self.LTM_flat, self.output_flat]
            names = ["rand", "LTM", "output"]
            x = self.get_data_of(data, names, select = self.basis_on)
            self.proj = PCA(n_components = self.PC_dim)
            self.proj.fit(np.transpose(x))

            self.basis = self.proj.components_
            self.data_mean = x.mean(1)
            print(f"explained variance: {self.proj.explained_variance_ratio_}")
            
    def get_minimal_flat(self):
            self.input_flat = self.flatten_trajectory(self.net_input[self.trial_inds])
            self.LTM_flat = self.flatten_trajectory(self.LTM[self.trial_inds])
            self.Q_flat = self.flatten_trajectory(self.Q_values[self.trial_inds])
            self.QDIFF_flat = self.Q_flat[1] - self.Q_flat[0]
            self.PGO_flat = self.flatten_trajectory(self.step_PGO[self.trial_inds])
            self.plant_PGO_flat = self.flatten_trajectory(self.step_plant_PGO[self.trial_inds])
            self.PSAFE_flat = self.flatten_trajectory(self.safe_backbone[self.trial_inds])
#            self.C_gate_flat = self.flatten_trajectory(self.c_gate[self.trial_inds])
#            self.O_gate_flat = self.flatten_trajectory(self.o_gate[self.trial_inds])
#            self.I_gate_flat = self.flatten_trajectory(self.i_gate[self.trial_inds])
#            self.F_gate_flat = self.flatten_trajectory(self.f_gate[self.trial_inds])
            self.input_flat = self.flatten_trajectory(self.net_input[self.trial_inds])
            self.output_flat = self.flatten_trajectory(self.net_output[self.trial_inds])
            self.consec_flat = self.flatten_trajectory(self.consec_stim[self.trial_inds])
            self.pot_RT_flat = self.flatten_trajectory(self.step_pot_RT[self.trial_inds])
            self.RT_flat = self.flatten_trajectory(self.step_RT[self.trial_inds])
            self.lick_prob_flat = self.flatten_trajectory(self.lick_prob[self.trial_inds])

            self.flow_belief_flat = self.flatten_trajectory(self.flow_belief_structured[self.trial_inds])
            self.flow_theta_flat = self.flatten_trajectory(self.flow_theta_structured[self.trial_inds])
            self.factorized_theta_flat = self.flatten_trajectory(self.factorized_theta_structured[self.trial_inds]).astype(np.float64)
            self.factorized_belief_flat = self.flatten_trajectory(self.factorized_belief_structured[self.trial_inds]).astype(np.float64)
            self.postprocess_bayes_flat()                      

    def run_regressions(self, regression_noise = 0, regress_to = 'ground_truth', regress_from = 'STM', get = False, cog_map = None, pre_act = False):
        self.cog_map = cog_map
        self.regress_to = regress_to
        self.regress_from = regress_from
        if get:
            self.get_indices(planted = False, flatten = False, prep_mus = False, cross_validation = "training", til_action = True)
            self.get_minimal_flat()

        self.PC_flat = self.proj.transform(self.output_flat.T).T
        self.get_GO_NOGO_inds()
        self.get_reg_vec()
                
        X = self.pred_from.T + regression_noise * np.random.randn(*self.pred_from.T.shape)
        if regress_to == 'bayes':
            DV_Y = self.flow_dist_from_opt_flat
            T_Y = self.prob_to_LLR(self.flow_opt_thresh_flat)
            Q_Y = self.log_to_LLR(self.flow_belief_flat)
            PGO_Y = self.flow_theta_flat
            Q_reg_type = LinearRegression()

        if regress_to == 'naive':
            DV_Y = self.factorized_dist_from_opt_flat
            T_Y = self.prob_to_LLR(self.factorized_opt_thresh_flat)
            Q_Y = self.log_to_LLR(self.factorized_belief_flat)
            PGO_Y = self.factorized_theta_flat
            Q_reg_type = LinearRegression()

        if regress_to == 'ground_truth':
            DV_Y = self.flow_dist_from_opt_flat
            T_Y =  self.opt_thresh_flat
            Q_Y = self.PSAFE_flat
            PGO_Y = self.PGO_flat
            Q_reg_type = LogisticRegression()
        
        if pre_act: 
            inds = np.where(self.ACTION_inds)[0]
            self.Q_reg = Q_reg_type.fit(X[inds], Q_Y[inds])
            self.PGO_reg = LinearRegression().fit(X[inds], PGO_Y[inds])  
            self.DV_reg = LinearRegression().fit(X[inds], DV_Y[inds])  
            self.T_reg = LinearRegression().fit(X[inds], T_Y[inds])
        else:
            self.Q_reg = Q_reg_type.fit(X, Q_Y)
            self.PGO_reg = LinearRegression().fit(X, PGO_Y)  
            self.DV_reg = LinearRegression().fit(X, DV_Y)  
            self.T_reg = LinearRegression().fit(X, T_Y)
      
        if self.cog_map == 'policy':
            self.PGO_reg = LinearRegression().fit(X, self.RT_flat)
        self.get_net_decodings()
        self.handle_cog_map()

    def get_net_decodings(self):
        self.net_theta_flat = self.PGO_reg.predict(self.pred_from.T)
        self.net_thresh_flat = self.T_reg.predict(self.pred_from.T)

        if self.regress_to == 'ground_truth':
            self.net_DV_est_flat = self.DV_reg.predict(self.pred_from.T)
            state = self.Q_reg.predict_proba(self.pred_from.T)[:,1]
            self.net_belief_flat = -np.log(1-state)
            thresh = self.net_thresh_flat
        else:
            self.net_DV_est_flat = -np.exp(self.DV_reg.predict(self.pred_from.T))
            state = self.LLR_to_prob(self.Q_reg.predict(self.pred_from.T))
            thresh = self.LLR_to_prob(self.net_thresh_flat)
            self.net_belief_flat = state
        self.net_dist_from_opt_flat = state - thresh

    # def get_bayes_dist_regression(self):
    #     self.belief_STD = self.dist_to_std(self.belief_dist_flat, np.arange(0,3, 1)[:,None])
    #     self.theta_STD =  self.dist_to_std(self.theta_dist_flat, np.linspace(0,1, self.bayes_resolution)[:,None])
    #     self.theta_std_reg = LinearRegression().fit(self.pred_from.T, self.theta_STD)
    #     self.belief_std_reg = LinearRegression().fit(self.pred_from.T, self.belief_STD)
    #     self.theta_std_reg = LinearRegression().fit(self.pred_from.T, self.theta_STD)
    #     self.belief_std_reg = LinearRegression().fit(self.pred_from.T, self.belief_STD)  
    #     self.bayes_psafe_non_log = 1-np.exp(-self.flow_belief_flat.astype(float))
    #     self.bayes_belief_reg = LinearRegression().fit(self.pred_from.T, self.bayes_psafe_non_log)
    #     self.bayes_theta_reg = LinearRegression().fit(self.pred_from.T, self.flow_theta_flat)

        
        
    # def fit_GMM(self):
    #     T = self.theta_dist_flat.shape[1]
    #     fit_alpha, fit_beta, loc, scale =[np.empty(T) for _ in range(4)]
    #     for t in tqdm(range(T)):         
    #         fit_alpha[t], fit_beta[t], loc[t], scale[t] = beta.fit(self.theta_dist_flat[:, t, None])
    #     fit = LinearRegression().fit(self.pred_from.T, self.fit_alpha.T)
    #     pred = fit.predict(self.pred_from.T)
            
    # def get_RMSE(self):
    #     self.get_indices(planted = False, flatten = True, prep_mus = True, cross_validation = "testing")
    #     self.belief_STD = self.dist_to_std(self.belief_dist_flat, np.arange(0,3, 1)[:,None])
    #     self.theta_STD =  self.dist_to_std(self.theta_dist_flat, np.linspace(0,1, self.bayes_resolution)[:,None])
    #     self.flow_theta_flat = self.flatten_trajectory(self.flow_theta_structured[self.trial_inds])
    #     self.flow_belief_flat = self.flatten_trajectory(self.flow_belief_structured[self.trial_inds])
    #     self.bayes_psafe_non_log = 1-np.exp(-self.flow_belief_flat.astype(float))
    #     self.state_RMSE = np.sqrt(np.mean((self.PSAFE_flat-self.Q_reg.predict_proba(self.pred_from.T)[:, 1])**2))#  / self.PSAFE_flat.std()
    #     self.theta_RMSE = np.sqrt(np.mean((self.PGO_flat-self.PGO_reg.predict(self.pred_from.T))**2))#   / self.PGO_flat.std()
    #     self.state_STD_RMSE = np.sqrt(np.mean((self.belief_STD-self.belief_std_reg.predict(self.pred_from.T))**2))#  / self.belief_STD.std()
    #     self.theta_STD_RMSE = np.sqrt(np.mean((self.theta_STD-self.theta_std_reg.predict(self.pred_from.T))**2))#   / self.theta_STD.std()
    #     self.bayes_state_RMSE = np.sqrt(np.mean((self.bayes_psafe_non_log-self.Q_reg.predict_proba(self.pred_from.T)[:, 1])**2))#    / self.bayes_psafe_non_log.std()
    #     self.bayes_theta_RMSE = np.sqrt(np.mean((self.flow_theta_flat-self.PGO_reg.predict(self.pred_from.T))**2))#   / self.flow_theta_flat.std()
    #     self.bayes_state_reg_RMSE = np.sqrt(np.mean((self.bayes_psafe_non_log-self.bayes_belief_reg.predict(self.pred_from.T))**2))#   / self.bayes_psafe_non_log.std()
    #     self.bayes_theta_reg_RMSE = np.sqrt(np.mean((self.flow_theta_flat-self.bayes_theta_reg.predict(self.pred_from.T))**2))#   / self.flow_theta_flat.std()
    #     print("state RMSE ", self.state_RMSE)
    #     print("theta RMSE ", self.theta_RMSE)
    #     print("state STD RMSE ", self.state_STD_RMSE)
    #     print("theta STD RMSE ", self.theta_STD_RMSE)
    #     print("bayes state RMSE ", self.bayes_state_RMSE)
    #     print("bayes theta RMSE ", self.bayes_theta_RMSE)
    #     print("bayes state reg RMSE ", self.bayes_state_reg_RMSE)
    #     print("bayes theta reg RMSE ", self.bayes_theta_reg_RMSE)
    #     """ logistic regression to full PGO distribution """
    #     # PGO_label = np.where(self.PGO_flat[:, None] == self.PGO_range[None,:])[1]
    #     # self.decoded_dist_reg = LogisticRegression().fit(self.pred_from.T, PGO_label)

    def dist_to_std(self, dist, xax):
        X = dist*xax
        return np.mean(X**2, 0) - np.mean(X, 0)**2       
        
    def get_theta_error(self):
        self.bayes_mse, self.net_mse, self.net_ratio, self.bayes_ratio = [np.zeros((self.PGO_N, 2)) for _ in range(4)]
        self.MF_mse = np.zeros((self.MF_num, self.PGO_N, 2))
        for trial_i, trial in enumerate([0, self.num_trials-1]):
            self.get_indices(col = 'block', From = trial, Til = trial,  flatten = True)
            for self.split_i, self.split_curr in enumerate(self.split):
                self.get_split_inds()
                inds = self.full_split_inds_flat.astype(int)
                self.bayes_mse[self.split_i, trial_i] = ((self.flow_theta_flat[inds] - self.split_curr)**2).mean()
                self.net_mse[self.split_i, trial_i] = ((self.net_theta_flat[inds] - self.split_curr)**2).mean()
                self.MF_mse[:, self.split_i, trial_i] = ((self.MF_theta_flat[:, inds] - self.split_curr)**2).mean(-1)
                opt_MF = np.argmin(self.MF_mse[:, self.split_i, trial_i])
                self.net_ratio[self.split_i, trial_i] = self.net_mse[self.split_i, trial_i] / self.MF_mse[opt_MF, self.split_i, trial_i] 
                self.bayes_ratio[self.split_i, trial_i] = self.bayes_mse[self.split_i, trial_i] / self.MF_mse[opt_MF, self.split_i, trial_i] 
                        
    def get_full_PCA(self, x):
        self.full_PCA = PCA(self.hid_dim).fit(x)
        self.full_proj = self.full_PCA.transform(x).T
        
    """ get mean trajectories """
    def get_trajectory_mus(self):
        self.preprocess_mus()
        for self.split_i, self.split_curr in enumerate(self.split):
            self.get_split_inds()
            self.get_trajectory_mu_per_step()
            self.postprocess_mus()

    def get_trajectory_mu_per_step(self):
        for self.step in range(self.max_traj_steps):   
            self.get_trajectory_step_inds()
            self.get_net_mus()
   
    def postprocess_mus(self):
        no_prob = self.emp_act_prob[self.split_i, :].sum() == 0
        self.emp_act_prob[self.split_i, :] = 0 if no_prob else self.emp_act_prob[self.split_i, :]/sum(self.emp_act_prob[self.split_i, :])
        max_prob = 1e-3 + np.max(self.emp_act_prob[self.split_i, :])
        self.act_col[self.split_i, :,:] = self.Cmap.to_rgba(self.emp_act_prob[self.split_i, :]/max_prob)
        self.activities = [self.I_gate_mus, self.F_gate_mus, self.C_gate_mus, self.O_gate_mus, self.LTM_mus, self.output_mus]
        self.PC_traj_diffs[:, self.split_i, :] = np.diff(self.PC_mus[:, self.split_i, :], prepend = 0, axis = -1)
        
    def get_net_mus(self):
        if len(self.step_inds) > 0:
            self.PGO_mus[self.split_i, self.step] = self.PGO_flat[self.step_inds].mean()
            self.PSAFE_mus[self.split_i, self.step] = self.PSAFE_flat[self.step_inds].mean()
            self.GO_mus[self.split_i, self.step] = self.input_flat[1, self.step_inds].mean(-1)

            """ for without full tensors: training data """
            # self.lick_prob_mus[self.split_i, self.step] = self.value_mus[self.split_i, self.step] = self.QDIFF_mus[self.split_i, self.step] =  0
            
            self.lick_prob_mus[self.split_i, self.step] = self.lick_prob_flat[self.step_inds].mean(-1)
            self.value_mus[self.split_i, self.step] = self.value_flat[self.step_inds].mean(-1)
            self.QDIFF_mus[self.split_i, self.step] =  (self.Q_flat[1, self.step_inds] - self.Q_flat[0, self.step_inds]).mean(-1) 
            self.N_mus[self.split_i, self.step] = len(self.step_inds)

            self.pred_from_mus[:, self.split_i, self.step] = self.pred_from[:, self.step_inds].mean(-1)
            self.output_mus[:, self.split_i, self.step] = self.output_flat[:, self.step_inds].mean(-1)
            self.LTM_mus[:, self.split_i, self.step] = self.LTM_flat[:, self.step_inds].mean(-1)
            self.I_gate_mus[:, self.split_i, self.step] = self.I_gate_flat[:, self.step_inds].mean(-1)
            self.F_gate_mus[:, self.split_i, self.step] = self.F_gate_flat[:, self.step_inds].mean(-1)
            self.O_gate_mus[:, self.split_i, self.step] = self.O_gate_flat[:, self.step_inds].mean(-1)
            self.C_gate_mus[:, self.split_i, self.step] = self.C_gate_flat[:, self.step_inds].mean(-1)
            self.PC_mus[:, self.split_i, self.step] = self.PC_flat[:, self.step_inds].mean(-1)    

            self.mechanistic_DV_mus[self.split_i, self.step]  = self.mechanistic_DV_flat[self.step_inds].mean(-1)

            self.flow_theta_mus[self.split_i, self.step]  = self.flow_theta_flat[self.step_inds].mean(-1)
            self.flow_opt_thresh_mus[self.split_i, self.step]  = self.flow_opt_thresh_flat[self.step_inds].mean(-1)
            self.flow_belief_mus[self.split_i, self.step]  = self.flow_belief_flat[self.step_inds].mean(-1)
            self.flow_dist_from_opt_mus[self.split_i, self.step]  =  self.flow_dist_from_opt_flat[self.step_inds].mean(-1)           
            self.net_dist_from_opt_mus[self.split_i, self.step]  = self.net_dist_from_opt_flat[self.step_inds].mean(-1)
            self.net_DV_est_mus[self.split_i, self.step]  = self.net_DV_est_flat[self.step_inds].mean(-1)
            self.flow_dist_from_opt_vars[self.split_i, self.step]  =  self.flow_dist_from_opt_flat[self.step_inds].var(-1)           
            self.net_dist_from_opt_vars[self.split_i, self.step]  = self.net_dist_from_opt_flat[self.step_inds].var(-1)
            
            self.factorized_dist_from_opt_mus[self.split_i, self.step]  =  self.factorized_dist_from_opt_flat[self.step_inds].mean(-1)
            self.flow_belief_vars[self.split_i, self.step]  = self.flow_belief_flat[self.step_inds].var(-1)
            self.flow_theta_vars[self.split_i, self.step]  = self.flow_theta_flat[self.step_inds].var(-1)
            self.factorized_theta_mus[self.split_i, self.step]  = self.factorized_theta_flat[self.step_inds].mean(-1)
            self.factorized_belief_mus[self.split_i, self.step]  = self.factorized_belief_flat[self.step_inds].mean(-1)
            self.factorized_belief_vars[self.split_i, self.step]  = self.factorized_belief_flat[self.step_inds].var(-1)
            self.factorized_theta_vars[self.split_i, self.step]  = self.factorized_theta_flat[self.step_inds].var(-1)            
            self.net_theta_mus[self.split_i, self.step]  = self.net_theta_flat[self.step_inds].mean(-1)
            self.net_belief_mus[self.split_i, self.step]  = self.net_belief_flat[self.step_inds].mean(-1)            
            self.net_theta_vars[self.split_i, self.step]  = self.net_theta_flat[self.step_inds].var(-1)
            self.net_belief_vars[self.split_i, self.step]  = self.net_belief_flat[self.step_inds].var(-1)            
        
            self.flow_belief_RMS_mus[self.split_i, self.step]  = self.flow_belief_RMS_flat[self.step_inds].mean(-1)
            """ bayes mus and vars in prob """
            # self.flow_belief_vars[self.split_i, self.step]  = self.log_to_prob(self.flow_belief_flat)[self.step_inds].var(-1)
            # self.net_belief_vars[self.split_i, self.step]  = self.log_to_prob(self.net_belief_flat)[self.step_inds].var(-1)            
            # self.flow_belief_mus[self.split_i, self.step]  = self.log_to_prob(self.flow_belief_flat)[self.step_inds].mean(-1)
            # self.net_belief_mus[self.split_i, self.step]  = self.log_to_prob(self.net_belief_flat)[self.step_inds].mean(-1)            
            
    """ LSTM projections """  
            
    def proj_PCs_onto_gates(self):
        self.gate_proj = np.zeros((self.PC_dim, 4, self.hid_dim))
        self.act_proj = self.basis @ self.normed_Wa.T
        for gate_i, gate in enumerate(self.mode_normed_weights):
            self.gate_proj[:, gate_i, :] = self.basis @ gate
            # self.gate_proj[:, gate_i, :] = self.proj.transform(gate).T
        # self.plot_net_angles()
        
    def get_LSTM_weights(self):
        self.Wa = self.net['to_action.weight'];
        W = self.net['actor.hgates.weight']
        B = self.net['actor.hgates.bias']
        self.Wf, self.Wi, self.Wc, self.Wo = (W,W,W,W) if self.mode != 'LSTM' else W.chunk(4,0)
        self.Bf, self.Bi, self.Bc, self.Bo = (B,B,B,B) if self.mode != 'LSTM' else B.chunk(4,0)
        self.Wa_cpu, self.Wi_cpu, self.Wo_cpu, self.Wc_cpu, self.Wf_cpu,\
        self.Wa_cpu, self.Bi_cpu, self.Bo_cpu, self.Bc_cpu, self.Bf_cpu =\
            self.to([self.Wa, self.Wi, self.Wo, self.Wc, self.Wf,\
            self.Wa, self.Bi, self.Bo, self.Bc, self.Bf], 'np')
        self.normed_Wa = self.norm_W(self.Wa_cpu)
        self.normed_Wi = self.norm_W(self.Wi_cpu)
        self.normed_Wf = self.norm_W(self.Wf_cpu)
        self.normed_Wc = self.norm_W(self.Wc_cpu) 
        self.normed_Wo = self.norm_W(self.Wo_cpu)
        self.mode_normed_weights = [self.normed_Wi, self.normed_Wf, self.normed_Wc, self.normed_Wo]
        self.mode_weights = [self.Wi_cpu, self.Wf_cpu, self.Wc_cpu, self.Wo_cpu]

    def fit_mechanistic(self, 
            alpha_0 = .5, beta_0 = 1, eta_0 = -.5, slope_0 = .1, mem_0 = .5, thresh_0 = -1,\
            use_ctx_thresh = False, use_ctx_slope = False, use_ctx_mem = False, fit_steps = None):
        init = [alpha_0, beta_0, eta_0]\
             + [slope_0]  * self.PGO_N\
             + [mem_0]    * self.PGO_N\
             + [thresh_0] * self.PGO_N

        thresh_bounds = tuple([(None, 0)] * self.PGO_N)
        slope_bounds = tuple([(.01, None)] * self.PGO_N)
        mem_bounds = tuple([(.01, .995)] * self.PGO_N)
        bounds = ((.001,.999), (.001, None), (None,None)) + \
            0. + mem_bounds  + thresh_bounds 

        self.use_ctx_thresh = use_ctx_thresh
        self.use_ctx_slope = use_ctx_slope
        self.use_ctx_mem = use_ctx_mem
        self.fit_steps = fit_steps 
        self.preprocess_mechanistic(init, bounds)
        self.preprocess_mechanistic(test = True)

    def preprocess_mechanistic(self, init = None, bounds = None, test = False):
        if test:
            self.store_mechanistic_DV = True 
            self.fit_steps = self.num_steps
            self.mechanistic_DV = np.zeros(self.num_steps)
            self.mechanistic_DV_structured = self.mechanistic_model(self.mechanistic_res.x)
            self.mechanistic_QDIFF_R2 = self.R2(self.QDIFF_traj, self.mechanistic_DV)
            print("mechanistic R2 = ", self.mechanistic_QDIFF_R2)
            pre_act_inds = np.where(self.trial_pre_act)[0]
            print("preact mechanistic R2 = ", self.R2(\
                self.QDIFF_traj[pre_act_inds], self.mechanistic_DV[pre_act_inds]))

        else:
            Qs = self.flatten_trajectory(self.Q_values)
            self.QDIFF_traj = np.diff(Qs, axis = 0)[0]
            self.store_mechanistic_DV = False 
            print("fitting mechanistic model")
            self.mechanistic_res = minimize(self.mechanistic_model, init, bounds = bounds)

    def mechanistic_model(self, params):
        loss = 0
        trace = 0 
        mem__PGO = 1
        slope__PGO = 1
        thresh__PGO = 0
        prev_PGO = -1
     
        alpha = params[0]
        beta = params[1]
        eta = params[2]
        
        end_0 = 3
        end_1 = end_0 + self.PGO_N
        end_2 = end_1 + self.PGO_N
        slopes = params[end_0: end_1]
        mems = params[end_1: end_2]
        thresholds = params[end_2:]
         
        steps = self.num_steps if self.fit_steps is None else self.fit_steps
        for i in range(steps):
            inp = self.backbone_flat[i]
            PGO = self.PGO_backbone[i]
            QDIFF = self.QDIFF_traj[i]
            if PGO != prev_PGO:
                prev_PGO = PGO
                PGO_ind = np.where(self.PGO_range == PGO)[0]

                if self.use_ctx_slope:
                    slope__PGO = slopes[PGO_ind]
                if self.use_ctx_mem:
                    mem__PGO = mems[PGO_ind]
                if self.use_ctx_thresh:
                    thresh__PGO = thresholds[PGO_ind]
                    
            trace = trace * alpha * mem__PGO + inp * (1-alpha * mem__PGO)
            DV = slope__PGO * beta * trace + thresh__PGO + eta

            if self.store_mechanistic_DV:
                self.mechanistic_DV[i] = DV
            else:
                loss += (DV - QDIFF)**2

        if self.store_mechanistic_DV:
            return self.raw_to_structured(self.mechanistic_DV)
        else:
            return loss / self.num_steps         
            