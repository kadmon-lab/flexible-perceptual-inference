import numpy as np;         from sklearn.decomposition import PCA;
from enzyme import FIGPATH, TEXTWIDTH; from matplotlib.colors import Normalize
from enzyme.src.helper import save_plot; from enzyme.src.mouse_task.data_processing import data_processing; import pylab as plt; import seaborn as sns; 
import scipy.linalg as lin;  import matplotlib as mpl;   from cycler import cycler;  import pickle as p; import matplotlib.colors as cm_cols
import torch; import torch.nn as nn; from scipy.stats import entropy as ent; import matplotlib.cm as cm;  import matplotlib.pyplot as pcm
import sklearn; from sklearn.linear_model import LinearRegression, LogisticRegression; from tqdm import tqdm;         import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap; from matplotlib.ticker import MaxNLocator; import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter; from matplotlib.ticker import FixedLocator

class mouse_plotting(data_processing):
    show = plt.show
    def __init__(self, **params):
        self.__dict__.update(params)
    
    """ Plot single trials"""   
    def plot_episode(self, manager = None):
        self.height = .25
        end = int(self.end_times[self.curr_trial])
        self.xax = np.arange(0, end + 1, 1/self.temp_resolution)
        self.beeps = np.repeat(self.backbone[self.curr_trial, :end + 1], self.temp_resolution)
        self.act_color = 'g' if self.rew[self.curr_trial] == 1 else 'r'
        ends = np.cumsum(self.end_times+1)
        s = int(ends[-2]) - 1
        for d, field in self.get_net_recent(manager, s):
            self.make_trial_plot(net_data = self.to(d, 'np').T, title = field)          
            
    """ Plot visualization of network activity"""     
    def get_net_recent(self, manager, s):
        fields = ["lick_prob", "pred value", "Q diff", "net output", "LSTM_LTM", "LSTM_f_gate", "LSTM_i_gate", "LSTM_c_gate", "LSTM_o_gate"]
        gates = manager.agent.gates.detach() 
        data = [manager.agent.action_probs[1, s :],
                manager.agent.values[s :],
                # """ exp """
                # manager.agent.values[:, s :],
                # """ exp """
                manager.agent.Q_values[1, s :] - manager.agent.Q_values[0, s :],
                manager.agent.outputs[:, s :], 
                manager.agent.LTMs[:, s :],\
                gates[0, :, s:], gates[1, :, s:], gates[2, :, s:], gates[3, :, s:]]
        return zip(data, fields)

    """ Plot visualization of trial"""     
    def make_trial_plot(self, net_data, title):
        if self.show: fig = plt.figure(figsize=(10,5)); self.ax = fig.add_subplot(1, 1, 1)
        self.ax.plot(net_data, alpha = .5)  
        self.ax.fill_between(self.xax, self.beeps, alpha = .5, color = 'C0')    
        self.ax.fill_between(np.arange(self.stim_end[self.curr_trial], self.prem_end[self.curr_trial]), y1 = self.height, color ='r', alpha = .5)
        self.ax.fill_between(np.arange(self.W4L_end[self.curr_trial], self.end_times[self.curr_trial] + 2), y1 = self.height, color ='r', alpha = .5)
        self.ax.fill_between(np.arange(self.stim_end[self.curr_trial], self.W4L_end[self.curr_trial] + 1),y1 =  self.height, color ='g', alpha = .5)
        self.ax.fill_between(np.arange(0, self.stim_end[self.curr_trial] + 1), y1 = self.height, color ='deeppink',  alpha = .5)
        self.ax.scatter(self.act_time[self.curr_trial]+1, 1.1, c = self.act_color)
        self.ax.set_ylim([min(0, net_data.min()-.5), max(1.2, net_data.max()+.5)])    
        self.ax.set_title(f"PGO = {self.PGO:.3f} with {title}")
        if self.show: plt.show()
        
###############################################################################################################################################

    """ all episode plotting"""  
    def plotting(self):
        self.plot_avg_act_rew_wait()
        self.plot_wait_from_last()
        self.plot_analytical()
        self.plot_from_switch()
                
    """ plot avg reward, action time and wait from last"""
    def plot_avg_act_rew_wait(self):
        self.make_standard_plot(ys = self.episode_rews, alphs = self.PGO_ALPHA, cols = self.PGO_COLOR,
            plots = self.PGO_PLOT, leg = self.PGO_LEG, xlab = r"episode (per $\theta$)", title = "avg Reward")
        self.make_standard_plot(ys = self.episode_rew_rate, alphs = self.PGO_ALPHA, cols = self.PGO_COLOR,
            plots = self.PGO_PLOT, leg = self.PGO_LEG, xlab = r"episode (per $\theta$)", title = "avg rew rate")
        self.make_standard_plot(ys = self.episode_action_times, alphs = self.PGO_ALPHA, cols = self.PGO_COLOR,
            plots = self.PGO_PLOT, leg = self.PGO_LEG, xlab = r"episode (per $\theta$)", title = "avg Action times")
        self.make_standard_plot(ys = self.episode_wait_from_last, alphs = self.PGO_ALPHA, cols = self.PGO_COLOR,
            plots = self.PGO_PLOT, leg = self.PGO_LEG, xlab = r"episode (per $\theta$)", title = "avg wait from last")

    """ plot wait from last behavior"""
    def plot_wait_from_last(self):
        self.make_standard_plot(xax = self.wait_xax, ys = self.wait_PDF, cols = self.PGO_COLOR, alphs =  [1]*self.PGO_N, xlim = [0, 20],
            plots = self.PGO_PLOT, leg = self.PGO_LEG, xlab = "wait from last nogo PDF")   
        
        self.make_standard_plot(xax = self.wait_xax, ys = self.wait_hazard, cols = self.PGO_COLOR, alphs =  self.PGO_ALPHA,   xlim = [0,35], ylim = [0,2],
            plots = self.PGO_PLOT, leg = self.PGO_LEG, xlab = "time", title = "Hazard function of wait from last NOGO")           
            
        self.make_standard_plot(xax = self.wait_pot_xax, ys = self.wait_pot_hazard, cols = self.PGO_COLOR, alphs =  self.PGO_ALPHA,  xlim = [-5,35], ylim = [0,2],
            plots = self.PGO_PLOT, leg = self.PGO_LEG, xlab = "time", title = "Hazard function of wait from last potential")                      
            
    """ plot network behavior as a function of trial from block switch"""
    def plot_from_switch(self):   
        self.make_standard_plot(ys = self.wait_from_switch, cols = self.PGO_COLOR, alphs =  self.PGO_ALPHA,
            plots = self.PGO_PLOT, leg = self.PGO_LEG, xlab = "trial from block switch", title = "avg wait from last nogo") 
                
        self.make_standard_plot(ys = [self.PGO_corr, self.last_PGO_corr], cols = ["C0", "C1"], alphs =  [1]*2 ,plots = ['line']*2,
            leg =  ["current PGO corr", "last PGO corr"], xlab = "trial from block switch", title = "correlation difference")             
        

    def plot_analytical(self):
        self.plot_numer_v_ana()
        self.plot_wait_from_last_dist()
                
    """ plot numerical, analytical and network comparisons"""
    def plot_numer_v_ana(self):
        self.make_numer_v_ana_plot(A_ys = self.ana_acts, N_ys = self.numer_acts, agent_x_mus = self.wait_mu, agent_y_mus = self.act_mu, ylab = "action time", xlim = 20)
        if self.show: plt.show()
        self.make_numer_v_ana_plot(A_ys = self.ana_rews, N_ys = self.numer_rews,  agent_x_mus = self.wait_mu, agent_y_mus = self.rew_mu, ylab = "P(reward)", xlim = 20)
        if self.show: plt.show()
        self.make_numer_v_ana_plot(A_ys = self.ana_rates, N_ys = self.numer_rates, agent_x_mus = self.wait_mu, agent_y_mus = self.rate_mu, ylab = "reward rate", xlim = 20)  
        # add errorbars
        ax = plt.gca()
        for i in range(self.PGO_N):
            c = self.PGO_COLOR[i]
            ax.errorbar(self.wait_mu[i], self.rate_mu[i], yerr = self.rate_std[i], xerr=self.wait_std[i], capsize=5, color=c, alpha = .5)  
        if self.show: plt.show()


        # rew rates as a function of theta
        self.make_standard_plot(ys = [self.ana_max_rate, self.rate_mu, self.fixed_thresh_rate],
            alphs = [1]*3, cols = [None]*3, xticks = self.PGO_range, ylim = [.95*self.fixed_thresh_rate.min(), 1.05*self.ana_max_rate.max()],
            title = "Reward Rate comparisons", xlab = "PGO", plots = ["scatter"]*3, 
            leg = ["analytical optimal rate", "network rate", "optimal fixed threshold"], show=False)
        ax = plt.gca()
        ax.errorbar(np.arange(len(self.rate_mu)), self.rate_mu, yerr = self.rate_std, capsize=5, color='C1')
        if self.show: plt.show()
        
    
        
    def make_numer_v_ana_plot(self, A_ys, N_ys,  agent_x_mus, agent_y_mus, ylab, xlim = None, ylim = None, SVG = False):
        if self.show: fig = plt.figure(figsize=(8, 8))
        for i in range(self.PGO_N):
            c = self.PGO_COLOR[i]
            plt.plot(self.ana_xax, A_ys[i], color = c, linewidth = 5, alpha = .5)
            plt.plot(self.numer_xax, N_ys[i], "o", color = c)                     # plotting numerical sanity check
        for i in range(self.PGO_N):
            plt.plot(agent_x_mus[i], agent_y_mus[i], color = self.PGO_COLOR[i], marker = 'o', markeredgecolor = 'k', markersize = 10, alpha = 1)

        xlabel = "Mean Consecutive GOs before First Lick"
        plt.xlabel(xlabel);  plt.ylabel(ylab)
        leg = np.array([f"PGO = {pgo} {d}" for pgo in self.PGO_range for d in ["analytical", "numerical", "Network"] ])
        # plt.legend(leg, loc = "lower right"); 
        plt.title(f"analytical vs network")        
        if ylim is not None:             plt.ylim([0, ylim])
        if xlim is not None:             plt.xlim([0, xlim])        
        if SVG:                          self.save_SVG(f'analytical {ylab}')
        # plt.xlim([0,20])
        
    def plot_wait_from_last_dist(self, SVG = False):
        if self.show: fig = plt.figure(figsize=(20,10))
        for i, PGO in enumerate(self.PGO_range):
            til = min(self.xax_range, self.dist_x_lim[i])
            # plt.hlines(self.wait_PDF[i].max(), np.argmax(self.wait_PDF[i]) + self.wait_xax[i].min(), self.ana_max_thresh[i], linewidth = 2, linestyle = '--', color = self.PGO_COLOR[i])
            # plt.scatter(self.ana_max_thresh[i], self.wait_PDF[i].max() , marker = '*', s = 200, color = self.PGO_COLOR[i], label = f'optimal {PGO} threshold')
            plt.plot(self.wait_xax[i], self.wait_PDF[i].T, c = self.PGO_COLOR[i], linewidth = 3, label = f'network {PGO} PDF')
            # plt.plot(self.flow_wait_mu[i], self.wait_PDF[i].max() , marker = '*', markeredgecolor = 'k', markersize = 16, color = self.PGO_COLOR[i], label = f'Joint bayes mean waiting time')
            # plt.hlines(self.wait_PDF[i].max(), np.argmax(self.wait_PDF[i]) + self.wait_xax[i].min(), self.flow_wait_mu[i], linewidth = 2, linestyle = '--', color = self.PGO_COLOR[i])

            # plt.step(self.wait_xax[i], self.wait_PDF[i].T, where ='mid', c = self.PGO_COLOR[i], linewidth = 3, label = f'network {PGO} PDF')
        plt.xlabel("Wait from last NOGO", fontdict =  self.font_dict); plt.ylabel("PDF", fontdict = self.font_dict)
        plt.xlim([1,25])
        # self.make_color_bar()
        if SVG: 
            self.save_SVG("net dists")
        if self.show: plt.show()
        
    def plot_bayes_network_rew_rate(self,  xlim = [1,21], SVG = False):
        if self.show: fig = plt.figure(figsize=(8, 8))
        plt.axvline(self.fixed_max_thresh.mean(), color = 'k', linewidth = 1, alpha = .4)
        # plt.axhline(self.fixed_thresh_rate.mean(), color = 'k', linewidth = 1, alpha = .4)
        # plt.plot( self.factorized_wait_mu.mean(),  self.factorized_rate_mu.mean(), marker = 'x', markeredgecolor = 'k', markersize = 20, alpha = 1)
        for i in range(self.PGO_N):
            c = self.PGO_COLOR[i]
            # plt.plot( self.factorized_wait_mu[i],  self.factorized_rate_mu[i], marker = 'X', color = c, markersize = 10, alpha = 1)

            # plt.plot(self.ana_xax, self.ana_curve_mu[i], color = c, linewidth = 5, alpha = 1)
            # plt.axhline(self.ana_rates[i].max(), linestyle = '--', color = c, linewidth = 1, alpha = .5)
            plt.plot(self.ana_xax, self.ana_rates[i], color = c, linewidth = 4, alpha = .8)
            # plt.plot(self.ana_xax[self.ana_rates[i].argmax()], self.ana_rates[i].max(), '+', color = c, markeredgecolor = 'k', markersize = 10, alpha = .8)
            # plt.plot( self.factorized_wait_mu[i],  self.factorized_rate_mu[i], marker = 'x', markeredgecolor = 'k', markersize = 13, alpha = 1)

        for i in range(self.PGO_N):
            c = self.PGO_COLOR[i]
            # plt.errorbar(self.wait_mu[i], self.rate_mu[i], yerr = self.rate_std[i], capsize=5, color=c, alpha = .45)  
            # plt.errorbar(self.flow_wait_mu[i], self.flow_rate_mu[i], yerr = self.flow_rate_std[i], capsize=5, color=c, alpha = .45)  
            # plt.errorbar(self.wait_mu[i], self.rate_mu[i], xerr = self.wait_std[i], capsize=5, color=c, alpha = .8)  
            # plt.errorbar(self.flow_wait_mu[i], self.flow_rate_mu[i], xerr = self.flow_wait_std[i], capsize=5, color=c, alpha = .8)  
            plt.plot( self.wait_mu[i],  self.rate_mu[i], color = c, marker = 'o', markeredgecolor = 'k', markersize = 13, alpha = 1)
            # plt.plot( self.flow_wait_mu[i],  self.flow_rate_mu[i], color = c, marker = '*', markeredgecolor = 'k', markersize = 16, alpha = 1)
            # plt.fill_between(self.ana_xax, self.ana_curve_mu[i]-self.ana_curve_var[i], self.ana_curve_mu[i]+self.ana_curve_var[i], color =  c, alpha = .5)         

        plt.xlabel("mean consecutive GOs before first lick");  plt.ylabel("reward rate")
        plt.title("analytical vs network")        
        if xlim is not None:             plt.xlim(xlim)        
        if self.show: plt.show()
        if SVG:                          self.save_SVG(f'analytical reward rate')
        
    def plot_opt_thresh_v_ITI(self):
        fig, ax = plt.subplots(1, figsize = (10, 10))
        cmap = pcm.get_cmap('RdYlGn_r')
        N = self.ITIs_for_optim.shape[0]
        hot_map = cm.ScalarMappable(cmap=cmap, norm = cm.colors.Normalize(0, N))
        hot_map = hot_map.to_rgba(np.linspace(0, N,  N))
        for i, thresholds in enumerate(self.ITIs_for_optim):
            plt.plot(self.PGO_range, thresholds, c = hot_map[i], linewidth = 5, 
                 label = f"mean ITI = {self.ITIs_for_optim_range[i, 0]:.0f} to {self.ITIs_for_optim_range[i, 1]:.0f}")
            plt.scatter(self.PGO_range, thresholds, color = hot_map[i], s = 200)
        plt.ylabel("optimal threshold")
        plt.xlabel(r"$\theta$")
        plt.legend()
        plt.show()
        
    def plot_threshold_adaptation(self):
        if self.show: fig = plt.figure(figsize=(10, 5))
        # bot = self.rand_similarity*.99
        top = self.ana_similarity*1.01
        bot = self.ana_similarity*.8
        plt.bar(0, self.ana_similarity)
        plt.bar(1, self.net_similarity)
        plt.bar(2, self.fixed_similarity)
        plt.ylim([bot, top]); plt.title("Cosin similarity with optimal average wait from last NOGO")
        plt.legend(["analytical", "network", "optimal fixed threshold"])
        if self.show: plt.show() 
        if self.show: fig = plt.figure(figsize=(10, 5))
        optimal = self.ana_max_rate.mean()
        plt.bar(0, optimal/optimal)
        plt.bar(1, self.rate_mu.mean()/optimal)
        plt.bar(2, self.fixed_thresh_rate.mean()/optimal)
        plt.bar(3, self.rand_max_rate.mean()/optimal) 
        plt.title("Percent optimal reward rate averaged over PGO")
        plt.legend(["analytical", "network", "optimal fixed threshold", "optimal random acting"])
        if self.show: plt.show()
        
        self.make_standard_plot(ys = [self.wait_PDF_avg, self.ana_max_thresh, [self.fixed_thresh]*self.PGO_N], alphs = [1]*3, cols = [None]*3, 
            plots = ["scatter"]*3, xlab = "PGO", xticks = self.PGO_range, leg = ["Wait from last nogo", "optimal flexible threshold", "optimal fixed threshold"])
       
    
    def plot_theta_MSE_fast_v_stable(self):
        fig, ax = plt.subplots(1, figsize = (15, 15))
        plt.scatter(self.MF_mse[:,:, 0].mean(1), self.MF_mse[:,:,1].mean(1), marker = 'x', s = (1+np.arange(self.MF_num))*10, alpha = .35)
        plt.scatter(self.net_mse[:, 0].mean(), self.net_mse[:, 1].mean(), marker = 'o',  s = 200)
        plt.scatter(self.bayes_mse[:, 0].mean(), self.bayes_mse[:, 1].mean(), marker ='*',  s = 200);        
        plt.xlabel("first trial from switch"); plt.ylabel(f"{self.num_trials} trial"); plt.title("MSE error"); plt.legend(["MF", "network", "bayes"]); 
        if self.show: plt.show()

        fig, ax = plt.subplots(1, figsize = (15, 15))
        for i, label in enumerate([1,  self.num_trials]):
            plt.plot(self.PGO_range, self.net_ratio[:, i], label = f"network ratio {label} trial"); 
            plt.plot(self.PGO_range, self.bayes_ratio[:, i], linestyle = self.bayes_style, label = f"bayes ratio {label} trial"); plt.ylabel("Estimate MSE / Model Free MSE")
        plt.axhline(1); plt.xlabel("theta"); plt.legend(); plt.title("MSE ratio for estimate over (optimal) MF window"); 
        if self.show: plt.show()
        
    def plot_PGO_vs_wait_fancy(self, SVG = None, linestyle = '-', marker = 'o', first_trial = False):
        fig, ax = plt.subplots(1, figsize = (5, 5))
        waiting = np.zeros(self.PGO_N)
        for p_i, p in enumerate(self.PGO_range):
            inds = self.where(self.PGOs, p)
            if first_trial:
                inds = np.intersect1d(inds, self.where(self.trial, 0))
            waiting[p_i] = self.wait_from_last[inds].mean()
        ax.plot(self.PGO_range, waiting, linestyle = linestyle, c = 'k', alpha = .5)
        for p_i, p in enumerate(self.PGO_range):
            ax.scatter(p, waiting[p_i], color = self.PGO_COLOR[p_i], marker = marker, s = 150)
        ax.set_xlabel("PGO"); ax.set_ylabel("wait from last nogo"); ax.set_title("Waiting vs PGO")
        ax.set_ylim([3, 12])
        if SVG is not None:
            self.save_SVG(SVG)
        plt.show()
        
    """ plotting for mechanistic models """ 
    def plot_fancy_wait_dist(self, SVG  = None, ylim = None): 
        fig, ax = plt.subplots(figsize = (3,2))
        for p_i, p in enumerate(self.PGO_range):
            c = self.PGO_COLOR[p_i]
            inds = self.where(self.PGOs, p)
            waits = self.wait_from_last[inds]
            n, x = np.histogram(waits, bins=np.arange(0, 30, 1), density=True)    
            bin_centers = 0.5*(x[1:]+x[:-1])
            ax.plot(bin_centers, n, color=c, alpha=1, linewidth = 3)
            ax.fill_between(bin_centers, 0, n, color=c, alpha=.3)
            ax.set_xlabel(r'waiting time ($\tau_w$)')
            ax.set_xticks([1,14], labels = [1,14])
            ax.set_ylabel(r'$P(\tau_w$)')
            ax.set_xlim([1, 14])
            ax.set_ylim([0, ylim])
            ax.set_yticks([])
        if SVG is not None:
            self.save_SVG(SVG)
                
###############################################################################################################################################
    """ representation plots """

    """ plot prediction of nth last input or average input """ 
    def plot_past_prediction(self, window = 40, predict = 'exact'):
        Ms = np.empty((window), dtype = object)
        self.past_pred = np.zeros((self.PGO_N, window))
        self.get_indices(cross_validation = None, flatten = True)
        avg_inp = self.smooth(self.input_flat[1], window//2)[:-window//2 + 1]
        entropies = np.zeros(self.PGO_N)
        
        for w in range(window):
            IV = self.pred_from[:, window+1:].T
            
            if predict == 'exact':
                DV = self.input_flat[1, window - w : -(w+1)].T
                Ms[w] = LogisticRegression().fit(IV, DV)
            if predict == 'avg':
                DV = avg_inp[window - w : -(w+1)].T
                Ms[w] = LinearRegression().fit(IV, DV)
                
        for p_i, PGO in enumerate(self.PGO_range):
            inds = np.where(self.PGO_flat == PGO)[0]
            prob = self.input_flat[1, inds].mean()
            entropies[p_i] = - prob * np.log2(prob) - (1-prob)*np.log2(1-prob)
        
            for w in range(window):
                IV = self.pred_from[:, inds][:, window+1:].T
                if predict == 'exact':
                    DV = self.input_flat[1, inds][window - w : -(w+1)].T
                if predict == 'avg':
                    DV = avg_inp[inds][window - w : -(w+1)].T
                self.past_pred[p_i, w] = self.R2(Ms[w].predict(IV), DV)
        
        # plot past pred for each context 
        fig, ax = plt.subplots(figsize = (10, 10))
        for p in range(self.PGO_N):
            e = entropies[p]
            w =  3*(e**2)
            plt.plot(self.past_pred[p, :].T, c = self.PGO_COLOR[p], label = f"entropy = {e:.3f}", linewidth = w)
            plt.plot(self.past_pred[p, :].T, c = self.PGO_COLOR[p], linewidth = w + 1, alpha = .5)
        plt.ylim([0, 1]); 
        if predict == 'exact':
            plt.title('GO/NOGO Explained Variance')
        if predict == 'avg':
            plt.title(f'Explained Variance of average of last {window//2} inputs')
        plt.xlabel("time from present"); 
        plt.legend()
        plt.show()
        
    """ plot R^2 of recent stimuli's predictability of estimate """
    def plot_R2_recent_past(self, window = 10, estimates = 'network'):
        self.get_indices(til_action = True, flatten = True)        
        fig, ax = plt.subplots(figsize = (15, 5))
        self.recent_stim_predictive_power = np.zeros((2, window))
        if estimates == 'network':
            NOGO_theta_est = self.net_theta_flat[self.NOGO_inds]
            GO_theta_est = self.net_theta_flat[self.GO_inds]
        if estimates == 'theory':
            NOGO_theta_est = self.flow_theta_flat[self.NOGO_inds]
            GO_theta_est = self.flow_theta_flat[self.GO_inds]
        
        for w in range(window):
            N_inds = np.clip(self.NOGO_inds - w, a_min = 0, a_max = None)
            G_inds = np.clip(self.GO_inds - w, a_min = 0, a_max = None)
            NOGO_stim = self.input_flat[1, None, N_inds]
            GO_stim = self.input_flat[1, None, G_inds]
            # NOGO_stim = self.consec_flat[None, N_inds].T
            # GO_stim = self.consec_flat[None, G_inds].T
                          
            self.recent_stim_predictive_power[0, -(w+1)] = self.R2(NOGO_stim, NOGO_theta_est)
            self.recent_stim_predictive_power[1, -(w+1)] = self.R2(GO_stim, GO_theta_est)

        xax = np.arange(0, window-1)
        ax.plot(xax, self.recent_stim_predictive_power[0, :-1], c = 'r', linestyle = '--')
        ax.plot(xax, self.recent_stim_predictive_power[1, :-1], c = 'r', linestyle  = '--', alpha = .5)
        ax.legend(["NOGO in current step", "GO in current step"])
        ax.set_xticks(ticks = xax, labels =np.arange(-window+1, 0) )
        ax.set_title("Theta explained variance by recent stimuli")
        ax.set_xlabel("Time before present")
        ax.set_ylabel("R^2")
        plt.show()

    def plot_dropout_on_regression(self, n = 20):
        self.dropout_corr = np.zeros((2,n))
        self.d_range = np.linspace(0,1,n)       
        s_regs, theta_regs = [np.empty(n, dtype = object) for _ in range(2)]
    
        self.get_indices(cross_validation = 'training', flatten = True)
        for i, d in enumerate(tqdm(self.d_range)):
            mask = np.random.rand(self.pred_from.shape[0], self.pred_from.shape[1]) > d
            IV = (self.pred_from*mask).T
            
            s_regs[i] = LogisticRegression().fit(IV, self.PSAFE_flat)
            theta_regs[i] = LinearRegression().fit(IV, self.PGO_flat)
     
        self.get_indices(cross_validation = 'testing', flatten = True)
        for i, d in enumerate(tqdm(self.d_range)):
            mask = np.random.rand(self.pred_from.shape[0], self.pred_from.shape[1]) > d
            IV = (self.pred_from*mask).T
            
            self.dropout_corr[0, i] = self.correlate(self.PSAFE_flat, s_regs[i].predict_proba(IV)[:, 1])
            self.dropout_corr[1, i] = self.correlate(self.PGO_flat, theta_regs[i].predict(IV))
            
        plt.plot(self.d_range, self.dropout_corr.T); plt.legend(["state correlation", "theta correlation"]); 
        plt.ylim([0,1]); plt.xlabel("dropout"); plt.show()
        
    def plot_consec_GO_Q_val_correlation(self, N_consec = 12, pred = "Q"):
        pre_act = np.where(self.pre_act)[0]
        post_act = np.where(self.post_act)[0]
        consec_corrs = np.zeros((2, self.PGO_N, N_consec))
        consec_corrs_agents = np.zeros((2, 2, self.PGO_N, N_consec))
        xax = np.arange(N_consec) + 1
        for (inds_i, alpha, inds_, title) in zip([0,1], [1, .5], [pre_act, post_act], ("PRE ACTION", "POST ACTION")):
            for p_i, PGO in enumerate(self.PGO_range):
                inds = np.intersect1d(inds_, np.where(self.PGO_flat == PGO))
                net_lick = self.Q_flat[1, inds]
                bayes_lick = self.log_to_prob(self.flow_belief_flat[inds])                
             
                for i in range(N_consec):
                    consec_GO = np.clip(self.consec_flat[inds, None], a_min = -1, a_max = i+1)
           
                    L = LinearRegression().fit(consec_GO, net_lick)
                    consec_corrs_agents[0, inds_i, p_i, i] = self.R2(L.predict(consec_GO), net_lick)
                    L = LinearRegression().fit(consec_GO, bayes_lick)
                    consec_corrs_agents[1, inds_i, p_i, i] = self.R2(L.predict(consec_GO), bayes_lick)                    
                        
                    if pred == "Q":                    
                        consec_corrs[inds_i, p_i, i] = consec_corrs_agents[0, inds_i, p_i, i]
                    if pred == "bayes":
                        lick_probs = bayes_lick
                        consec_corrs[inds_i, p_i, i] = consec_corrs_agents[1, inds_i, p_i, i]
         

        fig, ax = plt.subplots(1,figsize = (4, 5))
        for (inds_i, alpha, inds_, title) in zip([0, 1], [1, .5], [pre_act, post_act], ("PRE ACTION", "POST ACTION")):
          for p_i, PGO in enumerate(self.PGO_range):                    
                ax.plot(xax , consec_corrs[inds_i, p_i], c = self.PGO_COLOR[p_i], linestyle = '-', label = f"{title}", alpha = alpha, linewidth = 4);
                ax.plot(xax , consec_corrs[inds_i, p_i], c = 'C0', linestyle = '-', label = f"{title}", alpha = alpha, linewidth = 3);
        ax.set_ylim([.2,1]); 
        # ax.legend();
        ax.set_xlabel("max consec GOs"); ax.set_ylabel("R2(consec GO, Q(lick))")
        plt.show()
        
        fig, ax = plt.subplots(2, 1, figsize = (5, 5), tight_layout = True)
        for inds_i in [0,1]:
            for a_i, style in zip([0, 1], ['-o','--o']):
                Argmax = np.argmax(consec_corrs_agents[a_i, inds_i], axis = -1) + 1
                Max = np.max(consec_corrs_agents[a_i, inds_i], axis = -1)
                for p_i, PGO in enumerate(self.PGO_range):                    
                    ax[inds_i].scatter(PGO, Argmax[p_i], color = self.PGO_COLOR[p_i], s = 200, alpha = Max[p_i])
                    if p_i + 2 <= self.PGO_N:
                        ax[inds_i].plot(self.PGO_range[p_i:p_i+2], Argmax[p_i:p_i+2], style, alpha = Max[p_i])
                    
            ax[inds_i].set_ylim([0, N_consec+1])
            ax[inds_i].set_xlabel("PGO")
            ax[inds_i].set_ylabel("N")
        ax[0].set_title(f"Counting ceiling most predictive of\n Q value / bayesian State estimate\n PRE ACTION")
        ax[1].set_title(f"POST ACTION")
        plt.show()
        
        
    """ plot predictability of context at onset, action and offset """
    def plot_ctx_predictability_from_PCs(self):
        # self.get_indices(cross_validation=None, flatten = True, eps_init = 0)
        self.get_full_PCA(x = self.output_flat.T)
        Y = self.PGO_flat
        self.ctx_predictability = np.zeros((3, self.hid_dim))
        start_inds = self.cum_start_times
        act_inds = self.cum_act_times
        end_inds = self.cum_end_times
        for n in range(self.hid_dim):
            self.ctx_predictability[0, n] = self.R2(self.full_proj[:n+1, start_inds].T, Y[start_inds])
            self.ctx_predictability[1, n] = self.R2(self.full_proj[:n+1, act_inds].T,   Y[act_inds])
            self.ctx_predictability[2, n] = self.R2(self.full_proj[:n+1, end_inds].T,   Y[end_inds])
        plt.plot(self.ctx_predictability.T)     
        plt.title("R^2 of theta prediction from principle components")
        plt.legend(["trial onset", "action time", "trial offset"])
        plt.xlabel("Principle components used in prediction")
        plt.ylim([0, .9])
        plt.show()
        
    def plot_net_var_explained_by_latent(self, IV = 'ground_truth', name = 'trained', act_num = 15):
        # self.get_indices(cross_validation=None, flatten = True, eps_init = 0, til_action = True)
        safe_inds = self.where(self.PSAFE_flat)
        theta_R2_a, state_R2_a = [np.zeros(act_num) for _ in range(2)]

        if IV == 'estimates':
            m = LogisticRegression().fit(self.pred_from.T, self.input_flat[1])
            net_GO_flat = m.predict(self.pred_from.T)        
        for inds, title in zip([self.cum_act_times, self.ACTION_inds, safe_inds], ["at action", "pre-action", "in safe state"]):
            if IV == 'ground_truth':
                input_R2 = self.R2(self.input_flat[1, inds], self.pred_from.T[inds])
                theta_R2 = self.R2(self.PGO_flat[inds], self.pred_from.T[inds])
                state_R2 = self.R2(self.PSAFE_flat[inds], self.pred_from.T[inds])
                for a in range(act_num):
                    theta_R2_a[a] = self.R2(self.PGO_flat[self.cum_acts-a], self.pred_from.T[ self.cum_acts-a])
                    state_R2_a[a] = self.R2(self.PSAFE_flat[self.cum_acts-a], self.pred_from.T[ self.cum_acts-a])
        

            if IV == 'estimates':
                input_R2 = self.R2(net_GO_flat[inds], self.pred_from.T[inds])
                theta_R2 = self.R2(self.net_theta_flat[inds], self.pred_from.T[inds])
                state_R2 = self.R2(self.net_belief_flat[inds], self.pred_from.T[inds])
                for a in range(act_num):
                    theta_R2_a[a] = self.R2(self.net_theta_flat[self.cum_acts-a], self.pred_from.T[ self.cum_acts-a])
                    state_R2_a[a] = self.R2(self.net_belief_flat[self.cum_acts-a], self.pred_from.T[ self.cum_acts-a])

            if IV == 'ground_truth':
                variables = ['Theta', 'State', 'Input']
            if IV == 'estimates':
                variables = ['decoded Theta', 'decoded State', 'decoded Input']
            R2_values = [theta_R2, state_R2, input_R2]
            
            plt.figure(figsize=(4, 3))
            plt.bar(variables, R2_values, color=['C0', 'C1', 'C2'])
            plt.xlabel('Variables')
            plt.ylabel('R2 Values')
            plt.title(f"explained variance of {name} neural activity {title}")
            plt.ylim([0,1])
            plt.show()
        

        variables = [f'action time - {a}' for a in range(act_num)]
        plt.subplots(1, figsize = (5, 5))
        plt.plot(theta_R2_a)
        plt.plot(state_R2_a)
        plt.xticks(np.arange(act_num), -np.arange(act_num))
        plt.title(f"explained variance of {name} neural activity")
        plt.xlabel("time before action")
        plt.legend([f"variance explained by {IV} theta", f"variance explained by {IV} state"])
        plt.ylabel("R2")
        plt.ylim([0,1])
        plt.show()
            
    """ plot state estimate vs consec GOs for network and theory """
    def plot_N_consec_vs_est(self, consecs = 20, title = "", SVG = None, get = True, var = "QDIFF"):
        self.get_N_consec_vs_est(consecs, variable = var, get = get)

        fig, ax = plt.subplots(1,2, figsize = (15,10), tight_layout = True)
        cmap = pcm.get_cmap('RdYlGn_r')
        GOS_cmap = cm.ScalarMappable(cmap=cmap, norm = cm.colors.Normalize(0, 1))
        GOs_cmap= GOS_cmap.to_rgba(np.linspace(0, 1, consecs))        

        for p_i, p in enumerate(self.PGO_range):
            ax[0].plot(self.N_consec_xax, self.theory_safe_est_N_per_PGO[p_i], c = self.PGO_COLOR[p_i], linewidth = 6, alpha = .5)
            ax[0].plot(self.N_consec_xax, self.net_safe_est_N_per_PGO[p_i], c = self.PGO_COLOR[p_i], linewidth = 6, alpha = .5)
        ax[0].plot(self.N_consec_xax, self.theory_safe_est_N_per_PGO.T, c= 'k',linewidth = 2, marker = '*',  markersize = 10)
        ax[0].plot(self.N_consec_xax, self.net_safe_est_N_per_PGO.T, c= 'C0', linewidth = 2, marker = 'o')
        ax[0].plot(self.N_consec_xax, self.theory_safe_est_N_per_PGO[-1], c = 'k', label = "Theory", linewidth = 2, marker = '*', markersize = 12)
        ax[0].plot(self.N_consec_xax, self.net_safe_est_N_per_PGO[-1], c = 'C0', label = "Network", linewidth = 2, marker = 'o')
        ax[1].plot(self.N_consec_xax, self.theory_safe_est_N, c= 'k',linewidth = 2, marker = '*',  markersize = 10)
        ax[1].plot(self.N_consec_xax, self.net_safe_est_N, c= 'C0', linewidth = 2, marker = 'o')

        for axis, t in zip([ax[0], ax[1]], ["per theta", "all data"]):
            axis.set_xlabel("N consecutive GO cues", fontsize = 10)
            axis.set_ylabel("state estimate", fontsize = 10)
            axis.set_title(t, fontsize = 10)
            axis.legend(fontsize = 10)
            # axis.set_ylim([0, 1])
        fig.suptitle(title, fontsize = 20)
        
        if SVG is not None:
            self.save_SVG(SVG)
        plt.show()
            
    def plot_N_consec_vs_est_V2(self, consecs = 20, title = "", SVG = None, get = True, var = "QDIFF", show = "net", N = 2, backwards = 0):
        self.get_N_consec_vs_est(consecs, variable = var, get = get, backwards = backwards)
        min_r = 1
        max_r = 0
        for i in range(consecs - 1): 
            for p_i, p in enumerate(self.PGO_range):
                if self.samples_N_per_PGO[p_i, i+1] > N:
                    R2 = self.R2_N_per_PGO[p_i, i]
                    min_r = min(min_r, R2)
                    max_r = max(max_r, R2)
                    a = np.sqrt(R2)
                    xax = np.arange(i, i+2)+1 - backwards
                    if show == "theory":
                        y = self.theory_safe_est_N_per_PGO.T[i : i+2,p_i]
                    if show == "net":
                        y = self.net_safe_est_N_per_PGO.T[i : i+2,p_i]
                    # plt.plot(xax, y, linewidth = 2, c = 'r', alpha = .5)
                    # plt.plot(xax, y, alpha = a, linewidth = 2, c= 'k')
                    plt.plot(xax, y, alpha = 1, linewidth = 2, c= 'k')
                    plt.scatter(i+1 - backwards, y[0], color = self.PGO_COLOR[p_i]) 
        # b = mpatches.Patch(color='k', label=f'high R^2, (max = {max_r:.2f})')
        # r = mpatches.Patch(color=(1, 0, 0), label=f'low R^2 (min = {min_r:.2f})')
        # plt.legend(handles = [b, r], loc = 'lower right', fontsize = 10)
        if var == "DV" or var == "Q" or var == "QDIFF":
            plt.axhline(0, linestyle= '--')
        N_title = "" if N == 0 else f"for data where n > {N}"
        plt.xlabel("consecutive GOs")
        plt.title(f" {title} {show} \n {var} vs consec GOs \n {N_title}")
        plt.ylabel(f"{var}")
        if SVG is not None:
            self.save_SVG(SVG)
        plt.show()
        
    def get_N_consec_vs_est(self, consecs = 20, variable = "QDIFF", get = True, backwards = 0):
        if get:
            self.get_indices(eps_init = 0, flatten = True, cross_validation = "testing", prep_mus = True, run_traj = True, needs_acted = True, til_action = True)
        self.net_safe_est_N_per_PGO, self.theory_safe_est_N_per_PGO, self.net_safe_est_SE, self.theory_safe_est_SE, self.R2_N_per_PGO, self.samples_N_per_PGO = [np.zeros((self.PGO_N, consecs)) for _ in range(6)]
        self.net_safe_est_N, self.theory_safe_est_N, = [np.zeros(consecs) for _ in range(2)]
        theory_prob = self.flow_dist_from_opt_flat
        if variable == "state":
            theory_prob = self.log_to_prob(self.flow_belief_flat)
            net_prob = self.log_to_prob(self.net_belief_flat)
        if variable == "DV":
            net_prob = self.net_dist_from_opt_flat
        if variable == "Q":
            net_prob = self.Q_flat[1]
        if variable == "QDIFF":
            net_prob = self.QDIFF_flat
        if variable == "act":
            net_prob = self.lick_prob_flat
        if variable == "log_act":
            net_prob = np.log(self.lick_prob_flat)
            
        
        self.N_consec_xax = np.arange(consecs) + 1 - backwards
        for n_i, n in enumerate(range(consecs)):
            if (n + 1 - backwards) != 0:
                inds = np.where((self.consec_flat == (n + 1 -backwards)))[0]   
            inds = np.intersect1d(inds,  self.ACTION_inds)
           
            self.theory_safe_est_N[n_i] = theory_prob[inds].mean()
            self.net_safe_est_N[n_i] = net_prob[inds].mean()
            for p_i, p in enumerate(self.PGO_range):
                p_inds_ = np.where(self.PGO_flat == p)[0]
                p_inds = np.intersect1d(inds, p_inds_)
                
                if len(p_inds) > 2:
                    theory_est = theory_prob[p_inds]
                    net_est = net_prob[p_inds]
                    self.samples_N_per_PGO[p_i, n_i] = len(p_inds)
                    self.theory_safe_est_N_per_PGO[p_i, n_i] = theory_est.mean() 
                    self.net_safe_est_N_per_PGO[p_i, n_i] = net_est.mean() 
                    self.theory_safe_est_SE[p_i, n_i] = theory_est.std()
                    self.net_safe_est_SE[p_i, n_i] = net_est.std()
                    self.R2_N_per_PGO[p_i, n_i] = self.R2(theory_est, net_est)
                    
    """ comparison of theory and network estimates """ 
    def plot_net_vs_theory_estimate_scatter(self,  xlim = None, ylim = None, consec_GOs = 15, log_scale = True,  SVG = None, estimate = 'state', cmap = 'GO', at_action = False, at_safe = False, eps = 1e-8, get_inds = False, means = False, lines = False):
        fig, ax = plt.subplots(figsize = (10,10))
        if get_inds:
            self.get_indices(flatten = True, cross_validation = 'testing')
        if estimate == 'state':
            net_data =  self.net_belief_flat
            bayes_data = self.flow_belief_flat
            # bayes_data = self.factorized_belief_flat
            # opt_data = -np.log(np.clip(1-self.opt_belief, a_min = eps, a_max = 1 - eps))
            if not log_scale:
                net_data = self.log_to_prob(net_data)
                bayes_data = self.log_to_prob(bayes_data)
                # opt_data = self.log_to_prob(opt_data)
        if estimate == 'theta':
            bayes_data = self.flow_theta_flat
            net_data = self.net_theta_flat 
                
        if cmap == 'GO':
            cmap = pcm.get_cmap('RdYlGn_r')
            GOS_cmap = cm.ScalarMappable(cmap=cmap, norm = cm.colors.Normalize(0, 1))
            cmap = GOS_cmap.to_rgba(np.linspace(0, 1, consec_GOs))
            for n_i, n in enumerate(np.arange(consec_GOs)):
                inds  = np.where(self.consec_flat == n + 1)[0]                                
                self.scatterplot_data(inds, bayes_data, net_data, n_i, means, estimate, at_action, at_safe, cmap = cmap)               
                
        if cmap == 'PGO':
            cmap = self.PGO_COLOR
            for p_i, PGO in enumerate(self.PGO_range):                                
                inds = np.where(self.PGO_flat == PGO)[0]      
                self.scatterplot_data(inds, bayes_data, net_data, p_i, means, estimate, at_action, at_safe, cmap = cmap)               

        # postprocessing
        if log_scale:            
            plt.xlabel(f"theory {estimate} (log scale)", fontsize = 10)
            plt.ylabel(f"network {estimate} (log scale)", fontsize = 10)
        else:            
            plt.ylabel(f"network {estimate}",  fontsize = 10)
            plt.xlabel(f"theory {estimate}",  fontsize = 10)
        if xlim is not None:
            plt.xlim([xlim[0], xlim[1]])
        if ylim is not None:
            plt.ylim([ylim[0], ylim[1]])

        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=self.PGO_N, fontsize = 10, prop={'size': 8})                
        plt.plot(np.linspace(0, xlim, 10), np.linspace(0, ylim, 10), 'k', alpha = .5);
        if lines:
            plt.hlines(1, xmin = 0, xmax = 1, color = 'k', alpha = .5)
            plt.vlines(1, ymin = 0, ymax = 1, color = 'k', alpha = .5)
        if SVG is not None: self.save_SVG(SVG)
        plt.show()

    def scatterplot_data(self, inds, bayes_data, net_data, i, means, estimate, at_action, at_safe, cmap):
            if at_safe:
                inds = np.intersect1d(inds, np.where(self.PSAFE_flat)[0])                              
            if at_action: 
                inds = np.intersect1d(inds, self.cum_acts)   
            bayes = bayes_data[inds]
            net = net_data[inds]
            if means: 
                plt.scatter(bayes.mean(), net.mean(), color ='k',s = 20)
                plt.scatter(bayes.mean(), net.mean(), color = cmap[i], label = f"R^2:\n{self.R2(net, bayes) :.2f}", s = 15)
                plt.errorbar(bayes.mean(), net.mean(), xerr = bayes.std(), yerr = net.std(), color = cmap[i], alpha = .3)
            else: 
                plt.scatter(bayes, net, color = cmap[i], alpha = .5, s = 5)
                plt.scatter(bayes[0], net[0], color = cmap[i], label = f"R^2:\n{self.R2(net, bayes) :.2f}", s = 15)

    """ plots the log probability of action given the recent stim """
    def plot_action_triggering_stim(self, window = 40):
        probs = np.zeros((2, self.PGO_N, window)) 
        self.get_indices(cross_validation = None, flatten = True, til_action = True)
        for p_i, PGO in enumerate(self.PGO_range):
            for w in range(window):            
                NOGO_inds = np.clip(np.intersect1d(self.NOGO_inds, np.where(self.PGO_flat == PGO)[0]) + w + 1, a_min = 0, a_max = self.full_ind_num-1)
                GO_inds = np.clip(np.intersect1d(self.GO_inds, np.where(self.PGO_flat == PGO)[0]) + w + 1, a_min = 0, a_max = self.full_ind_num-1)
                probs[0, p_i, -(w+1)] = self.lick_prob_flat[NOGO_inds].mean()
                probs[1, p_i, -(w+1)] = self.lick_prob_flat[GO_inds].mean()
            plt.plot(np.log(probs[1, p_i]/probs[0, p_i]), c = self.PGO_COLOR[p_i], linewidth = 4)
            plt.xlabel("Time before action")
            plt.xticks(np.arange(0, window + 5, 5), labels =np.arange(-window, 5, 5) )
        plt.title("Log likelihood ratio: log [ P(action | stim(t) = GO) / P(action | stim(t) = NOGO) ] ")
        plt.ylim([-4, 8])
        plt.show()
        
    def Q_vs_distance_from_threshold(self,  get = False, title = None, consec = 10, SVG = None):
        if get:
            self.get_indices(col = 'block', From = 0,  Til = 50, stim_above = None, stim_below = None, eps_init = 0, til_action = True, needs_acted = True, cross_validation = None,
                             planted = False, plant_PGO = None, plant_ID = None, prev_PGO = None, curr_PGO = None, rew = None, align_on = 'action', flatten = True)
        
        GO_inds = self.GO_inds 
        ACT_inds = self.cum_acts
        x = self.flow_dist_from_opt_flat
#        Y = self.Q_flat[1]
        Y = self.QDIFF_flat

        net_targets = np.arange(-7, 3, .1)           
        indices = np.argmin(np.abs(Y[:, None] - net_targets), axis=1)
        net_round = net_targets[indices]       
        net_uniques = np.unique(net_round[GO_inds])
        GO_net_round = net_round[GO_inds]
        ACT_net_round = net_round[ACT_inds]
      
        bayes_targs = np.arange(-1, .5, .01)
        indices = np.argmin(np.abs(x[:, None] - bayes_targs), axis=1)
        bayes_round = bayes_targs[indices]        
        bayes_uniques = np.unique(bayes_round[GO_inds]) 
        GO_bayes_round = bayes_round[GO_inds]
        ACT_bayes_round = bayes_round[ACT_inds]
        
        ACT_PGO_flat = self.PGO_flat[ACT_inds]
        GO_PGO_flat = self.PGO_flat[GO_inds]

        self.GO_hist = np.zeros((consec, len(bayes_uniques), len(net_uniques))) 
        self.ACT_hist = np.zeros((len(bayes_uniques), len(net_uniques))) 
        self.X_hist = np.zeros((self.PGO_N, len(bayes_uniques)))
        self.Y_hist = np.zeros((self.PGO_N, len(net_uniques)))
        for p_i, p in enumerate(self.PGO_range):
            GO_p_inds = GO_PGO_flat == p
            ACT_p_inds = ACT_PGO_flat == p
            for b_i, b in enumerate(bayes_uniques):
                # self.X_hist[p_i, b_i] = len(np.where((GO_bayes_round == b)*GO_p_inds*(GO_net_round > 0))[0])
                self.X_hist[p_i, b_i] = len(np.where((ACT_bayes_round == b)*ACT_p_inds)[0])
            self.X_hist[p_i,:] = gaussian_filter(self.X_hist[p_i,:], sigma = 2)
            self.X_hist[p_i,:] = self.X_hist[p_i,:]/self.X_hist[p_i,:].sum()

            for n_i, n in enumerate(net_uniques):
                # self.Y_hist[p_i, n_i] = len(np.where((GO_net_round == n)*GO_p_inds*(GO_bayes_round > 0))[0])
                self.Y_hist[p_i, n_i] = len(np.where((ACT_net_round == n)*ACT_p_inds)[0])
            self.Y_hist[p_i,:] = gaussian_filter(self.Y_hist[p_i,:], sigma = 2)
            self.Y_hist[p_i,:] = self.Y_hist[p_i,:]/self.Y_hist[p_i,:].sum()
 
        for c in range(consec):
            c_bool = self.consec_flat[GO_inds] == (c+1)
            for n_i, n in enumerate(net_uniques):
                for b_i, b in enumerate(bayes_uniques):
                    self.ACT_hist[b_i, n_i] = len(np.where((ACT_net_round == n)*(ACT_bayes_round == b))[0])
                    self.GO_hist[c, b_i, n_i] = len(np.where((GO_net_round == n)*(GO_bayes_round == b)*c_bool)[0])
                               
        self.GO_hist = self.GO_hist/self.GO_hist.sum()
        self.ACT_hist = self.ACT_hist/self.ACT_hist.sum()

        self.heatmap_Q_distance_from_thresh(x, Y, GO_inds, ACT_inds, title, consec, bayes_uniques, net_uniques)
        if SVG is not None:
            self.save_SVG(SVG + " decision variables")
        plt.show()



    def heatmap_Q_distance_from_thresh(self, x, Y, GO_inds, ACT_inds, title, consec, bayes_uniques, net_uniques):
        green = [(1, 0, 0, 0), (0, 1, 0)]
        red = [(1, 0, 0, 0), (1, 0, 0, .6)]
        n_bins = 100
    
        vmin_GO = .00006
        vmin_ACT = .001#.002
    
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 3, width_ratios=[4, 1, 1], height_ratios=[1, 4, 0.1], wspace=0.1, hspace=0.1)
    
        ax_heatmap = plt.subplot(gs[1, 0])
        ax_x_hist = plt.subplot(gs[0, 0], sharex=ax_heatmap)
        ax_y_hist = plt.subplot(gs[1, 1], sharey=ax_heatmap)
    
        for vmax, data, color, cmap_name in zip([vmin_GO, vmin_ACT], [self.GO_hist, self.ACT_hist], [green, red], ["black_to_green", "black_to_red"]):
            if cmap_name == "black_to_green":
                data = gaussian_filter(data, sigma=3)
                for c in range(consec):
                    color = plt.cm.RdYlGn(1 - .8 * (c / consec) ** 3)
                    color = [(color[0], color[1], color[2], .01), (color[0], color[1], color[2], 1)]
                    cm = LinearSegmentedColormap.from_list(cmap_name, color, N=n_bins)
                    sns.heatmap(data[c].T, cmap=cm, vmax=vmax, cbar=False, ax=ax_heatmap)
            else:
                data = gaussian_filter(data, sigma=2)
                cm = LinearSegmentedColormap.from_list(cmap_name, color, N=n_bins)
                sns.heatmap(data.T, cmap=cm, vmax=vmax, cbar=False, ax=ax_heatmap)
    
        ax_heatmap.invert_yaxis()        
        ax_heatmap.axhline(np.where(np.isclose(net_uniques,0))[0]-2, c='r', linestyle='--', alpha=.5)
        ax_heatmap.axvline(np.where(np.isclose(bayes_uniques,0))[0]-2, c='r', linestyle='--', alpha=.5)
        ax_heatmap.set_xlabel("Bayesian distance from threshold")
        ax_heatmap.set_ylabel("Network Q value")
    
        ax_x_hist.set_title("Decision variables")
    
        F = LinearRegression().fit(Y[GO_inds][:, None], x[GO_inds][:, None])
        R2 = F.score(Y[GO_inds][:, None], x[GO_inds][:, None])
        rescale = F.predict(net_uniques[:, None])
        x_scaled = (rescale - bayes_uniques.min()) / (bayes_uniques.max() - bayes_uniques.min()) * (len(bayes_uniques) - 1)
        y_scaled =  (net_uniques - net_uniques.min()) / (net_uniques.max() - net_uniques.min()) * (len(net_uniques) - 1)
    
        ax_heatmap.plot(x_scaled, y_scaled, color='k', alpha=.8)
            
        for p_i, p in enumerate(self.PGO_range):
            # For x histogram
            inds_x = np.where(self.X_hist[p_i] > 0.002)[0]
            xax_x = np.arange(len(bayes_uniques))[inds_x]
            yax_x = self.X_hist[p_i, inds_x]
            ax_x_hist.fill_between(xax_x, 0, yax_x, color=self.PGO_COLOR[p_i], alpha=0.6)
            ax_x_hist.plot(xax_x, yax_x, color=self.PGO_COLOR[p_i], alpha=.8,  linewidth = 1)  # Adding normal line
            
            # For y histogram
            inds_y = np.where(self.Y_hist[p_i] > 0.002)[0]
            xax_y = np.arange(len(net_uniques))[inds_y]
            yax_y = self.Y_hist[p_i, inds_y]
            ax_y_hist.fill_betweenx(xax_y, 0, yax_y, color=self.PGO_COLOR[p_i], alpha=0.6)
            ax_y_hist.plot(yax_y, xax_y, color=self.PGO_COLOR[p_i], alpha=.8, linewidth = 1)  # Adding normal line

        ax_x_hist.set_xticks([])
        ax_x_hist.set_yticks([])
        ax_x_hist.tick_params(axis='both', which='both', length=0)
        ax_x_hist.spines['top'].set_visible(False)
        ax_x_hist.spines['right'].set_visible(False)
        ax_x_hist.spines['left'].set_visible(False)
        ax_x_hist.spines['bottom'].set_visible(False)
        
        ax_y_hist.set_xticks([])
        ax_y_hist.set_yticks([])
        ax_y_hist.tick_params(axis='both', which='both', length=0)
        ax_y_hist.spines['top'].set_visible(False)
        ax_y_hist.spines['right'].set_visible(False)
        ax_y_hist.spines['left'].set_visible(False)
        ax_y_hist.spines['bottom'].set_visible(False)
        
        ax_heatmap.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax_heatmap.yaxis.set_major_locator(MaxNLocator(nbins=10))    
        xticks = ax_heatmap.get_xticks().astype(int)
        yticks = ax_heatmap.get_yticks().astype(int)
        ax_heatmap.text(xticks.mean(), yticks.mean() - 10, f" R^2 = {R2:.2f}")
        ax_heatmap.set_xticklabels([f"{bayes_uniques[i]:.2f}" for i in xticks if i < len(bayes_uniques)])
        ax_heatmap.set_yticklabels([f"{net_uniques[i]:.1f}" for i in yticks if i < len(net_uniques)])
        for tick in ax_x_hist.get_xticklabels() + ax_x_hist.get_yticklabels():
            tick.set_visible(False)
        for tick in ax_y_hist.get_xticklabels() + ax_y_hist.get_yticklabels():
            tick.set_visible(False)






        
    """ theta distributions as a function of consec GOs prior to current input, split by current input """
    # def plot_bayes_vs_net_dists_consec_GOs(self, consec = 5, bins = 30):
    #     self.get_indices(cross_validation=True, flatten = True, eps_init = 0, til_action = True)

    #     self.net_dists, self.net_dists_xax = [np.zeros((consec, 2, bins)) for _ in range(2)]
    #     self.mu_vars = np.zeros((2, consec, 2, 2)) # bayes / net, consec, NOGO / GO, mu / var 
    #     self.bayes_dists = np.zeros((consec, 2, self.bayes_resolution))
    #     cols = self.Cmap.to_rgba(np.linspace(0, 1, consec))
    #     fig,ax = plt.subplots(2, 2, tight_layout = True, figsize = (15, 10))
    #     for c_i, c in enumerate(range(consec)):
    #         consec_inds = np.where(self.consec_flat == c + 1)[0] + 1
    #         for s_i, (stim_inds, stim_name) in enumerate(zip([self.NOGO_inds , self.GO_inds], ["NOGO", "GO"])):
    #             beep_inds = np.intersect1d(consec_inds, stim_inds)
    #             net_dist = np.histogram(self.net_theta_flat[beep_inds], bins = bins)
                
    #             self.net_dists[c_i, s_i] = net_dist[0]/net_dist[0].sum() 
    #             self.net_dists_xax[c_i, s_i] = net_dist[1][:-1]
    #             self.bayes_dists[c_i, s_i] = self.theta_dist_flat[:, beep_inds].mean(-1)

    #             if s_i:
    #                 linestyle = '-'
    #                 linewidth = 2
    #             else:
    #                 linestyle = '--'
    #                 linewidth = 3
    #             label =  f"{stim_name} after {c+1} GOs"
    #             ax[s_i, 0].plot(self.bayes_range, self.bayes_dists[c_i, s_i], c = cols[c_i], label = label, linestyle = linestyle, linewidth = linewidth)
    #             ax[s_i, 1].plot(self.net_dists_xax[c_i, s_i], self.net_dists[c_i, s_i], c = cols[c_i],  linestyle = linestyle, linewidth = linewidth)
                
    #             for agent_i, data in zip([0, 1], [self.flow_theta_flat, self.net_theta_flat]):
    #                 self.mu_vars[agent_i, c_i, s_i, 0] = data[beep_inds].mean()
    #                 self.mu_vars[agent_i, c_i, s_i, 1] = data[beep_inds].std()
                    
    #     ax[1, 0].set_xlabel("theta estimate")
    #     ax[1, 1].set_xlabel("theta estimate")
    #     ax[0, 0].set_title('Bayes Dists')
    #     ax[0, 1].set_title('Net Dists')
    #     ax[0, 0].set_ylabel('P(theta)')
    #     ax[1, 0].set_ylabel('P(theta)')
    #     ax[0, 0].legend()
    #     ax[1, 0].legend()
    #     ax[0, 0].set_ylim([0, .015])
    #     ax[0, 1].set_ylim([0, .1])
    #     ax[1, 0].set_ylim([0, .015])
    #     ax[1, 1].set_ylim([0, .1])
    #     ax[0, 1].set_xlim([0, 1])
    #     ax[1, 1].set_xlim([0, 1])
    #     plt.show()
        
    #     plt.plot(self.mu_vars[0, :, 0, 0], self.mu_vars[0, :, 0, 1], linestyle = '--', c = 'k', alpha = .35)
    #     plt.plot(self.mu_vars[0, :, 1, 0], self.mu_vars[0, :, 1, 1], c = 'k', alpha = .35)
    #     plt.plot(self.mu_vars[1, :, 0, 0], self.mu_vars[1, :, 0, 1], linestyle = '--',  c = 'k', alpha = .35, label = 'NOGO stats')
    #     plt.plot(self.mu_vars[1, :, 1, 0], self.mu_vars[1, :, 1, 1], c = 'k', alpha = .35, label = 'GO stats')
    #     plt.scatter(self.mu_vars[0, :, 0, 0], self.mu_vars[0, :, 0, 1], label = 'bayes', c=cols, s = 150, marker = '*')
    #     plt.scatter(self.mu_vars[0, :, 1, 0], self.mu_vars[0, :, 1, 1], c = cols, s = 40, marker = '*')
    #     plt.scatter(self.mu_vars[1, :, 0, 0], self.mu_vars[1, :, 0, 1], label = 'network', c = cols, s = 150)
    #     plt.scatter(self.mu_vars[1, :, 1, 0], self.mu_vars[1, :, 1, 1], c = cols, s = 40,)
    #     plt.xlabel("estimator mean"); plt.ylabel("estimator variance")
    #     plt.title("estimators mean vs variance")
    #     plt.xlim([0,1])
    #     plt.legend()
    #     plt.show()
        
    def plot_bayes_thresh_var(self, trials = 6):
        fig, ax = plt.subplots(1, 2, figsize = (15, 5))
        waiting, thresh, std = [np.zeros((trials, self.PGO_N)) for _ in range(3)]
        for p_i, p in enumerate(self.PGO_range):
            for s in range(trials):
                PGO_inds = np.where(self.PGO_flat == p)[0]
                PGO_inds = np.intersect1d(PGO_inds, self.cum_acts)
                trial_inds = np.where(self.trial_flat == s)[0]
                inds = np.intersect1d(PGO_inds, trial_inds)
              
                waiting[s, p_i] = self.RT_flat[inds].mean()
                std[s, p_i] = self.flow_thresh_RMS_flat[inds].mean()
                thresh[s, p_i] = (1+self.flow_thresh_flat[inds]).mean()

            pm =  std[:, p_i]
            for i, (mu, title) in enumerate(zip([waiting, thresh], ["emperical mean waiting time", "optimal threshold"])):
                mu = mu[:, p_i]
                ax[i].fill_between(x = np.arange(trials), y1 = mu - pm, y2 = mu + pm, color = self.PGO_COLOR[p_i], alpha = .25)
                ax[i].plot(np.arange(trials), mu, linestyle = '--', marker = '*', color = self.PGO_COLOR[p_i], markersize = 20)
                ax[i].set_title(title + " +/- optimal threshold STD (wrt. theta)")
        for i in range(2):
            ax[i].set_xlabel("trial from switch")
            ax[i].set_ylim([2, 15])
        plt.show()
    
###############################################################################################################################################
    def init_traj_plot(self):
        fig, ax = plt.subplots(1, 1, figsize = (self.fig_W, self.fig_H), tight_layout = True)
        self.trajectories_title = f"\naligned on {self.align_on}"
        xax = np.arange(self.max_traj_steps) 
        plt.xticks(ticks = xax, labels = xax - self.label_offset)
        plt.xlabel(self.traj_xlabel)
        self.alpha_mus = np.clip(self.split_num * self.N_mus / np.clip(self.N_mus.sum(0)[None, :], a_min = 0.01, a_max = 1), a_min = 0, a_max = 1)
        
    def init_3D_plot(self):
        if self.show: self.fig = plt.figure( figsize = (15,15));     ax = self.fig.add_subplot(111, projection='3d')
        ax.set_xlabel('PC 1', fontdict=self.font_dict)
        ax.set_ylabel('PC 2', fontdict=self.font_dict)
        ax.set_zlabel('PC 3', fontdict=self.font_dict)
        return ax

    """ plot state, theta, time """ 
    def plot_net_bayes_state_theta(self, alpha = 1, scatter = True):
        bayes_data = [self.flow_belief_mus, self.flow_theta_mus]
        net_data = [self.net_belief_mus, self.net_theta_mus]
        for name, data in zip(["regression", "theory"], [net_data, bayes_data]):
            traj_color = self.Cmap.to_rgba(np.linspace(0, 1, self.max_traj_steps))
            fig, ax = plt.subplots()
            for self.split_i in range(self.split_num):
                act_color = self.act_col[self.split_i, :self.max_traj_steps, :]
                split_color = self.split_cols[self.split_i]
                X = data[0][self.split_i]
                Y = data[1][self.split_i]
                if scatter:
                    ax.scatter(X, Y, s = 50, c= traj_color, alpha = 1)
                    ax.scatter(X, Y, s = 50, c= act_color, alpha = .5)
                ax.plot(X, Y, c = split_color, alpha = alpha)
            ax.set(xlabel='belief',ylabel='theta', title=name)
            if self.show: plt.show()
            
    """ plot pre action, action, post action trajectories along context manifold """
    def plot_gershman_comparison(self): 
        angle = [-0, -30]
        ax = self.init_3D_plot()
        traj_color = self.Cmap.to_rgba(np.linspace(0, 1, self.max_traj_steps))
        half = self.max_traj_steps//2
        for self.split_i in range(self.split_num):
            act_color = self.act_col[self.split_i, :self.max_traj_steps, :]
            split_color = self.split_cols[self.split_i]
            X, Y, Z = [self.PC_mus[dim, self.split_i, :] for dim in range(self.PC_dim)]
            ax.plot3D(X, Y, Z , c = split_color)     
            ax.plot(X[half:half+2], Y[half:half+2], Z[half:half+2], linewidth = 3,  markersize = 20, c= 'r', alpha = 1)
            dx, dy, dz = np.diff(X[half:half+2]), np.diff(Y[half:half+2]), np.diff(Z[half:half+2])
            L = .25*(dx**2 + dy**2 + dz**2)[0]
            if self.split_i == 0:
                ax.scatter3D(X[:half], Y[:half], Z[:half], s = 20, c= 'k', alpha = 1, label = 'stim')
                ax.scatter3D(X[half:], Y[half:], Z[half:], s = 20, c= 'b', alpha = 1, label = 'ITI')
                ax.quiver(X[half], Y[half], Z[half], dx, dy, dz, color = 'r', length = L, label = 'action')
            else:
                ax.scatter3D(X[:half], Y[:half], Z[:half], s = 20, c= 'k', alpha = 1)
                ax.scatter3D(X[half:], Y[half:], Z[half:], s = 20, c= 'b', alpha = 1)
                ax.quiver(X[half], Y[half], Z[half], dx, dy, dz, color = 'r', length = L)
        ax.plot(self.PC_mus[0, :, 0], self.PC_mus[1, :, 0], self.PC_mus[2, :, 0], c= 'g', linewidth = 7)
        ax.set_xlabel('PC 1');       ax.set_ylabel('PC 2');       ax.set_zlabel('PC 3')
        ax.legend(fontsize=25, markerscale=5)
        ax.view_init(angle[0], angle[1]);       
        plt.show()

    """ plot network activity in PCA basis"""
    def plot_on_PCA_basis(self, save_SVG = False, From = 0, Til = -1, half_col = 'r', start_col = True, act_prob_col = False, til_max_act = False):
        cmap = pcm.get_cmap('RdYlGn_r')
        act_prob_cmap = cm.ScalarMappable(cmap=cmap, norm = cm.colors.Normalize(0, self.emp_act_prob.max()))
        for angle in self.angles3D:
            ax = self.init_3D_plot()
            for self.split_i in range(self.split_num):
                if til_max_act:
                    Til = np.argmax(self.emp_act_prob[self.split_i, From:]) + From + 1
                
                split_color = self.split_cols[self.split_i]
                X, Y, Z = [self.PC_mus[dim, self.split_i, :] for dim in range(self.PC_dim)]
                if start_col:
                    ax.scatter3D(X[0], Y[0], Z[0], s = 200, c= 'g', alpha = .5)
                ax.scatter3D(X[From:Til], Y[From:Til], Z[From:Til], s = 20, c= split_color, alpha = .5)

                if act_prob_col:
                    c = act_prob_cmap.to_rgba(self.emp_act_prob[self.split_i, From:Til])
                    ax.scatter3D(X[From:Til], Y[From:Til], Z[From:Til], s = 20, c = c, alpha = 1)   
                
                if self.align_on == 'action' or self.align_on == 'W4L':
                    half = self.max_traj_steps // 2 
                    x_h = X[half]
                    y_h = Y[half]
                    z_h = Z[half]
                    dx =X[half+1] - x_h
                    dy =Y[half+1] - y_h
                    dz =Z[half+1] - z_h
                    ax.scatter3D(x_h, y_h, z_h, s = 200, c = half_col)
                    # ax.quiver(x_h, y_h, z_h, dx, dy, dz, color = 'r', label = 'action')
        
                ax.plot3D(X[From:Til], Y[From:Til], Z[From:Til] , c = split_color)     
            ax.set_xlabel('PC 1');       ax.set_ylabel('PC 2');       ax.set_zlabel('PC 3')
            ax.view_init(angle[0], angle[1]);   
            if save_SVG:
                self.save_SVG(name = str(angle))
            if self.show: plt.show()
            
    def plot_interactable_PCA(self):
        self.get_indices(col = 'block', From = 0,  Til = 50, stim_above = None, stim_below = None, eps_init = 0, 
            planted = False, plant_PGO = None, plant_ID = None, prev_PGO = None, curr_PGO = None, rew = None, align_on = 'action', flatten = True)
        self.run_trajectory(plot = False)
        ax = self.init_3D_plot()
        traj_color = self.Cmap.to_rgba(np.linspace(0, 1, self.max_traj_steps))
        for self.split_i in range(self.split_num):
            act_color = self.act_col[self.split_i, :self.max_traj_steps, :]
            split_color = self.split_cols[self.split_i]
            X, Y, Z = [self.PC_mus[dim, self.split_i, :] for dim in range(self.PC_dim)]
            ax.scatter3D(X, Y, Z, s = 50, c= traj_color, alpha = 1)
            ax.scatter3D(X, Y, Z, s = 50, c= act_color, alpha = .5)
            ax.plot3D(X, Y, Z , c = split_color)     
        ax.set_xlabel('PC 1');       ax.set_ylabel('PC 2');       ax.set_zlabel('PC 3')
        ax.view_init(0,-90)
        if self.show: plt.show()
        
    def plot_means_on_PCA_basis(self):
        for angle in self.angles3D:
            ax = self.init_3D_plot()
            for self.split_i, self.split_curr in enumerate(self.split):
                self.get_split_inds()
                split_color = self.split_cols[self.split_i]
                proj = np.array([self.proj.transform(traj.cpu().mean(-1, keepdims=True).T) for traj in self.net_output[self.trial_inds][self.split_inds]]).squeeze(1)
                
                X, Y, Z = [proj[:, dim]  for dim in range(self.PC_dim)]
                ax.scatter3D(X, Y, Z, s = 50, color = split_color, alpha = .5)
            ax.set_xlabel('PC 1');       ax.set_ylabel('PC 2');       ax.set_zlabel('PC 3')
            ax.view_init(angle[0], angle[1]);
            if self.show: plt.show()
                                
    """ plot projection onto each PC across time and gate mean activity"""
    def plot_trajectories(self):
        self.plot_PCA_trajectories()
        self.plot_net_v_bayes_trajectories()
        self.plot_network_trajectories()
        self.plot_behavior_trajectories()
    
    def plot_PCA_trajectories(self):
        self.plot_PCA_projections()
        # self.plot_PCA_derivatives()
        
        
    def plot_PCA_var_corrs(self, get_inds = True):
        from matplotlib.patches import Patch
        if get_inds:
            self.get_indices(run_traj = True, til_action = True, flatten = True, prep_mus = True, cross_validation = None)
        
        var_list = [self.Q_flat[1], self.flow_belief_flat, self.flow_theta_flat, -self.flow_dist_from_opt_flat]
        corrs = np.zeros((self.PC_dim, len(var_list), 3))  # Adjusted to store results for each inds group
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))  # Single subplot
        
        safe_inds = self.GO_inds
        unsafe_inds = self.NOGO_inds
        all_inds = self.ACTION_inds
        inds_groups = [all_inds, unsafe_inds, safe_inds]
        titles = ["GO inds", "NOGO inds", "all inds"]
        alphas = [.4, 0.7, 1.0]  # Different alpha levels for different inds groups
        
        colors = ['C0', 'C1', 'C2']  # Colors for PC1, PC2, PC3
        width =  .1*(len(colors) * len(inds_groups))  # Adjust width so that bars are adjacent with no space
        
        for var_i, var in enumerate(var_list):
            for p, c in enumerate(colors):
                for state_i, (inds, title) in enumerate(zip(inds_groups, titles)):
                    PC = self.PC_flat[p, inds]
                    V =  var[inds]

                    corrs[p, var_i, state_i] = self.R2(PC, V)
                    x = 2*var_i * (len(colors) * len(inds_groups) + 1) + p * len(inds_groups) + state_i * width
                    ax.bar(x, corrs[p, var_i, state_i], width=width, color=c, alpha=alphas[state_i])
        
        ax.set_xticks(np.arange(len(var_list))*2 * (len(colors) * len(inds_groups) + 1) + (len(colors) * len(inds_groups) - 1) / 2)
        ax.set_xticklabels(["Q lick", "state\nestimate", "context\nestimate", "DV"])
        
        pc_handles = [Patch(facecolor=color) for color in colors]
        pc_labels = [f"PC{p+1}" for p in range(self.PC_dim)]
        
        inds_handles = [Patch(facecolor='grey', edgecolor='grey', alpha=alpha) for alpha in alphas]
        inds_labels = titles
        ax.legend(pc_handles + inds_handles, pc_labels + inds_labels, loc='upper right')
        
        ax.set_title("R^2 between each PC & task variables")
        ax.set_ylim([0, 1])
        plt.show()
        
    def plot_PCA_PGO_corr(self):
        self.init_traj_plot()
        PCs = self.proj.transform(self.output_flat.T).T
        corr = np.zeros((self.PC_dim, self.max_traj_steps))
        self.get_split_inds(split_after = True)
        for PC in range(self.PC_dim):
            corr[PC, :] = self.get_PC_PGO_corr(PCs[PC,:])
        plt.plot(np.abs(corr.T), '-o');        plt.title(f"Strength of correlation of PCs to {self.col}");        plt.legend(["PC1", "PC2", "PC3"])
        if self.show: plt.show()
        
    def plot_PC2_PGO_corr(self):
        self.get_indices(col = 'block', planted = True, align_on = 'onset', flatten = True)
        self.run_trajectory(plot = False)
        PCs = self.proj.transform(self.output_flat.T).T
        block = self.get_PC_PGO_corr(PCs[1,:])
        self.get_indices(col = 'plant_PGO', planted = True, align_on = 'onset', flatten = True)
        self.run_trajectory(plot = False)
        PCs = self.proj.transform(self.output_flat.T).T
        plant = self.get_PC_PGO_corr(PCs[1,:])
        
        self.init_traj_plot();                          plt.plot(block.T);                                      plt.plot(plant.T)
        plt.title("Correlation of PC 2 to PGO");        plt.legend(["previous context", "new context"]);        
        if self.show: plt.show()
        
    def get_PC_PGO_corr(self, PC):
       corr = np.zeros(self.max_traj_steps)
       self.get_split_inds(split_after = True)
       for self.step in range(self.max_traj_steps):   
           self.get_trajectory_step_inds()
           if len(self.step_inds) > 0:
               corr[self.step] = self.correlate(X = PC[None, self.step_inds], Y = self.step_split_cond).squeeze()
       return corr

    def plot_PCA_projections(self, SVG = False):
        PC_num = self.PC_mus.shape[0] 
        for self.dim in range(PC_num):
            self.plot_general_trajectory(self.PC_mus[self.dim], name = f"{self.PC_labs[self.dim]}", SVG = SVG)
        
    def plot_network_trajectories(self):
        for name, activity in zip(self.activity_names, self.activities):
            self.init_traj_plot()
            for self.split_i in range(self.split_num):
                SE = abs(activity[:, self.split_i, :]).std(0)/np.sqrt(self.hid_dim)
                mu = activity[:, self.split_i, :].mean(0)
                self.plot_trajectory_split(mu, SE)
                plt.fill_between(np.arange(len(mu)), mu-SE, mu+SE, color =  self.split_cols[self.split_i, :], alpha = .05)   
                plt.title(f"{name} network activity" + self.trajectories_title)
            plt.legend()
            if self.show: plt.show()
            
    def plot_net_v_bayes_trajectories(self):
        self.plot_general_trajectory(self.flow_theta_mus, "bayes theta")
        self.plot_general_trajectory(self.net_theta_mus, "net theta")
        self.plot_general_trajectory(self.flow_belief_mus, "bayes belief")
        self.plot_general_trajectory(self.net_belief_mus, "net belief")
        self.plot_general_trajectory(self.PGO_mus, "True theta")
        self.plot_general_trajectory(self.GO_mus, "stimuli")
        
    def plot_net_bayes_same_plot(self, SVG = False):
        self.plot_general_trajectory(self.net_belief_mus, "net vs joint bayes belief",\
            other_mu = self.flow_belief_mus,  var = self.net_belief_vars, other_var = self.flow_belief_vars, SVG = SVG)
        self.plot_general_trajectory(self.net_theta_mus,"net vs joint bayes theta",\
            other_mu = self.flow_theta_mus, var =  self.net_theta_vars, other_var = self.flow_theta_vars, SVG = SVG, ylim = 1)
        self.plot_general_trajectory(self.net_belief_mus, "net vs factorized bayes belief",\
            other_mu = self.factorized_belief_mus,  var = self.net_belief_vars, other_var = self.factorized_belief_vars, SVG = SVG)
        self.plot_general_trajectory(self.net_theta_mus,"net vs factorized bayes theta",\
            other_mu = self.factorized_theta_mus, var =  self.net_theta_vars, other_var = self.factorized_theta_vars, SVG = SVG, ylim = 1)
        self.plot_general_trajectory(self.GO_mus, "stimuli")
        self.plot_general_trajectory(self.net_dist_from_opt_mus, "state est - thresh est", other_mu = self.flow_dist_from_opt_mus)
        self.plot_general_trajectory(self.net_dist_from_opt_mus - self.flow_dist_from_opt_mus, "TRAINED net DV - bayes DV", ylim = [-.3, .3], use_alpha_mus = True)
        self.plot_general_trajectory(self.net_dist_from_opt_mus,"net DV, bayes DV", other_mu = self.flow_dist_from_opt_mus, other_var = self.flow_belief_RMS_mus, hide_mu = [2])


    def plot_ICLR_NOGO_plot(self, SVG = False):
        belief_mus = [self.net_belief_mus, self.flow_belief_mus, self.factorized_belief_mus]
        belief_vars = [ self.net_belief_vars, self.flow_belief_vars, self.factorized_belief_vars]
        theta_mus = [self.net_theta_mus, self.flow_theta_mus, self.factorized_theta_mus]
        theta_vars = [self.net_theta_vars, self.flow_theta_vars, self.factorized_theta_vars]
        for i, (name, mu_data, var_data) in enumerate(zip(["belief", "theta"], [belief_mus, theta_mus], [belief_vars, theta_vars])):
            self.init_traj_plot()
            for self.split_i in range(self.split_num):
                self.plot_trajectory_split(mu_data[0][self.split_i, :], SE = var_data[0][self.split_i, :])
                self.plot_trajectory_split(mu_data[1][self.split_i, :], SE = var_data[1][self.split_i, :], bayes = True)
                self.plot_trajectory_split(mu_data[2][self.split_i, :], SE = var_data[2][self.split_i, :], bayes = True)
                plt.plot(mu_data[2][self.split_i, :], marker = 'x', c = 'k', linestyle = self.bayes_style, alpha = .6)

    def plot_behavior_trajectories(self):
        names = ["Q DIFF", "lick prob", "FIRST lick prob", "pred value"]
        mus =  [self.QDIFF_mus, self.lick_prob_mus, self.get_survival(self.lick_prob_mus), self.value_mus]
        for name, mu in zip(names, mus):
            self.plot_general_trajectory(mu, name)
            
    def plot_general_trajectory(self, mu = None, name = None, other_mu = None, var = None, other_var = None, SVG = False, ylim = None, var_alpha = .2, use_alpha_mus = False, hline = False, hide_mu = []):
        self.init_traj_plot()
        self.use_alpha_mus = use_alpha_mus
        for self.split_i in range(self.split_num):
            var_1 =  var[self.split_i, :] if np.any(var != None) else None
            self.plot_trajectory_split(mu[self.split_i, :] , SE = var_1, var_alpha = var_alpha, draw_mu = 1 not in hide_mu)

            has_var = np.any(other_var != None)
            if np.any(other_mu != None) or has_var: 
                var_2 =  other_var[self.split_i, :] if has_var else None
                self.plot_trajectory_split(other_mu[self.split_i, :] , SE = var_2, bayes = True, var_alpha = var_alpha, draw_mu = 2 not in hide_mu)
                
        if hline:
            plt.axhline(hline, c = 'r', linestyle = '--', alpha = .7, linewidth = 2)
        # plt.legend()
        if ylim is not None:
            plt.ylim(ylim)
        # self.make_color_bar()
        plt.title(f"{name} {self.trajectories_title}", fontdict=self.font_dict)
        if SVG:
            if (other_mu is not None) and self.factorize: 
                name = name + ' (factorized bayes)'
            self.save_SVG(name = f"{name}")
        if self.show: plt.show()

    def plot_trajectory_split(self, mu, SE = None, bayes = False, var_alpha = .2, draw_mu = True):
        xax = np.arange(self.max_traj_steps)
        col = self.split_cols[self.split_i, :]
        alpha = .05 if bayes else .15 
        linestyle = self.bayes_style if bayes else '-' 
        
        if self.align_on == "action" and self.til_action:
            xax = xax[:self.max_traj_steps//2 + 1]
            mu = mu[:self.max_traj_steps//2+1]
            if SE is not None:
                SE = SE[:self.max_traj_steps//2+1]
        if SE is not None: 
            # plt.errorbar(xax, mu, yerr=SE, color = col, alpha = var_alpha, linestyle = linestyle, capsize=5)
            plt.fill_between(xax, mu-SE, mu + SE, color = col, alpha = var_alpha)
            plt.plot(xax, mu-SE, color = col, linestyle = linestyle, linewidth = 1)
            plt.plot(xax, mu+SE, color = col, linestyle = linestyle, linewidth = 1)

        if draw_mu:
            if self.use_alpha_mus:
                for i in range(len(xax)-1):
                    a = self.alpha_mus[self.split_i]
                    plt.plot(xax[i:i+2], mu[i:i+2], color = col, linestyle = linestyle, alpha = a[i], linewidth = 2)
            else:
                plt.plot(mu, color = col, linestyle = linestyle, label = self.split_leg[self.split_i], linewidth = 4)
                
        plt.xlabel("time",  fontdict=self.font_dict)

    
#################################################################################################

    def plot_trajectory_with_velocity(self, cue = None, get_inds = False):
        if get_inds:
            self.get_indices(run_traj = True, til_action = True, flatten = True, prep_mus = True, cross_validation = None)
        self.run_trajectory(plot=False)
        self.get_trajectory_diffs()        
        ax = self.init_3D_plot()
        traj_color = self.Cmap.to_rgba(np.linspace(0, 1, self.max_traj_steps-1))
        
        for self.split_i, self.split_curr in enumerate(self.split):
            mu = self.PC_prev_mus[:,self.split_i,1:]
            d_go = self.GO_traj_diffs[:,self.split_i,1:]
            d_nogo = self.NOGO_traj_diffs[:,self.split_i,1:]
            
            """ CLAMP """
            # d_go = np.clip(d_go, a_min = -.3, a_max = .3)
            # d_nogo = np.clip(d_nogo, a_min = -.3 , a_max = .3)
            """ CLAMP """
            
            
            act_color = self.act_col[self.split_i, 1:self.max_traj_steps, :]
            split_color = self.split_cols[self.split_i]
            from enzyme.colors import c_go, c_nogo
        
            ax.scatter3D(mu[0], mu[1], mu[2], s=plt.rcParams['lines.markersize']**2, c=traj_color, alpha=.5)
            ax.scatter3D(mu[0], mu[1], mu[2], s=plt.rcParams['lines.markersize']**2, c=act_color, alpha=0.5)
            ax.plot3D(mu[0], mu[1], mu[2], linewidth=8, c=split_color, alpha=0.5)
            if cue == None or cue == 1:
                ax.quiver(mu[0], mu[1], mu[2], d_go[0], d_go[1], d_go[2], color=c_go, alpha=0.5)
            if cue == None or cue == 0:
                ax.quiver(mu[0], mu[1], mu[2],  d_nogo[0], d_nogo[1], d_nogo[2], color=c_nogo, alpha=0.5)
            ax.set_xlabel(self.PC_Xlab)
            ax.set_ylabel(self.PC_Ylab)
            ax.set_zlabel(self.PC_Zlab)        
            ax.view_init(90, 0);

        if self.show: plt.show()

    """ plot trajectory of constant go vs nogo"""
    def plot_constant_GO_NOGO_trajectory(self, GO_ID = 1, til_action = None, scatter_til = 100):
        for angle in self.angles3D:
            ax = self.init_3D_plot()
            for plant_ID in [0, 1]:
                self.get_indices(col = 'block', plant_ID = plant_ID, planted = True, align_on = 'onset', flatten = True, til_action = til_action)
                c = 'C2' if plant_ID == GO_ID else 'r'
                M = 'H' if plant_ID == GO_ID else 'X'
                self.run_trajectory(plot = False)
                self.get_trajectory_diffs()
        
                for self.split_i in range(self.split_num):
                    split_color = self.split_cols[self.split_i]
                    X, Y, Z = [self.PC_prev_mus[dim, self.split_i, 1:] for dim in range(self.PC_dim)]
                    
                    """ for sanity checking diffs """
                    diffs = self.GO_traj_diffs if plant_ID == GO_ID else self.NOGO_traj_diffs            
                    dX, dY, dZ = [diffs[dim, self.split_i, 1:] for dim in range(self.PC_dim)]
                    ax.quiver(X,Y,Z,dX,dY,dZ)
                    ax.quiver(X,Y,Z,dX,dY,dZ)
                    """ for sanity checking diffs """
                    
                    ax.plot3D(X, Y, Z , c = c, linewidth = 15, alpha = .25)     
                    ax.plot3D(X, Y, Z, c = split_color, alpha = 1, linewidth = 2)
                    ax.scatter3D(X[1:scatter_til],Y[1:scatter_til],Z[1:scatter_til], marker=M, color = split_color, s = 100, alpha = .5)
                    ax.scatter3D(X[0], Y[0], Z[0],  s = 500, color = 'g', alpha = .5)
                    
                X, Y, Z = [self.PC_mus[dim, :, 0] for dim in range(self.PC_dim)]
                ax.plot3D(X,Y,Z, linewidth = 5, color = 'g', alpha = 1)
                ax.set_xlabel('PC 1');       ax.set_ylabel('PC 2');       ax.set_zlabel('PC 3')
                ax.view_init(angle[0], angle[1]);
            self.save_SVG(name = str(angle))
            if self.show: plt.show()
       
        
    def plot_stim_specific_flow_field(self, name="NETWORK", til_action=True, from_action=False, SVG=False, xlim = None, ylim = None, x_round = 0, y_round = 1, min_count = 50, scale = 1, get_inds = True, get_R = False):
        cmap = pcm.get_cmap('cool')        
        norm = Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)        
        self.get_phase_space(til_action=til_action, from_action=from_action, x_round = x_round, y_round = y_round, min_count = min_count, get_inds = get_inds, get_R = get_R)
    
        fig, self.ax = plt.subplots(1, 2, figsize=(20, 10), tight_layout=True)
        for self.stim_i, (stim_name, c) in enumerate(zip(["NOGO", "GO"], ["r", "C2"])):
            self.plot_flow_field(X = self.uniques[0], Y = self.uniques[1], U=self.dPC1[self.stim_i].T, V=self.dPC2[self.stim_i].T, c=c, scale = scale)
            
            self.ax[self.stim_i].set_title(f"{name} {stim_name} flow\n {'(pre action)' if til_action else ''} {'(post action)' if from_action else ''}")
            for x_i, x in enumerate(self.uniques[0]):
                for y_i, y in enumerate(self.uniques[1]):
                    if self.PC_count[self.stim_i, x_i, y_i] > 0:
                        col = sm.to_rgba(self.PGO_mu[self.stim_i, x_i, y_i])
                        self.ax[self.stim_i].scatter(x, y, color = 'k', s = 15, alpha = .8)
                        self.ax[self.stim_i].scatter(x, y, color = col, s = 12, alpha = 1)
            self.set_flow_lims(self.ax[self.stim_i], xlim, ylim)

        if SVG:
            self.save_SVG(f"{name} flow")
        if self.show:
            plt.show()
            
        """ plot R """
       #  self.run_regressions(regress_to = "ground_truth", regress_from = "STM", get_inds = True, cog_map = "naive")
       # self.plot_stim_specific_flow_field(x_round = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], y_round = [.1, .3, .5, .7, .9], scale = 10, get_R = True)
       # fig, ax = plt.subplots(1, 2, figsize = (10, 5)); sns.heatmap(self.grid_R2[0].T, ax = ax[0], vmin = 0, vmax = .9);  sns.heatmap(self.grid_R2[1].T, ax = ax[1], vmin = 0, vmax = .9); plt.show()
        """ plot R """
        
    def plot_flow_field(self, X, Y, U, V, c = 'C1', a = 1, scale = 1):
        X, Y = np.meshgrid(X, Y)
        self.ax[self.stim_i].quiver(X,Y, U,V, color=c, alpha = a, scale = scale)
        self.set_cog_map_labels(self.ax[self.stim_i], subplot = True)
            
    def set_flow_lims(self, ax, xlim, ylim):
        count = np.where(self.PC_count[self.stim_i] > 0)
        if xlim is not None:
            ax.set_xlim([xlim[0], xlim[1]])
        else:
            min_x = self.uniques[0][count[0].min()]
            max_x = self.uniques[0][count[0].max()]
            ax.set_xlim([min_x - .1, max_x +.1])
        if ylim is not None:
            self.ax[self.stim_i].set_ylim([ylim[0], ylim[1]])
        else:
            min_y = self.uniques[1][count[1].min()]
            max_y = self.uniques[1][count[1].max()]
            ax.set_ylim([min_y - 1, max_y + 1])
        
    def get_flow_for_noises(self, net_name ="TRAINED", SVG=False, x_round = [.5, 1, 1.5], y_round = [.3, .4, .5], noises = [.01, .03, .05, .08], xlim = None, ylim = None, scale = 1):
        x_len = len(x_round)
        y_len = len(y_round)
        n_len = len(noises)
        
        self.net_mean_field = np.zeros((x_len, n_len, 2, 2))            # dims : state, noise, dx/dy, NOGO/GO
        self.net_full_field = np.zeros((x_len, y_len, n_len, 2, 2))     # dims : state, theta, noise, dx/dy, NOGO/GO
        self.bayes_mean_field = np.zeros((2, x_len, 2, 2))              # dims : BAYES/ FACTORIZED, state, dx/dy, NOGO/GO
        self.bayes_full_field = np.zeros((2, x_len, y_len, 2, 2))       # dims : BAYES-FACTORIZED, state, dx/dy, NOGO/GO
        self.mean_field_similarity = np.zeros((3, x_len, y_len, n_len, 2)) # dims: NET-BAYES / NET-FACTORIZED / BAYES-FACTORIZED, state, theta, NOGO / GO

        for n_i, n in enumerate(tqdm(noises)):
            self.run_regressions(regression_noise = n, regress_to = 'ground_truth', cog_map = 'regression', get_inds = True);
            self.get_phase_space(x_round = x_round, y_round = y_round, get_inds = True, disable = True)        
            self.net_mean_field[:, n_i, 0] =  self.dX_mu.T
            self.net_mean_field[:, n_i, 1] =  self.dY_mu.T
            self.net_full_field[:, :, n_i, 0] = np.transpose(self.dPC1, axes = (1, 2, 0))
            self.net_full_field[:, :, n_i, 1] = np.transpose(self.dPC2, axes = (1, 2, 0))
     
        for a_i, (theory_name, cog_map) in enumerate(zip(["full bayes", "naive bayes"], ["theory", "naive"])):
            self.cog_map = cog_map 
            self.handle_cog_map()
            self.get_phase_space(x_round = x_round, y_round = y_round, get_inds = False, disable = True)    
            self.bayes_mean_field[a_i, :, 0, :] =  self.dX_mu.T
            self.bayes_mean_field[a_i, :, 1, :] =  self.dY_mu.T
            self.bayes_full_field[a_i, :, :, 0] = np.transpose(self.dPC1, axes = (1, 2, 0))
            self.bayes_full_field[a_i, :, :, 1] = np.transpose(self.dPC2, axes = (1, 2, 0))
            
        for n_i in range(n_len):
            for stim_i, stim_name in enumerate(["NOGO", "GO"]):
                for x_i, x in enumerate(x_round):
                    for y_i, y in enumerate(y_round):
                        net = self.net_full_field[x_i, y_i, n_i, :, stim_i]
                        bayes = self.bayes_full_field[:, x_i, y_i, :, stim_i]
                        self.mean_field_similarity[0, x_i, y_i, n_i, stim_i] = self.get_cosin_similarity(net, bayes[0])
                        self.mean_field_similarity[1, x_i, y_i, n_i, stim_i] = self.get_cosin_similarity(net, bayes[1])
                        self.mean_field_similarity[2, x_i, y_i, n_i, stim_i] = self.get_cosin_similarity(bayes[1], bayes[0])
        
        self.plot_noise_fields(net_name, noises, n_len, x_round, y_round, SVG, xlim, ylim, scale)
                        
    def plot_noise_fields(self, net_name, noises, n_len, x_round, y_round, SVG, xlim, ylim, scale = 1):
        Cs = np.empty(2, dtype = object)
        cmap = LinearSegmentedColormap.from_list('red_to_gray', [(0, '#FF0000'), (1, '#808080')] )
        cmap = cm.ScalarMappable(cmap=cmap, norm = cm.colors.Normalize(0, 1))
        Cs[0] = cmap.to_rgba(np.linspace(0, 1, n_len))
        cmap = LinearSegmentedColormap.from_list('green_to_gray',  [(0, '#008000'), (1, '#808080')])
        cmap = cm.ScalarMappable(cmap=cmap, norm = cm.colors.Normalize(0, 1))
        Cs[1] = cmap.to_rgba(np.linspace(0, 1, n_len))
        
        
        fig, self.ax = plt.subplots(1, 2, figsize=(5, 3), tight_layout=True)
        for self.stim_i, stim_name in enumerate(["NOGO", "GO"]):
            for n_i, n in enumerate(tqdm(noises)):
                U = self.net_full_field[:, :, n_i, 0, self.stim_i].T
                V = self.net_full_field[:, :, n_i, 1, self.stim_i].T
                self.plot_flow_field(X = x_round, Y = y_round, U = U, V = V, c = Cs[self.stim_i][n_i], scale = scale)
            self.ax[self.stim_i].set_title(f"{net_name} {stim_name} flow")
        self.set_noise_field_plot_lims(xlim, ylim)
        if SVG is not False:
            self.save_SVG(f"{name} network fields")
        plt.show()
        
        fig, self.ax = plt.subplots(1, 2, figsize=(5, 3), tight_layout=True)
        for a_i, (theory_name, c) in enumerate(zip(["full bayes", "naive bayes"], [0, -1])):
            for self.stim_i, stim_name in enumerate(["NOGO", "GO"]):
                U = self.bayes_full_field[a_i, :, :, 0, self.stim_i].T
                V = self.bayes_full_field[a_i, :, :, 1, self.stim_i].T
                self.plot_flow_field(X = x_round, Y = y_round, U = U, V = V, c = Cs[self.stim_i][c], scale = scale)
                self.ax[self.stim_i].set_title(f"theory {stim_name} flow")
        self.set_noise_field_plot_lims(xlim, ylim)
        if SVG is not False:
            self.save_SVG(f"{name} theory fields")
        plt.show()
        
        full = np.nanmean(np.nanmean(np.nanmean(self.mean_field_similarity[0], 0), 0), -1).T
        naive = np.nanmean(np.nanmean(np.nanmean(self.mean_field_similarity[1], 0), 0), -1).T
        self.plot_net_bayes_mean_field_similarity(noises, full, naive, name = net_name, SVG = SVG)

    def set_noise_field_plot_lims(self, xlim, ylim):
        for i in range(2):
            if xlim is not None:
                self.ax[i].set_xlim(xlim[0], xlim[1])
            if ylim is not None:
                self.ax[i].set_ylim(ylim[0], ylim[1])

        
    def plot_net_bayes_mean_field_similarity(self, noises, full, naive, name, SVG = False, ylim = [.6, 1.05]):
        fig, self.ax = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)                    
        self.ax[0].plot(noises, full, label = "cosine similarity to full bayes", marker = 'o')
        self.ax[0].plot(noises, naive, label = "cosine similarity to naive bayes", marker = 'o')
        self.ax[0].set_title("network theory similarity")
        self.ax[0].set_ylim(ylim)
        self.ax[0].legend()
        self.ax[1].plot(noises, full - naive, marker = 'o')
        self.ax[1].axhline(0, c = 'r', alpha = .5)
        self.ax[1].set_xlabel("guassian noise standard deviation")
        self.ax[1].set_title("full - naive")
        self.ax[1].set_ylim([-.5, .3])
        if SVG:
            self.save_SVG(f"{name} mean field")
        plt.show()
                     
    def plot_quadrant_space(self, SVG = False):
        self.flow_quadrants = np.nan_to_num(self.flow_quadrants, nan = 0)
        belief_axis = np.array([-9, -2, 5])
        theta_axis = np.array([-6,0,6])
        X, Y = np.meshgrid(belief_axis, theta_axis)

        fig, ax = plt.subplots(1, 1, figsize = (10, 10), tight_layout = True)
        for stim_i, stim_col in zip([0,1], ['r', 'g']):
            # fig, ax = plt.subplots(1, 1, figsize = (10, 10), tight_layout = True)
            for agent_i, a, e, h, w in zip([2, 0,1], [.85,.85, .85], ['k', None,'w'], ['|||||', None, '|||||'],[.0085, .02,.0085]):
                belief_std = self.flow_quadrants[agent_i, stim_i, :, :, 0].std()
                U = self.flow_quadrants[agent_i, stim_i, :, :, 0].T/belief_std
                theta_std = self.flow_quadrants[agent_i, stim_i, :, :, 1].std()
                V = self.flow_quadrants[agent_i, stim_i, :, :, 1].T/theta_std
                plt.quiver(X,Y, U,V, color= stim_col, angles = 'xy',  scale_units = 'xy', scale = 1, edgecolor= e,  hatch= h, width = w)
                plt.xticks(ticks = belief_axis, labels = ["low belief", "med belief", "high belief"])
                plt.yticks(ticks = theta_axis, labels = ["low theta", "med theta", "high theta"])    
            plt.xlim([-11, 9]); plt.ylim([-11, 8])
        plt.legend(["naive", "net", "full bayes"])
        if SVG:
            self.save_SVG("quadrant flow")
        plt.show()

#################################################################################################
    """ PC clustering plotting """ 
    
    """ plot PC vs diff PC colored by different variables"""
    def plot_2D_stim_clusters(self):
        self.get_indices(planted = None, align_on = 'onset', flatten = True)
        self.run_trajectory(plot = False)
        self.get_GO_NOGO_inds()
      
        fig, ax = plt.subplots(3, 3, figsize = (20, 17), tight_layout = True)
        col_from = self.PGO_flat         # self.pre_act  # FOR COLOR BY PRE-ACT POST ACT
        
        for PC in range(self.PC_dim):
            for stim_col, inds in zip(['C2', 'C3'],[self.GO_inds, self.NOGO_inds]):
                inds = np.random.choice(inds, 5000)
                inner_col = self.Cmap.to_rgba([col_from[inds]]).squeeze(0)
                
                X = self.PC_flat[PC, inds]
                X_prev = self.PC_prev[PC, inds]
                Y = self.PC_flat[(PC+1) % self.PC_dim, inds]
                Y_prev = self.PC_prev[(PC+1) % self.PC_dim, inds]
                Z = self.PC_diff[(PC+2) % self.PC_dim, inds]
                
                ax[PC, 0].scatter(X, Y, s = 100, alpha = .5, color = stim_col)
                ax[PC, 0].scatter(X, Y, s = 10, alpha = 1, c = inner_col)
                ax[PC, 1].scatter(X_prev, Z, s = 100, alpha = .5, color = stim_col)
                ax[PC, 1].scatter(X_prev, Z, s = 10, alpha = 1, c = inner_col)
                ax[PC, 2].scatter(Y_prev, Z, s = 100, alpha = .5, color = stim_col)
                ax[PC, 2].scatter(Y_prev, Z, s = 10, alpha = 1, c = inner_col)
            ax[PC,0].set_title(f"PC{(PC)%self.PC_dim + 1} and PC{(PC+1)%self.PC_dim + 1} locations")
            ax[PC,0].set_xlabel(f"location in PC{(PC) % self.PC_dim + 1}")
            ax[PC,0].set_ylabel(f"location in PC{(PC+1)%self.PC_dim + 1}")
            ax[PC,1].set_title(f"effect of PC{(PC)%self.PC_dim + 1} location on PC{(PC+2)%self.PC_dim + 1} velocity")
            ax[PC,1].set_xlabel(f"location in PC{(PC) % self.PC_dim + 1}")
            ax[PC,1].set_ylabel(f"velocity of PC{(PC+2)%self.PC_dim + 1}")
            ax[PC,2].set_title(f"effect of PC{(PC + 1)%self.PC_dim + 1} location on PC{(PC+2)%self.PC_dim + 1} velocity")
            ax[PC,2].set_xlabel(f"location in PC{(PC+1) % self.PC_dim + 1}")
            ax[PC,2].set_ylabel(f"velocity of PC{(PC+2)%self.PC_dim + 1}")
            # self.save_SVG(name = f"{PC}")
        if self.show: plt.show()                   

    def plot_specific_2D_diffs(self, first = True, second = True, third = True):
        self.get_indices(planted = None, align_on = 'onset', flatten = True)
        self.run_trajectory(plot = False)
        self.get_GO_NOGO_inds()
     
        G_inds = np.random.choice(self.GO_inds, 5000)
        N_inds = np.random.choice(self.NOGO_inds, 5000)
        
        if first:
            """ PC3 on PC2 GO diff """
            fig, ax = plt.subplots(1, 1, figsize = (15, 15), tight_layout = True)
            ax.scatter(self.PC_prev[2, G_inds], self.PC_diff[1, G_inds], s = 1000, c = 'C2', alpha = .2)
            inner_col = self.Cmap.to_rgba([self.pre_act[G_inds]]).squeeze(0)
            ax.scatter(self.PC_prev[2, G_inds], self.PC_diff[1, G_inds], s = 100, c = inner_col, alpha = 1)
            ax.set_ylim([-.5, .5]);         ax.set_title("Effect on PC2 velocities");         ax.set_xlabel("PC3");        
            if self.show: plt.show()
        
        if second:
            """ PC2 on PC1 GO diff (use til action)"""
            fig, ax = plt.subplots(1, 1, figsize = (15, 15), tight_layout = True)
            ax.scatter(self.PC_prev[1, G_inds], self.PC_diff[0, G_inds], s = 1000, c = 'C2', alpha = .2)
            inner_col = self.Cmap.to_rgba([self.PGO_flat[G_inds]]).squeeze(0)
            ax.scatter(self.PC_prev[1, G_inds], self.PC_diff[0, G_inds], s = 100, c = inner_col, alpha = 1)
            ax.set_title("Effect on PC1 velocities");         ax.set_xlabel("PC2");        
            if self.show: plt.show()
        
        if third:
            """ PC1 on PC2 NOGO diff (use from action)"""
            fig, ax = plt.subplots(1, 1, figsize = (15, 15), tight_layout = True)
            ax.scatter(self.PC_prev[0, N_inds], self.PC_diff[1, N_inds], s = 1000, c = 'C3', alpha = .2)
            inner_col = self.Cmap.to_rgba([self.PGO_flat[N_inds]]).squeeze(0)
            ax.scatter(self.PC_prev[0, N_inds], self.PC_diff[1, N_inds], s = 100, c = inner_col, alpha = 1)
            ax.set_title("Effect on PC2 velocities");         ax.set_xlabel("PC1");        
            if self.show: plt.show()
            
        
###############################################################################################################################################

    """ plot evolution of PCs throughout training """ 
    def get_episode_PC_evolution(self, round_dim = 2):
        self.get_indices(planted = False, eps_init = 1500, flatten = False)
        self.run_PCA(override_indices=True)
        self.PC_yax = np.empty((self.episodes, 2, self.PC_dim, self.PC_dim), dtype = object)                         # dim: [episode, GO/NOGO, PC, diff][xax]
        self.PC_all = self.proj.transform(self.flatten_trajectory(self.net_output).T).T
        self.uniques = np.array([np.unique(np.round(PC, round_dim)) for PC in self.PC_all], dtype = object)       
        
        for e in range(self.episodes):
            e_inds = np.where(self.eps == e)
            self.output_flat = self.flatten_trajectory(self.net_output[e_inds])
            self.PC_flat = self.proj.transform(self.output_flat.T).T            
            self.PC_prev = np.concatenate((np.zeros((self.PC_dim, 1)), self.PC_flat[:, :-1]), -1)
            PC_round = np.round(self.PC_prev, round_dim)
            self.get_GO_NOGO_inds()
            
            for stim_i, stim_inds in enumerate([self.GO_inds, self.NOGO_inds]):
                for PC in range(self.PC_dim):
                    for PC_diff in range(self.PC_dim):
                        self.PC_yax[e, stim_i, PC, PC_diff] = np.zeros(len(self.uniques[PC]))                        
                    for u_i, u in enumerate(self.uniques[PC]):
                        PC_unique_inds = np.where(PC_round[PC, :] == u)[0]
                        inds = np.intersect1d(PC_unique_inds, stim_inds)
                        for PC_diff in range(self.PC_dim):
                            self.PC_yax[e, stim_i, PC, PC_diff][u_i] = 0 if len(inds) == 0 else self.PC_diff[PC_diff, inds].mean(-1)
        self.plot_episode_PC_evolution()

    def plot_episode_PC_evolution(self):
        PC_Z = self.smooth(self.PC_yax, 100, 0)
        PC_Y = np.arange(len(PC_Z))
        
        for PC in range(self.PC_dim):
            for PC_diff in range(self.PC_dim):
                norm_Z = np.array([np.max(np.abs(np.stack(PC_Z[e, :, PC, PC_diff]))) for e in range(len(PC_Z))])[:,None]
                norm_X = np.max(np.abs(self.uniques[PC]))
             #   norm_X = norm_Z = 1 
                fig, ax = plt.subplots(1, 1, figsize = (19,9), subplot_kw=dict(projection='3d'), tight_layout = True)
                for stim_i, (stim_name, c) in enumerate(zip(["GO", "NOGO"], ["C2", "r"])):        
                    Z = np.vstack(PC_Z[:, stim_i, PC, PC_diff])
                    Z = self.smooth(Z, 10, axis = 1)[:, 5:-4]
                    X, Y = np.meshgrid(self.uniques[PC], PC_Y)
                    ax.plot_surface(X/norm_X, Y, Z/norm_Z, color = c, alpha = .65)
                ax.set_xlabel(f"PC{PC+1}")
                ax.set_zlabel(f"Delta PC{PC_diff+1}")
                ax.set_ylabel("time")
                ax.set_title(f"effect of PC{PC+1} on PC{PC_diff+1} response")
                ax.set_box_aspect([1,2, 1])
                if self.show: plt.show()
            
#################################################################################################

    """ plot correlations between inputs and principle components"""
    def plot_corrs(self):       
        inps = ["GO", "LICK", "REWARD"]
        points = int(self.inp_cross.shape[-1])
        cross_corr_tick_labels = np.linspace(-points, points, 5)/2
        cross_corr_ticks = np.arange(len(cross_corr_tick_labels))
        xax = np.linspace(0, cross_corr_ticks[-1], points)
        mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green'])
        fig, ax = plt.subplots(4, 1, figsize = (10, 20), tight_layout = True)
        l1 = [f" PC{PC}" for PC in [1,2,3]]
        l2 = [f" PC{PC} auto correlation" for PC in [1,2,3]]
        ax[3].plot(self.inp_corrs, '-o', label = l1)

        for i, name in enumerate(inps):
            ax[i].plot(xax, self.inp_cross[i, :,:].T, label = l1)
            ax[i].set_xticks(cross_corr_ticks)
            ax[i].set_xticklabels(cross_corr_tick_labels)
            ax[i].set_title(f"Cross-correlation betwen {name} & PCs")
            ax[i].plot(xax, self.auto_corrs.T, '--', label = l2, alpha = .5)
            ax[i].legend(loc = "upper right")
            
        ax[3].set_xticks(np.arange(len(inps)));       ax[3].set_xticklabels(inps)
        ax[3].set_xlabel("input");                    ax[3].legend(loc = "upper right")
        ax[3].set_title("Correlation coeffs");
        mpl.rcParams['axes.prop_cycle'] =\
        cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',\
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']);        
        if self.show: plt.show()
        
    """ plot angle between LSTM output and each node in each LSTM gate"""
    def plot_net_angles(self, power = 3, sort = False):
        leg = ["PC1", "PC2", "PC3"]
        plt.plot(self.act_proj.T, '-o');                                plt.legend(leg)
        plt.xlabel("Action");plt.title("contribution to action");       
        if self.show: plt.show()
        for i, gate in enumerate(self.gate_names):
            contribution = self.gate_proj[:, i, :]**power
            plt.plot(np.sort(contribution, -1).T, '-o') if sort == True else plt.plot(contribution.T, '-o')
            plt.xlabel(f"Neuron (Sorted = {sort})");                    plt.legend(leg)
            plt.title(f"Contribution to {gate} to the power {power}");  
            if self.show: plt.show()
            
    """ plot eigenvalues of gates and their interactions"""
    def plot_gate_eigen(self):
        W_rand = np.random.randn(self.hid_dim, self.hid_dim)/np.sqrt(self.hid_dim)
        Ws = [self.Wi_cpu, self.Wo_cpu, self.Wc_cpu, self.Wf_cpu, W_rand]
        eigs = [lin.eig(w, right = True) for w in Ws];        
        eig_names = self.gate_names + ["Random guassian W / sqrt(N)"]  
        fig, ax = plt.subplots(2, 1, figsize = (8, 10), tight_layout = True)
        for v in eigs:
            v = v[0][abs(v[0]).argsort()[::-1]]   
            real_vals = np.real(v); imag_vals = np.imag(v)
            ax[0].scatter(real_vals, imag_vals, s = 85)
            ax[1].plot(real_vals**2 + imag_vals**2, '-o')    
        ax[0].set_title("eigenvalue real vs imaginary");      ax[1].set_title("Power spectrum")
        ax[0].set_xlabel("real");                             ax[1].set_xlabel("eigenvalue #")       
        ax[0].legend(eig_names);                              ax[1].legend(eig_names);               
        ax[0].set_ylabel("imaginary");                        
        if self.show: plt.show()

    """ Plot dynamic cellgate weight matrix variability"""
    def plot_weight_variability(self):
        sns.heatmap(self.split_var); plt.title(f"variance due to {self.col} (total variance: {self.split_var.sum():.3f})"); 
        if self.show: plt.show()
        sns.heatmap(self.time_var); plt.title(f"variance across time (total variance: {self.time_var.sum():.3f})"); 
        if self.show: plt.show()
        sns.heatmap(self.split_x_time_var); plt.title(f"ELEMENT-WISE: (normed {self.col} var) x (normed time var) \n(norm of result: {lin.norm(self.split_x_time_var):.3f})")
        if self.show: plt.show()
                
    """ Plot dynamic cellgate matrix dynamics"""
    def plot_eigen_variability(self):
        for val, vec, name in zip([self.mod_eig_vals_R, self.mod_eig_vals_I], [self.mod_eig_vec_R, self.mod_eig_vec_I], ['real', 'complex']):            
            self.make_standard_plot(ys = val.var(-1), cols = self.split_cols, \
                alphs = [.5] * self.split_num, leg =  self.split_leg, xlab = "time", plots = ['scatter'] * self.split_num, \
                title = f"variance between (modulated cellgate) {name} eigenvalues through time", traj = True)

            fig, ax = plt.subplots(1,max(2, self.split_num), figsize = (20,5), tight_layout = True)
            for split in range(self.split_num):
                ax[split].plot(np.abs(val[split]), linewidth = 3, alpha = .25)
                ax[split].set_title(f"{self.col} {split} (modulated cellgate) {name} abs(eigenvalues) through time"); ax[split].set_xlabel("time")
                ax[split].set_ylim([0, np.max(val**2)])
            if self.show: plt.show()
        
            # self.make_standard_plot(ys = (val**2).mean(-1), cols = self.split_cols, \
            #     alphs = [.5] * self.split_num, leg =  self.split_leg, xlab = "time", plots = ['scatter'] * self.split_num, \
            #     title = f"(modulated cellgate) {name} mean(eigenvalues^2) through time", traj = True)

            # self.make_standard_plot(ys = val.var(1), cols = self.split_cols, \
            #     alphs = [.5] * self.split_num, leg =  self.split_leg, xlab = "eigenvalue", plots = ['scatter'] * self.split_num, \
            #     title = f"variance of each (modulated cellgate) {name} eigenvalue across time")
            
###############################################################################################################################################

    def layer_theta_traj(self, est, dist = None):
        reps = np.floor(self.PGO_flat.shape[0]/ self.theta_traj_N).astype(int)
        print(f"{reps} reps")
        theta_est = np.zeros((reps,  self.theta_traj_N))
        theta_dist = 0 if dist is None else np.zeros((dist.shape[0],  self.theta_traj_N))
        for i in range(reps): 
            s = i * self.theta_traj_N
            e = (i+1) * self.theta_traj_N
            theta_est[i, :] = est[s:e] 
            theta_dist = theta_dist + (0 if dist is None else dist[:, s:e])           
        return theta_est, theta_dist/reps, theta_est.mean(0), theta_est.var(0)

    def get_p_TH_theta(self):
        return [self.theta_dist_flat[:, np.abs(self.flow_theta_flat - theta) < 1e-3].mean(1) for theta in self.bayes_range]

    def plot_fixed_theta_traj(self, s, e, SVG = False):
        _, bayes_theta_dist, bayes_mu, bayes_var = self.layer_theta_traj(self.flow_theta_flat, self.theta_dist_flat)
        # _, _, factorized_bayes_mu, factorized_bayes_var = self.layer_theta_traj(self.factorized_theta_flat)
        _, _, net_mu, net_var = self.layer_theta_traj(self.net_theta_flat)
        # _, _, input_mu, input_var = self.layer_theta_traj(self.input_flat[1])        
        # _, _, MF_mu, MF_var = self.layer_theta_traj(self.MF_theta_flat[25])
        xax = np.arange(self.theta_traj_N)[:e-s]

        # ax = plt.subplots(figsize = (30, 4))
        # sns.heatmap(bayes_theta_dist[:, s:e]); ax.invert_yaxis()
        x = xax
        y = self.bayes_range
        # norm = cm_cols.SymLogNorm(linthresh=1e-5, vmin=0, vmax=1)
        norm = cm_cols.Normalize(vmin=0)#, vmax =1)
        # plt.pcolormesh(x, y, bayes_theta_dist[:, s:e], norm=norm,  cmap=plt.get_cmap("viridis"), shading = 'nearest')

        # cmap_white_to_darker_green = LinearSegmentedColormap.from_list('white_to_darker_green', ['#FFFFFF', '#006400'])
        # plt.pcolormesh(x, y, bayes_theta_dist[:, s:e], norm=norm, cmap=cmap_white_to_darker_green, shading='nearest')
        # plt.show()

        ax = plt.subplots(figsize = (30,5))
        # dist_var =(bayes_theta_dist*(self.bayes_range[:,None]**2) -(bayes_theta_dist*self.bayes_range[:,None])**2).sum(0)
        # plt.fill_between(xax, (bayes_mu[s:e] - dist_var[s:e]), (bayes_mu[s:e] + dist_var[s:e]), alpha=.5)
        #sns.heatmap(bayes_theta_dist[:, s:e]); ax.invert_yaxis()
        # self.theta_fill(xax, MF_mu, MF_var, s, e, color = 'C3', label = "window (5x mean stim duration)")   
        # self.theta_fill(xax, bayes_mu, bayes_var, s, e, color = 'C2', label = "joint bayes", linestyle = '--')   
        # self.theta_fill(xax, factorized_bayes_mu, factorized_bayes_var, s, e, color = 'C3', label = "factorized bayes", linestyle = '--')   
        # self.theta_fill(xax, net_mu, net_var, s, e, color = 'C0', label = "network")   
        plt.plot(xax, bayes_mu[s:e],color = 'C2', label = "joint bayes", linestyle = '--')   
        plt.plot(xax, net_mu[s:e], color = 'C0', label = 'network')
        
        plt.plot(xax,self.PGO_flat[:self.theta_traj_N][s:e], label = "theta", linewidth = 2, c = 'C1')
        title = "fixed theta trajectory" + (" (factorized)" if self.factorize else "")
        plt.title(title); plt.legend(); 
        if SVG:
            self.save_SVG(title)
        plt.show()
        
        # fig, ax = plt.subplots(figsize = (25,10))
        # # s = 0
        # e = 200
        # plt.plot(self.PSAFE_flat[s:e])
        # plt.plot(1-np.exp(-self.flow_belief_flat[s:e].astype(float)))
        # plt.plot(1-np.exp(-self.net_belief_flat[s:e].astype(float)))
        # plt.legend(["State", "bayesian belief", "network belief"])
        # if self.show: plt.show()

    def theta_fill(self, xax, mu, var, s, e, color, label, linestyle = '-', alpha = 1):
        plt.fill_between(xax, (mu[s:e]-var[s:e]), (mu[s:e]+var[s:e]), alpha = .5, color = color)
        l = plt.plot(xax, mu[s:e], alpha = alpha, color = color, label = label, linestyle = linestyle)  
        return l
  
    """ plot bayesian posterior distribution and compare to network regression to PGO """
    # def plot_bayes_through_steps(self, end = 5000):
    #     fig, ax = plt.subplots(1,1, figsize = (30, 10), tight_layout = True)
    #     pred_from = self.C_gate_flat # predict from cell only 
    #     net_pred_PGO = self.regressor['TRUE PGO'].predict(pred_from.T).T[-1]
    #     plt.plot(self.bayes_resolution - net_pred_PGO[:end]*self.bayes_resolution, alpha = .75)        
    #     plt.plot(self.bayes_resolution - self.bayes_PGO_flat[-1, :end]*self.bayes_resolution, alpha = .75)        
    #     plt.plot(self.bayes_resolution - self.PGO_flat[:end]*self.bayes_resolution, '--', linewidth = 3)
    #     sns.heatmap(np.flip(self.dist_flat[:,:end], 0), yticklabels = np.round(np.linspace(1, 0, self.bayes_resolution), 2))
    #     if self.show: plt.show()
        
    #     end = 5000
    #     #self.preprocess_data(testing)
    #     self.cog_map = True
    #     # self.run_PCA()
    #     #self.get_bayes()
    #     self.get_bayes_flow();
    #     self.get_indices(flatten = True)
    #     self.flatten_bayes()
    #     fig, ax = plt.subplots(1,1, figsize = (30, 10), tight_layout = True)
    #     # net_pred_PGO = self.PGO_reg.predict(self.pred_from.T).T
    #     # plt.plot(net_pred_PGO[:end], alpha = .75)     
    #     plt.plot(self.PC_flat[1,:end], '--', linewidth = 3)        
    #     plt.plot(self.PGO_flat[:end], '--', linewidth = 3)
    #     if self.show: plt.show(); 
        
    #     self.plot_PC_phase_space()
              
    #     fig, ax = plt.subplots(1,1, figsize = (30, 5), tight_layout = True)
    #     plt.plot(20*self.PGO_flat[:end], '--', linewidth = 3)
    #     bayes_dist =np.vstack(self.flatten_trajectory(self.raw_to_structured(self.joint_dist_log.sum(0), dim = 2)[self.trial_inds]))
    #     sns.heatmap(bayes_dist[:end].T)
    #     if self.show: plt.show();
    #     fig, ax = plt.subplots(1,1, figsize = (30, 5), tight_layout = True)
    #     plt.plot(20*self.PGO_flat[:end], '--', linewidth = 3)
    #     net_dist = self.dist_reg.predict_proba(self.pred_from.T)
    #     sns.heatmap(net_dist[:end, :].T)
    #     if self.show: plt.show()
    #     sns.heatmap(self.correlate(bayes_dist.T, net_dist.T));
        
###############################################################################################################################################

    def make_color_bar(self):
        clb = plt.colorbar(self.Cmap,  ax=plt.gca())
        clb.set_label('P(GO|unsafe)\n',  fontdict=self.font_dict)
        
    def save_SVG(self, name):
        plt.savefig(f"/home/johns/anaconda3/envs/PFC_env/PFC/Data/{name}.svg")
    
    def addlabels(self, x,y, s):
        plt.text(x,y,s)
        
        
###############################################################################################################################################

    # def estimate_crossvalidation(self, folds = 10):
    #     self.run_regressions(regress_to = 'bayes', regress_from = 'STM')                       
    #     self.get_indices(cross_validation = None, flatten = True, til_action = True, full_prep = True)
    #     all_inds = self.ACTION_inds
    #     N = len(self.trial_inds)
    #     ind_range = np.arange(N)
    #     self.net_acc, self.bayes_acc, self.net_bayes_state_R2, self.net_bayes_theta_R2, self.net_theta_R2, self.bayes_theta_R2 = \
    #         [np.zeros(folds) for _ in range(6)]
    #     for f in tqdm(range(folds)):
    #         train_inds = np.random.rand(N) > .8
    #         test_inds = 1 - train_inds

    #         """ train """ 
    #         self.trial_inds = inds[np.where(train_inds)[0]]
    #         self.pred_from = self.flatten_trajectory(self.net_output[self.trial_inds])
    #         self.PGO_flat = self.flatten_trajectory(self.step_PGO[self.trial_inds])
    #         self.PSAFE_flat = self.flatten_trajectory(self.safe_backbone[self.trial_inds])
    #         self.Q_reg = LogisticRegression().fit(self.pred_from.T, self.PSAFE_flat)
    #         self.PGO_reg = LinearRegression().fit(self.pred_from.T, self.PGO_flat)  

    #         """ test """ 
    #         self.trial_inds = inds[np.where(test_inds)[0]]
    #         self.pred_from = self.flatten_trajectory(self.net_output[self.trial_inds])
    #         self.PGO_flat = self.flatten_trajectory(self.step_PGO[self.trial_inds])
    #         self.PSAFE_flat = self.flatten_trajectory(self.safe_backbone[self.trial_inds])
    #         self.net_belief_flat = -np.log(1-self.Q_reg.predict_proba(self.pred_from.T)[:,1])
    #         self.net_theta_flat = self.PGO_reg.predict(self.pred_from.T)
    #         self.flow_theta_flat = self.flatten_trajectory(self.flow_theta_structured[self.trial_inds]).astype(np.float64)
    #         self.flow_belief_flat = self.flatten_trajectory(self.flow_belief_structured[self.trial_inds]).astype(np.float64)
            
    #         """ logging """
    #         net_prob = self.log_to_prob(self.net_belief_flat) 
    #         bayes_prob = self.log_to_prob(self.flow_belief_flat)
    #         self.net_acc[f] = ((net_prob >  0.5) == self.PSAFE_flat).mean()
    #         self.bayes_acc[f] = ((bayes_prob > 0.5) == self.PSAFE_flat).mean()
    #         self.net_bayes_state_R2[f] = self.correlate(net_prob, bayes_prob)**2
    #         self.net_bayes_theta_R2[f] = self.correlate(self.net_theta_flat, self.flow_theta_flat)**2
    #         self.net_theta_R2[f] = self.correlate(self.net_theta_flat, self.PGO_flat)
    #         self.bayes_theta_R2[f] = self.correlate(self.flow_theta_flat, self.PGO_flat)


    def plot_Qrange(self, get = False):
        if get:
            self.get_indices(full_prep=True)
        counts, max_diffs, mean_diffs = [np.zeros(self.PGO_N)
        for _ in range(3)]

        p = self.step_PGO[self.trial_inds]
        q = self.Q_values[self.trial_inds]
        a = self.act_times[self.trial_inds]
        for i, (PGO_traj, Q_traj, acts) in enumerate(zip(p, q, a)):
            Q_diff = Q_traj[1] - Q_traj[0]
            if (acts is not None) and (acts > 0):
                Q_diff = Q_diff[:acts.astype(int)+1].cpu().numpy()
                max_diff =  Q_diff.max() - Q_diff.min()
                mean_diff = np.diff(Q_diff).mean()


                p_i = np.where(self.PGO_range == PGO_traj[0])[0]
                counts[p_i] += 1
                max_diffs[p_i] += max_diff
                mean_diffs[p_i] += mean_diff
        max_diffs = max_diffs/counts
        mean_diffs = mean_diffs/counts

        fig, ax = plt.subplots(1, 3, tight_layout = True, figsize = (5, 2))
        ax[0].plot(self.PGO_range, max_diffs)
        ax[0].set_title(r"$\langle Q_t \text{ range} \rangle_{\text{trials} | \theta}$")
        ax[0].set_ylabel(r"max $Q_t$ - min $Q_t$")
        ax[1].plot(self.PGO_range, mean_diffs)
        ax[1].set_title(r"$\langle Q_t \text{ diff}\rangle_{\text{trials} | \theta}$")
        ax[1].set_ylabel(r"$\mathbb{E}(Q_i - Q_{i-1})$")        
        ax[2].plot(self.PGO_range, max_diffs/mean_diffs)
        ax[2].set_title(r"$\langle \dfrac{Q_t \text{ range}}{Q_t \text{ diff}} \rangle_{\text{trials} | \theta}$")
        ax[2].set_ylabel("dynamic range")
        for i in range(3):
            ax[i].set_xlabel(r"$\theta$")
            ax[i].set_xticks([])

















    def estimate_crossvalidation(self, folds = 10):
        self.run_regressions(regress_to = 'bayes', regress_from = 'STM')                       
        self.get_indices(cross_validation = None, flatten = True, til_action = True, full_prep = True)

        bayes_prob = self.log_to_prob(self.flow_belief_flat)
        bayes_DV =  bayes_prob - self.flow_opt_thresh_flat
        T_Y = self.prob_to_LLR(self.flow_opt_thresh_flat)
        Q_Y = self.log_to_LLR(self.flow_belief_flat)
        PGO_Y = self.flow_theta_flat
        SAFE = self.PSAFE_flat
        PGO = self.PGO_flat
        X = self.pred_from     
        t_back = 10

        self.net_acc, self.bayes_acc, self.net_bayes_state_R2, self.net_bayes_theta_R2, self.net_theta_R2, self.bayes_theta_R2, self.DV_R2 = \
            [np.zeros(folds) for _ in range(7)]
        self.net_acc_til_act = np.zeros((folds, t_back))
        for f in tqdm(range(folds)):
            train_bool = np.random.rand(len(self.pre_act)) > .8
            train_inds = np.where(train_bool*self.pre_act)[0]
            test_inds = np.where((~train_bool)*self.pre_act)[0]

            """ train """ 
            Q_reg = LinearRegression().fit(X[:,train_inds].T, Q_Y[train_inds])
            T_reg = LinearRegression().fit(X[:,train_inds].T, T_Y[train_inds])  
            PGO_reg = LinearRegression().fit(X[:,train_inds].T, PGO_Y[train_inds])  

            """ test """ 
            net_theta = PGO_reg.predict(X[:, test_inds].T)
            net_state = self.LLR_to_prob(Q_reg.predict(X[:, test_inds].T))
            net_thresh = self.LLR_to_prob(T_reg.predict(X[:, test_inds].T))

            net_DV = net_state - net_thresh

            """ logging """
            self.net_acc[f] = ((net_state >  0.5) == SAFE[test_inds]).mean()
            self.bayes_acc[f] = ((bayes_prob[test_inds] > 0.5) == SAFE[test_inds]).mean()
            self.net_bayes_state_R2[f] = self.correlate(net_state, bayes_prob[test_inds])**2
            self.net_bayes_theta_R2[f] = self.correlate(net_theta, PGO_Y[test_inds])**2
            self.bayes_theta_R2[f] = self.correlate(PGO_Y[test_inds], PGO[test_inds])
            self.net_theta_R2[f] = self.correlate(net_theta, PGO[test_inds])
            self.DV_R2[f] = self.correlate(net_DV, bayes_DV[test_inds])

            for t in range(t_back):
                til_act_inds = np.where((~train_bool)*(self.time_til_act == -t))[0]
                net_state_til_act = self.LLR_to_prob(Q_reg.predict(X[:, til_act_inds].T))
                self.net_acc_til_act[f, t] = ((net_state_til_act >  0.5) == SAFE[til_act_inds]).mean()



""" optimal reward rate as a function of estimate uncertainty and estimate error "" "
     keep_mean = False
     for p in [.5, .7, .9]:
         
         ind = np.argmin((self.bayes_range - p)**2)
         if ind % 2 != 0:
             ind = ind - 1
             p = self.bayes_range[ind]
         

         p_theta = np.zeros((self.bayes_resolution, 1)); p_theta[ind] = 1
         weighted_p_safe = (self.total_ana_rews*p_theta).sum(0)
         self.weighted_rew_rate = (p_theta*(self.total_ana_rews/(self.total_ana_acts + self.ITI_mean))).sum(0)
         #plt.plot(self.weighted_rew_rate);
         N = min(self.bayes_resolution - ind, ind)
         rep = 50
         for r in range(rep):
             p_theta = np.zeros((self.bayes_resolution, 1)); p_theta[ind] = 1

             noise = (np.random.rand(N))/((1+r)**2)
             A = 1 + (1-keep_mean)
             noise = np.concatenate((noise/A, A*noise))
             
             p_theta[ind:N+ind] = p_theta[ind:N+ind] + noise[:N, None]
             p_theta[ind-N:ind] = p_theta[ind-N:ind] + noise[N:, None]
             
             p_theta = np.maximum(p_theta, 0)
             p_theta = p_theta/p_theta.sum()
         #    print(p, (p_theta*self.bayes_range[:,None]).sum())
             error = np.abs(p - (p_theta*self.bayes_range[:,None]).sum())*2e1
       
             weighted_p_safe = (self.total_ana_rews*p_theta).sum(0)
             self.weighted_rew_rate = (p_theta*(self.total_ana_rews/(self.total_ana_acts + self.ITI_mean))).sum(0)
     #        plt.plot(self.weighted_rew_rate, c = self.Cmap.to_rgba((rep-r)/rep), alpha = .5);
             plt.plot(self.weighted_rew_rate, c = self.Cmap.to_rgba(error), alpha = .8)
     plt.xlim([1,15]); plt.ylim([.02, None])
     plt.xlabel("threshold"); plt.ylabel("reward rate"); 
     plt.title("optimal reward rate as a function of theta estimate error")
     plt.show()
     """