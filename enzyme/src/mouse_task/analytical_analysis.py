# import numpy as np; from enzyme.src.mouse_task.memory_weighting_theory import memory_weighting_theory
import numpy as np; 
from enzyme.src.mouse_task.bayesian_analysis import bayesian_analysis
# from enzyme.src.mouse_world.environment import MouseWorld

class analytical_analysis(bayesian_analysis):
    def __init__(self, **params):
        self.__dict__.update(params)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Analytical and numerical simulation
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """ get analytical and numerical reward rates"""
    def get_rew_rate(self):
        """"
        thresh_N: Waiting time after the last nogo until action
        """
        self.ana_waits, self.numer_waits = [np.zeros((self.thresh_N, self.trial_dur)) for _ in range(2)]
        self.numer_acts, self.ana_acts, self.numer_rews, self.ana_rews = [np.zeros((self.PGO_N, self.thresh_N)) for _ in range(4)]
        for self.p_i, self.curr_PGO in enumerate(self.PGO_range):
            # self.numer_trial_loop()
            self.ana_trial_loop()
            
        self.ana_postprocessing()
        # self.numer_waits = self.ana_waits.copy()
        self.numer_postprocessing()
        self.get_lower_bounds()
           
    def numer_postprocessing(self):        
        num_denom = (self.numer_acts + self.analytical_ITI_mean)
        self.numer_rates = self.numer_rews/num_denom
        self.numer_max_rate = np.max(self.numer_rates, 1)
        self.numer_xax, self.numer_max_thresh = self.thresh_to_wait(self.numer_rates, self.numer_waits)
       
    def ana_postprocessing(self):        
       an_denom = (self.ana_acts + self.analytical_ITI_mean + self.prem_dur * (1-self.ana_rews) )      
       self.ana_rews = self.rews.max()*self.ana_rews + self.rews.min()*(1-self.ana_rews)      # includes negative reward
       
       self.ana_rates = self.ana_rews/an_denom
       self.ana_max_rate = np.max(self.ana_rates, 1)  # max rate for each PGO
       self.ana_xax, self.ana_max_thresh = self.thresh_to_wait(self.ana_rates, self.ana_waits)

    def numer_trial_loop(self, N = 10000):
        """
        Numerically (sampling based) compute the wait from last nogo distribution
        """

        stims = np.random.binomial(1, self.curr_PGO,  size = (N, self.trial_dur))
        exp_durs = np.random.exponential(self.exp_mean, size = (N,1)).astype(int)
        exp_durs = np.clip(exp_durs, a_min = None, a_max = self.trial_dur - self.thresh_N - 2)
        wait_from_last, wait_from_start = [np.zeros((N, self.thresh_N, self.trial_dur)) for _ in range(2)]
        consec_gos = np.zeros((N,1))
        numer_range = np.arange(N)
      
        for step in range(self.trial_dur - 1):
            stim_ongoing = (step <= exp_durs)
            NOGO = (stims[:, step, None] == 0)

            # GO is not(stim_ongoing and NOGO) = (not stim_ongoing) or (not NOGO) = stim_finished or not NOGO
            GO = 1 - stim_ongoing * NOGO
            
            consec_gos = GO * (consec_gos + 1)
            C = consec_gos.squeeze().astype(int) - 1
            prob = (consec_gos.astype(int) == self.thresh).astype(int)            
            wait_from_last[numer_range, :, np.clip(C, a_min = 0, a_max = None)] += prob
            # wait_from_start[:, :, max(1, step)] += prob
            wait_from_start[:, :, step] += prob
        
        wait_from_last = wait_from_last/wait_from_last.sum(-1, keepdims=True)
        wait_from_start = wait_from_start/wait_from_start.sum(-1, keepdims=True)
            
        self.numer_waits = wait_from_last.mean(0)
        self.numer_acts[self.p_i, :] = (wait_from_start * (self.x_axis-1)).mean(0).sum(-1)
        failure_prob = wait_from_start.cumsum(-1)[numer_range, :, exp_durs.squeeze()]

        self.numer_rews[self.p_i, :] = ((1 - failure_prob)*self.rews.max() + failure_prob*self.rews.min()).mean(0)

    """ analytical solution"""
    def ana_trial_loop(self):
        action_times, reward_prob = self.ana_of_consec_GO()

        for N_i in range(self.thresh_N):     
            wait_from_last = np.zeros(len(self.x_axis))
            wait_from_last[N_i] = 1     

            self.ana_waits[N_i, :] = wait_from_last

            # picks out the right action time via the product, effectively implementing a mask
            self.ana_acts[self.p_i, N_i] = sum(wait_from_last*action_times)
            self.ana_rews[self.p_i, N_i] = sum(wait_from_last*reward_prob)
    
    def ana_of_consec_GO(self):
        action_times, reward_prob = [np.zeros(len(self.x_axis)) for _ in range(2)]
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
    
    """ post processing"""
    def get_lower_bounds(self):
        mean_rate_over_PGO = self.ana_rates.mean(0)
        _, self.fixed_thresh = self.thresh_to_wait(mean_rate_over_PGO, self.ana_waits)
        self.fixed_thresh_rate = self.ana_rates[:, mean_rate_over_PGO.argmax()]
        self.fixed_max_thresh = np.array([self.fixed_thresh]*self.PGO_N)

        a = self.exp_mean 
        c = self.analytical_ITI_mean
        p = (np.sqrt((a+1)**2 + 4*a*c) - (a+1)) / (4 * a* c)
        rand_rate = ((1-p)**a) / (c + 1 /p)
        
        self.rand_max_thresh = np.array([1/p]*self.PGO_N) 
        self.rand_max_rate = np.array([rand_rate]*self.PGO_N)
         
    def thresh_to_wait(self, rew_rate, prob):
        """
        rew_rate: (#PGO, #thresh_N)
        prob: (#t_wait, #thresh_N): indicator arraw of whether a given waiting time is the threshold
        """
        max_loc = np.argmax(rew_rate, axis = -1)  # opt. threshold for each context
        weighted_dist = prob*self.x_axis[None,:]
        new_xax = weighted_dist.sum(-1)  # waiting times
        new_max = (weighted_dist[max_loc, :]).sum(-1)
        return new_xax, new_max
    
    def get_total_analytical_optim(self):
        # gets analytical optimality for large range of PGO 
        self.ana_waits = np.zeros((self.thresh_N, self.trial_dur))
        self.ana_acts, self.ana_rews = [np.zeros((self.bayes_resolution, self.thresh_N)) for _ in range(2)]
        # fills up the rew array
        for self.p_i, self.curr_PGO in enumerate(self.bayes_range):
            self.ana_trial_loop()              
        self.ana_postprocessing()
        self.total_ana_rews = self.ana_rews.copy()
        self.total_ana_acts = self.ana_acts.copy()
        self.total_ana_rates = self.ana_rates.copy()
        self.total_opt_thresh = np.argmax(self.ana_rates, 1)
        self.total_opt_belief = self.ana_rews[self.bayes_range_inds, self.total_opt_thresh]
        # gets standard analytical optimality for experienced PGO
        self.get_rew_rate()        
        self.opt_belief = self.ana_rews[np.arange(self.PGO_N), np.argmax(self.ana_rates, 1)]
        
    def get_total_analytical_optim_v_ITI(self, Max = 50, N = 5, units = "belief"):
        opt_threshes = np.zeros((Max - 1, self.PGO_N))
        threshs = np.arange(0, Max-1) + 1
        OG_ITI_mean = self.analytical_ITI_mean
        jump = Max//N
                
        for i, self.analytical_ITI_mean in enumerate( threshs ):
            self.get_rew_rate()
            if units == "N":
                opt_threshes[i, :] =  np.argmax(self.ana_rates, 1)
            if units == "belief":
                opt_threshes[i, :] = self.ana_rews[np.arange(self.PGO_N), np.argmax(self.ana_rates, 1)]    

        self.ITIs_for_optim_range = np.zeros((N, 2))
        self.ITIs_for_optim = np.zeros((N, self.PGO_N))
        for i in range(N):
            self.ITIs_for_optim_range[i, :] = [jump*i + 1, (i+1)*jump] 
            self.ITIs_for_optim[i, :] = opt_threshes[i*jump:(i+1)*jump, :].mean(0)        
        self.analytical_ITI_mean = OG_ITI_mean