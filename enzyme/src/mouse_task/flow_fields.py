from tqdm import tqdm; from enzyme.src.mouse_task.net_dynamics import net_dynamics; import numpy as np; import pylab as plt; import seaborn as sns
from scipy.ndimage import convolve; from sklearn.linear_model import LinearRegression
 
"""class for simulating flow field dynamics """ 
class flow_fields(net_dynamics):
    def __init__(self, **params):
        self.__dict__.update(params)

    def get_phase_space(self, til_action = True, from_action = False, x_round = 1, y_round = 1, min_count = 0, get_inds = True, get_grid_flow = True, disable = False, get_R = False):    
        if get_inds:
            self.get_indices(col = 'all', flatten = True, til_action = til_action, from_action = from_action)
        PC_round = self.PC_prev.copy()
        if type(x_round) == int:
            PC_round[0] = PC_round[0].round(x_round)
            x_unique = np.unique(PC_round[0])
        else:
            PC_round[0] = self.round_to_nearest(PC_round[0], x_round)
            x_unique = x_round
        if type(y_round) == int: 
            PC_round[1] = PC_round[1].round(y_round)
            y_unique = np.unique(PC_round[1])
        else:
            PC_round[1] = self.round_to_nearest(PC_round[1], y_round)
            y_unique = y_round
        self.uniques = np.array([x_unique, y_unique], dtype = object)    

        self.dPC1, self.dPC2, self.PC_count, self.grid_R1, self.grid_R2 = [np.zeros((2, len(self.uniques[0]), len(self.uniques[1]))) for _ in range(5)]       # dims: [Stim, PC1, PC2]
        self.PGO_mu, self.PSAFE_mu = [np.zeros((2, len(self.uniques[0]), len(self.uniques[1]))) for _ in range(2)]                                       # dims: [Stim, PC1, PC2]
        self.Y_mu, self.dY_mu, self.dX_mu = [np.zeros((2, len(self.uniques[0]))) for _ in range(3)]
        self.get_GO_NOGO_inds()
                
        for stim_i, stim_inds in enumerate([self.NOGO_inds, self.GO_inds]):
            if get_R:
                R1_M = LinearRegression().fit( self.net_belief_diff[stim_inds, None] , self.PC_diff[0, stim_inds])
                R2_M = LinearRegression().fit( self.net_theta_diff[stim_inds, None] , self.PC_diff[1, stim_inds])

            for PC1_i, PC1 in enumerate(tqdm(self.uniques[0], desc = 'computing vector fields', disable = disable)):
                if get_grid_flow:
                    for PC2_i, PC2 in enumerate(self.uniques[1]):
                        PC_unique_inds = self.where((PC_round[0, :] == PC1)*(PC_round[1, :] == PC2))
                        inds = np.intersect1d(PC_unique_inds, stim_inds)
                        if len(inds) > min_count:
                            self.dPC1[stim_i, PC1_i, PC2_i] = self.PC_diff[0, inds].mean(-1) /  self.PC_diff[0].std()
                            self.dPC2[stim_i, PC1_i, PC2_i] = self.PC_diff[1, inds].mean(-1) /  self.PC_diff[1].std()
                            self.PC_count[stim_i, PC1_i, PC2_i] = len(inds)
                            self.PGO_mu[stim_i, PC1_i, PC2_i] = self.PGO_flat[inds].mean()
                            self.PSAFE_mu[stim_i, PC1_i, PC2_i] = self.PSAFE_flat[inds].mean()
                            if get_R:
                                self.grid_R1[stim_i, PC1_i, PC2_i] = self.correlate(R1_M.predict(self.net_belief_diff[inds, None]), self.PC_diff[0, inds])
                                self.grid_R2[stim_i, PC1_i, PC2_i] = self.correlate(R1_M.predict(self.net_theta_diff[inds, None]), self.PC_diff[1, inds])
                            
                # get X axis means, diffs and vars
                inds = np.intersect1d(self.where(PC_round[0, :] == PC1), stim_inds)
                self.Y_mu[stim_i, PC1_i] = self.PC_prev[1, inds].mean()
                self.dX_mu[stim_i, PC1_i] =  self.PC_diff[0, inds].mean() /  self.PC_diff[0].std()
                self.dY_mu[stim_i, PC1_i] =  self.PC_diff[1, inds].mean() /  self.PC_diff[1].std()
                    
    def get_quadrant_space(self, y_axis = "latent"):    
        self.get_indices(col = 'all', flatten = True, til_action = True)
        # self.get_GO_NOGO_inds()                
        self.flow_quadrants, self.flow_quadrants_var = [np.zeros((3, 2, 3, 3, 2)) for _ in range(2)]
        # dims: agent (net/joint/factorized), stim (NOGO/GO), x_axis (belief), y_axis (theta), diff_variable (belief/theta)

        if y_axis == 'latent':
            net_where = [self.net_belief_prev, self.net_theta_prev]
            net_what = [self.net_belief_diff, self.net_theta_diff]
        else: 
            # for visualizing belief vs trial response times
            net_where = [self.net_belief_prev, self.RT_prev]
            net_what = [self.net_belief_diff, self.RT_diff]
        
        bayes_where = [self.flow_belief_prev, self.flow_theta_prev]
        factorized_where = [self.factorized_belief_prev, self.factorized_theta_prev]

        bayes_what = [self.flow_belief_diff, self.flow_theta_diff]
        factorized_what = [self.factorized_belief_diff, self.factorized_theta_diff]
        where_data = [net_where, bayes_where, factorized_where]
        what_data = [net_what, bayes_what, factorized_what]
        
        # write to self
        # self.theta_top = theta_top; self.theta_bot = theta_bot; self.belief_top = belief_top; self.belief_bot = belief_bot
        
        for agent_i, (self.agent_where, self.agent_what) in enumerate(zip(where_data, what_data)):
            for stim_i, stim_inds in enumerate([self.NOGO_inds, self.GO_inds]):
                self.what_belief, self.where_belief, self.what_theta, self.where_theta = self.agent_what[0], self.agent_where[0],  self.agent_what[1], self.agent_where[1]

                size = len(self.where_belief)
                sorted_belief = np.sort(self.where_belief)
                belief_top = (sorted_belief[int(.6*size)])
                belief_bot = (sorted_belief[int(.3*size)])
                
                size = len(self.where_theta)
                sorted_theta = np.sort(self.where_theta)
                theta_top = (sorted_theta[int(.6*size)])
                theta_bot = (sorted_theta[int(.3*size)])
                        
                W = self.where((self.where_belief > belief_top)*(self.where_theta > theta_top))
                self.top_right_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 2, 2, 0], self.flow_quadrants[agent_i, stim_i, 2, 2, 1] =\
                    self.what_belief[self.top_right_inds].mean(), self.what_theta[self.top_right_inds].mean()        
                    
                self.flow_quadrants_var[agent_i, stim_i, 2, 2, 0], self.flow_quadrants_var[agent_i, stim_i, 2, 2, 1] =\
                    self.what_belief[self.top_right_inds].var(), self.what_theta[self.top_right_inds].var()      
                    
                    
                W =  self.where((self.where_belief < belief_bot)*(self.where_theta > theta_top))
                self.top_left_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 0, 2, 0], self.flow_quadrants[agent_i, stim_i, 0, 2, 1] =\
                    self.what_belief[self.top_left_inds].mean(), self.what_theta[self.top_left_inds].mean()         
                    
                self.flow_quadrants_var[agent_i, stim_i, 0, 2, 0], self.flow_quadrants_var[agent_i, stim_i, 0, 2, 1] =\
                    self.what_belief[self.top_left_inds].var(), self.what_theta[self.top_left_inds].var()  


                W = self.where((self.where_belief > belief_top)*(self.where_theta < theta_bot))
                self.bot_right_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 2, 0, 0], self.flow_quadrants[agent_i, stim_i, 2, 0, 1] =\
                    self.what_belief[self.bot_right_inds].mean(), self.what_theta[self.bot_right_inds].mean()   
                    
                self.flow_quadrants_var[agent_i, stim_i, 2, 0, 0], self.flow_quadrants_var[agent_i, stim_i, 2, 0, 1] =\
                    self.what_belief[self.bot_right_inds].var(), self.what_theta[self.bot_right_inds].var()           
                    
                    
                W = self.where((self.where_belief < belief_bot)*(self.where_theta < theta_bot))
                self.bot_left_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 0, 0, 0], self.flow_quadrants[agent_i, stim_i, 0, 0, 1] =\
                    self.what_belief[self.bot_left_inds].mean(), self.what_theta[self.bot_left_inds].mean()    
                    
                self.flow_quadrants_var[agent_i, stim_i, 0, 0, 0], self.flow_quadrants_var[agent_i, stim_i, 0, 0, 1] =\
                    self.what_belief[self.bot_left_inds].var(), self.what_theta[self.bot_left_inds].var()          
                
                
                W = self.where((self.where_belief > belief_bot)*(self.where_belief < belief_top)*(self.where_theta < theta_top)*(self.where_theta > theta_bot))
                self.mid_mid_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 1, 1, 0], self.flow_quadrants[agent_i, stim_i, 1, 1, 1] =\
                    self.what_belief[self.mid_mid_inds].mean(), self.what_theta[self.mid_mid_inds].mean()          
                    
                self.flow_quadrants_var[agent_i, stim_i, 1, 1, 0], self.flow_quadrants_var[agent_i, stim_i, 1, 1, 1] =\
                    self.what_belief[self.mid_mid_inds].var(), self.what_theta[self.mid_mid_inds].var()          
                    
                    
                W = self.where((self.where_belief < belief_bot)*(self.where_theta < theta_top)*(self.where_theta > theta_bot))
                self.mid_left_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 0, 1, 0], self.flow_quadrants[agent_i, stim_i, 0, 1, 1] =\
                    self.what_belief[self.mid_left_inds].mean(), self.what_theta[self.mid_left_inds].mean()          
                    
                self.flow_quadrants_var[agent_i, stim_i, 0, 1, 0], self.flow_quadrants_var[agent_i, stim_i, 0, 1, 1] =\
                    self.what_belief[self.mid_left_inds].var(), self.what_theta[self.mid_left_inds].var()      
                    
                    
                W = self.where((self.where_belief > belief_bot)*(self.where_belief < belief_top)*(self.where_theta < theta_bot))
                self.bot_mid_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 1, 0, 0], self.flow_quadrants[agent_i, stim_i, 1, 0, 1] =\
                    self.what_belief[self.bot_mid_inds].mean(), self.what_theta[self.bot_mid_inds].mean()                              
                            
                self.flow_quadrants_var[agent_i, stim_i, 1, 0, 0], self.flow_quadrants_var[agent_i, stim_i, 1, 0, 1] =\
                    self.what_belief[self.bot_mid_inds].var(), self.what_theta[self.bot_mid_inds].var()        
                    
                    
                W = self.where((self.where_belief > belief_bot)*(self.where_belief < belief_top)*(self.where_theta > theta_top))
                self.mid_top_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 1, 2, 0], self.flow_quadrants[agent_i, stim_i, 1, 2, 1] =\
                    self.what_belief[self.mid_top_inds].mean(), self.what_theta[self.mid_top_inds].mean()     
               
                self.flow_quadrants_var[agent_i, stim_i, 1, 2, 0], self.flow_quadrants_var[agent_i, stim_i, 1, 2, 1] =\
                    self.what_belief[self.mid_top_inds].var(), self.what_theta[self.mid_top_inds].var()    
                    
                    
                W = self.where((self.where_belief > belief_top)*(self.where_theta < theta_top)*(self.where_theta > theta_bot))
                self.left_mid_inds = np.intersect1d(stim_inds, W)
                self.flow_quadrants[agent_i, stim_i, 2, 1, 0], self.flow_quadrants[agent_i, stim_i, 2, 1, 1] =\
                    self.what_belief[self.left_mid_inds].mean(), self.what_theta[self.left_mid_inds].mean()     

                self.flow_quadrants_var[agent_i, stim_i, 2, 1, 0], self.flow_quadrants_var[agent_i, stim_i, 2, 1, 1] =\
                    self.what_belief[self.left_mid_inds].var(), self.what_theta[self.left_mid_inds].var()                             
