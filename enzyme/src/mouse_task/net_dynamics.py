import scipy.linalg as lin; import matplotlib.animation as animation; import numpy as np; import torch; import pylab as plt; from mpl_toolkits.mplot3d.axes3d import Axes3D
from enzyme.src.mouse_task.helper_functions import helper_functions; import matplotlib.cm as cm; from scipy.stats import entropy as ent


class dynamics_visualizations(helper_functions):
    def __init__(self, **params):
        self.__dict__.update(params)

    """ weight matrix animations """
    def make_weight_animation(self):
        self.angle = -45
        self.z_max = .2
        self.animation_title = "Cellgate dynamic weight matrix"
        
        self.init_animation(self.WX, self.WY, Z = self.sorted_W[0, :,:, 0], initialize_animation = self.animate_weights)    
        self.ani = animation.FuncAnimation(self.fig, self.animate_weights, self.animation_steps, interval = 1000/self.fps)
        self.save_animation()

    def make_eigen_animation(self):
        R = self.mod_eig_vals_R[0, 0, :];  I = self.mod_eig_vals_I[0, 0, :]
        self.animation_title = "Cellgate dynamic weight eigenvalues (normalized by dynamic matrix norm)"
        self.scatter_size = list(reversed(25*(1+.1*np.arange(len(R)))))
        
        self.init_animation(R, I, initialize_animation = self.animate_eigenvalues)    
        self.ani = animation.FuncAnimation(self.fig, self.animate_eigenvalues, self.animation_steps, interval = 1000/self.fps)
        self.save_animation()
        
    def animate_weights(self, frame_number):
        split, frame = self.preprocess_frame(frame_number)
        self.postprocess_frame(self.WX, self.WY, self.sorted_W[split, :,:, frame], split, frame)
        
    def animate_eigenvalues(self, frame_number):
        split, frame = self.preprocess_frame(frame_number)
        norm = lin.norm(self.dynamic_W[split, :, :, frame])
        R = self.mod_eig_vals_R[split, frame, :]/norm
        I = self.mod_eig_vals_I[split, frame, :]/norm
        
        self.axes = [self.ax.scatter(R, I, color = self.split_cols[split], s = self.scatter_size)]
        self.ax.set_xlabel("real"); self.ax.set_ylabel("imaginary")
        self.ax.set_title(self.animation_title + f"\n{self.col} {self.split[split]  :.2f} time {frame - self.label_offset} "+\
            f"null vecs = {self.null_N} STM = {self.with_STM}, LTM = {self.with_LTM}, baseline = {self.with_baseline}")

    """ landscape animation """
    def make_land_animation(self):
        self.angle = 135 # 0
        PC_A = [0, 0, 1]
        PC_B = [1, 2, 2]
        # L = np.linspace(-self.span_range, self.span_range, self.land_span)
        # self.R = [-self.span_range, self.span_range]
        # self.LX, self.LY = np.meshgrid(L,L)
        self.title = self.effect_of + " modulation "

        for self.A, self.B in zip(PC_A, PC_B):
            self.get_heats()
            self.plot_landscape_modulation(self.smooth_heat, "time smoothened ")
        #    self.plot_landscape_modulation(self.extreme_heat, "split extremes ")
    
    def plot_landscape_modulation(self, heat2plot, title_extension = ""):
        self.heat2plot = heat2plot
        self.z_max = np.max(self.heat2plot)
        self.animation_title =  f"{title_extension} {self.title} onto PC{self.A+1} v PC{self.B+1}"                
        self.init_animation(self.LX, self.LY, Z = self.heat2plot[0,:,:,0], initialize_animation = self.animate_land)    
        self.ani = animation.FuncAnimation(self.fig, self.animate_land, self.animation_steps, interval = 1000/self.fps)
        self.ax.set_xlabel(f"PC{self.B+1}");    self.ax.set_ylabel(f"PC{self.A+1}")      
        self.save_animation()
        
    def animate_land(self, frame_number):
        split, frame = self.preprocess_frame(frame_number)
        self.postprocess_frame(self.LX, self.LY, self.heat2plot[split, :,:,frame], split, frame)
    
    # def plot_land_v_time(self): 
    #     L1 = np.linspace(0, self.max_traj_steps-1, self.max_traj_steps-1) - self.label_offset
    #     fig, ax = plt.subplots(self.PC_dim, max(2, self.split_num), figsize = (18,18), subplot_kw=dict(projection='3d'), tight_layout = True)
    #     for split in range(self.split_num):
    #         for PC in range(self.PC_dim):
    #             self.R = [self.mod_land[PC].min(), self.mod_land[PC].max()]
    #             L0 = np.linspace(self.R[0], self.R[1], self.land_span)
    #             LX, LY = np.meshgrid(L0,L1)
    #             self.title = f"{self.col} {self.split[split]  :.2f} {self.effect_of}"
    #             heat, base = [np.zeros((self.split_num, self.land_span, self.max_traj_steps - 1)) for _ in range(2)]
                
    #             for t in range(self.max_traj_steps-1):
    #                 heat[split, :, t] = np.histogram(self.mod_land[PC, split, t, :] , bins = self.land_span, range = self.R)[0]
    #                 if self.with_baseline == True:
    #                     base[split, :, t] = np.histogram(self.mod_base[PC, split, t, :], bins = self.land_span, range = self.R)[0]
    #             heat = heat - base

    #             ax[PC, split].set_xlabel(f"PC{PC + 1}")
    #             ax[PC, split].set_ylabel("time")
    #             ax[PC, split].view_init(45, 315)
    #             ax[PC, split].set_title(self.title)
    #             ax[PC, split].set_box_aspect((1,3,1)) 

    #             ax[PC, split].plot_surface(LX, LY, heat[split].T, cmap = 'coolwarm', rstride = 1, cstride = 1, alpha = 1)

    #     plt.show()
        
    def plot_land_v_time_heatmap(self):
        bin_num = 500
        R = 1.6#.7
        bin_range = np.linspace(-R,R, bin_num + 1)
        xax = np.arange(self.max_traj_steps)
        heat, base = [np.zeros((self.PC_dim, self.split_num, bin_num, self.max_traj_steps - 1)) for _ in range(2)]
        ents = np.zeros((self.PC_dim, self.split_num, self.max_traj_steps - 1))
        fig, ax = plt.subplots(self.PC_dim, self.split_num, figsize=(44, 20))        
        """ collect heat and entropy """ 
        for split in range(self.split_num):
            for PC in range(self.PC_dim):
                for t in range(self.max_traj_steps-1):
                    count = np.histogram(self.mod_land[PC, split, t, :], bins=bin_range)[0]
                    heat[PC, split, :, t] = count / count.sum()
                    ents[PC, split, t] = ent(heat[PC, split, :, t])
        """ plot heat """
        vmax = np.max(heat)
        for split in range(self.split_num):
            for PC in range(self.PC_dim):
                im = ax[PC, split].imshow(heat[PC, split], aspect='auto', vmax = vmax)
                ax[PC, split].set_xticks(ticks=xax, labels=xax - self.label_offset)
                ax[PC, split].set_yticks(ticks=np.linspace(0, bin_num, int(bin_num/100)), labels=np.round(np.linspace(-R, R, int(bin_num/100)), 1))
                ax[PC, split].set_ylabel(f"PC {PC+1}")
                ax[PC, split].set_title(f"{self.col} {self.split[split]}")
                ax[PC, split].set_xlabel(self.traj_xlabel)
                # Add colorbar as legend
                cbar = fig.colorbar(im, ax=ax[PC, split])
                cbar.set_label('density')
        plt.show()
        """ plot entropy """
        fig, ax = plt.subplots(1, self.PC_dim, figsize=(44, 5))        
        for PC in range(self.PC_dim):
            for split in range(self.split_num):
                ax[PC].plot(ents[PC, split].T, c  = self.split_cols[split], label = f"{self.col} {self.split[split]}")
            ax[PC].legend()
            ax[PC].set_title(f"PC{PC+1} entropy")
            ax[PC].set_xticks(ticks=xax, labels=xax - self.label_offset)
            ax[PC].set_xlabel(self.traj_xlabel)
        plt.show()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))        
        for split in range(self.split_num):
            ax.plot((ents[0, split]/ents[1, split]).T, c  = self.split_cols[split], label = f"{self.col} {self.split[split]}")
        ax.legend()
        ax.set_title(f"PC1 entropy / PC2 entropy")
        ax.set_xticks(ticks=xax, labels=xax - self.label_offset)
        ax.set_xlabel(self.traj_xlabel)
        plt.show()
        
        # """ collect and plot bayes dist """ 
        # self.dist_flat = np.dstack(self.dist_log[self.trial_inds])    
        # bayes_dist = np.zeros((self.split_num, self.bayes_resolution, self.max_traj_steps))
        # bayes_ents = np.zeros((self.split_num, self.max_traj_steps))
        # mem = 10
        # tick_num = 10
        # fig, ax = plt.subplots(1, self.split_num, figsize=(44, 5))        
        # for self.split_i, self.split_curr in enumerate(self.split):
        #     self.get_split_inds()
        #     for self.step in range(self.max_traj_steps):   
        #         self.get_trajectory_step_inds()
        #         bayes_dist[self.split_i, :, self.step] =self.dist_flat[mem, :, self.step_inds].mean(0)
        #         bayes_ents[self.split_i, self.step] = ent( bayes_dist[self.split_i, :, self.step])
        #     ax[self.split_i].imshow(bayes_dist[self.split_i,:,:], aspect='auto')
        #     ax[self.split_i].set_xticks(ticks=xax, labels=xax - self.label_offset)
        #     ax[self.split_i].set_yticks(ticks= np.linspace(0, len(self.bayes_range), tick_num), labels=np.round(np.linspace(0,1, tick_num), 1))
        #     ax[self.split_i].set_ylabel("P(theta)")
        #     ax[self.split_i].set_title(f"{self.col} {self.split[self.split_i]}")
        #     ax[self.split_i].set_xlabel(self.traj_xlabel)

        # """ plot bayes entropy """
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))        
        # for split in range(self.split_num):
        #     ax.plot(bayes_ents[split].T, c  = self.split_cols[split], label = f"{self.col} {self.split[split]}")
        # ax.legend()
        # ax.set_title(f"bayes entropy")
        # ax.set_xticks(ticks=xax, labels=xax - self.label_offset)
        # ax.set_xlabel("time from action")
        # plt.show()

        """ plot PCA diffs """ 
        fig,ax = plt.subplots(6,1, figsize = (15,20), tight_layout = True)
        # self.get_trajectory_diffs()
        for self.split_i in range(self.split_num):
            for PC in range(self.PC_dim):
                ax[PC].plot(self.PC_traj_diffs[PC, self.split_i, 1:].T, c = self.split_cols[self.split_i])     
                ax[PC].set_title(f"PC{PC+1} diff")
                
            ax[self.PC_dim].plot((self.PC_traj_diffs[:, self.split_i]**2).sum(0), c = self.split_cols[self.split_i])     
            ax[self.PC_dim].set_title("total squared diff")
            ax[self.PC_dim + 1].plot(self.PC_mus[:, self.split_i].sum(0), c = self.split_cols[self.split_i])     
            ax[self.PC_dim + 1].set_title("total mean")
            ax[self.PC_dim + 2].plot(np.diff(self.PC_mus[:, self.split_i].sum(0))**2, c = self.split_cols[self.split_i])     
            ax[self.PC_dim + 2].set_title("total mean squared diff")
            for i in range(6):
                ax[i].set_xlabel(self.traj_xlabel)
                ax[i].set_xticks(ticks=xax, labels=xax - self.label_offset)
        plt.show()                
        

        
                            
    """ landscape computations """ 
    def get_heats(self):
        self.get_heat_axes()
        self.heat, self.smooth_heat = [np.zeros((self.split_num, self.land_span, self.land_span, self.max_traj_steps-1)) for _ in range(2)]
        for t in range(self.max_traj_steps-1): 
            for split in range(self.split_num):   
                self.heat[split, :, :, t] = self.land2heat(split, t)    
                self.smooth_heat[:,:,:,t] = (1-self.smoothing)*self.heat[:,:,:,t] + self.smoothing*(t > 0)*self.smooth_heat[:,:,:,t-1]
        self.get_heat_extremes()
    
    def get_heat_axes(self):
        self.R0, self.R1 = [[self.mod_land[X].min(), self.mod_land[X].max()] for X in [self.B, self.A]]
        LS = [np.linspace(R[0], R[1], self.land_span) for R in [self.R0, self.R1]]
        self.LX, self.LY = np.meshgrid(LS[0], LS[1])
        
        
    def get_heat_extremes(self):
        for split in range(self.split_num):
            self.max_heat = np.repeat(self.smooth_heat.max(-1)[:,:,:,None], self.max_traj_steps-1, -1)
            self.min_heat = np.repeat(self.smooth_heat.min(-1)[:,:,:,None], self.max_traj_steps-1, -1)
            self.extreme_heat = self.max_heat - self.min_heat

    def land2heat(self, split, frame):
        X, Y = self.mod_land[self.A, split, frame, :], self.mod_land[self.B, split, frame, :]
        heat = np.histogram2d(X, Y,  bins = self.land_span, range = [self.R1, self.R0])[0]
        if self.with_baseline == True:
            X, Y = self.mod_base[self.A, split, frame, :], self.mod_base[self.B, split, frame, :]
            base = np.histogram2d(X, Y, bins = self.land_span, range = [self.R1, self.R0])[0]
            heat = heat - base
        return heat
    
    """ general animation functions """
    def init_animation(self, X, Y, Z = None, initialize_animation = None):
        self.fig = plt.figure(figsize = (15,15), tight_layout = True)
        
        if Z is not None:
            self.ax = self.fig.add_subplot(1, 1, 1, projection= '3d')    
            self.axes = [self.ax.plot_surface(X, Y, Z, cmap = 'coolwarm', rstride=1, cstride=1, alpha = 1)]
        else: 
            self.ax = self.fig.add_subplot(1, 1, 1); self.ax.grid()
            self.axes = [self.ax.scatter(X, Y, color = self.split_cols[0], s = self.scatter_size)]
            
        self.ax.set_title(self.animation_title + f"\n{self.col} {self.split[0]  :.2f} time {0 - self.label_offset} " +\
            f"null vecs = {self.null_N} STM = {self.with_STM}, LTM = {self.with_LTM}, baseline = {self.with_baseline}")
        initialize_animation(0)

    def preprocess_frame(self, frame_number):
        self.axes[0].remove()
        return int(frame_number / self.frames_per_block), frame_number % self.frames_per_block
    
    def postprocess_frame(self, X, Y, heat, split, frame):
        self.axes = [self.ax.plot_surface(X, Y, heat, cmap = 'coolwarm', rstride = 1, cstride = 1, alpha = 1)]
        self.ax.set_title(self.animation_title + f"\n{self.col} {self.split[split]  :.2f} time {frame - self.label_offset} "+\
            f"null vecs = {self.null_N} STM = {self.with_STM}, LTM = {self.with_LTM}, baseline = {self.with_baseline}")
        self.ax.view_init(45, self.angle)
        self.ax.set_zlim(-self.z_max, self.z_max)
        self.angle += self.rotation_speed

    def save_animation(self):
        fn = self.save_path + '/Trajectories/' + self.animation_title + '.gif'
        self.ani.save(fn, fps = self.fps);plt.show()        

    
    

    
class net_dynamics(dynamics_visualizations):
    def __init__(self, **params):
        self.__dict__.update(params)
    
    def get_dynamics(self, with_STM, with_baseline, with_LTM, effect_of):
        self.with_baseline = with_baseline
        self.with_STM = with_STM
        self.with_LTM = with_LTM
        self.effect_of = effect_of
        self.dynamics_preprocessing()
        for self.split_i, self.split_curr in enumerate(self.split):
            mus = [self.output_mus, self.LTM_mus, self.I_gate_mus, self.O_gate_mus, self.F_gate_mus]
            self.RE_STM, self.RE_LTM, self.I_mod, self.O_mod, self.F_mod = self.get_split_from_mu(mus)        
            self.I_mean, self.O_mean, self.F_mean = self.get_mean_across_dim([self.I_mod, self.O_mod, self.F_mod], dim = 0)
            mus = [self.I_gate_mus, self.O_gate_mus, self.F_gate_mus]
            for self.step in range(self.max_traj_steps-1):
                self.get_trajectory_step_inds()
                self.modulate_weights()   
                self.modulate_land()
                self.get_eigen()
        # self.analyze_weights()
        # self.plot_weight_variability()
        # self.plot_eigen_variability()
        # self.make_weight_animation()
        # self.make_eigen_animation()
        # self.make_land_animation()
        # self.plot_land_v_time()
        self.plot_land_v_time_heatmap()
        
    def modulate_weights(self):
        self.curr_I = self.I_mod[self.step+1, None, :]
        self.curr_O =  self.O_mod[self.step+1, None, :]
        self.curr_F =  self.F_mod[self.step+1, None, :]
        data = [self.curr_I, self.curr_O, self.curr_F]
        
        self.mod_W = self.Wc * self.get_data_of(data, self.gates, select = self.effect_of, tensor = True).T 
        self.dynamic_W[self.split_i, :,:, self.step] = self.to(self.mod_W, 'np')
        self.sorted_W[self.split_i, :, :, self.step] = self.sort_dynamic(self.mod_W)
    
    def modulate_land(self):
        nulls = [self.Wi_null, self.Wo_null, self.Wf_null]
        spread = self.get_data_of(nulls, self.gates, select = self.effect_of, tensor = True)
        self.spread = (spread[:, :, None] * self.to(self.null_noise[:, :, self.step], 'tensor')).sum(1).T 
        (self.mod_land[:, self.split_i, self.step, :], self.mod_base[:, self.split_i, self.step, :]) =  self.get_land()
        
    def get_land(self):
        L = self.with_LTM == True
        T = self.with_STM == True
        self.curr_STM = T * self.RE_STM[self.step, None, :]
        self.curr_LTM =  L * self.RE_LTM[self.step, None, :] 
        self.PCA_bias = 0 if L or T else self.data_mean
        return [self.artificial_LSTM(mean) for mean in [False, True]] 

    def artificial_LSTM(self, mean):
        I = self.I_mean if (self.effect_of == "INPUT" and mean) else self.curr_I
        O = self.O_mean if (self.effect_of == "OUTPUT" and mean) else self.curr_O
        F = self.F_mean if (self.effect_of == "FORGET" and mean) else self.curr_F
        x = self.get_layer(self.spread + self.curr_STM, self.Bc, self.Wc, torch.tanh) 
        LTM_update =  I*x + (self.spread + self.curr_LTM) * F
        if self.basis_on == "LTM":
            return self.proj.transform(self.to(LTM_update, 'np') + self.PCA_bias).T
        return self.proj.transform(self.to(O*torch.tanh(LTM_update) , 'np') + self.PCA_bias).T
        # if self.basis_on == "LTM":
        #     return self.proj.transform(self.to(LTM_update, 'np')).T
        # return self.proj.transform(self.to(O*torch.tanh(LTM_update) , 'np')).T
    
    """ analysis """
    def get_eigen(self, sort = False):
        eig = lin.eig(self.to(self.mod_W, 'np'))
        inds = abs(eig[0]).argsort()[::-1] if sort else np.arange(len(eig[0]))
        
        self.mod_eig_vec_R[self.split_i, self.step, :, :] = np.real(eig[1][inds])
        self.mod_eig_vec_I[self.split_i, self.step, :, :] = np.imag(eig[1][inds])
        self.mod_eig_vals_R[self.split_i, self.step, :] = np.real(eig[0][inds])
        self.mod_eig_vals_I[self.split_i, self.step, :] = np.imag(eig[0][inds])        
        
    def analyze_weights(self):
        self.time_var = self.sorted_W.var(-1).mean(0)
        self.split_var = self.sorted_W.var(0).mean(-1)
        self.time_var_normed = self.time_var / lin.norm(self.time_var)
        self.split_var_normed = self.split_var / ( 1 if self.split_num == 1 else lin.norm(self.split_var) )
        self.split_x_time_var = self.split_var_normed * self.time_var_normed
        
    """ preprocessing """
    def dynamics_preprocessing(self):
        self.gates = ["INPUT", "OUTPUT", "FORGET"]
        self.preprocess_animations()
        self.get_trajectory_mus()
        self.get_heat_xax()
        self.create_logs()
        
    def preprocess_animations(self):
        self.frames_per_block = self.max_traj_steps-1 
        self.animation_steps = self.frames_per_block * self.split_num
    
    def hyper_params(self):
        self.fps = 8
        self.null_N = 10
        self.smoothing = .9#0#.9
        # self.span_range = 5#.8
        self.land_span = 30#40 #100 
        self.rotation_speed = 0
        self.span_samples = 200000#200000
    
    def get_heat_xax(self):
        self.hid_xax = np.arange(self.hid_dim)
        self.sort_inds_0 = np.argsort(self.Wc_cpu, axis = 0)
        S0 = np.take_along_axis(self.Wc_cpu, self.sort_inds_0, 0)
        self.sort_inds_1 = np.flip(np.argsort(S0, axis = 1), 1)
        self.WX, self.WY = np.meshgrid(self.hid_xax, self.hid_xax)
        
    def create_logs(self):
        self.hyper_params()
        self.mod_eig_vals_R, self.mod_eig_vals_I = [np.zeros((self.split_num, self.max_traj_steps-1, self.hid_dim)) for _ in range(2)]
        self.mod_eig_vec_R, self.mod_eig_vec_I = [np.zeros((self.split_num, self.max_traj_steps-1, self.hid_dim, self.hid_dim)) for _ in range(2)]
        self.sorted_W, self.dynamic_W, self.noise_W = [np.zeros((self.split_num, self.hid_dim, self.hid_dim, self.max_traj_steps-1)) for _ in range(3)]
        self.preprocess_noise()

    def preprocess_noise(self):
        self.null_noise = 2 * (.5 - np.random.rand(self.null_N, self.span_samples, self.max_traj_steps-1))
        self.mod_land, self.mod_base = [np.zeros((3, self.split_num, self.max_traj_steps-1, self.span_samples)) for _ in range(2)] 
        self.get_null_spaces()

    def get_null_spaces(self):
        Wi_null = lin.null_space(self.Wi_cpu, rcond = 1)[:,-self.null_N:]
        Wo_null = lin.null_space(self.Wo_cpu, rcond = 1)[:,-self.null_N:]
        Wf_null = lin.null_space(self.Wf_cpu, rcond = 1)[:,-self.null_N:]
        Wc_null = lin.null_space(self.Wc_cpu, rcond = 1)[:,-self.null_N:]
        self.Wi_null, self.Wo_null, self.Wf_null, self.Wc_null = self.to([Wi_null, Wo_null, Wf_null, Wc_null], 'tensor')
        print(f"Wi null onto Wi:  {[round(n, 3)  for n in self.get_null_norms(self.Wi, self.Wi_null)]}") 
        print(f"Wo null onto Wo:  {[round(n, 3)  for n in self.get_null_norms(self.Wo, self.Wo_null)]}") 
        print(f"Wf null onto Wf:  {[round(n, 3)  for n in self.get_null_norms(self.Wf, self.Wf_null)]}") 
        print(f"Wc null onto Wc:  {[round(n, 3)  for n in self.get_null_norms(self.Wc, self.Wc_null)]}")
        print("\n")
        print(f"Wi null onto Wc:  {[round(n, 3)  for n in self.get_null_norms(self.Wc, self.Wi_null)]}") 
        print(f"Wo null onto Wc:  {[round(n, 3)  for n in self.get_null_norms(self.Wc, self.Wo_null)]}")   
        print(f"Wc null onto Wf:  {[round(n, 3)  for n in self.get_null_norms(self.Wf, self.Wc_null)]}")   
        print(f"Wf null onto Wc:  {[round(n, 3)  for n in self.get_null_norms(self.Wc, self.Wf_null)]}")   