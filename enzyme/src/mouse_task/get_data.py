# %% [markdown]
# # GENERATING AND SAVING DATA 
from pathlib import Path
import shutil
from enzyme.src.helper import load_pytree, save_pytree
import joblib
from enzyme import PRJ_ROOT
from tqdm import tqdm
import pickle
import numpy as np


# %%
def get_data(checkpointing = False, config=None, name = "plotting_data", eps = 5000, mini_eps = 5000, batches=None):
    import pickle
    import torch
    regress_to = "bayes"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    save_path = PRJ_ROOT / str('Data/')
    path = PRJ_ROOT / str('enzyme/src/mouse_task/' + name + '.pickle')
    alt_path = PRJ_ROOT / str('Data/ignore/' + name + '.pickle')
    if alt_path.exists():
        path = alt_path
    if not checkpointing and (path.exists() or path.with_suffix('').exists()):
        try:
            with open(path, 'rb') as f:
               print("loading data...")
               return pickle.load(f)
        except Exception as e:
            print(f"retrying to load from {path.with_suffix('')} as tree")
            try:
                return load_pytree(path)
            except Exception as e:
                print(e)
    else:
        """
        Simulate the data. 
        """
        print(f"generating data because {path} does not exist or try_cached is False")
        # %%
        from enzyme.src.main.run_simulation import run_simulation; import torch
        from enzyme.src.network.Actor_Critic import Actor_Critic;  
        from enzyme.src.mouse_task.mouse_task import mouse_task_; 
        from enzyme.src.mouse_task.data_structures import cage_data_struct
        save_path = PRJ_ROOT / str('Data/')
        if checkpointing and (path.exists() or path.with_suffix('').exists()):
            with open(path, 'rb') as f:
               plotting_data = pickle.load(f)
            print(f"checkpoints: {plotting_data['checkpoint']}")
        else:
            plotting_data = dict()
            plotting_data["checkpoint"] = 0
        
        def gen_data(prefix, subnets, N, give_ctx = False, argmax = False, end_nogo = True, ITI_PM = 10,
            inp_is_consec = False, use_mini_eps = True, plant = "random", plant_prob = 0, PGOs = None, ignore_action = False):
            print(f"task has ITI PM {ITI_PM}")
            inp_dim = 5 + give_ctx
            net_filename = prefix + '_net.pth'
            if inp_is_consec: 
                inp = 'raw'
            else:
                inp = 'one_hot'
            loss_params = {'discount' : .995, 'B_val' : .1, 'B_ent': 0, 'decrease_entropy' : False}
            network_params = {'inp_dim': inp_dim, 'hid_dim' : N, 'act_dim' : 2, 'device' : device, 'mode' : 'LSTM', 'get_dynamics_of' : 'actor', 'mechanistic': None,
                'expansion' : None, 'pavlovian_bias' : False, 'lesion': [], 'handmade': False, 'subnets': subnets, 'RAP' : 0, 'argmax' : argmax, 'exp_ITI': False,
                'use_vanilla_torch': False, 'train_recurrent' : True, 'lr_decay': 1, 'skip_storage' : False, 'cost_of_action' : 0}
            
            optim_params = {'lr' : .0001, 'alpha' : .99, 'eps' : 1e-5, 'weight_decay' : .0, 'momentum' : 0, 'centered' : False}
            agent_params = dict(loss_params, **optim_params, **network_params)  
            agent = Actor_Critic(**agent_params)
            mouse_task_params = {'sim_ID' : 3, 'save_path' : save_path, 'act_dim' : 2, 'exp_mean': 10, 'exp_max': 50,  'W4L': 35, 'ITI_mean' : 15, 'ITI_PM' : ITI_PM,
                                 'store_tensors' : False, 'ignore_action' : ignore_action,'prem_dur': 0, 'give_ctx': give_ctx, 'inp_is_consec': inp_is_consec,
                                 'plant_type' : plant, 'end_NOGO': end_nogo, 'start_NOGO': False, 'regress_on' : ["CELL"], 'max_traj_steps': 40, 'basis_on' : 'output', 
                                 'plant_prob': plant_prob, 'neg_rew' : 0, 'theta_traj' : None}
            manager_params = {'training' : False, 'input': inp, 'skip_first_trial_rew': False, 'device' : device,  'skip_storage': False} 
            agent.load_state_dict(torch.load(save_path / str(f'{net_filename}'), map_location=torch.device(device)))
            mouse_task_params['PGO_range'] = [.1, .2, .3, .4, .5, .6, .7, .8, .9] if PGOs is None else PGOs

            mouse_task_params['episodes'] = mini_eps if use_mini_eps else eps 
            mouse_task_params['num_trials'] = 20
            mouse_task_params['exp_min'] = 1 
            mouse_task_params['store_tensors'] = True
            testing_sim_params = dict(agent_params, **mouse_task_params, **manager_params)
            testing = run_simulation(mouse_task_, testing_sim_params, agent, plot_episode = False)
            testing.sim.preprocess_data(manager = testing, skip_bayes_full = False)
            testing.sim.run_PCA()           
            testing.sim.run_regressions(regress_to = regress_to, regress_from = 'STM', get = True, pre_act = True)
            testing.sim.get_indices(eps_init = 0, full_prep = True, align_on = "action", cross_validation = None, til_action = True)        
            return testing.sim
        
        def postprocess_networks(self, agent_name, i, plotting_data):
            if i > 0:
                agent_name = agent_name + suffix
            plotting_data[agent_name] = dict()
            agent = plotting_data[agent_name]
            append_common_fields(self, agent)
            append_common_fields2(self, agent)
            del(self)
            
        def postprocess_bayes(self, agent):
            append_common_fields(self, agent)
            del(self)

        def append_common_fields(self, agent):
            agent["PGO_range"] = self.PGO_range
            agent["analytical_rew_rate_curves_y_axis"] =  self.ana_rates
            agent["analytical_rew_rate_curves_x_axis"] =  self.ana_xax
            agent["analytical_rew_rate_points_y_axis"] =  self.rate_mu
            agent["analytical_rew_rate_points_x_axis"] =  self.wait_mu
            agent["wait_from_switch_per_pgo"] = self.wait_from_switch
            agent["trial_vec"] =  self.trial
            agent["trial_PGO_vec"] = self.PGOs            
            agent["net_wait_from_last_nogo_vec"] =  self.wait_from_last
            agent["net_wait_from_last_nogo_per_PGO"] =  self.wait_PDF_avg
            agent["behavior_dists_density"] =  self.wait_PDF
            agent["behavior_dists_edges"] =  self.wait_xax
            agent["analytical_rew_rate_curves_y_axis"] =  self.ana_rates
            agent["analytical_rew_rate_curves_x_axis"] =  self.ana_xax
            agent["analytical_rew_rate_points_y_axis"] =  self.rate_mu
            agent["analytical_rew_rate_points_x_axis"] =  self.wait_mu
            agent["wait_from_switch_per_pgo"] = self.wait_from_switch
            agent["analytical_optimal_avg_thresh"] =  self.fixed_max_thresh
            agent["wait_from_last_nogo_first_trial_from_switch"] =  self.wait_from_switch[:, 1]
            self.get_indices(eps_init = 0, full_prep = True, align_on = "action", til_action = True)  
            agent["QDIFF_mu"] = self.QDIFF_mus 
            agent["state_est_mu"] = self.net_belief_mus 
            agent["flow_state_est_mu"] = self.flow_belief_mus 
            agent["step_SAFE_STATE_vec"] =  self.PSAFE_flat
            agent["step_PGO_vec"] =  self.PGO_flat
            agent["step_trial_vec"] =  self.trial_flat
            agent["step_act_vec"] =  self.cum_acts
            agent["step_RT_vec"] =  self.RT_flat

            agent["step_thresh_RMS_vec"] = self.flow_thresh_RMS_flat         
            agent["step_thresh_vec"] = self.flow_thresh_flat   
            agent["DV_mu"] = self.flow_dist_from_opt_mus
            agent["DV_RMS"] = self.flow_belief_RMS_mus 
            agent["bayes_wait_from_last_nogo_vec"] =  self.wait_from_last
            agent["bayes_wait_from_last_nogo_per_PGO"] =  self.wait_PDF_avg

            agent["bayes_state_estimate"] =  self.flow_belief_flat
            agent["bayes_PGO_estimate"] =  self.flow_theta_flat
            agent["net_state_estimate_CV"] = self.net_belief_flat
            agent["net_PGO_estimate_CV"] = self.net_theta_flat
            agent["step_SAFE_STATE_CV"] = self.PSAFE_flat
            agent["step_stim_CV"] = self.input_flat[1]
            agent["step_PGO_CV"] = self.PGO_flat

            agent["net_wait_from_last_nogo_vec"] =  self.wait_from_last
            agent["action_prob_per_PGO"] =  self.lick_prob_mus
            agent["bayes_DV_mu"] = self.flow_dist_from_opt_mus
            agent["bayes_DV_RMS"] = self.flow_belief_RMS_mus 
            agent["net_DV_mu"] = self.net_dist_from_opt_mus 
            agent["mu_var_N"] = self.N_mus 
            agent["bayes_DV"] = self.flow_dist_from_opt_flat 
            agent["lick_prob_flat"] = self.lick_prob_flat 
            agent["consec_inds"] = self.consec_flat 
            agent["ACT_inds"] = self.cum_acts 
            agent["net_DV"] = self.Q_flat[1] 
            agent["GO_inds"] = self.GO_inds 
            agent["PGO_inds_for_heatmap"] = self.PGO_flat
            agent["net_DV_est_mu"] = self.net_DV_est_mus 

            agent["PC_flat"] = self.PC_flat
            agent["Q_flat"] = self.Q_flat[1] 
            agent["GO_flat"] = self.input_flat[1]
            agent["pre_act_flat"] = self.ACTION_inds
            agent["QDIFF_flat"] = self.QDIFF_flat 
            agent["STM_flat"] = self.output_flat
            agent["safe_flat"] = self.PSAFE_flat 
            agent["theta_flat"] = self.PGO_flat 
            agent["bayes_DV_flat"] = self.flow_dist_from_opt_flat 
            agent["net_DV_flat"] = self.net_dist_from_opt_flat 

            self.get_indices(eps_init = 0, full_prep = True, align_on = "last_nog", til_action = True)  
            agent["net_DV_mu_last_nog"] = self.net_dist_from_opt_mus 
            agent["DV_mu_last_nog"] = self.flow_dist_from_opt_mus
            agent["QDIFF_mu_last_nog"] = self.QDIFF_mus 

            self.estimate_crossvalidation()
            agent["10_fold_net_state_acc"] = self.net_acc
            agent["10_fold_bayes_state_acc"] = self.bayes_acc
            agent["10_fold_net_state_acc_til_acc"] = self.net_acc_til_act
            agent["10_fold_bayes_net_state_R2"] = self.net_bayes_state_R2
            agent["10_fold_bayes_net_theta_R2"] = self.net_bayes_theta_R2
            agent["10_fold_bayes_theta_R2"] = self.bayes_theta_R2
            agent["10_fold_net_theta_R2"] = self.net_theta_R2
            agent["10_fold_DV_R2"] = self.DV_R2

        def append_common_fields2(self, agent):
            half = self.max_traj_steps//2
            self.get_indices(eps_init = 0, cross_validation = None, full_prep = True, align_on = "action")        
            agent["net_wait_from_last_nogo_per_PGO"] =  self.wait_PDF_avg    
            agent["PC1_til_action_per_PGO"] =  self.PC_mus[0, :, :half+1]
            agent["PC2_til_action_per_PGO"] =  self.PC_mus[1, :, :half+1]
            agent["avg_stim_per_PGO"] =  self.GO_mus[:, :half+1]
            agent["PC1_from_action_per_PGO"] =  self.PC_mus[0, :, half:]
            agent["PC3_from_action_per_PGO"] =  self.PC_mus[2, :, half:]
            agent["avg_stim_per_PGO_from_act"] =  self.GO_mus[:, half:]
            agent["PC1_from_safe_per_PGO"] =  self.PC_mus[0, :, half-2:]
            agent["PC2_from_safe_per_PGO"] =  self.PC_mus[1, :, half-2:]
            agent["PC1_from_safe_per_PGO"] =  self.PC_mus[0, :, half-2:]
            agent["PC2_from_safe_per_PGO"] =  self.PC_mus[1, :, half-2:]

            self.get_N_consec_vs_est(get = True)
            agent["net_PSAFE_vs_N"] = self.net_safe_est_N
            agent["net_PSAFE_vs_N_per_PGO"] = self.net_safe_est_N_per_PGO
            agent["samples_vs_N_per_PGO"] = self.samples_N_per_PGO
            agent["R2_N_per_PGO"] = self.R2_N_per_PGO

            self.run_regressions(regress_to = regress_to, regress_from = 'PCA', get = True, pre_act = True)
            self.get_indices(eps_init = 0, full_prep = True, align_on = "action", til_action = True)        
            agent["PC_bayes_state"] =  self.flow_belief_flat
            agent["PC_bayes_PGO"] =  self.flow_theta_flat
            agent["PC_to_state"] = self.net_belief_flat
            agent["PC_to_PGO"] = self.net_theta_flat
            agent["PC_safe"] = self.PSAFE_flat
            agent["PC_input"] = self.input_flat[1]
            agent["PC_PGO"] = self.PGO_flat
            agent["PC_to_DV_est_mu"] = self.net_DV_est_mus 
            agent["PC_to_net_DV_mu"] = self.net_dist_from_opt_mus 


            self.run_regressions(regress_to = regress_to, regress_from = 'QDIFF', get = True, pre_act = True)
            self.get_indices(eps_init = 0, full_prep = True, align_on = "action", til_action = True)        
            agent["QDIFF_bayes_state"] =  self.flow_belief_flat
            agent["QDIFF_bayes_PGO"] =  self.flow_theta_flat
            agent["QDIFF_to_state"] = self.net_belief_flat
            agent["QDIFF_to_PGO"] = self.net_theta_flat
            agent["QDIFF_safe"] = self.PSAFE_flat
            agent["QDIFF_input"] = self.input_flat[1]
            agent["QDIFF_PGO"] = self.PGO_flat
            agent["QDIFF_to_DV_est_mu"] = self.net_DV_est_mus 
            agent["QDIFF_to_net_DV_mu"] = self.net_dist_from_opt_mus 

            self.fit_mechanistic(use_ctx_thresh = True, fit_steps = 100000)
            self.get_indices(align_on = "action", from_nog = True, til_action = True, full_prep = True, run_traj = True)
            agent["mechanistic_from_nog_to_act"] = self.mechanistic_DV_mus 
            agent["mechanistic_QDIFF_R2"] = self.mechanistic_QDIFF_R2 
            agent["QDIFF_from_nog_to_act"] = self.QDIFF_mus 

            self.fit_mechanistic(fit_steps = 100000)
            self.get_indices(align_on = "action", from_nog = True, til_action = True, full_prep = True, run_traj = True)
            agent["no_ctx_mechanistic_from_nog_to_act"] = self.mechanistic_DV_mus 
            agent["no_ctx_mechanistic_QDIFF_R2"] = self.mechanistic_QDIFF_R2 
            agent["no_ctx_QDIFF_from_nog_to_act"] = self.QDIFF_mus 


        def save_checkpoint(path, plotting_data):
            with open(path, 'wb') as d:
                pickle.dump(plotting_data, d, protocol=pickle.HIGHEST_PROTOCOL)
            plotting_data["checkpoint"] += 1
            with open(path, 'wb') as d:
                pickle.dump(plotting_data, d, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved checkpoint {plotting_data['checkpoint']-1}")

        """ change here """
        # default network names
        trained_net = 'LSTM_neg'
        readout_net = 'LSTM_readout_neg' 
        readout_ctx_net =  'LSTM_readout_ctx_neg'
        batches = 10 if batches is None else batches
        N = 100
        """ change here """

        if plotting_data["checkpoint"] < 1:
            # Bayes with optimal policy data plus trained net 
            prefix = trained_net + "_0"
            subnets = "bayes_optim"  
            self = gen_data(prefix, subnets, N, use_mini_eps = False)
            agent_name = "bayes_agent_and_trained_LSTM_representation"
            plotting_data[agent_name] = dict()
            agent = plotting_data[agent_name]
            postprocess_bayes(self, agent)
                               
            # # Bayes with optimal policy data plus random net 
            prefix = readout_net + "_0"
            subnets = "bayes_optim"
            self = gen_data(prefix, subnets, N)   
            agent_name = "bayes_agent_and_random_LSTM_representation"
            plotting_data[agent_name] = dict()
            agent = plotting_data[agent_name] 
            postprocess_bayes(self, agent)

            # # Bayes with optimal policy data plus random net + context
            prefix = readout_ctx_net + "_0"
            subnets = "bayes_optim"
            self = gen_data(prefix, subnets, N, give_ctx = True)    

            agent_name = "bayes_agent_and_random_LSTM_plus_ctx_representation"
            plotting_data[agent_name] = dict()
            agent = plotting_data[agent_name]
            postprocess_bayes(self, agent)

            save_checkpoint(path, plotting_data)
                            
        for i in range(batches):
            print(f"\nCOLLECTING DATA\nFOR NETWORK {i+1}/{batches} \n")
            suffix = f"_{i}"
            subnets = None       
            mini = i > 0

            if plotting_data["checkpoint"] < (i + 2):
                # # trained network 
                print("TRAINED ARGMAX")
                self = gen_data(trained_net + suffix, subnets, N, argmax = True, use_mini_eps = mini)        
                postprocess_networks(self, "trained_LSTM", i, plotting_data)
                # # trained network softmax
                print("TRAINED SOFTMAX")
                self = gen_data(trained_net + suffix, subnets, N, use_mini_eps = mini)       
                postprocess_networks(self, "trained_LSTM_soft", i, plotting_data)
                # # readout network 
                print("READOUT")
                self = gen_data(readout_net + suffix, subnets, N, use_mini_eps = mini)    
                postprocess_networks(self, "random_LSTM", i, plotting_data)
                # # readout network + ctx         
                print("READOUT PLUS CTX")
                self = gen_data(readout_ctx_net + suffix, subnets, N, give_ctx = True, use_mini_eps = mini)        
                postprocess_networks(self, "random_LSTM_plus_ctx", i, plotting_data)

                save_checkpoint(path, plotting_data)

        # # trained network 

        if plotting_data["checkpoint"] < (i + 3):

            plant_A = [0,1,0,1,0,0,0,0]
            plant_B = [0,1,1,1,1,0,0,1]
            PGO = [.3, .7]
            plant = np.array([plant_A, plant_B])
            self = gen_data(trained_net + "_0", subnets, N, use_mini_eps = False, plant = plant, plant_prob = .1, PGOs = PGO)        
            agent_name = "trained_LSTM_planted"
            plotting_data[agent_name] = dict()
            agent = plotting_data[agent_name]
            append_common_fields(self, agent)
            self.run_regressions(regress_to = regress_to, regress_from = 'STM', get = True, pre_act = True)
            self.get_indices(planted = True, plant_ID = 0, eps_init = 0, full_prep = True, align_on = "onset", til_action = True)        
            agent["plantA_GOs"] =  self.GO_mus 
            agent["plantA_bayes_safe_est"] = self.flow_belief_mus
            agent["plantA_net_safe_est"] = self.net_belief_mus
            self.get_indices(planted = True, plant_ID = 1, eps_init = 0, full_prep = True, align_on = "onset", til_action = True)        
            agent["plantB_GOs"] =  self.GO_mus 
            agent["plantB_bayes_safe_est"] = self.flow_belief_mus
            agent["plantB_net_safe_est"] = self.net_belief_mus
            agent["plantA_seq"] = plant_A
            agent["plantB_seq"] = plant_B

            self.run_regressions(regress_to = regress_to, regress_from = 'PCA', get = True, pre_act = True)
            self.get_indices(planted = True, plant_ID = 0, eps_init = 0, full_prep = True, align_on = "onset", til_action = True)        
            agent["PC_plantA_net_safe_est"] = self.net_belief_mus
            self.get_indices(planted = True, plant_ID = 1, eps_init = 0, full_prep = True, align_on = "onset", til_action = True)        
            agent["PC_plantB_net_safe_est"] = self.net_belief_mus
            del(self)
            save_checkpoint(path, plotting_data)

    return plotting_data
if __name__ == "__main__":
    plotting_data = get_data()