from enzyme.src.main.run_simulation import run_simulation; import torch; import numpy as np
from enzyme.src.network.Actor_Critic import Actor_Critic; from enzyme import PRJ_ROOT
from enzyme.src.mouse_task.mouse_task import mouse_task_; 

if __name__ == "__main__":
    cuda = 'cuda:0' 
    mode = 'LSTM'                      # ['LSTM', 'RNN']
    get_dynamics_of = 'actor'        # ['actor', 'critic']
    subnets = None                     # [None, 'bayes', 'bayes_optim', 'dynamics', 'factorized', 'factorized_optim']
    hid_dim = 100 
    give_ctx = True
    inp_dim = 5 + give_ctx
    train_recurrent = False
    inp = 'one_hot'
    discount =  .5 
    lr = .0005 
    neg_rew = -3  
    exp_mean = 10
    ITI_mean = 15
    ITI_PM = 10 
    exp_max = 50
    W4L = 100 
    act_dim = 2
    momentum =  1e-3
    ent_bonus = .05
    value_beta = .01
    weight_decay = 1e-3
    pre = "LSTM_readout_ctx_neg_"
    rep = 10

    for p in range(rep):
        prefix = pre + str(p)
        net_filename = prefix + '_net.pth'
        data_filename =  prefix
        device = cuda if torch.cuda.is_available() else 'cpu'
        save_path = PRJ_ROOT / str('Data/')
        
        loss_params = {'discount' : discount, 'B_val' : value_beta, 'B_ent': ent_bonus, 'decrease_entropy' : False}
        network_params = {'inp_dim': inp_dim, 'hid_dim' : hid_dim, 'act_dim' : act_dim, 'device' : device, 'mode' : mode,
            'get_dynamics_of' : get_dynamics_of, 'mechanistic': None, 'expansion' : None, 'pavlovian_bias' : False, 'lesion': [],
            'handmade': False, 'subnets': subnets, 'argmax' : False, 'exp_ITI': False, 'skip_storage' : False,
            'use_vanilla_torch': False, 'train_recurrent' : train_recurrent, 'cost_of_action': 0}
        optim_params = {'lr' : lr, 'alpha' : .99, 'eps' : 1e-5, 'weight_decay' : weight_decay,
            'momentum' : momentum, 'centered' : True, 'lr_decay': .99} 
        agent_params = dict(loss_params, **optim_params, **network_params)  
        agent = Actor_Critic(**agent_params)

        mouse_task_params = {'sim_ID' : 3, 'save_path' : save_path, 'act_dim' : act_dim, 'exp_mean': exp_mean, 'exp_max': exp_max,
            'W4L': W4L, 'ITI_mean' : ITI_mean, 'ITI_PM' : ITI_PM, "prem_dur": 0, 'exp_ITI': False, 'store_tensors' : False,
            'ignore_action' : False,  'give_ctx': give_ctx, 'inp_is_consec': False, 'plant_type' : 'random', 'end_NOGO': False,
            'start_NOGO': True, 'max_traj_steps': 20, 'basis_on' : 'output', 'plant_prob': 0, 'neg_rew' : neg_rew, 'theta_traj' : None}
        manager_params = {'training' : True, 'input': inp, 'skip_first_trial_rew': False, 'device' : device,  'skip_storage': False}  
        mouse_task_params['PGO_range'] =  np.linspace(0, 1, 20)
        mouse_task_params['episodes'] = 10000 
        mouse_task_params['num_trials'] = 20
        mouse_task_params['exp_min'] = 1 
        training_sim_params = dict(agent_params, **mouse_task_params, **manager_params)            
        training = run_simulation(mouse_task_, training_sim_params, agent, plot_episode = False)    
        torch.save(agent.state_dict(), save_path / str(f'{net_filename}'))         
        del(training); del(agent)