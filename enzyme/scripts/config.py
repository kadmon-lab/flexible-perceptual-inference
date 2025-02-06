import numpy as np
import torch
from enzyme import PRJ_ROOT

prefix = 'LSTM' #'LSTM_exp_25_iti_1_negrew1' # 'LSTM_25_exp'


if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available() and False:
    device = torch.device("mps")
else:
    device = 'cpu'


def get_config(prefix=prefix, device=device, **kwargs):

    neg_rew = 0 #-1
    ITI_mean = 15 # 1 
    ITI_PM = 10 #0
    exp_mean = 10 # 25 #20 

    max_traj_steps = 6
    net_filename = prefix + '_net.pth'
    data_filename =  prefix

    print(f"device = {device}")
    save_path = PRJ_ROOT / "Data"

    loss_params = {'discount' : .995, 'B_val' : .1, 'B_ent': 0, 'decrease_entropy' : False}
    network_params = {'inp_dim': 5, 'hid_dim' : 48, 'act_dim' : 2, 'device' : device, 'mode' : 'LSTM', 'lesion': [],\
        'handmade': False, 'subnets': None, 'RAP' : 0,  'use_vanilla_torch': False,  'train_recurrent' : True}
    optim_params = {'lr' : .0001, 'alpha' : .99, 'eps' : 1e-5, 'weight_decay' : .0, 'momentum' : 0, 'centered' : False}
    agent_params = dict(loss_params, **optim_params, **network_params)  
    agent_params['load_path'] = save_path / net_filename

    mouse_task_params = {'sim_ID' : 3, 'save_path' : save_path, 'act_dim' : 2, 'exp_mean': exp_mean, 'exp_max': 50,  'W4L': 100, 'ITI_mean' : ITI_mean, 'ITI_PM' : ITI_PM, 'store_tensors' : True, 
        'start_NOGO': False, 'regress_on' : ['STM'], 'max_traj_steps': max_traj_steps, 'basis_on' : 'output', 'plant_prob': 0, 'neg_rew' : neg_rew, 'theta_traj' : None}

    """Set up default testing parameters"""
    mouse_task_params['PGO_range'] = np.linspace(.1, .9, 100)
    mouse_task_params['ignore_action'] = False
    mouse_task_params['end_NOGO'] = True
    mouse_task_params['plant_type'] = 'random'
    mouse_task_params['plant_prob'] = 0
    mouse_task_params['episodes'] = 5000
    mouse_task_params['num_trials'] = 50
    mouse_task_params['exp_min'] = 1


    manager_params = {'training' : False, 'input': 'one_hot', 'device' : device}  

    # update
    for k, v in kwargs.items():
        if k in agent_params:
            agent_params[k] = v
        elif k in mouse_task_params:
            mouse_task_params[k] = v
        elif k in manager_params:
            manager_params[k] = v
        else:
            raise ValueError(f"Unknown parameter {k}")

    return agent_params, mouse_task_params, manager_params

