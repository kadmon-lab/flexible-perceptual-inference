from tqdm import tqdm
from enzyme.src.mouse_task.mouse_task import mouse_task_
import torch; from enzyme.src.main.manager import manager; import numpy as np

class run_simulation(manager):
    sim: mouse_task_
    def __init__(self, simulation, simulation_params, agent, plot_episode = True, skip_run = False):
        super(run_simulation, self).__init__(simulation, simulation_params, agent)
        self.preprocess_observation_dims()
        self.sim = simulation(**simulation_params)                                                             # generate simulation
        if not skip_run:
            self.run_episodes(plot_episode)                                                            # Run all episodes
        
    def preprocess_observation_dims(self, rew_dims = 1):
        act_dims = self.agent.act_dim
        inps =  self.agent.inp_dim - act_dims - rew_dims - self.give_ctx
        self.last_O = 0        
        if self.sim.input == 'one_hot':
            self.O_hot = torch.arange(0, inps).to(self.device)                                                      # Create arange vector for all possible one-hot encoded observations 
            self.A_hot = torch.arange(0, self.agent.act_dim).to(self.device)                                             # Create arange vector for all possible one-hot encoded actions 

    def run_episodes(self, plot_episode):  
        self.plot_episode = plot_episode
        for self.e in tqdm(range(self.episodes), desc="Running episodes with agent Network"):                                                                    # for each episode
            self.sim.reset(self.e)                                                                             # reset episode logs and parameters 
            self.agent.reset()                                                                                 # detach LSTM gradients
            self.run_trials()                                                                                  # run all trials
            self.log_trials()                                                                                  # log trial outcomes
            self.end_episode()
                
    def end_episode(self):
        if self.plot_episode and self.e % int(1+self.episodes/5) == 0:                                         # if plotting episode
            self.sim.plot_episode(self)                                                                        # tell simulation to plot episode
        if self.sim.training != 'by_trial' and self.sim.training == True: 
            self.agent.backwards(episode_ratio = self.e/self.episodes)                                         # update network                       
            
    def run_trials(self):
        act, rew = [torch.zeros(1, device = self.device) for _ in range(2)]                                    # initialize previous action, reward to 0 
        for self.t in range(self.sim.num_trials):                                                              # for each trials
            self.s = 0                                                                                         # initialize trial's step number to 0       
            while self.sim.continue_trial(self.s):                                                             # for each step in trial
                act, rew, state, ctx = self.step(act, rew)                                                     # process step of trial
                # expensive saving call, better to preallocate but data size is behavior-dependent
                self.agent.store_outcome(rew, state, ctx)                                                      # tell agent to store trial's outcome and net activity 
                self.s += 1                                                                                    # increment step number                                                                         

    def step(self, action, reward):
        observation, state, ctx = self.sim.get_observation()                                                   # get observation from simulation       
        net_input = self.to_state(observation, action, reward, ctx)                                            # One-hot encoding 
        action, _, value = self.agent.forward(net_input)                                                       # get network output 
        if self.sim.ignore_action:
            if self.sim.plant_inds[self.t]:
                action = 0         
        reward = self.sim.get_reward(observation, action)                                                      # get reward from simulation
        if self.skip_first_trial_rew:
            reward = reward * (self.t > 0)
        return action, reward, state, ctx

    def to_state(self, O, A, R, C):                                                                            # Recieve an Observation, previous action and previous Reward
        if self.sim.input == 'diff': 
            O, A = self.manage_diff(O, A)
        if self.sim.input == 'raw':
            O = torch.tensor(O).reshape(1)
        if self.sim.input == 'one_hot':                                                                        # If the simulation can represent observations in a 1 hot manner
            A = self.A_hot == A                                                                                # Create a boolean vector with a True (1) at the action index
            O = self.manage_one_hot(O)
        O = O.to(self.device)
        if self.give_ctx:
            inputs = (O, A, R, torch.tensor([C], device = self.device))
        else:
            inputs = (O, A, R)
        state = torch.cat(inputs, 0).float().to(self.device)                                                # concatenate observation, action, and reward, along dim 0
        return state                                                                                           # Return a copy of the state for the actor and critic 

    def manage_diff(self, O, A):
        O_diff = O - self.last_O
        self.last_O = O
        return torch.tensor(O_diff, device = self.device).reshape(1), torch.tensor(A, device = self.device).reshape(1)

    def manage_one_hot(self, O):
        return self.O_hot == O                                                                                # Create a boolean vector with a True (1) at the sampled Observation index