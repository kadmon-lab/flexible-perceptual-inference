# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, fields
from types import SimpleNamespace
from typing import NamedTuple, Optional, Sequence, Tuple

import chex
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from jax import lax

from jumanji import specs
from jumanji.env import Environment
from jumanji.types import StepType, TimeStep, restart, termination, transition
from jumanji.viewer import Viewer
from jumanji.wrappers import AutoResetWrapper
from jumanji.specs import Array, DiscreteArray, Spec

from enzyme.src.mouse_world.lib import p_end
from enzyme.src.bayes.actor import Observation


import numpy as np
from enzyme.src.bayes.actor import *
from enzyme.src.bayes.config import THETAS
import equinox as eqx
from enzyme.src.mcmc import find_end_N_cons_GOs, generate_trajs_nb, get_poisson_trials

import logging
logger = logging.getLogger(__name__)

from jax import dtypes
max_int32 = dtypes.iinfo(jnp.int32).max

dt = 1.

reg = 1e-10

theta_safe = .0005
N_trials = 10000

class EnvKey(NamedTuple):
    key_s: chex.PRNGKey
    key_th: chex.PRNGKey
    key_x: chex.PRNGKey
    key_trial: chex.PRNGKey
    key_ITI: chex.PRNGKey
    key_lmbd_s: chex.PRNGKey
    key_lmbd_th: chex.PRNGKey


class State(NamedTuple):
    """
    s:
    x:
    theta:
    key: random key used to generate random numbers at each step and for auto-reset.
    """

    s: chex.Scalar
    x: chex.Scalar
    r: chex.Scalar
    theta: chex.Scalar
    ITI_counter: chex.Scalar
    trial_counter: chex.Scalar
    keys: EnvKey
    step_count: chex.Numeric  # ()


class TransitionAtoms(NamedTuple):
    T_S: chex.Array
    T_theta: chex.Array
    P_X__S_TH: chex.Array
    T_r: chex.Array


def construct_transition_atoms(lmbd_s, lmbd_theta, theta_safe, ITI_mean, ITI_PM):
    """
    Use JS's constructions. 
    """
    import numpy as np

    
    # BINARY = np.array([0, 1])
    # THETAS_ = BINARY[None,:]*THETAS[:,None] + (1-BINARY)[None:]*(1-THETAS[:,None])     


    I, ONES = np.eye(2), np.ones((2, 2))
    micro_N = int(2 + ITI_mean + ITI_PM)
    
    T_r = np.eye(micro_N)
    T_r[0,0] = 0
    T_r[1,1] = 0
    T_r[2, 0] = 1
    T_r[2, 1] = 1

    if ITI_mean == 0:
        ITI_mean = 1. # immediate leave

    min_leave_time = ITI_mean - ITI_PM
    max_leave_time = ITI_mean + ITI_PM
    delta_max = 2 * ITI_PM

    T_S = np.zeros((micro_N, micro_N))
    T_S[0, 0] = 1 - lmbd_s
    T_S[0, -1] = 1

    T_S[1, 0] = lmbd_s
    T_S[1, 1] = 1  

    # populate microstates
    for ITI_counter in range(3, micro_N):        # i is state we transition to 
        # ITI_time = i - 3 
        # if ITI_time < self.min_leave_time:
        #     ITI_leave_prob = 0
        # else:
        #     ITI_leave_prob = 1/(self.delta_max -  (ITI_time - self.min_leave_time))
        # ITI_time = ITI_counter - 2 
        # ITI_leave_prob = max(0, ITI_time - min_leave_time) / max(1, delta_max)

        ITI_leave_prob = p_end(ITI_counter - min_leave_time, delta_max)
        
        T_S[ITI_counter, ITI_counter-1] = 1 - ITI_leave_prob
        T_S[0, ITI_counter-1] = ITI_leave_prob

    # dims: x  x  s  x  th
    P_X__S_TH = np.zeros((2, micro_N, THETAS.size))                                                                                
    P_X__S_TH[0,0,:] = 1 - THETAS
    P_X__S_TH[1,0,:] = THETAS
    P_X__S_TH[0,1,:] = 0
    P_X__S_TH[1,1,:] = 1

    # ITI states, same x-statistics as unsafe state
    P_X__S_TH[0,2:,:] = 1 - THETAS
    P_X__S_TH[1,2:,:] = THETAS


    """ softenings """ 
    I = np.eye(THETAS.size)
    ONES = np.ones((THETAS.size, THETAS.size))
    T_theta = I * (1 - lmbd_theta*dt) + (ONES-I) * lmbd_theta*dt /( THETAS.size - 1)
    

    P_X__S_TH = P_X__S_TH
    P_X__S_TH = P_X__S_TH / P_X__S_TH.sum(0, keepdims=True)

    return TransitionAtoms(**jax.tree_util.tree_map_with_path(lambda kp, x: jnp.array(x), dict(T_S=T_S, T_theta=T_theta, P_X__S_TH=P_X__S_TH, T_r=T_r)))


class MouseWorld(Environment[State]):

    def __init__(
        self,
        lmbd_s=0.1,
        lmbd_th=0.0001,
        r_rat=0.0,
        gamma=1.0,
        episode_length=-1,
        THETAS=THETAS,
        backend="thy",
        plant_x=None,
        plant_theta=None,
        plant_start=0,
        ITI_mean=15.,
        ITI_is_exp=False,
        ITI_PM=10,
        min_trial_dur=0,
        max_trial_dur=150,
    ) -> None:

        self.lmbd_s = lmbd_s
        self.lmbd_act = lmbd_s
        self.lmbd_theta = lmbd_th
        self.theta_safe = theta_safe
        self.lmbd_ITI = (ITI_mean + 1e-3)**-1

        self.tau_safe = self.lmbd_s**-1
        self.tau_act = (self.lmbd_act + 1e-10)**-1
        self.tau_theta = (self.lmbd_theta + 1e-10)**-1
        if ITI_mean == 0: logger.warning("ITI mean is 0, this will not lead to an optimal reward rate")
        self.tau_ITI = ITI_mean
        self.ITI_PM = ITI_PM
        self.ITI_min = self.tau_ITI - ITI_PM
        self.ITI_max = self.tau_ITI + ITI_PM

        self.ITI_is_exp = ITI_is_exp

        self.r_rat = r_rat
        R_SUCCESS = 1.
        R_FAIL = - R_SUCCESS * r_rat
        self.R_SUCCESS = R_SUCCESS
        self.R_FAIL = R_FAIL

        self.gamma = gamma

        self.THETAS = jnp.array(THETAS)

        self.episode_length = int(episode_length)
        self.backend = backend
        self.max_trial_dur = max_trial_dur
        self.min_trial_dur = min_trial_dur

        if plant_x is not None:
            assert len(plant_x) > 1
        if plant_theta is not None:
            assert len(plant_theta) > 1

        self.plant_x = jnp.array(plant_x) if plant_x is not None else jnp.array([False])  # dirty hack
        self.plant_theta = jnp.array(plant_theta) if plant_theta is not None else jnp.array([-1.])
        self.plant_start = plant_start

        self.num_actions = 2

        # transition matrices and observation generation
        # The likelihood can be viewed as a diagonal matrix with a multiindex (s,theta; s',theta')
        self.T_S, self.T_theta, self.P_X__S_TH, self.T_r = construct_transition_atoms(
            self.lmbd_s, self.lmbd_theta, self.theta_safe, self.tau_ITI, self.ITI_PM,
        )

        # some analytical probs
        @jnp.vectorize
        def p_s0__x1_(n, theta):
            a, b, c =  lmbd_s, (1-lmbd_s)*theta, (1-lmbd_s)*(1-theta)
            sum_n_ = lax.fori_loop(0, n, lambda n_, val: val + c*b**n_, 0.0)
            return (b**n) / (1 - sum_n_)

        @jnp.vectorize
        def p_s1__x1_(t, theta):
            return 1. - p_s0__x1_(t, theta)

        # ensure that the probabilities sum to 1
        def p_s0__x1_thy(t, theta):
            return p_s0__x1_(t, theta) / (p_s0__x1_(t, theta) + p_s1__x1_(t, theta))

        def p_s1__x1_thy(t, theta):
            return p_s1__x1_(t, theta) / (p_s0__x1_(t, theta) + p_s1__x1_(t, theta))

        @jnp.vectorize
        def p_s1_thy(t):
            # prod = 1. - jnp.exp(-lmbd_s*(t))
            prod = lax.fori_loop(0, t + 1, lambda i, prod: prod * (1 - lmbd_s * dt) ** i, 1.0)
            return jnp.where(t >= 0., 1 - prod, 0.)

        @np.vectorize
        def p_s1_mc(t):
            x, s = get_poisson_trials(N_trials, int(self.lmbd_s**-1*10), theta=0, lmbd_s=self.lmbd_s)
            return jnp.mean(s, axis=0)[t]

        @np.vectorize
        def t_act_mc(t, theta):
            theta = float(theta)
            x, s = get_poisson_trials(N_trials, int(self.lmbd_s**-1*10), theta, lmbd_s)
            return find_end_N_cons_GOs(x, t).mean(0)

        @np.vectorize
        def p_s0__x1_mc(t, theta):
            theta = float(theta)
            x, s = get_poisson_trials(N_trials, int(self.lmbd_s**-1*10), theta, lmbd_s)
            # get the index where x is 1 for t steps
            idx = jnp.where(jnp.sum(x[:, :t], axis=1) == t)[0]
            # get the corresponding s
            ps = s[idx, t].mean()
            return 1. - ps

        def p_s1__x1_mc(t, theta):
            return 1 - p_s0__x1_mc(t, theta)

        @jnp.vectorize
        # @lru_cache
        def t_act_thy(n, theta):
            a, b, c =  lmbd_s, (1-lmbd_s)*theta, (1-lmbd_s)*(1-theta)
            sum_a = lax.fori_loop(0, n, 
                                  lambda i, val: val + b**i*(n*a + c*(i + 1)), 
                                  0.0)
            sum_b = lax.fori_loop(0, n, 
                                    lambda i, val: val + b**i,
                                    0.0)

            return (n*b**n + sum_a) / (1 - c*sum_b)

            # ts = jnp.linspace(0, n, 1000)
            # dt = ts[1] - ts[0]
            # return (n*b**n + jnp.sum(b**ts*(n*a + c*(ts + 1)))*dt) / (1 - c*jnp.sum(b**ts)*dt)

        @jnp.vectorize
        # @lru_cache
        def t_act_inv_thy(n, theta):
            a, b, c = lmbd_s, (1-lmbd_s)*theta, (1-lmbd_s)*(1-theta)
            i = jnp.arange(n)
            from scipy.optimize import root_scalar
            def zero(A_inv):
                lhs = A_inv
                rhs = ((1/n)*b**n + jnp.sum(b**i*((1/n)*a + c*(1/(i + 1 + A_inv**-1)))))
                return lhs - rhs
            n_max = 1/a
            sol = root_scalar(zero, x0=(n_max + n + 2)**-1, x1=1e2)
            assert sol.converged
            A_inv = sol.root
            return A_inv

        # make the functions available to the class
        if self.backend == "mc":

            # precompute lookup tables
            t_ = jnp.arange(20)
            theta_ = jnp.linspace(0., 1., 5)

            from jax.scipy.interpolate import RegularGridInterpolator
            p_s1_interp = RegularGridInterpolator((t_,), p_s1_mc(t_))
            p_s1__x1_interp = RegularGridInterpolator((t_, theta_), p_s1__x1_mc(t_[:, None], theta_[None, :]))
            p_s0__x1_interp = RegularGridInterpolator((t_, theta_), p_s0__x1_mc(t_[:, None], theta_[None, :]))
            t_act_interp = RegularGridInterpolator((t_, theta_), t_act_mc(t_[:, None], theta_[None, :]))

            # wrap in lambdas for the right signatures without the tuples
            p_s1 = lambda t: p_s1_interp((t,))
            p_s1__x1 = lambda t, theta: p_s1__x1_interp((t, theta))
            p_s0__x1 = lambda t, theta: p_s0__x1_interp((t, theta))
            t_act = lambda t, theta: t_act_interp((t, theta))

            # test
            t_act(t_[:, None], theta_[None, :])
            p_s1(t_)
            p_s1__x1(t_[:, None], theta_[None, :])
            p_s0__x1(t_[:, None], theta_[None, :])

            t_wait_inv = t_act_inv_thy
        elif self.backend == "thy":
            p_s1 = p_s1_thy
            p_s1__x1 = p_s1__x1_thy
            p_s0__x1 = p_s0__x1_thy
            t_act = t_act_thy
            t_wait_inv = t_act_inv_thy
        else:
            raise ValueError("backend not recognized")

        self.p_s1 = p_s1
        self.p_s1__x1 = p_s1__x1
        self.p_s0__x1 = p_s0__x1
        self.t_act = t_act
        self.t_wait_inv = t_wait_inv

        @jnp.vectorize
        def r_t(ta):
            """
            The reward rate a context-insensitive agent would get at time t.
            """
            return (1/(ta + reg + self.tau_ITI)) * (gamma**ta * R_SUCCESS * p_s1(ta) + (1. - p_s1(ta))*(R_FAIL))

        # https://www.wolframalpha.com/input?i=maximize+%282*%281-e%5E%28-x%29%29+%2B+-5*e%5E%28-x%29%29%2F%28x%29
        # W = lambertw
        # e = jnp.exp(1)
        # def t_star():
        #     return -lmbd_s**-1 * W(z=-1/e * (1 - jnp.abs(r_rat)), k=-1) - 1

        # t_act = lambda t: t - ITI

        def R_t__x1(tw, theta):
            return gamma**tw * R_SUCCESS * p_s1__x1(tw, theta) + p_s0__x1(tw, theta)*R_FAIL

        def r_t__X1(tw, theta=None): 
            if theta is not None:
                ta = t_act(tw, theta)
                r_t_ = (1/(ta + reg + self.tau_ITI)) * R_t__x1(tw, theta)
            else:
                tw = jnp.atleast_1d(tw)[:, None]
                thetas = jnp.atleast_1d(THETAS)[None, :]
                ta = t_act(tw, THETAS)
                r_t_ = (1/(ta + reg + self.tau_ITI)) * R_t__x1(tw, thetas)
                r_t_ = r_t_.mean(-1)  # average over thetas

            # jax.debug.print("t_act: {}", (t, t_act_, r_t_))
            return r_t_

        # R, r = rm, r = rp/rm
        # r_t__X1 = lambda t, theta: (1/(t_act(t) + reg + ITI)) * R*(1 - p_s1__x1_theta(t, theta)*(1 - r))

        @jnp.vectorize
        def _t_wait_opt__X1(theta): 
            t_wait_cand = jnp.arange(100)[:, None]
            t_wait = jnp.argmax(r_t__X1(t_wait_cand, theta), axis=0).squeeze()
            # jax.debug.print("t_star: {}", t_star)
            return t_wait
        
        # for caching
        _THETAS = jnp.linspace(0., 1., 100)
        _t_wait_opt__X1_vals = _t_wait_opt__X1(_THETAS)
        def t_wait_opt__X1(theta):
            return jnp.round(jnp.interp(theta, _THETAS, _t_wait_opt__X1_vals)).astype(int)


        def p_s1_thsd__theta(theta):
            t_wait_opt = t_wait_opt__X1(theta)  # opimal policy at that theta
            p_tshd = p_s1__x1(t_wait_opt, theta)
            return p_tshd

        self.r_t = r_t

        self.R_t__x1 = R_t__x1
        self.r_t__X1 = r_t__X1
        self.t_wait_opt__X1 = t_wait_opt__X1
        self.p_s1_thsd__theta = p_s1_thsd__theta

    def action_spec(self) -> Spec:
        return DiscreteArray(num_values=self.num_actions, dtype=jnp.int32)

    def observation_spec(self) -> Spec:
        return Spec(
            Observation,
            "ObservationSpec",
            agent_view=Array(shape=(2,), dtype=jnp.bool),
            action_mask=Array(shape=(self.action_spec().num_values,), dtype=jnp.float32),
            step_count=Array(shape=(), dtype=jnp.int32),
        )

    def reset(self, key=None, key_env=None) -> Tuple[State, TimeStep[Observation]]:

        if key_env is None:
            _, *keys = jax.random.split(key, 1 + len(EnvKey._fields))
            keys = EnvKey(*keys) 
        else:
            keys = key_env
            key = jax.random.key(399807987)

        x = jnp.array([True, False])  # dont use key
        theta = jax.random.choice(keys.key_th, THETAS)
        s = jnp.array(False)  # dont use key

        obs = Observation(agent_view=x,
                          action_mask=jnp.ones(self.num_actions),
                          step_count=0,
        )

        state = State(
            s=s,
            x=x,
            r=0., 
            ITI_counter=-1,
            theta=theta,
            keys=keys,
            step_count=0,
            trial_counter=0,
        )

        timestep = restart(observation=obs, extras={})

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Updates the environment state after the agent takes an action.

        Args:
            state: the current state of the environment.
            action: the action taken by the agent.

        Returns:
            state: the new state of the environment.
            timestep: the next timestep.
        """
        keys = jax.tree.map(lambda k: jax.random.split(k)[1], state.keys)  # produce new keys for the next step
        key_s, key_th, key_x, key_ITI, key_lmbd_s, key_lmbd_th = keys.key_s, keys.key_th, keys.key_x, keys.key_ITI, keys.key_lmbd_s, keys.key_lmbd_th
        
        # do not split the trial key, only on action that ends a trial
        key_trial = state.keys.key_trial # TODO

        # repackage
        keys = EnvKey(key_s, key_th, key_x, key_trial, key_ITI, key_lmbd_s, key_lmbd_th)

        # jax.debug.print("s: {}", key.key_s)

        def ITI_leave_cond_uniform(ITI_counter, key):
            # return ITI_counter >= self.tau_ITI  # fixed leave probability

            # ITI_counter starts at 0

            # hazard for a uniform distribution
            a = self.ITI_max - self.ITI_min

            if a == 0:
                # shortcut
                end_ITI_now = ITI_counter == (self.ITI_max - 1)
                return end_ITI_now            

            end_ITI_now = lax.cond(ITI_counter >= self.ITI_min, 
                             lambda: jax.random.uniform(key) < p_end(ITI_counter - self.ITI_min, a),  # leave
                             lambda: False,) 

            return end_ITI_now

        def ITI_leave_cond_exponential(ITI_counter, key):
            end_ITI_now =  ((jax.random.uniform(key) < self.lmbd_ITI) & (ITI_counter >= 1)) | (ITI_counter == self.ITI_max)  # ! <= max ITI length in BayesActor (20), too high causes numerical issues
            return end_ITI_now

        ITI_leave_cond = ITI_leave_cond_uniform if not self.ITI_is_exp else ITI_leave_cond_exponential

        # jax.debug.print("ITI_counter: {}", ITI_counter)

        if self.max_trial_dur:
            action = jax.lax.cond(
                (state.trial_counter >= self.max_trial_dur),
                lambda _: 1,
                lambda _: action,
                None)
            trial_counter = jax.lax.cond(             
                (state.trial_counter >= self.max_trial_dur),
                lambda _: 0,
                lambda _: state.trial_counter + 1,
                None)
        else:
            trial_counter = state.trial_counter + 1

        def forward_or_leave_ITI(ITI_counter):
            # forward the ITI counter if in ITI
            ITI_counter = lax.select(ITI_leave_cond(ITI_counter, key_ITI), -1, ITI_counter + 1,)
            return ITI_counter

        def enter_ITI(ITI_counter):
            return (0 if self.tau_ITI > 0 else -1)  # enter ITI or reset, and reset state

        # get reward and update state depending on action
        def did_act_in_ITI(s, ITI_counter):
            r = 0.
            return s, r, forward_or_leave_ITI(ITI_counter)

        def did_act_not_in_ITI(s, ITI_counter):
            r = jax.lax.select(s, self.R_SUCCESS, self.R_FAIL)
            s_next = False
            return jax.lax.cond(state.trial_counter >= self.min_trial_dur, 
                                lambda: (s_next, r, enter_ITI(ITI_counter)), 
                                lambda: (s, 0., ITI_counter),  # just do nothing
                                )

        def did_act(s, ITI_counter):
            s, r, ITI_counter = jax.lax.cond(
                ITI_counter >= 0,
                did_act_in_ITI,
                did_act_not_in_ITI,
                s, ITI_counter
            )
            return False, r, ITI_counter

        def did_not_act(s, ITI_counter): 
            # s update
            s = jax.lax.select(
                (jax.random.uniform(key_s) < self.lmbd_s) & (ITI_counter == -1),
                True,
                s,
            )

            ITI_counter = jax.lax.select(ITI_counter >= 0, forward_or_leave_ITI(ITI_counter), ITI_counter)
            return (s, 0., ITI_counter)

        ### -> process action and update dependants
        s, r, ITI_counter = lax.cond(
            action, 
            did_act, 
            did_not_act,
            state.s, state.ITI_counter
        )

        # theta update
        theta = jax.lax.cond(
            (self.plant_theta.size > 1) & (state.step_count >= self.plant_start),
            lambda _: self.plant_theta[state.step_count % self.plant_theta.size],
            lambda _: jax.lax.cond(
                jax.lax.lt(jax.random.uniform(key_lmbd_th), self.lmbd_theta),
                lambda x: jax.random.choice(key_th, self.THETAS),
                lambda x: state.theta,
                operand=None,
        ),
        operand=None
        )

        # finally expose the new state to the worlds
        x = jax.lax.cond(
            (self.plant_x.size > 1) & (state.step_count >= self.plant_start),
            lambda _: self.plant_x[state.step_count % self.plant_x.size],
            lambda _: jax.lax.cond(
                (s == 0) | (s == 2),  # use new state here
                lambda _: jax.lax.select(
                    jax.random.uniform(key_x) < theta,
                    True,
                    False,
                ),
                lambda _: True,
                operand=None,
            ),
            operand=None
        )

        x = jnp.eye(2)[x.astype(int)].astype(jnp.bool)

        # Build the next state.
        state = State(
            s=s,
            x=x,
            r=r,
            ITI_counter=ITI_counter,
            step_count=state.step_count + 1,
            trial_counter=trial_counter,
            theta=theta,
            keys=keys,
        )

        done = False
        extras = dict()

        episode_max = jax.lax.select(self.episode_length != -1, self.episode_length, max_int32)
        timestep = TimeStep(
            step_type=jnp.where(state.step_count < episode_max, StepType.MID, StepType.LAST),
            observation=Observation(
                agent_view=x,
                action_mask=jnp.ones(self.num_actions),
                step_count=state.step_count,
            ),
            reward=r,
            discount=1.,
            extras={},
        )

        return state, timestep


class RolloutSlice(eqx.Module):
    x: chex.Array
    r: chex.Array
    s: chex.Array
    ITI_counter: chex.Array
    theta: chex.Array
    a: chex.Array
    t: chex.Array

    actor_state: AgentState
    observer_state: AgentState

class Tape(RolloutSlice):
    actor: Any
    observer: Any

base_key = jax.random.key(0)
actor_key_ = jax.random.key(23)
observer_key_ = jax.random.key(42)

def environment_loop(env=None, actor=None, episode_length=None, observer=None, key_env=None, key_actor=actor_key_, key_observer=observer_key_, record_vec_state=True, handicap=max_int32):    
    if not handicap:
        handicap = max_int32
    episode_length = int(episode_length)

    if key_env is None:
        key_env = EnvKey(*jax.random.split(base_key, len(EnvKey._fields) + 1)[1:])

    def step_fn(state, key):
        world_state, agent_state, R = state
        w_s = world_state
        actor_state, observer_state = agent_state

        x, r, s, theta, ITI_counter = w_s.x, w_s.r, w_s.s, w_s.theta, w_s.ITI_counter
        R += r

        ## ACT ##
        action, actor_state_n = actor._policy(actor._params, observation=(x, r, s, theta, ITI_counter), state=actor_state, handicap=w_s.step_count >= handicap)

        if observer is not None:
            ## OBSERVE ##
            observer_state_with_actor_action = eqx.tree_at(lambda t: t.last_action, observer_state, actor_state.last_action,)
            action_o, observer_state_n = observer._policy(observer._params, observation=(x, r, s, theta, ITI_counter), state=observer_state_with_actor_action, handicap=w_s.step_count >= handicap)
        else:
            action_o, observer_state_n = None, observer_state

        ## evolve the world state ##
        # TODO control random state here
        w_sn, timestep = env.step(w_s, action)
        if not record_vec_state:
            actor_state_n_ = eqx.tree_at(lambda t: t.vec_state, actor_state_n, None)
            observer_state_n_ =  eqx.tree_at(lambda t: t.vec_state, observer_state_n, None)
        else:
            actor_state_n_ = actor_state_n
            observer_state_n_ = observer_state_n

        # record the last world state paired with the resulting observer state through the action in response to that world state
        rollout_slice = RolloutSlice(x=w_s.x, r=w_s.r, s=w_s.s, ITI_counter=w_s.ITI_counter, theta=w_s.theta, a=action, t=world_state.step_count, actor_state=actor_state_n_, observer_state=observer_state_n_)
        world_state_n = w_sn

        # if world_state.step_count % 100 == 0:
        #     loss = ...
        #     actor.step()
        #     R = 0.

        return (world_state_n, (actor_state_n, observer_state_n), R), rollout_slice

    def run_n_steps(state_0, key, steps):
        # random_keys = jax.random.split(key, n) 
        # todo control randomness here for best granularity
        # step_fn_ = checkify.checkify(step_fn, errors=checkify.all_checks)
        env_state_0, agent_state_0 = state_0

        # actor.observe_first(timestep) 

        with jax.disable_jit(False):
            R = 0.
            step_fn_ = scan_tqdm(episode_length, desc=f"Running {episode_length}")(step_fn)  # prev
            out = jax.lax.scan(step_fn_, (*state_0, R), steps)

        (world_state_f, agent_state_f, R), rollout = out
        return rollout
    
    # add a batch dimension if not present
    key_actor = jnp.atleast_1d(key_actor)
    key_observer = jnp.atleast_1d(key_observer)
    key_env = jax.tree.map(lambda x: jnp.atleast_1d(x), key_env)

    vals, treedef = jax.tree_util.tree_flatten(key_env)

    # tile the keys through broadcasting if no batch dimension is specified
    *vals, key_actor, key_observer = jnp.broadcast_arrays(*vals, key_actor, key_observer)

    key_env = treedef.unflatten(vals)

    # Instantiate a batch of environment states
    n_batch = key_actor.shape[0]
    logger.info(f"Running simulation of size (batch, steps)={(n_batch, episode_length)}")

    env_state_0, timestep = jax.vmap(env.reset)(key_env=key_env)

    actor_state_0 = jax.vmap(actor._init)(key_actor)
    observer_state_0 = jax.vmap(observer._init)(key_observer) if observer is not None else None
    agent_state_0 = (actor_state_0, observer_state_0)
    
    steps = jnp.arange(episode_length)
    rollout = jax.jit(eqx.filter_vmap(run_n_steps, in_axes=(0, 0, None)))((env_state_0, agent_state_0), key_env, steps)

    # Shape and type of given rollout:
    # TimeStep(step_type=(7, 5), reward=(7, 5), discount=(7, 5), observation=(7, 5, 6, 6, 5), extras=None)
    rollout = jax.tree_map(lambda x: x.squeeze(), rollout)
    tape = Tape(**{f.name: getattr(rollout, f.name) for f in fields(rollout)}, actor=actor, observer=observer)

    return tape


from jax_tqdm import scan_tqdm
if __name__ == "__main__":
    env = MouseWorld(backend="thy", ITI_mean=10.)
    actor = RandomAgent()     
    actor_core = make_bayes_actor(env)
    actor = GenericActor(actor_core, jit=True, random_key=jax.random.key(1))
    env_key = EnvKey(jax.random.key(0))
    state, timestep = env.reset(env_key)
    actor.observe_first(timestep)   

    tape_shape = (int(1e6), int(1e1))
    rollout = environment_loop(env, actor, int(1e6))
    rollout


    # import jax.numpy as jnp
    # thetas = jnp.linspace(.1, .9, 100)

    # t_wait = env.t_star__X1(thetas)
    # t_act = env.t_N_cons(t_wait, thetas)
    # p_s1 = env.p_s1_thsd__theta(thetas)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # ax.plot(thetas, t_wait, label="t_wait")
    # ax.plot(thetas, t_act, label="t_act")
    # ax2.plot(thetas, p_s1, label="p_s1", color="C2")

    # ax.set_xlabel("theta")
    
    # fig.legend(loc="outside upper right")
    # plt.show()
