from functools import lru_cache
import numba as nb
import numba 
from numba import njit, vectorize
import numpy as np


dt = 1
@njit
def find_end_N_cons_GOs_(x, N_cons):
    if x.shape[-1] < N_cons:
        return np.nan
    for i in range(len(x) - N_cons + 1):
        sequence = x[i:i+N_cons]
        if (sequence == 1).all():
            return i + N_cons
    return np.nan

# @np.vectorize
# @lru_cache
@njit
def find_end_N_cons_GOs(x, N_cons_GO):
    avgs = np.zeros(x.shape[0])
    for i_tr in range(x.shape[0]):
        avgs[i_tr] = find_end_N_cons_GOs_(x[i_tr], N_cons_GO)
    return avgs

# @nb.jit(nopython=True)
def infer_nb(x, llh, T_S, T_theta, s=None, a=None):
    N_theta = T_theta.shape[0]
    N_trials, T_tot = x.shape[:2]
    p_S_TH = np.zeros((N_trials, T_tot, 2, N_theta))

    #initial condition
    p_S_TH[:, 0] = 1
    p_S_TH[:, 0] /= np.sum(p_S_TH[:, 0])

    for i_tr in range(N_trials):
        for ti in range(1, T_tot):
            llh_cue = llh[1] if x[i_tr, ti] else llh[0]
            p_S_TH[i_tr, ti] = llh_cue*(T_S@p_S_TH[i_tr, ti-1]@T_theta.T)
            p_S_TH[i_tr, ti] /= np.sum(p_S_TH[i_tr, ti])

            assert p_S_TH[i_tr, ti].sum() > 0

            if a is not None:
                if a[i_tr, ti] == 1:
                    if s[i_tr, ti-1] == 1:
                        p_S_TH[i_tr, ti][0, :] = p_S_TH[i_tr, ti][1, :] 
                    elif s[i_tr, ti-1] == 0:
                        p_S_TH[i_tr, ti][0, :] = p_S_TH[i_tr, ti][0, :] 
                    else:
                        raise ValueError
                    p_S_TH[i_tr, ti][1,:] = 0.
                    p_S_TH[i_tr, ti] /= np.sum(p_S_TH[i_tr, ti])
                elif a[i_tr, ti] == 0:
                    pass
                else:
                    raise ValueError
            assert p_S_TH[i_tr, ti].sum() > 0
    return p_S_TH

@nb.jit(nopython=True)
def generate_trajs_nb(N_trials, T_tot, lmbd_theta, lmbd_safe, lmbd_act, THETAS):
    
    T_tot = int(T_tot)
    ts = np.arange(T_tot) * dt

    # thetas = np.empty((N_trials, T_tot), )
    # s = np.empty((N_trials, T_tot), )
    # x = np.empty((N_trials, T_tot), )
    # a = np.zeros((N_trials, T_tot), )

    # all_trans_x = np.empty((N_trials, T_tot), )
    # all_trans_th = np.empty((N_trials, T_tot), )
    # all_trans_s = np.empty((N_trials, T_tot), )
    # all_trans_a = np.empty((N_trials, T_tot),)

    # N_trans_th = np.empty((N_trials,), )
    # N_trans_s = np.empty((N_trials,), )
    # N_trans_x = np.empty((N_trials,), )
    # N_trans_a = np.empty((N_trials,), )

    thetas = np.zeros((N_trials, T_tot), dtype=numba.float32)
    s = np.zeros((N_trials, T_tot), dtype=numba.boolean)
    x = np.zeros((N_trials, T_tot), dtype=numba.boolean)
    a = np.zeros((N_trials, T_tot), dtype=numba.boolean)

    all_trans_x = np.empty((N_trials, T_tot), dtype=numba.int16)
    all_trans_th = np.empty((N_trials, T_tot), dtype=numba.int16)
    all_trans_s = np.empty((N_trials, T_tot), dtype=numba.int16)
    all_trans_a = np.empty((N_trials, T_tot), dtype=numba.int16)

    N_trans_th = np.empty((N_trials,), dtype=numba.int16)
    N_trans_s = np.empty((N_trials,), dtype=numba.int16)
    N_trans_x = np.empty((N_trials,), dtype=numba.int16)
    N_trans_a = np.empty((N_trials,), dtype=numba.int16)


    for i_tr in range(N_trials):
        ti = 0
        t = 0
        i_context = 0
        i_trans_s = 0

        while ti < T_tot:
            t_act = np.random.exponential(scale=lmbd_theta**-1)
            ti_act = int(max(t_act // dt, 1))

            if ti + ti_act > T_tot:
                break

            theta = np.random.choice(THETAS)
            all_trans_th[i_tr, i_trans_s] = ti
            i_context += 1
            i_trans_s += 1

            thetas[i_tr, ti:ti+ti_act] = theta
            ti += ti_act
            t += t_act

        N_trans_th[i_tr] = i_trans_s

    # generate random Poissonian actions and states
    for i_tr in range(N_trials):
        t = 0
        ti = 0
        i_trans_s = 0
        i_trans_a = 0

        t_trans_s = np.random.exponential(scale=lmbd_safe**-1)
        while t_act < 1:
            t_act = np.random.exponential(scale=lmbd_act**-1)
            
        ti_trans_s = int(max(t_trans_s // dt, 1))
        ti_act = int(max(t_act // dt, 1))

        while ti + ti_act < T_tot:
            if t_act < t_trans_s:
                s[i_tr, ti:ti+ti_act] = 0
            else:
                s[i_tr, ti:ti+ti_trans_s] = 0
                s[i_tr, ti+ti_trans_s:ti+ti_act] = 1
                all_trans_s[i_tr, i_trans_s] = ti+ti_trans_s
                i_trans_s += 1

            a[i_tr, ti+ti_act] = 1
            all_trans_a[i_tr, i_trans_a] = ti+ti_act
            i_trans_a += 1

            ti += ti_act
            t += t_act
            
            t_trans_s = np.random.exponential(scale=lmbd_safe**-1)
            while t_act < 1:
                t_act = np.random.exponential(scale=lmbd_act**-1)
            ti_trans_s = int(max(t_trans_s // dt, 1))
            ti_act = int(max(t_act // dt, 1))
            

        N_trans_s[i_tr] = i_trans_s
        N_trans_a[i_tr] = i_trans_a

    
    for i_tr in range(N_trials):
        for ti in range(T_tot):
            if s[i_tr, ti] == 1:
                x[i_tr, ti] = 1
            else:
                x[i_tr, ti] = np.random.rand() < thetas[i_tr, ti]

    # get an array of up-down tuples for x as well
    for i_tr in range(N_trials):
        # True if there is a change
        delta_x = np.diff(x[i_tr])
        trans_x_ = np.where(delta_x)[0]
        i_trans_x = 0
        for ti in trans_x_:
            t = ts[ti]
            all_trans_x[i_tr, i_trans_x] = ti
            i_trans_x += 1

        N_trans_x[i_tr] = i_trans_x

    return x, s, thetas, a, all_trans_x, all_trans_s, all_trans_th, all_trans_a, N_trans_x, N_trans_s, N_trans_th, N_trans_a

@njit
def get_poisson_trials(N_trials, T_tot, theta, lmbd_s):
    s = np.zeros((N_trials, T_tot), dtype=np.int32)
    i_trans = np.random.exponential(lmbd_s**-1, size=(N_trials,)).astype(np.int32)

    for i_tr in range(N_trials):
        s[i_tr, i_trans[i_tr]:] = 1

    x = ((s == 1) | (np.random.uniform(0, 1, size=s.shape) < theta))
    return x, s
