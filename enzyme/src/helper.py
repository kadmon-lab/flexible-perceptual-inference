from matplotlib.collections import LineCollection
import torch
from enzyme import CACHE_DIR, FIGPATH, PRJ_ROOT

import joblib
from pathlib import Path

def kp_to_filename(kp):
    import jax
    id_hash = joblib.hash(kp)[:4]

    keystr_human_readable = jax.tree_util.keystr(kp).replace('[', '').replace(']', '').replace(', ', '_').replace(' ', '_').replace("'", "")
    id_hash += ('__' + keystr_human_readable)

    return id_hash

def save_pytree(tree, path):
    import jax
    # Create a directory named after the pickle file (without the .pkl extension)
    dir_name = path.with_suffix('')  # Remove the extension
    dir_name.mkdir(exist_ok=True)

    # delete directory if it exists
    # Check if all files in the directory are .npz files
    all_npz_files = all(f.suffix == "" for f in dir_name.iterdir() if f.is_file())
    has_subdirs = any(f.is_dir() for f in dir_name.iterdir())
    
    if all_npz_files and not has_subdirs:  # some safeguards
        import shutil
        import pickle
        # If only .npz files are in the directory, proceed with deletion
        shutil.rmtree(dir_name)
    else:
        # Otherwise, raise an error or handle the situation as appropriate
        raise ValueError(f"Directory {dir_name} does not seem to be right")
    dir_name.mkdir(exist_ok=True)
    
    # Save each item in the dictionary as a separate .npz file
    def save_leaf(kp, v):
        filename = kp_to_filename(kp)
        try: 
            size = np.prod(v.shape)
            if size > 1e6: print(f"Warning: Saving large field {filename} of size {size}")
        except AttributeError:
            pass

        joblib.dump(v, dir_name / f'{filename}.joblib', 
                    compress=9,  # maybe disable
                    protocol=pickle.HIGHEST_PROTOCOL)  # save the value
        return

    jax.tree_util.tree_map_with_path(save_leaf, tree)


    # also save the pytree structure so that we can reinstantiate it!
    empty_tree = jax.tree_util.tree_map(lambda x: None, tree)
    joblib.dump(empty_tree, dir_name / 'treedef')

def load_pytree(path):
    import jax

    # Convert path to a Path object
    path = Path(path)
    
    # Directory path where .npz files would be saved
    dir_name = path.with_suffix('')  # Remove the extension
    
    empty_tree = joblib.load(dir_name / 'treedef')
    
    def load_leaf(kp, _):
        filename = kp_to_filename(kp)
        v = joblib.load(dir_name / f'{filename}.joblib')
        return v

    tree = jax.tree_util.tree_map_with_path(load_leaf, empty_tree, is_leaf=lambda x: x is None)
    
    return tree

def test_save_and_load_almost_pytree_joblib():
    import tempfile
    import shutil
    import jax
    import numpy as np
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    # test_path = Path(temp_dir) / "test_pytree.pkl"
    test_path = Path().home() / "test_pytree.pkl"
    
    try:
        # Create a small pytree (dictionary with JAX arrays)
        pytree = {
            'a': np.array([1.0, 2.0, 3.0]),
            'b': {'c': np.array([4.0, 5.0])}
        }
        
        # Save the pytree
        save_pytree(pytree, test_path)

        # Load the pytree
        loaded_pytree = load_pytree(test_path)

        print(loaded_pytree)
        print(pytree)

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

# test save and load
if __name__ == "__main__":
    test_save_and_load_almost_pytree_joblib()


def save_plot(
        name,
        path=FIGPATH,
        fig=None,
        file_formats=["svg", "pdf", "png"],
        **save_args
):  
    transparent = save_args.get("transparent", True)
    for file_format in file_formats:
        fig.savefig(path / (name + f".{file_format}"), transparent=True, **save_args)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.scale import ScaleBase, register_scale
import matplotlib.transforms as mtransforms
from matplotlib.ticker import LogLocator, LogFormatterMathtext

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import ScaleBase, register_scale
import matplotlib.transforms as mtransforms
from matplotlib.ticker import LogLocator, LogFormatter

def riemann(integrand, a, b, n, args=()):
    dx = (b - a) / n
    x = np.linspace(a, b, n)
    return np.sum(integrand(x, *args))*dx, 0.

tr_s = lambda s: -jnp.log(jnp.abs(1-s) + 1e-6)  # jnp.finfo(jnp.float32)
tr_s_inv = lambda s: 1 - jnp.exp(-jnp.abs(s))
tr_th = lambda theta: jnp.log(jnp.abs(theta/(1-theta + 1e-6)) + 1e-6)
th_th_inv = lambda theta: jnp.abs(jnp.exp(theta)/(1+jnp.exp(theta)))

class CustomLogLocator(LogLocator):
    def tick_values(self, vmin, vmax):
        # Transform the range to cluster close to 0
        vmin = 1 - vmin
        vmax = 1 - vmax

        # Get the tick locations in the transformed range
        ticks = super().tick_values(vmin, vmax)

        # Transform the tick locations back to the original range
        return 1 - ticks
    
from scipy.stats import beta
from math import gamma

from numba import njit, vectorize
beta = lambda alpha, beta, x: gamma(alpha+beta)/(gamma(alpha)*gamma(beta)) * x**(alpha-1) * (1-x)**(beta-1)
# beta = vectorize(nopython=True)(beta)
    
def get_beta_prior(theta_mean, theta_vals, N_pseudo=1):
    a_beta_dist = N_pseudo*theta_mean + 0
    b_beta_dist = N_pseudo*(1-theta_mean) + 0
    p_TH_prior = beta(a_beta_dist, b_beta_dist, theta_vals)
    p_TH_prior /= p_TH_prior.sum()
    return p_TH_prior
    
def logitspace(a, b, n, eps=1e-3):
    if n % 2 == 0: n += 1
    x = np.concatenate([a + np.geomspace(eps, (b - a)/2, (n + 1) // 2), b - np.geomspace(eps*b, (b - a)/2, (n + 1) // 2)[:-1][::-1]])
    return x

def mystep(ax, x,y, where='post', colors=None, **kwargs):
    assert where in ['post', 'pre']
    x = np.array(x)
    y = np.array(y)
    if where=='post': y_slice = y[:-1]
    if where=='pre': y_slice = y[1:]
    X = np.c_[x[:-1],x[1:],x[1:]]
    Y = np.c_[y_slice, y_slice, np.zeros_like(x[:-1])*np.nan]
    if not ax: ax=plt.gca()

    x_, y_ = X.flatten(), Y.flatten()
    if colors is not None:
        points = np.array([x_, y_]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=colors)

        # Set the values used for colormapping
        # lc.set_array(dydx)
        # lc.set_linewidth(2)
        line = ax.add_collection(lc)
    else:
        line = ax.plot(x_, y_, **kwargs)
    return (line,)


def save_plot(
        name,
        path=FIGPATH,
        fig=None,
        file_formats=["svg", "pdf", "png"],
        **save_args
):  
    transparent = save_args.get("transparent", True)
    if not path.exists():
        path.mkdir(parents=True)
    for file_format in file_formats:
        if file_format == "png":
            transparent = False
            save_args.update({"dpi": 600})
        fig.savefig(path / (name + f".{file_format}"), transparent=transparent, **save_args)

from pathlib import Path
from joblib import Memory
memory = Memory(CACHE_DIR, verbose=0)

def get_manager(agent_params, mouse_task_params, manager_params, manager=None, ):
    sim_params = dict(agent_params, **mouse_task_params, **manager_params) 
    
    agent = Actor_Critic(**agent_params)
    device = manager_params['device']
    agent.load_state_dict(torch.load(agent_params['load_path'], map_location=torch.device(device)))

    if manager is None:
        empty_manager = run_simulation(mouse_task, sim_params, agent, plot_episode = False, run=False)
        data = get_data_dict(agent_params, mouse_task_params, manager_params)
        empty_manager.data.from_dictionary(data)
        empty_manager.sim.preprocess_data(empty_manager)
        manager = empty_manager
    else:
        # uses the passed manager object to store the data
        _ = get_data_dict(agent_params, mouse_task_params, manager_params, manager = manager)

    return manager

@memory.cache(ignore=['manager'], verbose=10)
def get_data_dict(agent_params, mouse_task_params, manager_params, manager=None):
    from enzyme.src.main.run_simulation import run_simulation
    from enzyme.src.mouse_task.mouse_task import mouse_task_
    from enzyme.src.network.Actor_Critic import Actor_Critic
    agent = Actor_Critic(**agent_params)
    device = manager_params['device']
    agent.load_state_dict(torch.load(agent_params['load_path'], map_location=torch.device(device)))

    sim_params = dict(agent_params, **mouse_task_params, **manager_params)
    if manager is None: 
        manager = run_simulation(mouse_task_, sim_params, agent, plot_episode = False)
    data = manager.data.to_dictionary()   

    keys = list(data.keys())
    purge_keys = ['backbone', 'stim','gos', 'nogos', "lick_prob", "value", "Qs", "LTM", "f_gate", "i_gate", "c_gate", "o_gate"]
    purge_keys = ["LTM", "f_gate", "i_gate", "c_gate", "o_gate"]


    for k in keys:
        v = data[k]
        print(v.dtype)
        elem = v[0]
        shape = elem.shape if isinstance(elem, torch.Tensor) else np.array(elem).shape
        dtype = elem.dtype if isinstance(elem, torch.Tensor) else np.array(elem).dtype

        if (len(shape) > 1 and False) or (k in purge_keys):
            # network weights, do not persist
            print(f"deleting {k} of shape ({len(v)}, {shape}, {dtype})")
            del data[k]
        else:
            print(f"persisting {k} of shape ({len(v)}, {shape}, {dtype})")

    return data

def notebook_cache(mem, module, **mem_kwargs):
    """
    https://stackoverflow.com/questions/75202475/joblib-persistence-across-sessions-machines/
    """
    def cache_(f):
        f.__module__ = module
        f.__qualname__ = f.__name__
        return mem.cache(f, **mem_kwargs)
    return cache_