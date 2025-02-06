from typing import NamedTuple
from matplotlib import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from enzyme.colors import *
from enzyme.plot_helper import fill_between
from enzyme.scripts.plot_config import *
from enzyme.src.bayes.utils import calc_waiting_times

import logging
logger = logging.getLogger(__name__)

def batch_to_MU_STD(all_data, data_field, agent_name, agent_num):
    d = all_data[agent_name][data_field]
    if isinstance(d, (float, int)):
        data = np.zeros(agent_num)
    else:
        size = d.shape
        if len(size) == 1:
            data = np.zeros((agent_num, size[0]))
        if len(size) == 2:
            data = np.zeros((agent_num, size[0], size[1]))

    for a in range(agent_num):
        name = agent_name
        if a > 0:
            name = name + "_" + str(a) 
        data[a] = all_data[name][data_field]
    return data.mean(0), data.std(0), data

def batch_to_concat(all_data, data_field, agent_name, agent_num):
    data = all_data[agent_name][data_field]
    for a in range(1, agent_num):
        name = agent_name + "_" + str(a) 
        next_data = all_data[name][data_field]
        data = np.concatenate((data, next_data), axis = -1)
    return data

def batch_to_concat_group(all_data, data_fields, agent_name, agent_num, index):
    reference_field = data_fields[0] # must not be index field
    init_inds = np.zeros(agent_num, dtype = int)
    next_len = len(all_data[agent_name][reference_field])

    for a in range(1, agent_num):
        name = agent_name + "_" + str(a) 
        init_inds[a] = init_inds[a-1] + next_len
        next_len = len(all_data[name][reference_field])

    datas = np.zeros(len(data_fields), dtype = object)
    for d_i, (data_field, is_index) in enumerate(zip(data_fields, index)):
        data = all_data[agent_name][data_field]
        for a in range(1, agent_num):
            init = is_index * init_inds[a]
            name = agent_name + "_" + str(a) 
            next_data = all_data[name][data_field] + init
            data = np.concatenate((data, next_data), axis = -1)
        datas[d_i] = data
    return [data for data in datas]

def batch_to_object_array(all_data, data_field, agent_name, agent_num):
    data = np.empty(agent_num, dtype = object)
    for a in range(agent_num):
        name = agent_name
        if a > 0:
            name = name + "_" + str(a) 
        data[a] = all_data[name][data_field]
    return data

def plot_heatmap_and_dists(agent, ax_Q_heatmap, ax_x_hist, ax_y_hist, 
        batch = False, agent_name = None, agent_num = None, all_data = None, ylim = None, return_bayes_DV = False):

    PGOs = agent["PGO_range"]
    PGO_N = len(PGOs)

    for ax in [ax_Q_heatmap, ax_x_hist, ax_y_hist]:
        ax.cla()

    # axes
    ax_x_hist.sharex(ax_Q_heatmap)
    ax_y_hist.sharey(ax_Q_heatmap)
    logger.info("collecting variables")
    Y_var = "QDIFF_flat" 
    if batch:
        vars = [Y_var, "bayes_DV", "consec_inds", "PGO_inds_for_heatmap", "ACT_inds", "GO_inds"]
        index = [False]*4 + [True, True]
        Y, x, consec_flat, PGO_flat, ACT_inds, GO_inds = batch_to_concat_group(\
                        all_data, vars, agent_name, agent_num, index)
    else:
        x = agent["bayes_DV"]
        consec_flat = agent["consec_inds"]
        ACT_inds = agent["ACT_inds"] 
        Y = agent[Y_var]
        GO_inds  = agent["GO_inds"]
        PGO_flat = agent["PGO_inds_for_heatmap"]

    upres = 1 


    # x
    bayes_res = .01/upres
    bayes_bin_edges = np.arange(-2, .5, bayes_res)
    bayes_bin_centers = (bayes_bin_edges[1:] + bayes_bin_edges[:-1]) / 2

    # y
    net_res = .1/upres
    net_bin_edges = np.arange(-12, 5, net_res)
    net_bin_centers =  (net_bin_edges[1:] + net_bin_edges[:-1]) / 2


    consec = 10
    GO_hist = []
    logger.info("getting heatmaps")

    INDS = np.arange(x.size)
    GO_where = np.isin(INDS, GO_inds)
    ACT_where = np.isin(INDS, ACT_inds)

    for c in range(consec):
        where_c = (consec_flat == (c+1))
        H, _, _ = np.histogram2d(x[GO_where & where_c & ~ACT_where], 
                                 Y[GO_where & where_c & ~ACT_where], 
                                 bins=(bayes_bin_edges, net_bin_edges), 
                                 density=False,  # ensures things sum to 1 if True
                                 )
        GO_hist.append(H)

    GO_hist = np.stack(GO_hist, axis=0)  # (c, nx, ny)
    GO_hist /= GO_hist.sum()

    # now ACT_hist
    ACT_hist, _, _ = np.histogram2d(x[ACT_where], 
                                 Y[ACT_where], 
                                 bins=(bayes_bin_edges, net_bin_edges), 
                                 density=False)  # (nx, ny)
    ACT_hist /= ACT_hist.sum()  # we don't get a proper density
    
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    from sklearn.linear_model import LinearRegression
    from scipy.ndimage import gaussian_filter

    smooth_GO = gaussian_filter(GO_hist, sigma = 2, 
                                # axes=(-2, -1)
                                )
    
    smooth_ACT = gaussian_filter(ACT_hist, sigma = 1)

    ax = ax_Q_heatmap
    ax.cla()
    R_alph = .6
    green = [(1, 0, 0, 0), (0, 1, 0)] 
    red =[(1, 0, 0, 0), (1, 0, 0, R_alph)]
    n_bins = 100  

    vmax_GO =.00006 
    vmax_ACT = .002

    logger.info("plotting heatmaps")
    for vmax, data, cmap_name in zip([vmax_GO, vmax_ACT], [smooth_GO, smooth_ACT], ["black_to_green", "black_to_red"]):
        if cmap_name == "black_to_green":
            for c in range(consec):
                color = plt.cm.RdYlGn(1-.8*(c/consec)**3)
                color = [(color[0], color[1], color[2], .01), (color[0], color[1], color[2], 1)]
                cm = LinearSegmentedColormap.from_list(cmap_name, color, N=n_bins)
                # ax.imshow(data[c].T, cmap=cm, vmax = vmax, aspect='auto', extent=(net_uniques.min(), net_uniques.max()
                ax.pcolormesh(bayes_bin_centers, net_bin_centers, data[c].T, 
                              cmap=cm, rasterized=True,
                              vmax = vmax)
        else:
            cm = LinearSegmentedColormap.from_list(cmap_name, red, N=n_bins)
            # ax.imshow(data.T, cmap=cm, vmax=vmax, aspect='auto')
            ax.pcolormesh(bayes_bin_centers, net_bin_centers, data.T, cmap=cm, vmax=vmax, rasterized=True)
        
    ax.axhline(0, c = 'r', linestyle = '--', alpha = .5)
    ax.axvline(0, c = 'r', linestyle = '--', alpha = .5)
    ax.set_xlabel(r"Bayesian distance from threshold $\delta(\mathcal{P}_t)$")
    ax.set_ylabel(r"network output $Q_t$")
    ax.set_xlim([-.7, x[ACT_where].max()])
    if ylim is not None: 
        ax.set_ylim(ylim)
    
    """ show p(action)
    ax.axvline(-.84, c = 'k', linewidth = 2, zorder = 100)
    from matplotlib import colors as C
    transformed_Y = net_targs 
    transformed_Y = 1 / (1 + np.exp(-2 * transformed_Y))
    X_mesh, Y_mesh = np.meshgrid([-.9, -.86], transformed_Y)  
    norm = C.Normalize(vmin=-.1, vmax=.5)
    
    ax.pcolormesh(X_mesh, Y_mesh,
        np.vstack([transformed_Y, transformed_Y]).T, 
        cmap= plt.cm.YlOrRd, norm=norm, shading='auto')
    ax.set_xticks([]);    ax.set_yticks([])
    """
    logger.info("getting regression")

    """ GO cue regresssion """
    F = LinearRegression().fit(Y[GO_inds][:, None], x[GO_inds][:, None])
    R2 = F.score(Y[GO_inds][:, None], x[GO_inds][:, None])


    ax.text(0.5, 0.1, f" $R^2$ = {R2:.2f}", transform=ax.transAxes, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=.7))
    ax.plot(F.predict(net_bin_centers[:,None]).squeeze(), net_bin_centers, color= 'k', alpha =.8)  # @ JS: Now everything is in data coordinates, but I'm still too stupid to draw the 45° line. Probably you see it straightaway!

    # replace with subfig.suptitle if needed
    # ax.text(.5, 1.1, f"decision variables", ha='center', va='bottom', transform=ax.transAxes, fontsize=plt.rcParams['axes.titlesize'])

    # distributions on top of heatmap
    PGO_range = PGOs.astype(float)
    PGO_COLOR = cmap_cool(PGO_range) 
    logger.info("getting histograms")

    X_hist = []
    Y_hist = []
    for p_i, p in enumerate(PGO_range):
        H, _ = np.histogram(x[(PGO_flat == p) & ACT_where], bayes_bin_edges, density=False)
        H = H / H.sum()
        H = gaussian_filter(H, sigma = 2)
        X_hist.append(H)

        H, _ = np.histogram(Y[(PGO_flat == p) & ACT_where], net_bin_edges, density=False)
        H = H / H.sum()
        H = gaussian_filter(H, sigma = 2)
        Y_hist.append(H)

    X_hist = np.stack(X_hist, 0)  # (PGO, ...)
    Y_hist = np.stack(Y_hist, 0)  # (PGO, ...)


    logger.info("plotting histograms")
    for p_i, p in enumerate(PGO_range):
        # For x histogram
        inds_x = np.where(X_hist[p_i] > 0.002)[0]
        xax_x = bayes_bin_centers[inds_x]
        yax_x = X_hist[p_i, inds_x]
        ax_x_hist.fill_between(xax_x, 0, yax_x, color=PGO_COLOR[p_i], alpha=0.6)
        ax_x_hist.plot(xax_x, yax_x, color=PGO_COLOR[p_i], alpha=.8,  linewidth = 1)  # Adding normal line
        
        # For y histogram
        inds_y = np.where(Y_hist[p_i] > 0.002)[0]
        xax_y = net_bin_centers[inds_y]
        yax_y = Y_hist[p_i, inds_y]
        ax_y_hist.fill_betweenx(xax_y, 0, yax_y, color=PGO_COLOR[p_i], alpha=0.6)
        ax_y_hist.plot(yax_y, xax_y, color=PGO_COLOR[p_i], alpha=.8, linewidth = 1)  # Adding normal line

    ax_x_hist.set_xticks([])
    ax_x_hist.set_yticks([])
    ax_x_hist.tick_params(axis='both', which='both', length=0)
    ax_x_hist.spines['top'].set_visible(False)
    ax_x_hist.spines['right'].set_visible(False)
    ax_x_hist.spines['left'].set_visible(False)
    ax_x_hist.spines['bottom'].set_visible(False)

    ax_y_hist.set_xticks([])
    ax_y_hist.set_yticks([])
    ax_y_hist.tick_params(axis='both', which='both', length=0)
    ax_y_hist.spines['top'].set_visible(False)
    ax_y_hist.spines['right'].set_visible(False)
    ax_y_hist.spines['left'].set_visible(False)
    ax_y_hist.spines['bottom'].set_visible(False)


    for tick in ax_x_hist.get_xticklabels() + ax_x_hist.get_yticklabels():
        tick.set_visible(False)
    for tick in ax_y_hist.get_xticklabels() + ax_y_hist.get_yticklabels():
        tick.set_visible(False)

    if return_bayes_DV:
        bayes_DV_all = x
        bayes_DV_at_act = x[ACT_where]
        bayes_DV_pre_act = x[~ACT_where]
        bayes_pre_act_PGO = PGO_flat[~ACT_where]
        bayes_at_act_PGO = PGO_flat[ACT_where]
        return bayes_DV_at_act, bayes_at_act_PGO, bayes_DV_pre_act, bayes_pre_act_PGO, bayes_DV_all, PGO_flat

def plot_learning(ax, GOs_2, training_first_waits, testing_first_waits, **largs):
    # plot_first_adapt(ax, np.array([.3, .7]), training_first_waits, testing_first_waits, marker=marker_mice)

    largs = dict(linewidth=1, edgecolor='k', capsize=2, width=.9)
    pads = [0, 0]
    PGOs_ = np.array([.3, .7])
    pos = [[0, 3], 
        [1, 4]]
    pos_flat_sort = np.sort(np.array(pos).flatten())

    def mode_kde(data):
        kde = stats.gaussian_kde(data.flatten())
        x = np.linspace(data.min(), data.max(), 1000)
        y = kde(x)
        mode = x[np.argmax(y)]
        return mode

    SY_trains = []
    MY_trains = []
    SY_tests = []
    MY_tests = []

    for i_GO, PGO in enumerate():
        i_GO = np.argmin(np.abs(PGOs_ - PGO))
        c = cmap(PGO)

        if training_first_waits is not None:
            train_waits = training_first_waits[:,i_GO]
            
            MY_train = np.nanmean(train_waits, axis=0)
            SY_train = np.nanstd(train_waits, axis=0) / np.sqrt(train_waits.shape[0])
            SY_trains.append(SY_train)
            MY_trains.append(MY_train)

            # compute the mode after kde smoothing
            mode = mode_kde(train_waits)


            ax.bar(pos[i_GO][0], mode, color=c, alpha=.4, **largs, linestyle='--', yerr=SY_train)

        test_waits = testing_first_waits[:, i_GO]
        MY_test = np.nanmean(test_waits, axis=0)
        SY_test = np.nanstd(test_waits, axis=0) / np.sqrt(test_waits.shape[0])
        SY_tests.append(SY_test)
        MY_tests.append(MY_test)

        mode = mode_kde(test_waits)
        ax.bar(pos[i_GO][1], mode, color=c, alpha=1., **largs, linestyle='-', yerr=SY_test)


    # calculate p-values
    pvals = []
    for i_GO, PGO in enumerate(PGOs_2):
        i_GO = np.argmin(np.abs(PGOs_ - PGO))
        test_waits_PGO = testing_first_waits[:, i_GO]
        train_waits_PGO = training_first_waits[:, i_GO]

        pval = stats.ttest_ind(test_waits_PGO, train_waits_PGO, axis=0, equal_var=False).pvalue
        print(pval)
        pvals.append(pval)

        

    # make blended transform
    trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)

    strings = ['ns', r'$\ast$', r'$\ast\ast$', r'$\ast\ast\ast$', r'$\ast\ast\ast\ast$']
    thsds = [1., .05, .01, .001, .0001]

    pad_ = .1
    pad__ = .125

    # train
    ax.plot([pos_flat_sort[0], pos_flat_sort[0], pos_flat_sort[1], pos_flat_sort[1]], [1-pad__, 1-pad_, 1-pad_, 1-pad__], 'k-', lw=1, transform=trans)
    ax.text((pos_flat_sort[0] + pos_flat_sort[1]) / 2, 1-pad_ + .01, 
            strings[np.where(pvals[0] < thsds)[0][-1]],  # get the index of the p-values that are below the threshold
            ha='center', va='bottom', transform=trans)

    # test
    ax.plot([pos_flat_sort[2], pos_flat_sort[2], pos_flat_sort[3], pos_flat_sort[3]], [1-pad__, 1-pad_, 1-pad_, 1-pad__], 'k-', lw=1, transform=trans)
    ax.text((pos_flat_sort[2] + pos_flat_sort[3]) / 2, 1-pad_ + .01, 
            strings[np.where(pvals[1] < thsds)[0][-1]],  # get the index of the p-values that are below the threshold
            ha='center', va='bottom', transform=trans)
            


    ax.margins(y=.2)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel(r"$\tau_w$")

def plot_t_wait_dist(ax, PGOs_data, t_waits, pad=0., bins=None, bw_method=None, mode_mode='raw', dist_mode="hist", alpha_fill=.2, PGO_local = None, **largs,):
    """
    t_wait = N_PGO, N_mice, 
    """
    from matplotlib.mlab import GaussianKDE

    metrics = {}
    class Metrics(NamedTuple):
        mode: float
        mean: float
        std: float
        std_l: float
        std_r: float

    PGOs = PGO_show if PGO_local is None else PGO_local
    if PGOs_data.ndim == 1 and not hasattr(t_waits[0], '__len__'):
        PGOs_unique = np.unique(PGOs_data, return_index=True)[0].astype(float)
        t_waits_flat = t_waits.astype(float)
        if bins is None: 
            bins = np.concatenate([PGOs_unique - .01, [PGOs_unique[-1] + .01]])
        PGO_idcs = np.digitize(PGOs_data, bins=bins)
        t_waits = [t_waits_flat[PGO_idcs == i] for i in range(1, len(PGOs_unique) + 1)]
        PGOs_data = np.round(bins, 1)
    
    for i_GO, PGO in enumerate(PGOs):
        c = cmap_cool(PGO)

        handles, labels = [], []
        i_GO = np.argmin(np.abs(PGOs_data - PGO))
        t_waits_ = t_waits[i_GO]
        t_waits_ = t_waits_[~np.isnan(t_waits_)]

        PGO = PGOs_data[i_GO]

        # estimate a de nsity
        kde = GaussianKDE(t_waits_, bw_method=bw_method)
        x = np.linspace(t_waits_.min(), t_waits_.max(), 100)
        y = kde(x)
        

        # get the most frequent value
        if mode_mode == 'raw':
            mode, count = stats.mode(t_waits_)
            # print(mode, count)
            mode = mode
        else:
            mode = x[np.argmax(y)]

        mean = np.mean(t_waits_)
        std = np.std(t_waits_)
        std_r = (t_waits_ - mean)[t_waits_ > mean].std()
        std_l = (t_waits_ - mean)[t_waits_ < mean].std()

        offset = 0.
        # print(len(mice.all_trial_waits[:, i_mice]))
        if dist_mode == 'violin':
            vplot_kwargs = dict(
                points=60,
                widths=0.1,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            
            parts = ax.violinplot([t_waits_], positions=[PGO + pad], vert=False, **dict(vplot_kwargs, bw_method=bw_method))
            for pc in parts['bodies']:
                pc.set_facecolor(c)
                # pc.set_edgecolor('black')
                pc.set_alpha(.2)


            ax.plot(mode, PGO + pad, color=(c, .8), markerfacecolor=(c, 1.),  markeredgecolor=mec, **largs)
            ax.set_yticks(PGOs)
            ax.set_yticklabels(PGOs)
            ax.set_ylabel(r"$\theta$", rotation=0, va="center", labelpad=6.)

        elif dist_mode == 'hist':
            n, x = np.histogram(t_waits_, bins=np.arange(0, 30, 1), density=True)


            bin_centers = 0.5*(x[1:]+x[:-1])
            ax.plot(bin_centers, n, color=c, alpha=.5)
            ax.fill_between(bin_centers, 0, n, color=c, alpha=alpha_fill)
            ax.set_ylabel(r'$P(\tau_w$)')
            ax.set_yticks([])
        elif dist_mode == 'kde':
            ax.plot(x, y, color=c, alpha=.5)
            ax.fill_between(x, 0, y, color=c, alpha=alpha_fill)
            ax.set_ylabel(r'$P(\tau_w$)')
            ax.set_yticks([])
        else:
            raise ValueError
        
       
        metrics[PGO] = Metrics(mode, mean, std, std_l, std_r)


    ax.dataLim.x1 = xlim_dist_wait
    ax.dataLim.x0 = 0
    ax.xaxis.set_major_locator(plt.MaxNLocator(1))
    ax.autoscale_view()

    
    ax.set_xlabel(r"waiting time $\tau_w$")

    return metrics


def plot_first_adapt(ax, PGOs, training_first_waits=None, testing_first_waits=None, **largs):
    bar_width = 0.35
    pos = np.arange(len(PGOs_2))

    for p, PGO in enumerate(PGOs_2):

        if training_first_waits is not None:
            training_means = np.nanmean(training_first_waits[:, (PGOs[:, None] == PGOs_2[None, :]).any(1)], axis=0)
            training_std = np.nanstd(training_first_waits[:, (PGOs[:, None] == PGOs_2[None, :]).any(1)], axis=0) / np.sqrt(training_first_waits.shape[0])
            ax.bar(0 - bar_width/2, training_means, bar_width, yerr=training_std, label='Pre-Training', color='lightblue', capsize=5)

        if testing_first_waits is not None:
            testing_means = np.nanmean(testing_first_waits[:, (PGOs[:, None] == PGOs_2[None, :]).any(1)], axis=0)
            testing_std = np.nanstd(testing_first_waits[:, (PGOs[:, None] == PGOs_2[None, :]).any(1)], axis=0) / np.sqrt(testing_first_waits.shape[0])
            ax.bar(index + bar_width/2, testing_means, bar_width, yerr=testing_std, label='Post-Training', color='lightgreen', capsize=5)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"waiting time $\tau_w$")
    ax.set_xticks(index)
    ax.set_xticklabels()
    ax.legend()
    ax.set_title('First Trial')
    


def plot_waiting_time(ax, waiting_times_THETA, world, kde=False):
    THETAS = world.THETAS
    th_var_bins = np.arange(0, 30, 1)

    modes = []
    for ith, theta in enumerate(THETAS):
        n, x = np.histogram(waiting_times_THETA[ith], bins=th_var_bins)

        from scipy import stats
        dataset = waiting_times_THETA[ith]
        if kde and dataset.size > 0:
            pdf = stats.gaussian_kde(waiting_times_THETA[ith], bw_method=1.0)
            x = np.linspace(0, 100, 1000)
            pdf_ = pdf(x)
            mode = x[np.argmax(pdf_)]
        else:
            bin_centers = 0.5*(x[1:]+x[:-1])
            mode = bin_centers[np.argmax(n)]
            # ax.plot(x, pdf(x), color=cmap(ith/(len(THETAS)-1)), alpha=.5)

    t_wait_mu = np.array([np.mean(waiting_times_THETA[ith]) for ith in range(len(THETAS))])
    t_wait_std = np.array([np.std(waiting_times_THETA[ith]) for ith in range(len(THETAS))])
    fill_between(ax, THETAS, y_mean=t_wait_mu, y_std=t_wait_std,)
    # ax.plot(THETAS, modes,  label='mode')
    # ax.errorbar(THETAS, [np.mean(waiting_times_THETA[ith]) for ith in range(len(THETAS))], 
    #             yerr=[np.std(waiting_times_THETA[ith]) for ith in range(len(THETAS))], fmt='o', label='std')
    
    # for ith, theta in enumerate(THETAS):
    #     if len(waiting_times_THETA[ith]) > 0:
    #         parts = ax.violinplot(waiting_times_THETA[ith], positions=[theta], widths=0.1, showmeans=False, showextrema=False, showmedians=True, )
    #         for pc in parts['bodies']:
    #             pc.set_facecolor(cmap(ith/(len(THETAS)-1)))
    #             # pc.set_alpha(1)

    # ax.set_yscale('symlog', linthresh=1)
    ax.legend(loc="upper left")
    ax.set_ylabel("waiting time")
    # ax.dataLim.y0 = 0

def plot_waiting_time_dist(ax, waiting_times_THETA, world, kde=False):
    bins = np.arange(0, 30, 1)
    THETAS = world.THETAS

    for ith, theta in enumerate(THETAS[::max(len(THETAS)//10, 1)]):
        
        n, x = np.histogram(waiting_times_THETA[ith], bins=bins)
        bin_centers = 0.5*(x[1:]+x[:-1])
        

        # kde estimation
        if kde:
            from scipy import stats
            dataset = waiting_times_THETA[ith]
            if dataset.size > 0:
                pdf = stats.gaussian_kde(waiting_times_THETA[ith], bw_method=1.0)
                x = np.linspace(0, 100, 1000)
                ax.plot(x, pdf(x), color=cmap(theta), alpha=1.)
                # ax.plot(bin_centers, n, color=cmap(ith/(len(THETAS)-1)), alpha=.5)
        else:
            ax.plot(bin_centers, n, color=cmap(theta), alpha=.5)
            
        ax.axvline(np.mean(waiting_times_THETA[ith]), color=cmap(theta), alpha=.5, ymax=.1)
        mode = bin_centers[np.argmax(n)]
        ax.axvline(mode, color=cmap(theta), alpha=.5, linestyle="--",  ymax=.1)
        ax.set_xlabel("waiting time")

def plot_rew_rate(ax, tape, world):
    all_t_waits = []
    THETAS = world.THETAS
    for i_th, theta in enumerate(THETAS):
        where_theta = np.isclose(tape.theta, theta)

        r_t_theta_emp = tape.r[where_theta]
        t_theta_emp = where_theta[where_theta]  # just ones

        # chunk to get variance
        r_t_theta_split = np.array_split(r_t_theta_emp, 10)
        t_theta_split = np.array_split(t_theta_emp, 10)

        r_t_emp = np.array([np.sum(r) / np.sum(T) for r, T in zip(r_t_theta_split, t_theta_split)])

        c = cmap_cool(theta)
        t_waits = np.arange(0, 30).astype(int)
        ax.plot(t_waits, world.r_t__X1(t_waits, theta), label="optimal reward rate in context", ls="-", c=c)

        ax.axhline(world.r_t__X1(world.t_wait_opt__X1(theta), theta), label="optimal reward rate in context", ls="--", c=c)
        mu_r, std_r = r_t_emp.mean(), r_t_emp.std()

        t_waits_ = calc_waiting_times(tape)[i_th]
        all_t_waits.append(t_waits_.flatten())
        mu_t, std_t = t_waits_.mean(), t_waits_.std()
        ax.errorbar(mu_t, mu_r, yerr=std_r, xerr=std_t, fmt="o", c=c, label=f"reward rate in context {theta}")


    # overall reward rate  
    ax.plot(t_waits, world.r_t__X1(t_waits), label="optimal reward rate", ls="-", c="k")
    where = tape.r == tape.r
    # chunk to get variance
    r_t_split = np.array_split(tape.r[where], 10)
    t_split = np.array_split(where, 10)
    r_t_emp = np.array([np.sum(r) / np.sum(T) for r, T in zip(r_t_split, t_split)])
    mu_r, std_r = r_t_emp.mean(), r_t_emp.std()
    all_t_waits = np.concatenate(all_t_waits)
    ax.errorbar(np.mean(all_t_waits), mu_r, yerr=std_r, xerr=np.std(all_t_waits), fmt="o", c="k", label="reward rate")

    ax.set_xlabel(r"waiting time $\tau_w$")
    ax.set_ylabel(r"reward rate $r_t$")

    return r_t_emp
