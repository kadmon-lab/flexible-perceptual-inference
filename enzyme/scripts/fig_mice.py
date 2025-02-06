import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from pathlib import Path

from enzyme import PRJ_ROOT, TEXTHEIGHT, TEXTWIDTH
from enzyme.src.helper import save_plot


MICE_PATH = PRJ_ROOT / "Data/mice/analysis_january_25"
# MICE_PATH = Path.home() / "Downloads"

# File selection
batch1 = "batch1.mat"
batch2 = "batch2.mat"

# Select which file to use
file = batch2

from enzyme.plot_helper import plot_significance, plotting_context
import seaborn as sns
ctx = plotting_context('paper', font_scale=1., scale=.5)
sns.set_context(ctx)

fig = plt.figure(layout='constrained', figsize=(TEXTWIDTH, TEXTHEIGHT*.3))


"""
A      B     C
----------------
D1 D2 | E1 E2 E3
"""

subfigs = fig.subfigures(2, 1,)
axd = {}
axd_ = subfigs[0].subplot_mosaic([['A', 'B', 'C'],], )
axd.update(axd_)

meta_subfigs = subfigs[1].subfigures(1, 2, wspace=0.1)
axd_ = meta_subfigs[0].subplot_mosaic([['D1', 'D2'],], )
axd.update(axd_)

axd_ = meta_subfigs[1].subplot_mosaic([['E1', 'E2', 'E3'],], 
                                      gridspec_kw=dict(wspace=0.0),
                                      width_ratios=[1, 1, 1],
                                      )
axd.update(axd_)


ax_task = axd['A']
ax_waits = axd['B']

ax_mouse_pic = axd['A']
ax_first_early = axd['D1']
ax_first_late = axd['D2']

ax_corr = axd['C']
ax_learn_1 = axd['E1']
ax_learn_2 = axd['E2']
ax_learn_diff = axd['E3']

for file in [batch1, batch2]:

    if (MICE_PATH / file).with_suffix('.npy').exists():
        data_dict = np.load((MICE_PATH / file).with_suffix('.npy'), allow_pickle=True).item()
    else:
        """
        % MATLAB CODE
        % FILL DIRECTORY HERE
        dir = "/MATLAB Drive/analysis_jan_2025/";

        %% BATCH 1
        batch1 = "batch1.mat";
        %% BATCH 2
        batch2 = "batch2.mat";

        %% SELECT HERE
        file = batch1;

        T_ = load(dir + file);

        T = T_.('T')


        % Get the variable names
        varNames = T.Properties.VariableNames;

        % Create a struct to hold the data
        data = struct();

        % Loop through all fields
        for i = 1:numel(varNames)
            fieldName = varNames{i};
            fieldData = T.(fieldName);
            
            % Convert table column to cell array if it's categorical
            if iscategorical(fieldData)
                fieldData = cellstr(fieldData);
            end
            
            % Assign to the struct
            data.(fieldName) = fieldData;
        end

        % Save the struct

        save(dir + 'converted_' + file, 'data', '-v7.3');
        
        """
        ## MATLAB conversion code, super hacky
        # Load the .mat file
        from scipy.io import loadmat
        # from mat73 import loadmat

        mat_contents = loadmat(str((MICE_PATH / file)))
        raw_dict = mat_contents['data_dict']

        # Create a dictionary to hold the data
        data_dict = {}

        # Populate the dictionary
        for k, v in raw_dict.items():
            # Convert from MATLAB's Unicode to Python string
            print(k)

            # Extract the data for this field
            try:
                field_data = np.array(v).squeeze()
            except:
                field_data = v
            
            # Add to dictionary
            data_dict[k] = field_data

        # save dict to disk
        np.save((MICE_PATH / file).with_suffix('.npy'), data_dict)

        # Now data_dict contains all your table fields
        print(data_dict.keys())

    T = data_dict
    T['data'] = T['wait_from_last_NoGo_duration']

    for k, v in T.items():
        try:
            print(k, v.shape)
        except:
            print(k, len(v))

    T = {k: v.squeeze() if type(v) == np.ndarray else v for k, v in T.items()}

    T_1D = {k: v for k, v in T.items() if type(v) != list}
    T_1D = {k: v for k, v in T_1D.items() if v.ndim !=2}
    T = pd.DataFrame(T_1D)


    mouse_num = int(T['mouse_number'].max())
    PGOs = np.unique(T['PGo'])
    PGO_N = len(PGOs)
    from enzyme.colors import cmap_cool
    cmap = cmap_cool(np.linspace(0, 1, PGO_N))
    trial_N = 8
    N_back = trial_N // 2

    x = np.arange(1, N_back + 1)
    xax = np.arange(1, trial_N + 1)
    labels = np.arange(-N_back, N_back)

    # Save PGOs and labels
    # np.save('PGOs_x_axis.npy', PGOs)
    # np.save('trial_number_x_axis.npy', labels)

    # Initial indices
    init_inds = (T['any_licks'] == 1) & (T['trial_number'] < 300) & ~np.isnan(T['data'])
    switch_inds = (T['trial_number'] > trial_N/2)

    data_per_mouse_trial_PGO = np.zeros((mouse_num, trial_N, PGO_N))
    data_per_mouse_trial = np.zeros((mouse_num, trial_N))
    data_per_mouse_PGO = np.zeros((mouse_num, PGO_N))



    # %%
    if file == batch1:
        # %% initialization %%
        wait_per_mouse_PGO = np.zeros_like(data_per_mouse_PGO)
        wait_per_mouse_trial_PGO = np.zeros_like(data_per_mouse_trial_PGO)
        PGO_curr_corr_per_mouse_trial = np.zeros_like(data_per_mouse_trial)
        wait_curr_corr_per_mouse_trial = np.zeros_like(data_per_mouse_trial)
        PGO_prev_corr_per_mouse_trial = np.zeros_like(data_per_mouse_trial)
        wait_prev_corr_per_mouse_trial = np.zeros_like(data_per_mouse_trial)
        
        # %% collection %%
        for m in range(1, mouse_num + 1):
            mouse_inds = T['mouse_number'] == m
            for p, PGO in enumerate(PGOs):
                PGO_inds = T['PGo'] == PGO
                inds = init_inds & mouse_inds & PGO_inds
                wait_per_mouse_PGO[m-1, p] = np.nanmean(T['data'][inds])
            
            session_N = int(np.max(T['session_number'][mouse_inds]))
            block_N = int(np.max(T['block_number'][mouse_inds]))
            wait_prev_vec = []
            wait_curr_vec = []
            PGO_prev_vec = []
            PGO_curr_vec = []
            trial_vec = []
            IV_vec = []
            
            for s in range(1, session_N + 1):
                session_inds = T['session_number'] == s
                for b in range(2, block_N + 1):
                    curr_inds = T['block_number'] == b
                    prev_inds = T['block_number'] == (b-1)
                    default_inds = mouse_inds & init_inds & session_inds & switch_inds
                    curr_trial_N = np.max(T['trial_number_from_switch'][curr_inds & default_inds])
                    prev_trial_N = np.max(T['trial_number_from_switch'][prev_inds & default_inds])
                    
                    if (curr_trial_N > N_back) and (prev_trial_N > N_back):
                        prev_til_inds = (T['trial_number_from_switch'] < (prev_trial_N - N_back))
                        curr_til_inds = (T['trial_number_from_switch'] < (curr_trial_N - N_back))
                        from_inds = (T['trial_number_from_switch'] > N_back)
                        
                        prev_mean_inds = prev_inds & default_inds & from_inds & prev_til_inds
                        curr_mean_inds = curr_inds & default_inds & from_inds & curr_til_inds
                        if (np.sum(prev_mean_inds) > N_back) and (np.sum(curr_mean_inds) > N_back):
                            prev_mean = np.nanmean(T['data'][prev_mean_inds])
                            curr_mean = np.nanmean(T['data'][curr_mean_inds])
                            prev_PGO = np.nanmean(T['PGo'][prev_inds & default_inds])
                            curr_PGO = np.nanmean(T['PGo'][curr_inds & default_inds])
                            for t in range(1, trial_N + 1):
                                shifted_block_inds = np.roll(T['block_number'], -N_back)
                                shifted_trial_inds = np.roll(T['trial_number_from_switch'], -N_back)
                                shift_inds = (shifted_block_inds == b) & (shifted_trial_inds == t)
                                trial_ind = np.where(default_inds & shift_inds)[0]
                                if len(trial_ind) > 0:
                                    wait_prev_vec.append(prev_mean)
                                    wait_curr_vec.append(curr_mean)
                                    PGO_prev_vec.append(prev_PGO)
                                    PGO_curr_vec.append(curr_PGO)
                                    trial_vec.append(t)
                                    IV_vec.append(T['data'][trial_ind[0]])
            
            wait_prev_vec = np.array(wait_prev_vec)
            wait_curr_vec = np.array(wait_curr_vec)
            PGO_prev_vec = np.array(PGO_prev_vec)
            PGO_curr_vec = np.array(PGO_curr_vec)
            trial_vec = np.array(trial_vec)
            IV_vec = np.array(IV_vec)
            
            for t in range(1, trial_N + 1):
                inds = np.where(trial_vec == t)[0]
                if len(inds) > 1:
                    IV = IV_vec[inds]
                    wait_prev_corr = np.corrcoef(IV, wait_prev_vec[inds])[0, 1]
                    wait_curr_corr = np.corrcoef(IV, wait_curr_vec[inds])[0, 1]
                    PGO_prev_corr = np.corrcoef(IV, PGO_prev_vec[inds])[0, 1]
                    PGO_curr_corr = np.corrcoef(IV, PGO_curr_vec[inds])[0, 1]
                    wait_prev_corr_per_mouse_trial[m-1, t-1] = wait_prev_corr
                    wait_curr_corr_per_mouse_trial[m-1, t-1] = wait_curr_corr
                    PGO_prev_corr_per_mouse_trial[m-1, t-1] = PGO_prev_corr
                    PGO_curr_corr_per_mouse_trial[m-1, t-1] = PGO_curr_corr
        
        wait_per_PGO_mean_over_mouse = np.nanmean(wait_per_mouse_PGO, axis=0)
        wait_per_PGO_std_over_mouse = np.nanstd(wait_per_mouse_PGO, axis=0)
    
        # %% plotting %%
        means = wait_per_PGO_mean_over_mouse
        standard_errors = wait_per_PGO_std_over_mouse / np.sqrt(mouse_num)
        
        ax = ax_waits
        for i in range(PGO_N):
            ax.bar(i+1, means[i], color=cmap[i], edgecolor='none')
            ax.errorbar(i+1, means[i], yerr=standard_errors[i], color='tab:red')
        
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'waiting time $\tau_w$ (s)')
        
        ax.set_title(r'waiting time')
        ax.xaxis.set_major_locator(plt.FixedLocator(range(1, PGO_N + 1)))
        ax.xaxis.set_major_formatter(plt.FixedFormatter([PGOs[i] if PGOs[i] in [0., .3, .6, .9] else '' for i in range(PGO_N)]))
        # ax.tick_params(axis='x', labelsize=5)
        # ax.grid(True)

        
        # np.save('PGO_bar_plot_means.npy', means)
        # np.save('PGO_bar_plot_standard_errors.npy', standard_errors)
        
        # %% correlation plots
        PGO_curr_means = np.mean(PGO_curr_corr_per_mouse_trial, axis=0)
        PGO_prev_means = np.mean(PGO_prev_corr_per_mouse_trial, axis=0)
        wait_curr_means = np.mean(wait_curr_corr_per_mouse_trial, axis=0)
        wait_prev_means = np.mean(wait_prev_corr_per_mouse_trial, axis=0)
        color_curr = [0, 0.4470, 0.7410]  # Blue for current
        color_prev = [0.8500, 0.3250, 0.0980]  # Red for previous

        PGO_curr_corr_flat = PGO_curr_corr_per_mouse_trial.flatten()
        PGO_prev_corr_flat = PGO_prev_corr_per_mouse_trial.flatten()
        wait_curr_corr_flat = wait_curr_corr_per_mouse_trial.flatten()
        wait_prev_corr_flat = wait_prev_corr_per_mouse_trial.flatten()

        ax = ax_corr
        ax.plot(xax, wait_curr_means, '-o', color=color_curr, linewidth=1.5, label=r"previous $\theta$")
        ax.plot(xax, wait_prev_means, '-o', color=color_prev, linewidth=1.5, label=r"current $\theta$")
        ax.fill_between(x, 0, 1, color='k', alpha=0.1)

        ax.set_xlabel('trial from switch')
        ax.set_ylabel('correlation')
        ax.set_title('correlation with average waiting')
        # ax.grid(True)
        ax.set_xticks(xax, labels)
        ax.set_ylim(0, 0.5)
        ax.legend(fontsize='small', loc='center left')
        
        # np.save('after_switch_correlations.npy', wait_curr_means)
        # np.save('before_switch_correlations.npy', wait_prev_means)

    # %%
    if file == batch2:
        # %% initialization %%
        training_sessions = (T['session_number'] <= 11) & (T['session_number'] != 10) & (T['session_number'] > 5)
        testing_sessions = T['session_number'] > 14
        train_wait_per_mouse_trial_PGO = np.zeros_like(data_per_mouse_trial_PGO)
        test_wait_per_mouse_trial_PGO = np.zeros_like(data_per_mouse_trial_PGO)

        # %% collection %%
        for m in range(1, mouse_num + 1):
            mouse_inds = T['mouse_number'] == m
            for p, PGO in enumerate(PGOs):
                shifted_PGO_inds = np.roll(T['PGo'], -N_back) == PGO
                for t in range(1, trial_N + 1):
                    shifted_trial_inds = np.roll(T['trial_number_from_switch'], -N_back) == t
                    shifted_inds = shifted_trial_inds & shifted_PGO_inds
                    inds = init_inds & mouse_inds & shifted_inds & switch_inds

                    # %% training collection %%
                    train_inds = inds & training_sessions
                    train_wait_per_mouse_trial_PGO[m-1, t-1, p] = np.mean(T['data'][train_inds])

                    # %% testing collection %%
                    test_inds = inds & testing_sessions
                    test_wait_per_mouse_trial_PGO[m-1, t-1, p] = np.mean(T['data'][test_inds])

        # %% analysis %%
        train_wait_per_trial_PGO_mean_over_mouse = np.nanmean(train_wait_per_mouse_trial_PGO, axis=0)
        test_wait_per_trial_PGO_mean_over_mouse = np.nanmean(test_wait_per_mouse_trial_PGO, axis=0)

        train_wait_per_trial_PGO_std_over_mouse = np.nanstd(train_wait_per_mouse_trial_PGO, axis=0)
        test_wait_per_trial_PGO_std_over_mouse = np.nanstd(test_wait_per_mouse_trial_PGO, axis=0)

        # %% plotting %%
        train_means = train_wait_per_trial_PGO_mean_over_mouse
        test_means = test_wait_per_trial_PGO_mean_over_mouse
        train_standard_errors = train_wait_per_trial_PGO_std_over_mouse / np.sqrt(mouse_num)
        test_standard_errors = test_wait_per_trial_PGO_std_over_mouse / np.sqrt(mouse_num)

        ax1, ax2 = ax_first_early, ax_first_late
        for i, PGO_ in enumerate(PGOs):
            y = train_standard_errors[:, i]
            m = train_means[:, i]
            ax1.errorbar(xax, m, yerr=y, fmt='-o', color=cmap_cool(PGO_),)
            ax1.fill_between(xax, m-y, m+y, color=cmap_cool(PGO_), alpha=0.3)

        ax1.set_ylim(0.15, 0.85)
        ax1.set_xlabel('trial from switch')
        ax1.set_ylabel('waiting time (s)')
        ax1.set_title('novice')
        # ax1.grid(True)
        ax1.set_xticks(xax)
        ax1.set_xticklabels(labels)
        ax1.fill_between(x, 0, 1, color='k', alpha=0.1)
        ax1.set_xlim(1, trial_N)

        # np.save('early_exposure_means.npy', train_means)
        # np.save('early_exposure_standard_errors.npy', train_standard_errors)

        for i, PGO_ in enumerate(PGOs):
            y = test_standard_errors[:, i]
            m = test_means[:, i]
            ax2.errorbar(xax, m, yerr=y, fmt='-o', color=cmap_cool(PGO_), )
            ax2.fill_between(xax, m-y, m+y, color=cmap_cool(PGO_), alpha=0.3)

        ax2.set_ylim(0.15, 0.85)
        ax2.set_xlabel('trial from switch')
        ax2.set_ylabel('waiting time (s)')
        ax2.set_title('expert')
        # ax2.grid(True)
        ax2.fill_between(x, 0, 1, color='k', alpha=0.1)
        ax2.set_xticks(xax)
        ax2.set_xticklabels(labels)
        ax2.set_xlim(1, trial_N)

        # alter spine thickness
        for spine in ax2.spines.values():
            spine.set_linewidth(plt.rcParams['lines.linewidth']*1)


        # np.save('late_exposure_means.npy', test_means)
        # np.save('late_exposure_standard_errors.npy', test_standard_errors)

        # %% first trial plot
        First_trial = trial_N - N_back
        first_train = train_wait_per_mouse_trial_PGO[:, First_trial, :]
        first_test = test_wait_per_mouse_trial_PGO[:, First_trial, :]

        train_means_first = np.mean(first_train, axis=0)
        test_means_first = np.mean(first_test, axis=0)
        train_error_first = np.std(first_train, axis=0) / np.sqrt(mouse_num)
        test_error_first = np.std(first_test, axis=0) / np.sqrt(mouse_num)

        train_diffs = first_train[:, 1] - first_train[:, 0]
        test_diffs = first_test[:, 1] - first_test[:, 0]
        train_diffs_standard_error = np.std(train_diffs) / np.sqrt(mouse_num)
        test_diffs_standard_error = np.std(test_diffs) / np.sqrt(mouse_num)
        train_diffs_mean = np.mean(train_diffs)
        test_diffs_mean = np.mean(test_diffs)

        _, p_train_PGO = stats.wilcoxon(first_train[:, 0], first_train[:, 1])
        _, p_test_PGO = stats.wilcoxon(first_test[:, 0], first_test[:, 1])
        _, p_diff_diff = stats.wilcoxon(train_diffs, test_diffs)

        ax1, ax2, ax3 = ax_learn_1, ax_learn_2, ax_learn_diff

        for i, PGO_ in enumerate(PGOs):
            ax1.bar(i+1, train_means_first[i], color=cmap_cool(PGO_))
            ax1.errorbar(i+1, train_means_first[i], yerr=train_error_first[i], fmt='tab:red')

        ax1.set_xlabel(r'$\theta$')
        ax1.set_ylabel(r'waiting time $\tau_w$ (s)')
        ax1.set_title('novice')
        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(PGOs)
        # ax1.grid(True)

        # ax1.axhline(y=0.65, xmin=.25, xmax=.75, color='k', linewidth=1.5)
        # ax1.text(1.5, 0.65, f'p<{p_train_PGO:.3f}', ha='center', va='bottom')  # put ** notation TODO @JB
        plot_significance(ax1, p_train_PGO, xl=1, xr=2)

        ax1.set_ylim(0, 0.7)

        for i, PGO_ in enumerate(PGOs):
            ax2.bar(i+1, test_means_first[i], color=cmap_cool(PGO_), edgecolor='k', linewidth=plt.rcParams['lines.linewidth']*0)
            ax2.errorbar(i+1, test_means_first[i], yerr=test_error_first[i], fmt='tab:red',)

        ax2.set_xlabel(r'$\theta$')
        ax2.set_title('expert')
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(PGOs)
        # ax2.grid(True)

        # ax2.axhline(y=0.65, xmin=.25, xmax=.75, color='k', linewidth=1.5)
        # ax2.text(1.5, 0.65, f'p<{p_test_PGO:.3f}', ha='center', va='bottom')
        plot_significance(ax2, p_test_PGO, xl=1, xr=2)

        ax2.set_ylim(0, 0.7)

        ax3.bar(1, train_diffs_mean, color=[0.7, 0.7, 0.7])
        ax3.errorbar(1, train_diffs_mean, yerr=train_diffs_standard_error, fmt='tab:red',)
        ax3.axhline(y=0, color='k')
        ax3.bar(2.5, test_diffs_mean, color=[0.7, 0.7, 0.7], edgecolor='k', linewidth=plt.rcParams['lines.linewidth']*0)
        ax3.errorbar(2.5, test_diffs_mean, yerr=test_diffs_standard_error, fmt='tab:red',)
        ax3.set_xticks([1, 2.5])
        ax3.set_xticklabels(["novice", "expert"])
        # ax3.grid(True)

        # ax3.axhline(y=0.35, xmin=.2, xmax=.8, color='k', linewidth=1.5)
        # ax3.text(1.75, 0.35, f'p<{p_diff_diff:.3f}', ha='center', va='bottom')
        plot_significance(ax3, p_diff_diff, xl=1, xr=2.5)

        ax3.set_ylim(-0.3, 0.4)
        ax3.set_title('difference')


        # np.save('early_exposure_first_trial_means.npy', train_means_first)
        # np.save('early_exposure_standard_errors.npy', train_error_first)
        # np.save('late_exposure_first_trial_means.npy', test_means_first)
        # np.save('late_exposure_standard_errors.npy', test_error_first)
        # np.save('early_exposure_diff_means.npy', train_diffs_mean)
        # np.save('early_exposure_diff_standard_errors.npy', train_diffs_standard_error)
        # np.save('late_exposure_diff_means.npy', test_diffs_mean)
        # np.save('late_exposure_diff_standard_errors.npy', test_diffs_standard_error)

# %%
from enzyme.plot_helper import LABEL_KWARGS, add_panel_label
for letter_, ax in axd.items():
    # insert an underscore before any numbers
    import re
    letter_ = re.sub(r"(\d+)", r"_\1", letter_)

    letter = r"$\mathrm{\mathbf{" + letter_.lower() + "}}$"
    ax.text(-.2, 1.05, letter, transform=ax.transAxes, fontsize=LABEL_KWARGS['size'], fontweight='bold', va='bottom', ha='right')


# fig.set_layout_engine('none')
# JB can't get the title to work, let's do it in inkscape
# ax2.text(0.5, 1.2, "First trial after context switch", transform=ax2.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='center')

# %%
# save_plot("_fig_mice", fig=fig, 
#           path=PRJ_ROOT / "tex_REFACTORIZED/figures/",
#           file_formats=[
#     "svg", 
#     "png", 
#     "pdf"
# ])

# place SVGs in the figure
use_skunk = True
if use_skunk:
    import skunk
    
    axd['A'].axis('off')
    skunk.connect(axd['A'], 'A')
    svg_path = PRJ_ROOT / "tex_REFACTORIZED/figures/mouse.svg"
    svg = skunk.insert({'A': str(svg_path)})

# save figure
out_path = PRJ_ROOT / "tex_REFACTORIZED/figures/_fig_mice.svg"
with open(out_path, 'w') as f:
    f.write(svg) if use_skunk else fig.savefig(out_path)

plt.show()

