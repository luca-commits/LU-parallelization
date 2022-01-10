import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from os import listdir
from os.path import isdir
from cycler import cycler

names = {
    'mpi': 'MPI',
    'scalapack': 'ScaLAPACK',
    'hybrid': 'Hybrid',
    'omp': 'OpenMP'
}

# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(4 * 0.8, 3 * 0.8), dpi=72)
# ax.set_prop_cycle('color',[plt.get_cmap('viridis').colors[i] for i in range(0, 192, int(192 / 3))])
ax.set_prop_cycle('color',plt.get_cmap('Dark2').colors)

experiments = [ dir for dir in listdir('.') if isdir(f'./{dir}') ]

for experiment in experiments:

    timings_files = [ file for file in listdir(f'{experiment}/.') if ".csv" in file ]

    df_singles = []

    for tf in timings_files:
        df_size = pd.read_csv(f'{experiment}/{tf}').pivot(columns='p', values='time')
        df_singles.append(df_size)

    df = pd.concat(df_singles, axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.transform(np.sort)

    n = df.shape[0]
    size = df.shape[1]
    alpha = 0.05

    medians = df.median(axis=0)

    conf_intervals = pd.DataFrame()
    lower_ci_idx = 7
    upper_ci_idx = 18

    for col in df:
        lower_ci = medians[col] - df[col].iloc[lower_ci_idx]
        upper_ci = df[col].iloc[upper_ci_idx] - medians[col]

        conf_intervals[col] = [lower_ci, upper_ci]


    # ax.boxplot(df, widths=0.5, showfliers=False)

    ax.errorbar(medians.index, medians.values, label=names[experiment], yerr=conf_intervals.values, marker='.', linewidth=0.0, elinewidth=1.0, capsize=2.0)

ax.set_xlabel('number of ranks')
ax.set_ylabel('runtime [s]')
# ax.set_title('Number of ranks vs. Runtime [s]')
ax.set_yscale('log')
ax.set_facecolor('whitesmoke')
ax.grid(axis='y', color='white')

fig.legend(ncol=2, loc=1, bbox_to_anchor=(1,1), fontsize=8, handlelength=2)
fig.tight_layout(pad=0.0)



plt.savefig('plot.pdf', bbox_inches='tight', pad_inches=0.025)
plt.savefig('plot.png', bbox_inches='tight', pad_inches=0.025)
# plt.show()