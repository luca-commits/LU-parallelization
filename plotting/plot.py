import numpy as np
import scipy.stats as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir

sns.set_theme(style="whitegrid")

timings_files = [ file for file in listdir('.') if ".csv" in file ]

df_singles = []

for tf in timings_files:
    df_size = pd.read_csv(f'{tf}').pivot(columns='p', values='time')
    df_singles.append(df_size)

df = pd.concat(df_singles, axis=1)
df = df.reindex(sorted(df.columns), axis=1)
df = df.transform(np.sort)

print(df)

n = df.shape[0]
size = df.shape[1]
alpha = 0.05

conf_intervals = np.zeros((size, 2))

print(st.norm.isf(0.025))
# lower_ci_idx = int(np.floor(0.5 * (n - st.norm.isf(0.5 * alpha) * np.sqrt(n))))
# upper_ci_idx = int(np.ceil(1 + 0.5 * (n + st.norm.isf(0.5 * alpha) * np.sqrt(n))))

lower_ci_idx = 7
upper_ci_idx = 18

print([lower_ci_idx, upper_ci_idx])

# for col in df:
#     lower_ci = df[col].iloc[lower_ci_idx]
#     upper_ci = df[col].iloc[upper_ci_idx]

#     conf_intervals[col - 1, :] = [lower_ci, upper_ci]

print(df.values.T.shape)
print(conf_intervals.shape)

# ax = sns.boxplot(x='p', y='value', data=pd.melt(df), notch=True, conf_intervals=conf_intervals)
# ax = sns.boxplot(x='p', y='value', data=pd.melt(df), notch=True)
# ax = sns.boxplot(x=df.values, notch=True, conf_intervals=conf_intervals.T)
# ax = df.boxplot(notch=True, conf_intervals=conf_intervals)
ax = df.boxplot(notch=True, showfliers=False)
ax.set_xlabel('number of ranks')
ax.set_ylabel('runtime [s]')

plt.show()