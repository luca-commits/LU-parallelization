import numpy as np
import scipy.stats as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir

sns.set_theme(style="whitegrid")

timings_files = listdir('./timings/')

df_singles = []

for tf in timings_files:
    df_size = pd.read_csv(f'./timings/{tf}').sort_values('time').pivot(columns='size', values='time')
    df_singles.append(df_size)

df = pd.concat(df_singles, axis=1)
df = df.reindex(sorted(df.columns), axis=1)

print(df)

n = df.shape[0]
size = df.shape[1]
alpha = 0.05

conf_intervals = np.zeros((n, 2))

print(st.norm.isf(0.025))
lower_ci_idx = int(np.floor(0.5 * (n - st.norm.isf(0.5 * alpha) * np.sqrt(n))))
upper_ci_idx = int(np.ceil(1 + 0.5 * (n + st.norm.isf(0.5 * alpha) * np.sqrt(n))))

for col in df:
    lower_ci = df[col].iloc[lower_ci_idx]
    upper_ci = df[col].iloc[upper_ci_idx]

    conf_intervals[col, :] = [lower_ci, upper_ci]

print(len(pd.melt(df)["size"]))
print(df.columns)

ax = sns.boxplot(x="size", y="value", data=pd.melt(df), notch=True, conf_intervals=conf_intervals)
ax.set_xlabel('number of ranks')
ax.set_ylabel('runtime [s]')

plt.show()