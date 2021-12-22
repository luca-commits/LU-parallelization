import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    'input',
    help="CSV file to plot",
    type=str,
)
parser.add_argument(
    "-o", "--output",
    help="Plot file to output",
    type=str,
    required=True,
)
args = parser.parse_args()

df = pd.read_csv(Path(args.input), delimiter=',')
df = df.pivot(index=['p'], columns='name', values='value')
df['speed-up'] = df['time'].iloc[0] / df['time']
df['optimum'] = df.index
print(df)

df.plot(
    #x='p',
    y=['speed-up', 'optimum'],
    kind='line',
    ## update title
    title='Speedup - LU with PRP, OpenMP',
    logy=True,
    #ylabel="Speed-up",
    logx=False,
    xlabel="#processors",
    legend=True,
)
plt.savefig(Path(args.output))