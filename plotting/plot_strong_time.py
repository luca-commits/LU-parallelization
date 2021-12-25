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
print(df)

df['time'].plot(
    kind='line',
    ## update title
    title='Strong scaling - Jacobi 2D, MPI',
    logy=False,
    ylabel="time in sec",
    logx=False,
    xlabel="#processors",
    legend=False,
    style='o'
)

plt.savefig(Path(args.output))