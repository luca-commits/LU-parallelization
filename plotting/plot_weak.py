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
df = df.pivot(index=['n', 'p'], columns='name', values='value')
df['serial time'] = df['time'].iloc[0]

df.plot(
    y=['time', 'serial time'],
    kind='line',
    title='Weak scaling',
    #logy=True,
    #ylabel="Speed-up",
    #logx=False,
    xlabel="#processors",
    legend=True,
)
plt.savefig(Path(args.output))