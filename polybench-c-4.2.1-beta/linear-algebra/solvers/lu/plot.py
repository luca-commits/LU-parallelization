import pandas as pd
import matplotlib.pyplot as plt
 
df = pd.read_csv('results_summary.txt')

x = df.iloc[:, [0]]
y = df.iloc[:, [1]]

plt.scatter(x,y, label='MPI-LU-Decomposition')

plt.xlabel("#processors")
plt.ylabel("time [s]")
plt.legend()



plt.savefig("mpi_lu.png")