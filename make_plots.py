"""
This program creates plots for the HPC report.
"""

import pandas as pd
from matplotlib import pyplot as plt

#-------------------------
# Stock Results
# -------------------------

# Set stock
stock = "U(0, 1, 1, 2)"

# load from csv
d = pd.read_csv("./outputs/deterministic_baseline/counts_deterministic_female_white_y1_s12.csv")
dm = pd.read_csv("./outputs/stochastic_baseline/counts_stochastic_n10_female_white_y1_s12.csv")
dh = pd.read_csv("./outputs/stochastic_baseline/counts_high_stochastic_n10_female_white_y1_s12.csv")
dl = pd.read_csv("./outputs/stochastic_baseline/counts_low_stochastic_n10_female_white_y1_s12.csv")

# Extract time series
d = d[d["Unnamed: 0"] == stock].values.tolist()[0][1:-1]
dm = dm[dm.index == stock].values.tolist()[0][1:-1]
dh = dh[dh.index == stock].values.tolist()[0][1:-1]
dl = dl[dl.index == stock].values.tolist()[0][1:-1]
t = range(len(dm))

# Set colors
S_rgb = [0.7, 1.0, 1.0]
alp = 0.3

# Plot
plt.plot(t, dh, color=S_rgb, alpha=alp)
plt.plot(t, dm, color='c', label='Stochastic Mean')
plt.plot(t, d, color='k', label='Determistic')
plt.plot(t, dl, color=S_rgb, alpha=alp)
plt.fill_between(t, dl, dh, color=S_rgb, alpha=alp)
plt.xlabel('Months')
plt.ylabel('Stock Level')
legend = plt.legend(loc=1)
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/model_results.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by years deterministic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/deterministic_years.csv")

# Plot
plt.plot(df['year'], df['time'], color='k', label='S')
plt.xlabel('Years')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/deterministic_years.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by foi deterministic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/deterministic_foi.csv")

# Plot
plt.plot(df['num_foi'], df['time'], color='k', label='S')
plt.xlabel('Influence')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/deterministic_foi.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by age groups deterministic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/deterministic_age.csv")

# Plot
plt.plot(df['age'], df['time'], color='k', label='S')
plt.xlabel('Number of Age Groups')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/deterministic_age.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by addiction deterministic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/deterministic_addiction.csv")

# Plot
plt.plot(df['addiction'], df['time'], color='k', label='S')
plt.xlabel('Number of Addiction Levels')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/deterministic_addiction.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by years stochastic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/stochastic_years.csv")

# Plot
plt.plot(df['year'], df['time'], color='k', label='S')
plt.xlabel('Years')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/stochastic_years.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by foi stochastic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/stochastic_foi.csv")

# Plot
plt.plot(df['num_foi'], df['time'], color='k', label='S')
plt.xlabel('Influence')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/stochastic_foi.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by age groups deterministic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/stochastic_age.csv")

# Plot
plt.plot(df['age'], df['time'], color='k', label='S')
plt.xlabel('Number of Age Groups')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/stochastic_age.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by addiction stochastic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/stochastic_addiction.csv")

# Plot
plt.plot(df['addiction'], df['time'], color='k', label='S')
plt.xlabel('Number of Addiction Levels')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/stochastic_addiction.png', dpi=300)
plt.clf()

#-------------------------
# Runtime by number of MC simulations stochastic
# -------------------------

# Read data
df = pd.read_csv("./outputs/timings/stochastic_mcsims.csv")

# Plot
plt.plot(df['sruns'], df['time'], color='k', label='S')
plt.xlabel('Number of Monte Carlo Simulations')
plt.ylabel('Runtime (Seconds)')
fig = plt.gcf()
fig.set_size_inches(11, 5)
plt.savefig('./outputs/plots/stochastic_mcsims.png', dpi=300)
plt.clf()
