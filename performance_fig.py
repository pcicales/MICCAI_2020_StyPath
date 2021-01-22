from matplotlib.pyplot import errorbar
import numpy as np
import matplotlib.pyplot as plt
x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800])
y = np.array([85.23, 86.3, 86.83, 88.49, 87.69, 88.09, 87.76, 87.96, 88.42])
stds = np.array([6.28, 3.74, 4.2, 3.75, 4.58, 3.32, 2.9, 3.52, 4.26])/2


fig, ax = plt.subplots(nrows=1, ncols=1)
ax.grid(linestyle='-', linewidth='0.5', color='lightgrey')
# Don't allow the axis to be on top of your data
ax.set_axisbelow(True)
line1, = ax.plot(x, y, '-o', lw=1, label='Random', markersize=3, color='darkred',)
ax.fill_between(x,
                y - stds,
                y + stds,
                color='red', alpha=0.3)
ax.set_xlabel('ST Image Count (Per Class)')
ax.set_ylabel('Prediction Accuracy')
ax.set_ylim([82,92])
ax.set_xlim([0,800])
fig.show()

temp