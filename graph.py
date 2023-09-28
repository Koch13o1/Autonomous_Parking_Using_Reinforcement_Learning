import numpy as np
import matplotlib.pyplot as plt

# define the functions for y=3 and y=-1 as before
x = np.linspace(-10, 10, 1000)
y = np.zeros_like(x)
y[x <= -4] = 3
y[(x > -4) & (x < 4)] = -1
y[x >= 4] = 3

# define the arrays for the new line
x_arr = [-9, -6, -3, 0, 3, 6, 9]
y_arr = [2, 1, 0, -1, 0, 1, 2]

fig, ax = plt.subplots()       # create a figure and axis object

ax.plot(x, y, color='black')    # plot the x and y arrays on the axis
ax.plot(x_arr, y_arr, color='green') # plot the new x and y arrays on the same axis

ax.set_xlim(-10, 10)           # set the x limits of the plot
ax.set_ylim(-2, 4)             # set the y limits of the plot

plt.show()                     # display the plot
