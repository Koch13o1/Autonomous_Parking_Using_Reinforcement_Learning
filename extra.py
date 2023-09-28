import numpy as np
from scipy.interpolate import CubicSpline

# Define some sample points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 2, 1, 4, 3, 5])

# Create the cubic spline interpolator
cs = CubicSpline(x, y)

# Generate a high-resolution x-axis
x_hr = np.linspace(0, 5, 100)
print(x_hr)

# Interpolate the y-values at the high-resolution x-values
y_hr = cs(x_hr)
print(y_hr)

# Plot the original points and the interpolated curve
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='data')
plt.plot(x_hr, y_hr, label='interpolated')
plt.legend()
plt.show()



"""import numpy as np
from scipy.interpolate import interp1d

# generate some data for x and y
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# create an interpolation function for the data
f = interp1d(x, y, kind='linear', fill_value='extrapolate')

# generate some test data for extrapolation
x_test = np.array([6, 7, 8])

# use the interpolation function to extrapolate y values for x_test
y_test = f(x_test)

print(y_test)
"""