import allantoolkit
import numpy as np

# Compute a deviation using the Dataset class
a = allantoolkit.Dataset(data=np.random.rand(1000))
a.compute("mdev")

# Plot it using the Plot class
b = allantoolkit.Plot()
b.plot(a, errorbars=True, grid=True)
# You can override defaults before "show" if needed
b.ax.set_xlabel("Tau (s)")
b.show()
