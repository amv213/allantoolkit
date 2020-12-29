# 🎈 Basic usage

As a general rule of thumb, `allantoolkit` expects the user to provide input 
data either as evenly spaced phase measurements - in units of second - or 
as fractional frequency measurements. Deviations are then calculated by 
`allantoolkit` in seconds, over a requested range of averaging times.

## 🧩 Minimal Examples

### Phase data

We can call `allantoolkit` with only two parameters - an array of phase 
data and its associated data sampling rate. 

For example, to calculate the overlapping allan deviation (OADEV) for a 
time-interval measurement at 1 Hz from a time-interval-counter measuring the 
1PPS output of two clocks:

```python
import allantoolkit

# Generate some mock phase data
y = allantoolkit.noise.white(10000)  # frequency
x = allantoolkit.utils.frequency2phase(y, rate=1)  # phase

# Compute the overlapping Allan deviation (OADEV)
out = allantoolkit.devs.oadev(x, rate=1)

# Print deviation results
print(out)
```

By default, the deviations will be computed at `octave` averaging times. 
One can also choose another of the in-built options (`all`, `many`, `decade`) 
or provide custom averaging times to the `taus` parameter of the deviation 
of choice.

.. seealso:
    
    :func:`allantoolkit.utils.tau_generator`

### Frequency data

For input fractional frequency data, it is important to remember to set 
explicitly the `data_type` argument to `freq`. The default value is `phase`,
for input phase data. 

Note that `allantoolkit` assumes non-dimensional frequency data input, so
normalization and scaling is left to the user.

```python
import allantoolkit

# Generate some mock frequency data
y = allantoolkit.noise.white(10000)

# Compute the overlapping Allan deviation (OADEV)
out = allantoolkit.devs.oadev(y, rate=1, data_type='freq')

# Print deviation results
print(out)
```

### API

`allantoolkit` also offers a top-level `API` which allows to conveniently 
handle data, stability analysis results, and plots all at once.

For example, the minimal frequency data example from above can be 
re-implemented as follows:

```python
import allantoolkit
import matplotlib.pyplot as plt

# Generate some mock frequency data
y = allantoolkit.noise.white(10000)

# Store data in API wrapper
y = allantoolkit.api.Dataset(y, rate=1, data_type='freq')

# Show raw data
y.show()

# Compute the overlapping Allan deviation (OADEV)
y.calc('oadev', taus='octave')

# Plot deviation results
y.plot()

# Show plots
plt.show()
```
