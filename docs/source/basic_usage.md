# 🎈 Basic usage

As a general rule of thumb, `allantoolkit` expects the user to provide input 
data either as evenly spaced phase measurements - in units of second - or 
as fractional frequency measurements. Deviations are then calculated by 
`allantoolkit` in seconds, over a requested range of averaging times.

To avoid confusion, phase data - in seconds - is usually denoted as `x`. 
Fractional frequency data is instead usually denoted as `y`.


## 🧩 Minimal Examples

### Phase data

We can call `allantoolkit` with only two parameters - an array of phase 
data and its associated data sampling rate. 

For example, to calculate the overlapping allan deviation (OADEV) for a 
time-interval measurement at 1 Hz from a time-interval-counter measuring the 
1PPS output of two clocks:

```python
import allantoolkit

# Generate some mock pink phase data
noise = allantoolkit.noise.pink(10000, rate=1) 
x = noise.data

# Compute the overlapping Allan deviation (OADEV)
out = allantoolkit.devs.oadev(x, rate=1)

# Print deviation results
print(out)
```

By default, the deviations will be computed at `octave` averaging times. 
One can also choose another of the in-built options (`all`, `many`, `decade`) 
or provide custom averaging times to the `taus` parameter of the deviation 
of choice.

```{seealso}
{mod}`allantoolkit.devs` : a complete list of available deviations
```

### Frequency data

For input fractional frequency data, it is important to remember to set 
explicitly the `data_type` argument to `freq`. The default value is `phase`,
for input phase data. 

Note that `allantoolkit` assumes non-dimensional frequency data input, so
normalization and scaling is left to the user.

```python
import allantoolkit

# Generate some mock pink frequency data
noise = allantoolkit.noise.pink(10000, rate=1, data_type='freq')
y = noise.data

# Compute the overlapping Allan deviation (OADEV)
out = allantoolkit.devs.oadev(y, rate=1, data_type='freq')

# Print deviation results
print(out)
```

```{seealso}
{func}`allantoolkit.utils.frequency2fractional` 
{func}`allantoolkit.utils.scale` :
    some helpful utility functions to rescale your frequency data
```

### API

`allantoolkit` also offers a top-level `API` which allows to conveniently 
handle data, stability analysis results, and plots all at once.

For example, the minimal frequency data example from above can be 
re-implemented as follows:

```python
import allantoolkit
import matplotlib.pyplot as plt

# Generate some mock pink frequency data
noise = allantoolkit.noise.pink(10000, rate=1, data_type='freq')

# Store data in API wrapper
y = allantoolkit.api.Dataset(noise)

# Show raw data
y.show()

# Compute the overlapping Allan deviation (OADEV)
y.calc('oadev', taus='octave')

# Plot deviation results
y.plot()

# Show plots
plt.show()
```

Note how you can feed the ``Noise`` object directly to the API wrapper, 
without having to explicitely set the ``data_type`` and data sampling ``rate``.

```{seealso}
{mod}`allantoolkit.api` for a complete list of available API methods.
```