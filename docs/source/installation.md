# 📚 Installation

Install the latest development version of `allantoolkit`, using pip to 
download it from this repo:

 ```bash
 python -m pip install git+https://gitlab.com/amv213/allantoolkit.git
 ```

These commands should be run as root for system-wide installation, or
you can use the `--user` option to install for your account only. Exact 
command names may vary depending on your OS / package manager / target python 
version.


You can then test your installation running the following minimal script:

```python
import allantoolkit

# Generate some pink noise phase data
noise = allantoolkit.noise.white(10000)
x = noise.data

# Compute overlappig Allan deviation
out = allantoolkit.devs.oadev(x)

# Display analysis results
print(out)
```