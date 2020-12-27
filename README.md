# 📑 Welcome to Allantoolkit!

Allantoolkit is a fork of Anders E.E. Wallin's [AllanTools](https://github.com/aewallin/allantools)

A python library for calculating Allan deviation and related time & frequency statistics toolkit.

## 🚀 Quick Start

1. 📚 Install the latest development version of `allantoolkit`, using pip to 
   download it from this repo:

    ```bash
    $ python -m pip install git+https://gitlab.com/amv213/allantoolkit.git
    ```
   
2. 🐍 Test your installation running the following minimal script:
    
   ```python
      import allantoolkit
       
      # Generate some frequency data
      y = allantoolkit.noise.white(10000)
      r = 1. # data sampling rate, in Hz
      
      # Compute overlappig Allan deviation
      out = allantoolkit.allantools.oadev(y, rate=r, data_type='freq')
      
      # Display analysis results
      print(out)
   ```

## 🔥 Changelog

- new low-level API (based on NamedTuple)
- fully featured object-oriented high-level API
- Auto noise ID for all deviation types (Lag1 ACF, B1, R(n))
- Bias correction for all deviation types
- Non-naive EDF / confidence interval calculation for all deviation types
- Basic plotting and visualisation tools
- Fast Theo1 algorithm
- Many-tau tau generation
- Streamlined code architecture
- Support for tables and constant mappings from .yaml file
- Updated documentation and docstrings
- Type hinting
- Streamlined pytests
- Additional tests
- New demo test dataset based on optical lattice clock data
- General gap resistance for most deviation types (to be tested)
- Logging
   
## 📚 Documentation

To learn more about the package head over to the [official documentation](https://amv213.gitlab.io/allantoolkit)!