# 📑 Welcome to Allantoolkit!

Allantoolkit is a fork of **Anders E.E. Wallin's** [AllanTools](https://github.com/aewallin/allantools): a python library for calculating Allan deviation 
and related time & frequency statistics. Don't hesitate to head over to his 
Github repo to check out the original project and to drop a star! 🌟

## 🚀 Quick Start

1. 📚 Install the latest development version of `allantoolkit`, using pip to 
   download it from this repo:

    ```bash
    $ python -m pip install git+https://gitlab.com/amv213/allantoolkit.git
    ```
   
2. 🐍 Test your installation running the following minimal script:

   ```python
   import allantoolkit
    
   # Generate some pink noise phase data
   noise = allantoolkit.noise.pink(10000)
   x = noise.data
   
   # Compute overlappig Allan deviation
   out = allantoolkit.devs.oadev(x)
   
   # Display analysis results
   print(out)
   ```

## 📚 Documentation

To learn more about the package and all new features head over to the 
[official documentation](https://amv213.gitlab.io/allantoolkit)!