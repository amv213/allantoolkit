# 📑 Welcome to AllanToolkit!

Allantoolkit is a fork of **Anders E.E. Wallin's** [AllanTools](https://githubcom/aewallin/allantools): a python library for calculating Allan deviation 
and related time & frequency statistics. Don't hesitate to head over to his 
Github repo to check out the original project and to drop a star! 🌟

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

## 🚀 Quick Start

1. 📚 Install the latest development version of `allantoolkit`, using pip to 
   download it from this repo:

    >```bash
    >python -m pip install git+https://gitlab.com/amv213/allantoolkit.git
    >```
   
2. 🐍 Test your installation running the following minimal script:

   >```python
   >import allantoolkit
   > 
   ># Generate some frequency data
   >y = allantoolkit.noise.white(10000)
   >r = 1. # data sampling rate, in Hz
   >
   ># Compute overlappig Allan deviation
   >out = allantoolkit.devs.oadev(y, rate=r, data_type='freq')
   >
   ># Display analysis results
   >print(out)
   >```

3. 🎉 If everything went well you are now all set-up to use `allantoolit`! Enjoy!

## 📚 Table of Contents


```{toctree}
:caption: MAIN DOCS
:maxdepth: 2

Readme <readme_copy>
Documentation <documentation.md>
Development <dev.rst>
```

```{toctree}
:caption: TUTORIALS
:maxdepth: 2

First Steps <tutorial_intro.md>
```

## ✏ Authors

* Anders E.E. Wallin, anders.e.e.wallin "at" gmail.com , https://github.com/aewallin
* Danny Price, https://github.com/telegraphic
* Cantwell G. Carson, carsonc "at" gmail.com
* Frédéric Meynadier, https://github.com/fmeynadier
* Yan Xie, https://github.com/yxie-git
* Erik Benkler, https://github.com/EBenkler
* Alvise Vianello, https://gitlab.com/amv213