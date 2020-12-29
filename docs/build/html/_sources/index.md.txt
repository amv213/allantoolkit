# 📑 Welcome to AllanToolkit!

Allantoolkit is a fork of **Anders E.E. Wallin's** 
[`allantools`](https://githubcom/aewallin/allantools): a python library for 
calculating Allan deviation and related time & frequency statistics.

* Development at [https://github.com/aewallin/allantools](https://github.com/aewallin/allantools)
* Installation package at [https://pypi.python.org/pypi/AllanTools](https://pypi.python.org/pypi/AllanTools)
* Discussion group at [https://groups.google.com/d/forum/allantools](https://groups.google.com/d/forum/allantools)
* Documentation available at [https://allantools.readthedocs.org](https://allantools.readthedocs.org)

Don't hesitate to head over to the original Github repo to check out the 
project and to drop a star! 🌟

This fork was born in an effort to implement some key missing features: 
automatic noise identification for all frequency stability analysis types, 
bias correction for all biased frequency stability analysis types, and 
calculation of confidence intervals based on $`\Chi^2`$ statistics.

```{dropdown} 🔥 CHANGELOG
:title: text-center
:animate: fade-in-slide-down
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
```

---

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

3. 🎉 If everything went well you are now all set-up to use `allantoolkit`! 
   Enjoy!

---

## 📚 Table of Contents


```{toctree}
:caption: MAIN DOCS
:maxdepth: 2

Installation <installation.md>
Basic Usage <basic_usage.md>
Development <development.md>
Documentation <documentation.md>
```

```{toctree}
:caption: TUTORIALS
:maxdepth: 2

First Steps <tutorial_intro.md>
```

---

## 📝 Authors

* Anders E.E. Wallin, anders.e.e.wallin "at" gmail.com , 
  [https://github.com/aewallin](https://github.com/aewallin)
* Danny Price, [https://github.com/telegraphic](https://github.com/telegraphic)
* Cantwell G. Carson, carsonc "at" gmail.com
* Frédéric Meynadier, [https://github.com/fmeynadier](https://github.com/fmeynadier)
* Yan Xie, [https://github.com/yxie-git](https://github.com/yxie-git)
* Erik Benkler, [https://github.com/EBenkler](https://github.com/EBenkler)
* Alvise Vianello, [https://gitlab.com/amv213](https://gitlab.com/amv213)