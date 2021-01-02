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
calculation of confidence intervals based on $\\chi^2$ statistics. These 
features have now all been implemented, but would benefit from further 
testing before being released.

This fork also wanted to modernise the whole package - streamlining the code, 
adopting more modern coding practices, implementing better design patterns, 
and re-designing existing documentation. This should make future development 
easier, and code more maintainable. For this reason a lot of the original 
codebase structure has been heavily changed.

I have made sure not to break any of the original tests during development, 
and have added new ones as I went along. Nonetheless, given the magnitude 
of changes, the eventual merging with the original project is likely to
have to happen in a monolithic way, rather than through incremental changes.
This means that additional extensive testing of every aspect of the fork 
needs to take place, to make sure that everything works flawlessly.

Anyone who would like to help is more than welcome! Just install this fork 
and start opening issues on the GitLab repo. A non-exhaustive list of new 
features in this fork is here just below!

```{dropdown} 🔥 CHANGELOG
:title: text-center
:animate: fade-in-slide-down
- new low-level API (based on NamedTuple)
- fully featured object-oriented high-level API
- Auto noise ID for all deviation types (Lag1 ACF, B1, R(n))
- Bias correction for all deviation types
- Non-naive EDF / confidence interval calculation for all deviation types
- Support for Stable32-like implementation of confidence interval calculation
- More modern plotting and visualisation tools
- Fast Theo1 / TheoBR algorithm
- Fast MTIE binary decomposition algorithm
- Many-tau and Fast u tau generation
- Streamlined noise generation, based on Kasdin & Walter algorithm
- Support for discrete simulation of arbitrary power law noise, in 
addition to utility functions for violet, blue, white, pink, brown, flfm and, 
rwfm noise generation
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
   ># Generate some pink noise phase data
   >noise = allantoolkit.noise.white(10000)
   >x = noise.data
   >
   ># Compute overlappig Allan deviation
   >out = allantoolkit.devs.oadev(x)
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