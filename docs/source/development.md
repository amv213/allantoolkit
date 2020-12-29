# 🛠 Development 

## Documentation generation

See /docs for documentation in sphinx format. On Ubuntu this requires 
the **python-sphinx** and **python-numpydoc** packages.
html/pdf documentation using sphinx can be built locally with::

    /docs$ make html
    /docs$ make latexpdf

this generates html documentation in docs/_build/html and pdf 
documentation in docs/_build/latex.

The sphinx documentation is also auto-generated online

* https://allantools.readthedocs.org

## Tests

The tests compare the output of allantools to other programs such
as Stable32. Tests may be run using py.test (http://pytest.org). 
Package managers may install it with different binary names (e.g. pytest-3 
for the python3 version on debian).


Slow tests are marked 'slow' and tests failing because of a known
reason are marked 'fails'. To run all tests::
    
    $ py.test

To exclude known failing tests::

    $ py.test -m "not fails" --durations=10

To exclude tests that run slowly::

    $ py.test -m "not slow" --durations=10

To exclude both (note option change) and also check docstrings is ReST files ::

    $ py.test -k "not (slow or fails)" --durations=10 --doctest-glob='*.rst'

To run the above command without installing the package::

    $ python setup.py test --addopts "-k 'not (fails or slow)'"

Test coverage may be obtained with the 
(https://pypi.python.org/pypi/coverage) module::

    coverage run --source allantools setup.py test --addopts "-k 'not (fails or slow)'"
    coverage report # Reports on standard output 
    coverage html # Writes annotated source code as html in ./htmlcov/

On Ubuntu this requires packages **python-pytest** and 
**python-coverage**.