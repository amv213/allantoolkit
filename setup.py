import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="allantoolkit",
    version="0.0.1",
    description="A fork of Anders E.E. Wallin's [AllanTools](https://github.com/aewallin/allantools): "
                "Allan deviation and related time/frequency statistics toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alvise Vianello",
    author_email="alvise@vianello.ai",
    url="https://gitlab.com/amv213/allantoolkit",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later " 
        "(GPLv3+)",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        ],
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'matplotlib',
        'numpy',
        'pytest',
        'setuptools',
    ],
)

# to build the package run the following:
# python setup.py sdist bdist_wheel
# twine check dist/*
