# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'allantoolkit'
copyright = '2020, Alvise Vianello'
author = 'Alvise Vianello'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [

]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = "classic"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# https://sphinx-book-theme.readthedocs.io/en/latest/index.html
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://gitlab.com/amv213/allantoolkit",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
}

extra_navbar = "<p>Your HTML</p>"

html_sidebars = {
    "**": ["sidebar-search-bs.html",
           "sbt-sidebar-nav.html",
           "sbt-sidebar-footer.html"]
}
# extra_navbar = "<div>Icons made by <a
# href='https://www.flaticon.com/authors/eucalyp' title='Eucalyp'>Eucalyp</a> from <a href='https://www.flaticon.com/' title='Flaticon'>www.flaticon.com</a></div>"


html_title = "AllanToolkit Documentation"
html_logo = "_static/logo.png"
html_favicon = "_static/favico.png"
show_navbar_depth = 2

# To build the documentation run the following:
# cd docs/
# sphinx-apidoc -o source/ ../ "../setup.py"
# sphinx-build -b html source build/html

