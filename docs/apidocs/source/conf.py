# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import better_apidoc

midnite_dir = os.path.abspath("../../../src/midnite")

sys.path.insert(0, os.path.abspath(midnite_dir))


def run_apidoc(app):
    """Generate API documentation"""

    better_apidoc.APP = app
    better_apidoc.main(
        [
            "better-apidoc",
            "-t",
            os.path.join(".", "templates"),
            "--force",
            "--no-toc",
            "--separate",
            "-o",
            os.path.join(".", "source", "api"),
            midnite_dir,
        ]
    )


# -- Project information -----------------------------------------------------

project = "midnite"
copyright = "2019"
author = "Christina Aigner, Fabian Huch"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinxcontrib.apidoc"]

# Configure apidocs
apidoc_module_dir = midnite_dir
apidoc_exlcude_paths = ["tests"]
apidoc_separate_modules = True
apidoc_toc_file = False

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# Register better apidoc
def setup(app):
    app.connect("builder-inited", run_apidoc)
