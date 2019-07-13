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
from unittest import mock

import better_apidoc
from recommonmark.transform import AutoStructify

midnite_dir = os.path.abspath("../../src/midnite")

sys.path.insert(0, midnite_dir)
sys.path.insert(0, os.path.abspath(midnite_dir + "/.."))


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
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.apidoc",
    "sphinxcontrib.bibtex",
    "recommonmark",
    "nbsphinx_link",
    "nbsphinx",
]

# Configure apidocs
apidoc_module_dir = midnite_dir
apidoc_exlcude_paths = ["tests"]
apidoc_separate_modules = True
apidoc_toc_file = False

autodoc_mock_imports = ["tqdm", "cv2"]

# Use short class/function names in modules
add_module_names = False

# Add package to latex output that is capable of drawing jupyter notebook outputs
latex_elements = {"preamble": r"\usepackage{pmboxdraw}"}

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_materialdesign_theme"

html_theme_options = {
    "header_links": [
        ("Home", "index", False, "home"),
        ("GitLab", "https://gitlab.com/luminovo/midnite", True, "link"),
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_favicon = "assets/images/favicon.ico"

# markdown support
source_suffix = [".rst", ".md"]


# Register better apidoc
def setup(app):
    # Sys-patching is required as better-apidoc does not respect autodoc_mock_imports
    for mod_name in autodoc_mock_imports:
        sys.modules[mod_name] = mock.Mock()
    app.connect("builder-inited", run_apidoc)
    app.add_config_value(
        "recommonmark_config",
        {"enable_eval_rst": True, "enable_auto_toc_tree": True},
        True,
    )
    app.add_transform(AutoStructify)
