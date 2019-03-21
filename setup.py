# -*- coding: utf-8 -*-
from distutils.core import setup

package_dir = {"": "src"}

packages = ["interpretability_framework"]

package_data = {"": ["*"]}

install_requires = [
    "assertpy>=0.14.0,<0.15.0",
    "numpy>=1.16,<2.0",
    "pillow>=5.4,<6.0",
    "pytest-mock>=1.10,<2.0",
    "torch>=1.0,<2.0",
    "torchvision>=0.2.2,<0.3.0",
]

setup_kwargs = {
    "name": "interpretability-framework",
    "version": "0.1.0",
    "description": "This is a framework for visualization and uncertainty in CNNs.",
    "long_description": None,
    "author": "Christina Aigner",
    "author_email": "christina@luminovo.ai >, Fabian Huch <fabian@luminovo.ai ",
    "url": None,
    "package_dir": package_dir,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.7,<4.0",
}


setup(**setup_kwargs)
