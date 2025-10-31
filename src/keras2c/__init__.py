"""__init__.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c
"""

from . import keras2c_main
from .keras2c_main import k2c

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2019, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"
__version__ = "1.0"
