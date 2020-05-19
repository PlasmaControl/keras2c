"""k2c_tensor.y
This file is part of keras2c
Allows the k2c_tensor struct to be used in python
"""
# emulate ../include/k2c_tensor_include.h for python ctypes functionality
# imports
from ctypes import Structure, c_size_t, c_void_p, CDLL, POINTER, byref

__author__ = "Mitchell Clement"
__copyright__ = "Copyright 2020, Mitchell Clement"
__license__ = "GNU GPLv3"
__maintainer__ = "Mitchell Clement, https://github.com/mdclemen/keras2c"
__email__ = "mclement@pppl.gov"

K2C_MAX_NDIM = 5
SIZE_T_MAX_NDIM = c_size_t*K2C_MAX_NDIM

class k2c_tensor(Structure):
    _fields_ = [("array", c_void_p),
                ("ndim", c_size_t),
                ("numel", c_size_t),
                ("shape", SIZE_T_MAX_NDIM)]
