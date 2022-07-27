""" Define Preprocess module """

import torch
from typing import Tuple, Iterable, Union, Callable, List
from torch.autograd import Variable, grad
import torch.nn as nn
from augfeatures import _Augment


class Scaler(_Augment):
    """A custom data scaler"""

    def __init__(self, *args, **kwargs):
        """ See base class docstring for all args and kwargs"""
        super(Scaler, self).__init__(*args, **kwargs)
    
    # TODO: needs to be properly written!
    def initialize(self, x):
        
        self.scaler = None
    
    def forward(self, x, inverse=False):

        if inverse:
            x *= self.scaler
        else:
            x /= self.scaler

        return x
