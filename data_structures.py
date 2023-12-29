"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Useful structires for the project. This is a utility file defining the
classes for linear systems, polytopes, and other useful structures.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from dataclasses import dataclass


@dataclass
class LinearSystem:
    """
    This class defines a system with a state space, an input space,
    and linear dynamics.
    """
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __init__(self):
        self.a, self.b, self.c = [None]*3


@dataclass
class Polytope:
    """
    This class defines a polytope {x | Hx <= g}.
    Default is the whole euclidian space.
    """
    h: np.ndarray
    g: np.ndarray

    def __init__(self):
        self.h, self.g = [0] * 2
