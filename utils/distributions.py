"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Useful structures for testing the project. This is a utility file defining a
function that provides closures returning samples from a given distribution.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
rng = np.random.default_rng(123)


def get_distribution(name: str, param=None):
    """
    Provides samples from a distribution or profile with a given name.

    :param name: str, name of the distribution, must be one of the following:
        - 'gaussian': Gaussian distribution with mean 0 and variance 1
        - 'uniform': Uniform distribution on the unit interval
        - 'truncated_gaussian': Truncated Gaussian distribution with mean 0 and
            variance param (default=1), truncated on the unit interval
        - 'bimodal_gaussian': Bimodal Gaussian distribution with means -1 and 1
            and variance param (default=1)
        - 'constant': Constant distribution with value 1
        - 'sine': Sine profile with frequency param (default=1)
        - 'sawtooth': Sawtooth profile with frequency param (default=1)
        - 'triangle': Triangle profile with frequency param (default=1)
        - 'step': Step profile with frequency param (default=1)
    :param param: float, parameter of the distribution or profile
    :return: closure with signature (n) -> xis, where xis is a matrix of n
        samples of the distribution.
    """

    # TODO: transform param into mutable to store current time for profiles
    if name.lower() == "gaussian":
        def distribution(n):
            return rng.standard_normal((n, 1))
    else:
        raise NotImplementedError(f"Distribution with name {name} "
                                  f"is not in our library.")

    return distribution

