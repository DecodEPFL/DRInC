"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Useful structures for testing the project. This is a utility file defining a
function that provides closures returning samples from a given distribution.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
rng = np.random.default_rng(123)
implemented = ['uniform', 'truncated_gaussian', 'bimodal_gaussian',
               'constant', 'sine', 'sawtooth', 'triangle', 'step']
#implemented = ["gaussian"]
# Add line saying why didn't use a Gaussian or any unbounded distribution
# (compare, especially in the bounded support case, which has not been studied in the literature)


def get_random_int(n: int):
    """
    generate a random integer from 0 to n-1
    :param n: int, upper bound
    :return: int, random integer
    """
    return rng.integers(n-1)


def get_distribution(name: str, param=None):
    """
    Provides samples from a distribution or profile with a given name.

    :param name: str, name of the distribution, must be one of the following:
        - 'gaussian': Gaussian distribution with mean 0 and variance 1
        - 'uniform': Uniform distribution on the unit interval
        - 'beta': Beta distribution with parameters param[0] and param[1]
        - 'truncated_gaussian': Truncated Gaussian distribution with mean 0 and
            variance param[0] (default=1), truncated on the unit interval
        - 'bimodal_gaussian': Bimodal Gaussian distribution with means
            0 and 0.8*param[1] and standard deviation 0.2*param[0] (default=1)
        - 'constant': Constant distribution with value 1
        - 'sine': Sine profile with frequency param[0] (default=1),
            phase param[1] (default=0), and unit sampling frequency
        - 'sawtooth': Sawtooth profile with frequency param[0] (default=1),
            phase param[1] (default=0), and unit sampling frequency
        - 'triangle': Triangle profile with frequency param[0] (default=1),
            phase param[1] (default=0), and unit sampling frequency
        - 'step': Step profile with phase param[1] (default=0), the phase is
            the number of zero values before the step. If param[1] == 0, 'step'
            is equivalent to 'constant'
    :param param: list, parameters of the distribution or profile. This must be
        a list even if there is zero or only one parameter. This argument is
        mutable, which means that it will store the phase of the profile and
        increment it at each call.
    :return: closure with signature (n) -> xis, where xis is a matrix of n
        samples of the distribution.
    """

    # Check that the name is a string
    if not isinstance(name, str):
        raise ValueError("The name must be a string.")

    # Check that param is a list or None
    param = [] if param is None else param
    if not isinstance(param, list):
        raise ValueError("The parameters must be a list.")

    # Fill in default parameters
    if len(param) < 2:
        param[:] = [1.0, 0.0] if len(param) == 0 else [param[0], 0.0]

    if name.lower() == "gaussian":
        def distribution(n):
            return rng.standard_normal((n, 1))
    elif name.lower() == "uniform":
        def distribution(n):
            return rng.uniform(-1.0, 1.0, (n, 1))
    elif name.lower() == "beta":
        def distribution(n):
            return rng.beta(param[0], param[1], (n, 1))
    elif name.lower() == "truncated_gaussian":
        def distribution(n):
            num = param[0] * rng.standard_normal((n, 1))
            # Replace out of bounds values by uniform samples
            # This "normalizes" the integral of the distribution to 1.0
            num[np.abs(num[:, 0]) > 1, 0] = \
                rng.uniform(-1.0, 1.0, (np.sum(np.abs(num) > 1),))
            return num
    elif name.lower() == "bimodal_gaussian":
        def distribution(n):
            # Bimodal Gaussian
            num = rng.uniform(-1.0, 1.0, (n, 1))
            num = 0.2*rng.standard_normal((n, 1)) * param[0] \
                + 0.8*(num > 0) * param[1]

            # Truncate the tails at 2 sigmas
            to_trunc = ((num[:, 0] > 0.8*param[1] + 0.4*param[0])
                        | (num[:, 0] < -0.4*param[0]))
            num[to_trunc, 0] = \
                rng.uniform(-0.4*param[0], 0.8*param[1] + 0.4*param[0],
                            (np.sum(to_trunc),))
            return num
    elif name.lower() == "constant":
        def distribution(n):
            return np.ones((n, 1))
    elif name.lower() == "sine":
        def distribution(n):
            param[1] += n * 2*np.pi * param[0]  # increment phase
            return np.sin(np.arange(-n, 0) * 2*np.pi * param[0]
                          + param[1])[:, None]
    elif name.lower() == "sawtooth":
        def distribution(n):
            param[1] = (n * param[0] * 2 + param[1]) % 2  # increment phase
            return (np.arange(-n, 0) * param[0] * 2 + param[1])[:, None] % 2 - 1
    elif name.lower() == "triangle":
        def distribution(n):
            s = 0
            param[1] = (n * param[0] * 4 + param[1]) % 4  # increment phase
            num = (np.arange(-n, 0) * param[0] * 4 + param[1])[:, None] % 2 - 1
            return num * np.sign((np.arange(-n, 0) * param[0] * 4 + param[1])
                                 % 4 - (2 - np.finfo(float).eps))[:, None]
    elif name.lower() == "step":
        def distribution(n):
            _t = rng.integers(1, n-1)  # random phase
            return np.vstack([np.zeros((_t, 1)),
                              np.ones((n-_t, 1))])
    else:
        raise NotImplementedError(f"Distribution with name {name} "
                                  f"is not in our library.")

    return distribution

