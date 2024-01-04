import numpy as np
import matplotlib.pyplot as plt
from utils.distributions import get_distribution, implemented


def test_distributions(verbose=False):
    """
    tests the achievability constraints generation for a random closed loop map.
    :param verbose: bool, if True, prints the optimization verbose.
    """
    _n = 13
    ds = dict()
    ps = dict()

    for n in implemented:
        if n in ['constant', 'sine', 'sawtooth', 'triangle', 'step']:
            d = get_distribution(n, [3] if n == 'step' else [0.1])
            ps[n] = np.vstack([d(_n), d(_n), d(_n)])
        else:
            d = get_distribution(n, [])
            ds[n] = 10 * np.vstack([d(_n), d(_n), d(_n)]) + _n*3/2

    plt.figure()
    for n, points in ps.items():
        plt.plot(points, label=n)
    for i, (n, points) in enumerate(ds.items()):
        plt.scatter(points, -(i/2+2)*np.ones_like(points), label=n)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

    if verbose:
        print("All tests passed")
        print("The profiles are shown at the top, "
              "over time (x axis) and around y=0.")
        print(f"The distributions are shown at the bottom,"
              f"centered around {_n*3/2} and scaled by 10.")

# Press the green button in the gutter to run the test.
if __name__ == '__main__':
    test_distributions(True)


