import numpy as np
from utils.simulate import split_clm, simulate
from utils.data_structures import LinearSystem


def test_simulate(verbose=False):
    """
    tests the achievability constraints generation for a random closed loop map.
    :param verbose: bool, if True, prints the optimization verbose.
    """
    sys = LinearSystem()
    sys.a, sys.b, sys.c = np.eye(2), np.array([[1], [0]]), np.array([[2, 0]])
    xi = np.ones((6*3, 1))
    x, u, y = simulate(np.kron(np.arange(6), np.ones((3, 1))), sys, xi)

    if verbose:
        print("states:")
        print(x)
        print("inputs:")
        print(u)
        print("outputs:")
        print(y)

    assert np.all(x == np.vstack((np.zeros((4, 1)), 15*np.ones((8, 1)))))
    assert np.all(u == np.vstack((np.zeros((2, 1)), 15*np.ones((4, 1)))))
    assert np.all(y == np.vstack((np.zeros((2, 1)), 30*np.ones((4, 1)))))

    if verbose:
        print("All tests passed")


# Press the green button in the gutter to run the test.
if __name__ == '__main__':
    # Not using generators for unit tests
    np.random.seed(123)

    test_simulate(True)
