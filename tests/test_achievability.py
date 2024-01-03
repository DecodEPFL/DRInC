import numpy as np
import cvxpy as cp
from achievability import achievability_constraints
from utils.simulate import split_clm
from utils.data_structures import LinearSystem


def test_achievability_constraints(verbose=False):
    """
    tests the achievability constraints generation for a random closed loop map.
    :param verbose: bool, if True, prints the optimization verbose.
    """
    np.random.seed(123)

    # Parameters
    _m, _n, _p = 2, 3, 2
    t_fir = 4

    # Define a system
    sys = LinearSystem()
    sys.a = np.random.randn(_n, _n)
    sys.b = np.random.randn(_n, _m)
    sys.c = np.random.randn(_p, _n)

    # Generate the constraints
    mkcons = achievability_constraints(sys, t_fir)

    # Generate a random closed loop map
    phi = cp.Variable(((_n + _m), (_n + _p) * t_fir))
    phi_tar = np.random.randn((_n + _m), (_n + _p) * t_fir)

    # Try to be random but still satisfy constraints
    cp.Problem(cp.Minimize(cp.norm(phi - phi_tar, 'fro')),
               mkcons(phi)).solve(verbose=verbose)
    # Transform to list and store shifted values
    phi = list(reversed(split_clm(phi.value, _n, t_fir)))
    phis = phi[1:] + [0*phi[0]]

    # Check constraints
    # Short notation
    def _ass(x, y):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-8)

    for i, (p, ps) in enumerate(zip(phi, phis)):
        if i == 0:
            # first delay term z^0
            _ass(p[:_n, :], 0)
            _ass(p[_n:, :_n], 0)

            # Second delay term z^1
            _ass(ps[:_n, :_n], np.eye(_n))
            _ass(ps[:_n, _n:], sys.b @ p[_n:, _n:])
            _ass(ps[_n:, :_n], p[_n:, _n:] @ sys.c)
        else:  # Other terms, phi(t_fir+1) is zero
            # Pxw(k+1) = A Pxw(k) + B Puw(k)
            _ass(ps[:_n, :_n], sys.a @ p[:_n, :_n] + sys.b @ p[_n:, :_n])
            # Pxv(k+1) = A Pxv(k) + B Puv(k)
            _ass(ps[:_n, _n:], sys.a @ p[:_n, _n:] + sys.b @ p[_n:, _n:])

            # Pxw(k+1) = Pxw(k) A + Pxv(k) C
            _ass(ps[:_n, :_n], p[:_n, :_n] @ sys.a + p[:_n, _n:] @ sys.c)
            # Puw(k+1) = Puw(k) A + Puv(k) C
            _ass(ps[_n:, :_n], p[_n:, :_n] @ sys.a + p[_n:, _n:] @ sys.c)

    if verbose:
        print("All tests passed")


# Press the green button in the gutter to run the test.
if __name__ == '__main__':
    for k in range(10):
        test_achievability_constraints()
