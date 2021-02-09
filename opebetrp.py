import numpy as np
from collections import defaultdict, namedtuple
from numba import njit
import cvxpy as cp
import warnings
from scipy.linalg import cho_factor, cho_solve

def solve(wmax, A, b):
    psi = 2 - 4 * np.log(2)
    G = np.array([(-1.0, -1.0, 0.0),
                  (-1.0, -1.0, 1.0),
                  (0.0, -1.0, -1.0),
                  (0.0, -1.0, 0.0),
                  (0.0, 0.0, -1.0),
                  (0.0, 0.0, 0.0),
                  (0.0, 0.0, 1.0),
                  (0.0, wmax-1, -1.0),
                  (0.0, wmax-1, 0.0),
                  (0.0, wmax-1, wmax-1),
                  (0.0, wmax-1, wmax),
                  (wmax-1, wmax-1, -wmax),
                  (wmax-1, wmax-1, -(wmax-1)),
                  (wmax-1, wmax-1, 0.0),
                  (wmax-1, wmax-1, 1.0)])
    """
    G = np.array([(-1.0,   -1.0,   -1.0),
                  (-1.0,   -1.0,    0.0),
                  ( 0.0,   -1.0,   -1.0),
                  ( 0.0,   -1.0,    0.0),
                  ( 0.0,   wmax-1, -1.0),
                  ( 0.0,   wmax-1, wmax),
                  (wmax-1, wmax-1, -1.0),
                  (wmax-1, wmax-1, wmax)])
    """
    #success = True
    #xunc = np.zeros(3)
    #try:
    #    cf = cho_factor(-2*psi*A)
    #    xunc = cho_solve(cf, b, overwrite_b=False)
    #    if np.any(G@xunc <= -0.49):
    #        success = False
    #except Exception as e:
    #    warnings.warn(str(e))
    #    success = False
    #if success:
    #    warnings.warn('accepted unconstrained solution!')
    #    return xunc
    x = cp.Variable(3)
    constraints = [1+G@x >= 0.5]
    problem = cp.Problem(cp.Maximize(psi * cp.quad_form(x, A) + x.T @ b), constraints)
    try:
        problem.solve()
        #if min(1+G@x.value) > 0.501:
        #    print(x.value, xunc)
        return x.value
    except Exception as e:
        warnings.warn(str(e))
        return np.zeros(3)


def solve2(wmax, A, b):
    psi = 2 - 4 * np.log(2)
    G = np.array([(-1.0, -1.0),
                  (wmax-1, -wmax)])

    x = cp.Variable(2)
    constraints = [G@x >= -0.5, x[1] >= 0]
    problem = cp.Problem(cp.Maximize(psi * cp.quad_form(x, A) + x.T @ b), constraints)
    try:
        problem.solve()
        return x.value
    except Exception as e:
        warnings.warn(str(e))
        return np.zeros(2)


def solve_0(wmax, A, b):
    psi = 2 - 4 * np.log(2)
    G = np.array([(-1.0, -1.0),
                  (wmax-1, -1)])

    x = cp.Variable(2)
    constraints = [G@x >= -0.5, x[1] >= 0]
    problem = cp.Problem(cp.Maximize(psi * cp.quad_form(x, A) + x.T @ b), constraints)
    try:
        problem.solve()
        return x.value
    except Exception as e:
        warnings.warn(str(e))
        return np.zeros(2)


def solve_c(wmax, A, b):
    psi = 2 - 4 * np.log(2)

    G = np.array([[0., 1.],
           [-1., 1.],
           [0., 0.],
           [-1., 0.],
           [wmax-1, 0.],
           [wmax-1, 1.],
           [wmax-1, -(wmax-1)],
           [wmax-1, wmax]])


    #G = np.array([(-1.0, 0.0),
    #              (wmax-1, -(wmax-1))])

    x = cp.Variable(2)
    constraints = [G@x >= -0.5]
    problem = cp.Problem(cp.Maximize(psi * cp.quad_form(x, A) + x.T @ b), constraints)
    try:
        problem.solve()
        return x.value
    except Exception as e:
        warnings.warn(str(e))
        return np.zeros(2)



def solve_gd(wmax, A, b):
    psi = 2 - 4 * np.log(2)
    G = np.array([(-1.0, 1.0),
                  (-1.0, -2.0),
                  (wmax-1, -1.0),
                  (wmax-1, wmax)])

    x = cp.Variable(2)
    constraints = [G@x >= -0.5]
    problem = cp.Problem(cp.Maximize(psi * cp.quad_form(x, A) + x.T @ b), constraints)
    try:
        problem.solve()
        return x.value
    except Exception as e:
        warnings.warn(str(e))
        return np.zeros(2)


@njit
def update_lb(log_wealth, vs, neg_log_alpha):
    return vs[np.argmax(log_wealth < neg_log_alpha)]


@njit
def update_wealth(log_wealth, bet, vs, wi, ri, ci):
    log_wealth += np.log1p(bet[0]*0 + bet[1] * (wi - 1) + bet[2] * (wi * ri - ci - vs))


@njit
def update_wealth2(log_wealth, bet, vs, wi, ri, ci):
    log_wealth += np.log1p(bet[0] * (wi - 1) + bet[1] * (wi * ri - ci - vs))


@njit
def update_wealth_gd(log_wealth, bet, vs, wi, ri):
    log_wealth += np.log1p(bet[0] * (wi - 1) + bet[1] * (wi * ri - ri - vs))


def lcs4d(wrc, wmin, wmax, alpha):
    bet = np.zeros(3)
    v = 0
    vs = np.linspace(0, 1, 10000)
    log_wealth = np.log(0.5) * np.ones(len(vs))
    neg_log_alpha = np.log(1 / alpha)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    A0 = np.zeros((3, 3))
    A1 = np.zeros((3, 3))
    A2 = np.zeros((3, 3))
    b0 = np.zeros(3)
    b1 = np.zeros(3)
    z = np.zeros(3)
    z2 = np.zeros((3, 3))
    z2[-1, -1] = 1
    z3 = np.zeros(3)
    z3[-1] = -1
    z4 = np.zeros((3, 3))

    for t,(w,r,c) in enumerate(wrc):
        update_wealth(log_wealth, bet, vs, w, r, c)
        v = max(v, update_lb(log_wealth, vs, neg_log_alpha))
        yield v

        z[:] = [0, w - 1, w * r - c]
        z4 *= 0
        z4[:, -1] -= z
        z4[-1, :] -= z
        s = t / (t+1.0)
        r = 1 / (t+1.0)
        A0 *= s
        A1 *= s
        A2 *= s
        b0 *= s
        b1 *= s
        A0 += r * np.outer(z, z)
        A1 += r * z4
        A2 += r * z2
        b0 += r * z
        b1 += r * z3
        A[...] = A0 + v * A1 + (v*v) * A2
        b[...] = b0 + v * b1

        bet = solve(wmax, A, b)

def reflect(wrc):
    w,r,c = wrc
    return w, 1-r, w-1-c


def wealth_lb_3d(wrc, wmin, wmax, alpha):
    import tqdm
    from itertools import tee
    lwrc, lwrc2 = tee(wrc)
    uwrc = map(reflect, lwrc2)
    lb = []
    ub = []
    for vmin, vminprime in tqdm.tqdm(zip(lcs4d(lwrc, wmin, wmax, alpha), lcs4d(uwrc, wmin, wmax, alpha)), total=len(wrc), ncols=80):
        vmax = 1 - vminprime
        #vmin = min(vmax,vmin)
        #vmax = max(vmin,vmax)
        lb.append(vmin)
        ub.append(vmax)
    return lb, ub





def lcs2(wrc, wmin, wmax, alpha):
    bet = np.zeros(2)
    v = 0
    vs = np.linspace(0, 1, 10000)
    log_wealth = np.log(0.5) * np.ones(len(vs))
    neg_log_alpha = np.log(1 / alpha)

    A = np.zeros((2, 2))
    b = np.zeros(2)
    A0 = np.zeros((2, 2))
    A1 = np.zeros((2, 2))
    A2 = np.zeros((2, 2))
    b0 = np.zeros(2)
    b1 = np.zeros(2)
    z = np.zeros(2)
    z2 = np.zeros((2, 2))
    z2[-1, -1] = 1
    z3 = np.zeros(2)
    z3[-1] = -1
    z4 = np.zeros((2, 2))

    for t,(w,r,c) in enumerate(wrc):
        update_wealth2(log_wealth, bet, vs, w, r, c)
        v = max(v, update_lb(log_wealth, vs, neg_log_alpha))
        yield v

        z[:] = [w - 1, w * r - c]
        z4 *= 0
        z4[:, -1] -= z
        z4[-1, :] -= z
        s = t / (t+1.0)
        r = 1 / (t+1.0)
        A0 *= s
        A1 *= s
        A2 *= s
        b0 *= s
        b1 *= s
        A0 += r * np.outer(z, z)
        A1 += r * z4
        A2 += r * z2
        b0 += r * z
        b1 += r * z3
        A[...] = A0 + v * A1 + (v*v) * A2
        b[...] = b0 + v * b1

        bet = solve2(wmax, A, b)


def reflect2(wrc):
    w,r,c = wrc
    return w, 1-r, w-1-c


def wealth_lb_rp(wrc, wmin, wmax, alpha):
    import tqdm
    from itertools import tee
    lwrc, lwrc2 = tee(wrc)
    uwrc = map(reflect2, lwrc2)
    lb = []
    ub = []
    for vmin, vminprime in tqdm.tqdm(zip(lcs2(lwrc, wmin, wmax, alpha), lcs2(uwrc, wmin, wmax, alpha)), total=len(wrc), ncols=80):
        vmax = 1-vminprime
        lb.append(vmin)
        ub.append(vmax)
    return lb, ub


def lcs_double(wrc, wmin, wmax, alpha):
    bet = np.zeros(3)
    v = 0
    vs = np.linspace(0, 1, 10000)
    log_wealth = np.log(0.25) * np.ones(len(vs))
    neg_log_alpha = np.log(1 / alpha)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    A0 = np.zeros((3, 3))
    A1 = np.zeros((3, 3))
    A2 = np.zeros((3, 3))
    b0 = np.zeros(3)
    b1 = np.zeros(3)
    z = np.zeros(3)
    z2 = np.zeros((3, 3))
    z2[-1, -1] = 1
    z3 = np.zeros(3)
    z3[-1] = -1
    z4 = np.zeros((3, 3))

    for t,(w,r,c) in enumerate(wrc):
        update_wealth(log_wealth, bet, vs, w, r, c)
        v = max(v, update_lb(log_wealth, vs, neg_log_alpha))
        yield v

        z[:] = [0, w - 1, w * r - c]
        z4 *= 0
        z4[:, -1] -= z
        z4[-1, :] -= z
        s = t / (t+1.0)
        r = 1 / (t+1.0)
        A0 *= s
        A1 *= s
        A2 *= s
        b0 *= s
        b1 *= s
        A0 += r * np.outer(z, z)
        A1 += r * z4
        A2 += r * z2
        b0 += r * z
        b1 += r * z3
        A[...] = A0 + v * A1 + (v*v) * A2
        b[...] = b0 + v * b1

        bet = solve(wmax, A, b)


def r2(wrc):
    w, r, c = wrc
    return w, r, 0


def r3(wrc):
    w, r, c = wrc
    return w, 1-r, w-1-c


def r4(wrc):
    w, r, c = wrc
    return w, 1-r, 0


def wealth_lb_rp_double_hedge(wrc, wmin, wmax, alpha):
    import tqdm
    from itertools import tee
    lwrc, lwrc2, lwrc3, lwrc4 = tee(wrc, 4)
    lwr0 = map(r2, lwrc2)
    uwrc = map(r3, lwrc3)
    uwr0 = map(r4, lwrc4)
    lb = []
    ub = []
    for vmin, vmin0, vminprime, vminprime0 in tqdm.tqdm(zip(lcs_double(lwrc, wmin, wmax, alpha),
                                                            lcs_double(lwr0, wmin, wmax, alpha),
                                                            lcs_double(uwrc, wmin, wmax, alpha),
                                                            lcs_double(uwr0, wmin, wmax, alpha),
                                                            ), total=len(wrc), ncols=80):
        vmax = 1 - vminprime
        vmax0 = 1 - vminprime0
        #print(vmin, vmin0, vmax, vmax0)
        lb.append(max(vmin,vmin0))
        ub.append(min(vmax,vmax0))
    return lb, ub


def lcs_gd(wr, wmin, wmax, alpha):
    bet = np.zeros(2)
    v = -1
    vs = np.linspace(-1, 1, 10000)
    log_wealth = np.log(0.5) * np.ones(len(vs))
    neg_log_alpha = np.log(1 / alpha)

    A = np.zeros((2, 2))
    b = np.zeros(2)
    A0 = np.zeros((2, 2))
    A1 = np.zeros((2, 2))
    A2 = np.zeros((2, 2))
    b0 = np.zeros(2)
    b1 = np.zeros(2)
    z = np.zeros(2)
    z2 = np.zeros((2, 2))
    z2[-1, -1] = 1
    z3 = np.zeros(2)
    z3[-1] = -1
    z4 = np.zeros((2, 2))

    for t,(w,r) in enumerate(wr):
        update_wealth_gd(log_wealth, bet, vs, w, r)
        v = max(v, update_lb(log_wealth, vs, neg_log_alpha))
        yield v

        z[:] = [w - 1, w * r - r]
        z4 *= 0
        z4[:, -1] -= z
        z4[-1, :] -= z
        s = t / (t+1.0)
        r = 1 / (t+1.0)
        A0 *= s
        A1 *= s
        A2 *= s
        b0 *= s
        b1 *= s
        A0 += r * np.outer(z, z)
        A1 += r * z4
        A2 += r * z2
        b0 += r * z
        b1 += r * z3
        A[...] = A0 + v * A1 + (v*v) * A2
        b[...] = b0 + v * b1

        bet = solve_gd(wmax, A, b)

def reflect_gd(wr):
    w,r = wr
    return w, 1-r


def wealth_lb_gd(wr, wmin, wmax, alpha):
    import tqdm
    from itertools import tee
    lwr, lwr2 = tee(wr)
    uwr = map(reflect_gd, lwr2)
    lb = []
    ub = []
    for vmin, vminprime in tqdm.tqdm(zip(lcs_gd(lwr, wmin, wmax, alpha), lcs_gd(uwr, wmin, wmax, alpha)), total=len(wr), ncols=80):
        vmax = -vminprime
        lb.append(vmin)
        ub.append(vmax)
    return lb, ub