from collections import namedtuple, defaultdict
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from numba import njit
import cvxpy as cp

Problem = namedtuple('Problem', ['A', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x', 'wmax', 'cands'])


def update_stats(d, w, r, tau):
    f = np.log1p(-tau) + tau
    d['c'] += tau * w * r
    d['s'] += tau
    d['q'] += f * w * w * r * r
    d['t'] += f * w * r
    d['u'] += f
    d['n'] += 1
    d['z'] += w * r
    d['y'] += w * w * r * r
    d['x'] += w * w
    d['w'] += w
    d['v'] += w * w * r
    return d


def fill_quad(d, v, A, b):
    a = 8 * np.log(2) - 4
    A[0, 0] = a * (d['x'] + d['n'] - 2 * d['w'])
    A[0, 1] = a * (d['v'] - d['z'] + v * (d['n'] - d['w']))
    A[1, 0] = A[0, 1]
    A[1, 1] = a * (d['y'] - 2 * v * d['z'] + v * v * d['n'])
    b[0] = d['w'] - d['n']
    b[1] = d['z'] - v * d['n']


def update_tau(d, v):
    s = d['z'] - d['n'] * v
    q = d['n'] * v * v + d['y'] - 2 * v * d['z']
    return max(0.0, min(0.9, s / (s + q))) if q > 0 else 0


def update_lb(d, alpha):
    a = d['u']
    if a == 0:
        return 0
    b = -(2 * d['t'] + d['s'])
    c = d['q'] + d['c'] - np.log(2 / alpha)
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0
    else:
        return (-b - np.sqrt(disc)) / (2 * a)


def wealth_lb_1d(wr, wmin, wmax, alpha):
    vmin = 0
    vmax = 0
    taumin = 0
    taumax = 0
    dmin = defaultdict(float)
    dmax = defaultdict(float)
    lb = []
    ub = []
    for t, (wi, ri) in enumerate(wr):
        dmin = update_stats(dmin, wi, ri, taumin)
        dmax = update_stats(dmax, wi, 1 - ri, taumax)
        vmin = min(1-vmax, max(vmin, update_lb(dmin, alpha)))
        vmax = min(1-vmin, max(vmax, update_lb(dmax, alpha)))
        lb.append(vmin)
        ub.append(1 - vmax)
        taumin = update_tau(dmin, vmin)
        taumax = update_tau(dmax, vmax)
    return lb, ub


def core_solve(prob):
    A = prob.A
    try:
        L, low = cho_factor(A, lower=True)
        L[0, 1] = 0
    except:
        return np.zeros(2)
    b = prob.b
    c = prob.c
    d = prob.d
    x = cho_solve((L, low), b)
    e = x.dot(c) - d
    if np.all(e >= 0):
        return x

    g = cho_solve((L, low), c)
    l = -e / np.sum(g * c, axis=0)
    cands = prob.cands
    cands[:3, :] = np.reshape(x, (1, -1)) + np.reshape(l, (-1, 1)) * g.T

    # replace infeasible stuff with feasible stuff
    if cands[0, 0] < -0.5 / (prob.wmax - 1) or cands[0, 0] > 0.5:
        cands[0, :] = 0
    if cands[1, 0] < 0 or cands[1, 1] < 0:
        cands[1, :] = 0
    if cands[2, 0] > 0 or cands[2, 1] < 0:
        cands[2, :] = 0
    # search over feasible candidates for the best objective
    q = cands @ L
    am = np.argmin(0.5 * np.sum(q * q, axis=1) - cands @ b)
    ret = cands[am]
    return ret


@njit
def core_solve_jit(A, b, c, d, e, f, g, h, x, wmax, cands):
    if A[0, 0] <= 0 or A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1] <= 0:
        t = np.copy(cands[3:, :])
        tAt = np.sum((t @ A) * t, axis=1)
        obj = 0.5 * tAt - t @ b
        am = np.argmin(obj)
        return t[am]
    A[0, 0] = np.sqrt(A[0, 0])
    A[1, 0] /= A[0, 0]
    A[1, 1] -= A[1, 0] * A[1, 0]
    A[1, 1] = np.sqrt(A[1, 1])
    A[0, 1] = 0

    p = b[0] / A[0, 0]
    x[1] = (b[1] - A[1, 0] * p) / (A[1, 1] * A[1, 1])
    x[0] = (p - A[1, 0] * x[1]) / A[0, 0]
    np.dot(x, c, out=e)
    e -= d
    if np.all(e >= 0):
        return x

    for i in range(3):
        p = c[0, i] / A[0, 0]
        g[1, i] = (c[1, i] - A[1, 0] * p) / (A[1, 1] * A[1, 1])
        g[0, i] = (p - A[1, 0] * g[1, i]) / A[0, 0]
        l = -e[i] / (g[0, i] * c[0, i] + g[1, i] * c[1, i])
        cands[i, :] = x + l * g[:, i]

    # replace infeasible stuff with feasible stuff
    if cands[0, 0] < -0.5 / (wmax - 1) or cands[0, 0] > 0.5:
        cands[0, :] = 0
    if cands[1, 0] < 0 or cands[1, 1] < 0:
        cands[1, :] = 0
    if cands[2, 0] > 0 or cands[2, 1] < 0:
        cands[2, :] = 0
    # search over feasible candidates for the best objective
    np.dot(cands, A, out=f)
    np.dot(cands, b, out=h)
    for i in range(6):
        h[i] = 0.5 * (f[i, 0] * f[i, 0] + f[i, 1] * f[i, 1]) - h[i]
    am = np.argmin(h)
    return cands[am]


def solve_quad(prob, d, v):
    fill_quad(d, v, prob.A, prob.b)
    x = core_solve_jit(prob.A, prob.b, prob.c, prob.d, prob.e, prob.f, prob.g, prob.h, prob.x, prob.wmax, prob.cands)
    return x


@njit
def update_lb_2d(log_wealth, vs, neg_log_alpha):
    return vs[np.argmax(log_wealth < neg_log_alpha)]


@njit
def update_wealth(log_wealth, bet, vs, wi, ri):
    log_wealth += np.log1p(bet[0] * (wi - 1) + bet[1] * (wi * ri - vs))


@njit
def update_wealth2(log_wealth, bet, vs, wi, ri):
    log_wealth += np.log1p(bet[0] * (wi - 1) + bet[1] * (wi * ri - vs))


def wealth_lb_2d(wr, wmin, wmax, alpha):
    vmin = 0
    vmax = 0
    dmin = defaultdict(float)
    dmax = defaultdict(float)
    lb = []
    ub = []
    vs = np.linspace(0, 1, 1000)
    long_bet = np.zeros(2)
    short_bet = np.zeros(2)
    log_long = np.log(0.5) * np.ones(len(vs))
    log_short = np.log(0.5) * np.ones(len(vs))

    Ap = np.zeros((2, 2))
    bp = np.zeros(2)
    cp = np.array([[0.0, -1.0, wmax - 1.0], [1.0, -1.0, -1.0]])
    dp = np.array([0.0, -0.5, -0.5])
    ep = np.zeros_like(dp)
    gp = np.zeros_like(cp)
    xp = np.zeros(2)
    cands = np.zeros((6, 2))
    cands[3, :] = [0.5, 0]
    cands[4, :] = [0, 0.5]
    cands[5, :] = [-0.5 / (wmax - 1), 0.0]
    fp = np.zeros_like(cands)
    hp = np.zeros(6)

    prob = Problem(A=Ap, b=bp, c=cp, d=dp, e=ep, f=fp, g=gp, h=hp, x=xp, wmax=wmax, cands=cands)
    neg_log_alpha = np.log(1 / alpha)
    for t, (wi, ri) in enumerate(wr):
        update_wealth(log_long, long_bet, vs, wi, ri)
        update_wealth(log_short, short_bet, vs, wi, 1 - ri)
        newvmin = update_lb_2d(log_long, vs, neg_log_alpha)
        newvmax = update_lb_2d(log_short, vs, neg_log_alpha)
        vmin = min(1-vmax,max(vmin, newvmin))
        vmax = min(1-vmin,max(vmax, newvmax))
        dmin = update_stats(dmin, wi, ri, 0)
        dmax = update_stats(dmax, wi, 1 - ri, 0)
        lb.append(vmin)
        ub.append(1 - vmax)
        long_bet = solve_quad(prob, dmin, vmin)
        short_bet = solve_quad(prob, dmax, vmax)
    return lb, ub


@njit
def multi_update_wealth(log_wealth, bets, vs, wi, ri):
    log_wealth += np.log1p(bets[:,0] * (wi - 1) + bets[:,1] * (wi * ri - vs))


def lcs_ind_qps(wr, wmin, wmax, alpha):
    N = 200
    T = len(wr)
    vmin = 0
    vs = np.linspace(0, 1, N)
    bets = np.zeros((T+1, len(vs), 2))
    log_wealth = np.log(0.5) * np.ones(len(vs))
    neg_log_alpha = np.log(1 / alpha)
    psi = -0.77258872224  # 2 - 4 * ln(2)

    for i,v in enumerate(vs):
        from qpsolvers import solve_qp
        G = -np.array([[ww - 1, ww * rr - v] for ww in (0, wmax) for rr in (0, 1)]+[(0,1)])
        h = -np.array([-0.5 for ww in (0, wmax) for rr in (0, 1)]+[0])

        A = np.zeros((2,2))
        b = np.zeros(2)
        A0 = np.zeros((2,2))
        A1 = np.zeros((2,2))
        A2 = np.zeros((2,2))
        b0 = np.zeros(2)
        b1 = np.zeros(2)
        z = np.zeros(2)
        z2 = np.zeros((2,2))
        z2[1,1] = 1
        z3 = np.zeros(2)
        z3[1] = -1
        z4 = np.zeros((2,2))
        reg = 1e-6*np.eye(2)
        cur_bet = np.zeros(2)
        local_capital = np.log(0.5)
        for t, (wi, ri) in enumerate(wr):
            local_capital += np.log1p(cur_bet[0]*(wi-1)+cur_bet[1]*(wi*ri-v))
            if local_capital > neg_log_alpha:
                break
            z[:] = [wi - 1, wi*ri]
            z4 *= 0
            z4[:,1] -= z
            z4[1,:] -= z
            A0 = t/(t+1.0) * A0 + np.outer(z, z)/(t+1.0)
            A1 = t/(t+1.0) * A1 + z4/(t+1.0)
            A2 = t/(t+1.0) * A2 + z2/(t+1.0)
            b0 = t/(t+1.0) * b0 + z/(t+1.0)
            b1 = t/(t+1.0) * b1 + z3/(t+1.0)
            A[...] = 2*(-psi * (A0 + v * (A1 + v * A2)) + reg)
            b[...] = -(b0 + v * b1)
            cur_bet[:] = solve_qp(A, b, G, h, solver='quadprog')
            bets[t, i, :] = cur_bet

    these_bets = np.zeros((N, 2))
    for t, (wi, ri) in enumerate(wr):
        multi_update_wealth(log_wealth, these_bets, vs, wi, ri)
        newvmin = update_lb_2d(log_wealth, vs, neg_log_alpha)
        vmin = max(vmin, newvmin)
        yield vmin
        these_bets[...] = bets[t,:,:]

def wealth_lb_2d_individual_qps(wr, wmin, wmax, alpha):
    lwr = wr
    uwr = [(w,1-r) for w, r in wr]
    vmin = 0
    vmax = 0
    lb = []
    ub = []
    for t, (lt, ut) in enumerate(zip(lcs_ind_qps(lwr, wmin, wmax, alpha), lcs_ind_qps(uwr, wmin, wmax, alpha))):
        vmin = min(1-vmax, lt)
        vmax = min(1-vmin, ut)
        lb.append(vmin)
        ub.append(1 - vmax)
    return lb, ub

def solve_log(hist, v, wmax):
    wrs = list(hist.keys())
    z = np.array([[w-1.0, w*r-v] for w,r in wrs])
    c = np.array([hist[wr] for wr in wrs])
    c = c/np.sum(c)

    lam = cp.Variable(2)
    constraints = [lam.T@ np.array([w-1.0, w*r - vv]) >= -0.5 for w in (0, wmax) for r in (0, 1) for vv in (0,1)]
    objective = c.T@cp.log1p(z@lam)
    p = cp.Problem(cp.Maximize(objective), constraints)
    p.solve()
    return lam.value


def lcs_log(wr, wmin, wmax, alpha):
    from collections import Counter
    N = 1000
    T = len(wr)
    vmin = 0
    vs = np.linspace(0, 1, N)
    log_wealth = np.log(0.5) * np.ones(len(vs))
    neg_log_alpha = np.log(1 / alpha)

    bet = np.zeros(2)
    hist = Counter()
    for t, (wi, ri) in enumerate(wr):
        hist[(wi,ri)] +=1
        update_wealth2(log_wealth, bet, vs, wi, ri)
        newvmin = update_lb_2d(log_wealth, vs, neg_log_alpha)
        vmin = max(vmin, newvmin)
        yield vmin
        bet = solve_log(hist, vmin, wmax)


def wealth_2d(wr, wmin, wmax, alpha):
    lwr = wr
    uwr = [(w,1-r) for w, r in wr]
    vmin = 0
    vmax = 0
    lb = []
    ub = []
    for t, (lt, ut) in enumerate(zip(lcs_log(lwr, wmin, wmax, alpha), lcs_log(uwr, wmin, wmax, alpha))):
        vmin = min(1-vmax, lt)
        vmax = min(1-vmin, ut)
        lb.append(vmin)
        ub.append(1 - vmax)
    return lb, ub
