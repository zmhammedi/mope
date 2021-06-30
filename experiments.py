import environments.ControlledRangeVariance
from opebet import wealth_lb_1d, wealth_lb_2d, wealth_lb_2d_freegrad, wealth_2d, wealth_lb_2d_individual_qps
from cs_via_supermartingale import cs_via_supermartingale, cs_via_EWA, cs_via_EWA_debug, cs_via_supermartingale_debug, cs_via_supermartingale_1d
import pickle
import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def getenv(wsq, tv=None):
    wsupport = [0, 0.5, 2, 100]
    env = environments.ControlledRangeVariance.ControlledRangeVariance(seed=90210, wsupport=wsupport, expwsq=wsq, tv=tv)
    return env, env.getpw(), env.range(), env.expectedwsq()


def compress(data):
    # could be improved but it's used only for debugging.
    sd = sorted(tuple(datum) for datum in data)
    from itertools import groupby
    return [(len(list(g)),) + tuple(map(float, k)) for k, g in groupby(sd)]


def produce_results(env, method, alpha, ndata=100, reps=10):
    wmin, wmax = env.range()
    ubd = np.zeros(ndata)
    lbd = np.zeros(ndata)
    cov = np.zeros((reps, ndata))
    width = np.zeros((reps, ndata))
    bounds = []
    for i in range(reps):
        (truevalue, data) = env.sample(ndata)
        try:
            cs = method(data=data, wmin=wmin, wmax=wmax, alpha=alpha)
            assert np.isfinite(cs[0]).all() and np.isfinite(cs[1]).all()
            assert np.all(cs[1] >= cs[0] - 1e-4)
            assert cs[1][-1] <= 1 + 1e-4
            assert cs[0][-1] >= -1e-4
        except:
            import json
            with open('bad_case.json', 'w') as out:
                perm_state = list(env.perm_state)
                perm_state[1] = list(map(int, perm_state[1]))
                out.write(json.dumps((float(truevalue), compress(data), perm_state, float(wmin), float(wmax), alpha)))
            print('truevalue was {}'.format(truevalue))
            print('data was {}'.format(compress(data)))
            print('wmin, wmax was {} {}'.format(wmin, wmax))
            print('ci was {} {}'.format(cs[0][-1], cs[1][-1]))
            raise
        np.greater_equal(cs[1], truevalue, out=ubd)
        np.less_equal(cs[0], truevalue, out=lbd)
        cov[i, :] = ubd * lbd
        width[i, :] += np.subtract(cs[1], cs[0])
        bounds.append((truevalue, cs[0], cs[1]))

    upper_ends = [d[2][-1] for d in bounds]
    lower_ends = [d[1][-1] for d in bounds]
    upperbounded = [1 if d[0] <= d[2][-1] else 0 for d in bounds]
    lowerbounded = [1 if d[1][-1] <= d[0] else 0 for d in bounds]
    covered = [1 if u * l > 0 else 0 for (u, l) in zip(upperbounded, lowerbounded)]
    final_width = [d[2][-1] - d[1][-1] for d in bounds]

    def std_mean(x):
        return np.std(x, ddof=1) / np.sqrt(len(x) - 1)

    dbg = {
        'cov': np.mean(covered),
        'covstd': std_mean(covered),
        'ubcov': np.mean(upperbounded),
        'lbcov': np.mean(lowerbounded),
        'final_width': np.mean(final_width),
        'widthstd': std_mean(final_width),
        'widthlo': np.quantile(final_width, q=[0.05])[0],
        'widthhi': np.quantile(final_width, q=[0.95])[0],
        'ub': np.mean(upper_ends),
        'lb': np.mean(lower_ends),
    }

    verbose = True
    if verbose:
        print('{}'.format((ndata, {k: np.round(v, 4) for k, v in dbg.items()})), flush=True)

    return (ndata,
            {
                'cov': np.mean(cov, axis=0),
                'covstd': np.std(cov, axis=0, ddof=1) / np.sqrt(cov.shape[0] - 1),
                'width': np.mean(width, axis=0),
                'widtstd': np.std(width, axis=0, ddof=1) / np.sqrt(width.shape[0] - 1),
            },
            )


def produce_results_ci(env, method, alpha, ndata=100, reps=10):
    wmin, wmax = env.range()
    ubd = np.zeros(1)
    lbd = np.zeros(1)
    cov = np.zeros(reps)
    width = np.zeros(reps)
    bounds = []
    for i in range(reps):
        (truevalue, data) = env.sample(ndata)
        try:
            cs = method(data=data, wmin=wmin, wmax=wmax, alpha=alpha)
            assert np.isfinite(cs[0]) and np.isfinite(cs[1])
            assert cs[1] >= cs[0] - 1e-4
            assert cs[1] <= 1 + 1e-4
            assert cs[0] >= -1e-4
        except:
            import json
            with open('bad_case.json', 'w') as out:
                perm_state = list(env.perm_state)
                perm_state[1] = list(map(int, perm_state[1]))
                out.write(json.dumps((float(truevalue), compress(data), perm_state, float(wmin), float(wmax), alpha)))
            print('truevalue was {}'.format(truevalue))
            print('data was {}'.format(compress(data)))
            print('wmin, wmax was {} {}'.format(wmin, wmax))
            print('ci was {} {}'.format(cs[0], cs[1]))
            raise
        np.greater_equal(cs[1], truevalue, out=ubd)
        np.less_equal(cs[0], truevalue, out=lbd)
        cov[i] = ubd * lbd
        width[i] += np.subtract(cs[1], cs[0])
        bounds.append((truevalue, cs[0], cs[1]))

    upper_ends = [d[2] for d in bounds]
    lower_ends = [d[1] for d in bounds]
    upperbounded = [1 if d[0] <= d[2] else 0 for d in bounds]
    lowerbounded = [1 if d[1] <= d[0] else 0 for d in bounds]
    covered = [1 if u * l > 0 else 0 for (u, l) in zip(upperbounded, lowerbounded)]
    final_width = [d[2] - d[1] for d in bounds]

    def std_mean(x):
        return np.std(x, ddof=1) / np.sqrt(len(x) - 1)

    dbg = {
        'cov': np.mean(covered),
        'covstd': std_mean(covered),
        'ubcov': np.mean(upperbounded),
        'lbcov': np.mean(lowerbounded),
        'final_width': np.mean(final_width),
        'widthstd': std_mean(final_width),
        'widthlo': np.quantile(final_width, q=[0.05])[0],
        'widthhi': np.quantile(final_width, q=[0.95])[0],
        'ub': np.mean(upper_ends),
        'lb': np.mean(lower_ends),
    }

    verbose = True
    if verbose:
        print('{}'.format((ndata, {k: np.round(v, 4) for k, v in dbg.items()})), flush=True)

    return (ndata,
            {
                'cov': np.mean(cov, axis=0),
                'covstd': np.std(cov, axis=0, ddof=1) / np.sqrt(cov.shape[0] - 1),
                'width': np.mean(width, axis=0),
                'widtstd': np.std(width, axis=0, ddof=1) / np.sqrt(width.shape[0] - 1),
            },
            )

def bet_1d(data, wmin, wmax, alpha):
    lb, ub = wealth_lb_1d(data, wmin, wmax, alpha)
    return np.array(lb), np.array(ub)


def bet_2d(data, wmin, wmax, alpha):
    lb, ub = wealth_lb_2d(data, wmin, wmax, alpha)
    return np.array(lb), np.array(ub)

def bet_2d_freegrad(data, wmin, wmax, alpha):
    lb, ub = wealth_lb_2d_freegrad(data, wmin, wmax, alpha)
    return np.array(lb), np.array(ub)

def bet_log(data, wmin, wmax, alpha):
    lb, ub = wealth_2d(data, wmin, wmax, alpha)
    return np.array(lb), np.array(ub)


def bet_iqp(data, wmin, wmax, alpha):
    lb, ub = wealth_lb_2d_individual_qps(data, wmin, wmax, alpha)
    return np.array(lb), np.array(ub)

# Copied from
# https://github.com/pmineiro/elfcb
# Why not import it? I modified some code in asymptoticconfidenceinterval below
# TODO: send a PR.
def estimate(datagen, wmin, wmax, rmin=0, rmax=1, raiseonerr=False, censored=False):
    import numpy as np
    from scipy.optimize import brentq

    assert wmin >= 0
    assert wmin < 1
    assert wmax > 1
    assert rmax >= rmin

    num = sum(c for c, w, r in datagen())
    assert num >= 1

    # solve dual

    def sumofw(beta):
        return sum((c * w)/((w - 1) * beta + num)
                   for c, w, _ in datagen()
                   if c > 0)

    # fun fact about the MLE:
    #
    # if \frac{1}{n} \sum_n w_n < 1 then \beta^* wants to be negative
    # but as wmax \to \infty, lower bound on \beta^* is 0
    # therefore the estimate becomes
    #
    # \hat{V}(\pi) = \left( \frac{1}{n} \sum_n w_n r_n \right) +
    #                \left( 1 - \frac{1}{n} \sum_n w_n \right) \rho
    #
    # where \rho is anything between rmin and rmax

    def graddualobjective(beta):
        return sum(c * (w - 1)/((w - 1) * beta + num)
                   for c, w, _ in datagen()
                   if c > 0)

    betamax = min( ((num - c) / (1 - w)
                    for c, w, _ in datagen()
                    if w < 1 and c > 0 ),
                   default=num / (1 - wmin))
    betamax = min(betamax, num / (1 - wmin))

    betamin = max( ((num - c) / (1 - w)
                    for c, w, _ in datagen()
                    if w > 1 and c > 0 ),
                   default=num / (1 - wmax))
    betamin = max(betamin, num / (1 - wmax))

    gradmin = graddualobjective(betamin)
    gradmax = graddualobjective(betamax)
    if gradmin * gradmax < 0:
        betastar = brentq(f=graddualobjective, a=betamin, b=betamax)
    elif gradmin < 0:
        betastar = betamin
    else:
        betastar = betamax

    remw = max(0.0, 1.0 - sumofw(betastar))

    if censored:
        vnumhat = 0
        vdenomhat = 0

        for c, w, r in datagen():
            if c > 0:
                if r is not None:
                    vnumhat += w*r* c/((w - 1) * betastar + num)
                    vdenomhat += w*1* c/((w - 1) * betastar + num)

        if np.allclose(vdenomhat, 0):
            vhat = vmin = vmax = None
        else:
            vnummin = vnumhat + remw * rmin
            vdenommin = vdenomhat + remw
            vmin = min([ vnummin / vdenommin, vnumhat / vdenomhat ])

            vnummax = vnumhat + remw * rmax
            vdenommax = vdenomhat + remw
            vmax = max([ vnummax / vdenommax, vnumhat / vdenomhat ])

            vhat = 0.5*(vmin + vmax)
    else:
        vhat = 0
        for c, w, r in datagen():
            if c > 0:
                vhat += w*r* c/((w - 1) * betastar + num)

        vmin = vhat + remw * rmin
        vmax = vhat + remw * rmax
        vhat += remw * (rmin + rmax) / 2.0

    return vhat, {
            'betastar': betastar,
            'vmin': vmin,
            'vmax': vmax,
            'num': num,
            'qfunc': lambda c, w, r: c / (num + betastar * (w - 1)),
           }


# Copied from
# https://github.com/pmineiro/elfcb/blob/d0daf9e634b2382001f9b336a715e35fa2fd8619/MLE/MLE/asymptoticconfidenceinterval.py
# NB: that was the git HEAD when I copied it
# NB: a small modification was done to avoid numerical issues with scipy.stats.f.isf when dfd > 23000
def asymptoticconfidenceinterval(datagen, wmin, wmax, alpha=0.05,
                                 rmin=0, rmax=1, raiseonerr=False):
    #from scipy.special import xlogy
    from scipy.stats import f, chi2
    from math import exp, log
    import numpy as np

    assert wmin >= 0
    assert wmin < 1
    assert wmax > 1
    assert rmax >= rmin

    vhat, qmle = estimate(datagen=datagen, wmin=wmin, wmax=wmax,
                          rmin=rmin, rmax=rmax, raiseonerr=raiseonerr)
    num = qmle['num']
    if num < 2:
        return ((rmin, rmax), (None, None))
    betamle = qmle['betastar']

    if num > 23000:
        Delta = 0.5 * chi2(df=1).isf(q=alpha)
    else:
        #There are numerical issues with isf for num > 23000
        Delta = 0.5 * f.isf(q=alpha, dfn=1, dfd=num-1)

    sumwsq = sum(c * w * w for c, w, _ in datagen())
    wscale = max(1.0, np.sqrt(sumwsq / num))
    rscale = max(1.0, np.abs(rmin), np.abs(rmax))

    # solve dual

    tiny = 1e-5
    logtiny = log(tiny)

    def safedenom(x):
        return x if x > tiny else exp(logstar(x))

    def logstar(x):
        return log(x) if x > tiny else -1.5 + logtiny + 2.0*(x/tiny) - 0.5*(x/tiny)*(x/tiny)

    def jaclogstar(x):
        return 1/x if x > tiny else (2.0 - (x/tiny))/tiny

    def hesslogstar(x):
        return -1/(x*x) if x > tiny else -1/(tiny*tiny)

    def dualobjective(p, sign):
        gamma, beta = p
        logcost = -Delta

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

        assert n == num

        if n > 0:
            logcost /= n

        return (-n * exp(logcost) + gamma + beta / wscale) / rscale

    def jacdualobjective(p, sign):
        gamma, beta = p
        logcost = -Delta
        jac = np.zeros_like(p)

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)

        assert n == num

        if n > 0:
            logcost /= n
            jac /= n

        jac *= -(n / rscale) * exp(logcost)
        jac[0] += 1 / rscale
        jac[1] += 1 / (wscale * rscale)

        return jac

    def hessdualobjective(p, sign):
        gamma, beta = p
        logcost = -Delta
        jac = np.zeros_like(p)
        hess = np.zeros((2,2))

        n = 0
        for c, w, r in datagen():
            if c > 0:
                n += c
                denom = gamma + (beta + sign * wscale * r) * (w / wscale)
                mledenom = num + betamle * (w - 1)
                logcost += c * (logstar(denom) - logstar(mledenom))

                jaclogcost = c * jaclogstar(denom)
                jac[0] += jaclogcost
                jac[1] += jaclogcost * (w / wscale)

                hesslogcost = c * hesslogstar(denom)
                hess[0][0] += hesslogcost
                hess[0][1] += hesslogcost * (w / wscale)
                hess[1][1] += hesslogcost * (w / wscale) * (w / wscale)

        assert n == num

        if n > 0:
            logcost /= n
            jac /= n
            hess /= n

        hess[1][0] = hess[0][1]
        hess += np.outer(jac, jac)
        hess *= -(n / rscale) * exp(logcost)

        return hess

    consE = np.array([
        [ 1, w / wscale ]
        for w in (wmin, wmax)
        for r in (rmin, rmax)
    ], dtype='float64')

    retvals = []

    easybounds = [ (qmle['vmin'] <= rmin + tiny, rmin),
                   (qmle['vmax'] >= rmax - tiny, rmax) ]
    for what in range(2):
        if easybounds[what][0]:
            retvals.append((easybounds[what][1], None))
            continue

        sign = 1 - 2 * what
        d = np.array([ -sign*w*r + tiny
                       for w in (wmin, wmax)
                       for r in (rmin, rmax)
                     ],
                     dtype='float64')

        minsr = min(sign*rmin, sign*rmax)
        gamma0, beta0 = ( num - qmle['betastar'] + 2 * tiny,
                          wscale * (qmle['betastar'] - (1 + 1 / wscale) * minsr)
                        )

        x0 = np.array([ gamma0, beta0 ])

        if raiseonerr:
           active = np.nonzero(consE.dot(x0) - d < 0)[0]
           from pprint import pformat
           assert active.size == 0, pformat({
                   'cons': consE.dot(x0) - d,
                   'd': d,
                   'consE.dot(x0)': consE.dot(x0),
                   'active': active,
                   'x0': x0
               })

#        from .gradcheck import gradcheck, hesscheck
#        gradcheck(f=lambda p: dualobjective(p, sign),
#                  jac=lambda p: jacdualobjective(p, sign),
#                  x=x0,
#                  what='dualobjective')
#
#        hesscheck(jac=lambda p: jacdualobjective(p, sign),
#                  hess=lambda p: hessdualobjective(p, sign),
#                  x=x0,
#                  what='jacdualobjective')

        # NB: things i've tried
        #
        # scipy.minimize method='slsqp': 3.78 it/s, sometimes fails
        # sqp with quadprog: 1.75 it/s, sometimes fails
        # sqp with cvxopt.qp: 1.05 s/it, reliable
        # cvxopt.cp: 1.37 s/it, reliable <= seems most trustworthy
        # minimize_ipopt: 4.85 s/it, reliable

##       from ipopt import minimize_ipopt
##       optresult = minimize_ipopt(
##                           options={
##                              'tol': 1e-12,
#        from scipy.optimize import minimize
#        optresult = minimize(method='slsqp',
#                             options={
#                               'ftol': 1e-12,
#                               'maxiter': 1000,
#                            },
#                            fun=dualobjective,
#                            x0=x0,
#                            args=(sign,),
#                            jac=jacdualobjective,
#                            #hess=hessdualobjective,
#                            constraints=[{
#                                'type': 'ineq',
#                                'fun': lambda x: consE.dot(x) - d,
#                                'jac': lambda x: consE
#                            }],
#                   )
#        if raiseonerr:
#            from pprint import pformat
#            assert optresult.success, pformat(optresult)
#
#        fstar, xstar = optresult.fun, optresult.x

#        from .sqp import sqp
#        fstar, xstar = sqp(
#                f=lambda p: dualobjective(p, sign),
#                gradf=lambda p: jacdualobjective(p, sign),
#                hessf=lambda p: hessdualobjective(p, sign),
#                E=consE,
#                d=d,
#                x0=x0,
#                strict=True,
#                condfac=1e-9,
#        )

        from cvxopt import solvers, matrix
        def F(x=None, z=None):
            if x is None: return 0, matrix(x0)
            p = np.reshape(np.array(x), -1)
            f = dualobjective(p, sign)
            jf = jacdualobjective(p, sign)
            Df = matrix(jf).T
            if z is None: return f, Df
            hf = z[0] * hessdualobjective(p, sign)
            H = matrix(hf, hf.shape)
            return f, Df, H

        soln = solvers.cp(F,
                          G=-matrix(consE, consE.shape),
                          h=-matrix(d),
                          options={'show_progress': False})

        if raiseonerr:
            from pprint import pformat
            assert soln['status'] == 'optimal', pformat(soln)

        xstar = soln['x']
        fstar = soln['primal objective']

        gammastar = xstar[0]
        betastar = xstar[1] / wscale
        kappastar = (-rscale * fstar + gammastar + betastar) / num

        qfunc = lambda c, w, r, kappa=kappastar, gamma=gammastar, beta=betastar, s=sign: kappa * c / (gamma + (beta + s * r) * w)

        vbound = -sign * rscale * fstar

        retvals.append(
           (vbound,
            {
                'gammastar': gammastar,
                'betastar': betastar,
                'kappastar': kappastar,
                'qfunc': qfunc,
            })
        )

    return (retvals[0][0], retvals[1][0]), (retvals[0][1], retvals[1][1])


def pointwise_asym_ci(data, wmin, wmax, alpha):
    n = len(data)
    grid = np.round(np.geomspace(1, n)).astype(np.int32)
    lbs = []
    ubs = []
    for t in grid:
        cd = compress(data[:t])
        (lb, ub), (_, _) = asymptoticconfidenceinterval(lambda: cd, wmin=wmin, wmax=wmax, alpha=alpha, rmin=0, rmax=1)
        lbs.append(lb)
        ubs.append(ub)
    t = 1+np.arange(n)
    lb = np.interp(t, grid, np.array(lbs))
    ub = np.interp(t, grid, np.array(ubs))
    return lb, ub


def evaluate(name, method, alpha, ndata, reps, wsq, tv=None):
    env, _, _, _ = getenv(wsq, tv)
    z = produce_results(env, method, alpha, ndata, reps)
    with open(name, 'wb') as pkl:
        pickle.dump(z, pkl)
    return z


def evaluate_ci(name, method, alpha, ndata, reps, wsq, tv=None):
    env, _, _, _ = getenv(wsq, tv)
    z = produce_results_ci(env, method, alpha, ndata, reps)
    with open(name, 'wb') as pkl:
        pickle.dump(z, pkl)
    return z


def plotit(d, title, ax=None):
    sns.set_theme(style='ticks')
    columns = []
    values = []
    for k in d:
        columns.append(k)
        values.append(d[k])
    values = np.stack(values, axis=1)
    data = pd.DataFrame(values, columns=columns)
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(data=data, palette="tab10", linewidth=2.5, ci=None, ax=ax)
    ax.set(xlabel="samples", ylabel="width", ylim=(-0.04,1.04), title=title)
    ax.legend(loc='lower left')
    return ax

def coverage_experiment():
    res2d = evaluate('cov2d.pkl', bet_2d, alpha=0.05, ndata=100000, reps=1000, wsq=10)
    res1d = evaluate('cov1d.pkl', bet_1d, alpha=0.05, ndata=100000, reps=1000, wsq=10)
    return res2d, res1d

def width_experiment(n, wsq, tv):
    res1d = evaluate(f'width1d_{wsq}_{tv}.pkl', bet_1d, alpha=0.05, ndata=n, reps=20, wsq=wsq, tv=tv)
    res2d = evaluate(f'width2d_{wsq}_{tv}.pkl', bet_2d, alpha=0.05, ndata=n, reps=20, wsq=wsq, tv=tv)
   # res2dfreegrad = evaluate(f'width2d_freegrad_{wsq}_{tv}.pkl', bet_2d_freegrad, alpha=0.05, ndata=n, reps=20, wsq=wsq, tv=tv)
    reslog = evaluate(f'widthlog_{wsq}_{tv}.pkl', bet_log, alpha=0.05, ndata=n, reps=5, wsq=wsq, tv=tv)
    resiqp = evaluate(f'widthiqp_{wsq}_{tv}.pkl', bet_iqp, alpha=0.05, ndata=n, reps=5, wsq=wsq, tv=tv)


