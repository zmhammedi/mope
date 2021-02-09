import numpy as np
import cvxpy as cp
from scipy.special import softmax
from pprint import pformat
import random


class ControlledRangeVariance:
    def __init__(self, seed, wsupport, expwsq, rvala=1, rvalb=1, tv=None):
        wmax = max(wsupport)
        assert wmax > 1
        assert wmax >= expwsq
        assert min(wsupport) < 1

        self.wsupport = np.sort(np.array(wsupport))

        wnice = self.wsupport / wmax

        A = np.array([wnice, wnice * wnice]).reshape(2, -1)
        b = np.array([1 / wmax, expwsq / (wmax * wmax)])
        mu = cp.Variable(len(b))
        prob = cp.Problem(cp.Maximize(mu.T @ b - cp.log_sum_exp(mu.T @ A)), [])
        tol = 5e-12
        prob.solve(solver='ECOS', verbose=False, max_iters=1000, feastol=tol, reltol=tol, abstol=tol)
        assert prob.status == 'optimal'
        logits = np.asarray((mu.T @ A).value).ravel()

        self.pw = softmax(logits)

        assert np.allclose(self.pw.dot(self.wsupport * self.wsupport), expwsq), pformat(
            {
                'self.pw.dot(self.wsupport * self.wsupport)': self.pw.dot(self.wsupport * self.wsupport),
                'expwsq': expwsq
            }
        )

        assert np.allclose(self.pw.dot(self.wsupport), 1), pformat(
            {
                'self.pw.dot(self.wsupport)': self.pw.dot(self.wsupport),
            }
        )

        assert np.allclose(np.sum(self.pw), 1), pformat(
            {
                'np.sum(self.pw)': np.sum(self.pw)
            }
        )

        self.rvala = rvala
        self.rvalb = rvalb
        self.tv = tv
        self.state = np.random.RandomState(seed)
        self.perm_state = None
        self.seed = seed

    def getpw(self):
        return self.pw, self.wsupport

    def range(self):
        return min(self.wsupport), max(self.wsupport)

    def expectedwsq(self):
        return self.pw.dot(self.wsupport * self.wsupport)

    def sample(self, ndata):
        if self.tv is None:
            truevalue = self.state.beta(a=self.rvala, b=self.rvalb)
        else:
            truevalue = self.tv

        pw = self.pw
        wsupport = self.wsupport

        rempw = 1
        remtv = truevalue
        pr = []

        for wi, pwi in zip(wsupport[:-1], pw[:-1]):
            if wi > 0 and pwi > 0:
                rempw -= wi * pwi
                rmin = 0 if rempw > remtv else (remtv - rempw) / (wi * pwi)
                rmax = 1 if (wi * pwi) <= remtv else remtv / (wi * pwi)
                pri = self.state.uniform(rmin, rmax)
                pr.append(pri)
                remtv -= wi * pwi * pri
            else:
                pr.append(0)

        prlast = 0 if pw[-1] == 0 else remtv / (wsupport[-1] * pw[-1])
        pr.append(prlast)
        pr = np.clip(pr, a_min=0, a_max=1)

        assert np.allclose(truevalue,
                           pw.dot(wsupport * pr),
                           atol=1e-6), pformat(
            {'truevalue': truevalue,
             'pw': pw,
             'pr': pr,
             'np.all(pr <= 1)': np.all(pr <= 1),
             'np.all(pr >= 0)': np.all(pr >= 0),
             'actualvalue': pw.dot(wsupport * pr),
             })

        counts = self.state.multinomial(n=ndata, pvals=pw)
        sufstat = []
        for prw, cw, w in zip(pr, counts, wsupport):
            if cw > 0:
                ones = self.state.binomial(cw, prw)
                zeros = cw - ones
                if zeros > 0:
                    sufstat.append((zeros, w, 0))
                if ones > 0:
                    sufstat.append((ones, w, 1))

        self.perm_state = self.state.get_state()
        perm = self.state.permutation(ndata)
        wr = np.empty((ndata, 2), dtype=np.float64)
        offset = 0
        for c, w, r in sufstat:
            slc = perm[offset:offset+c]
            wr[slc, 0] = w
            wr[slc, 1] = r
            offset += c
        return truevalue, wr
