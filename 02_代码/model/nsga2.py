# -*- coding: utf-8 -*-
"""紧凑 NSGA-II（无 pymoo 依赖）。目标按最小化处理，约束用 violation。"""

import numpy as np


def dominates(i, j, F, CV):
    if CV[i] == 0 and CV[j] > 0:
        return True
    if CV[i] > 0 and CV[j] == 0:
        return False
    if CV[i] > 0 and CV[j] > 0:
        return CV[i] < CV[j]
    return np.all(F[i] <= F[j]) and np.any(F[i] < F[j])


def fast_nds(F, CV):
    N = len(F)
    S = [[] for _ in range(N)]
    n = np.zeros(N, int)
    rank = np.zeros(N, int)
    fronts = [[]]
    for p in range(N):
        for q in range(p + 1, N):
            if dominates(p, q, F, CV):
                S[p].append(q); n[q] += 1
            elif dominates(q, p, F, CV):
                S[q].append(p); n[p] += 1
        if n[p] == 0:
            rank[p] = 0; fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1; nxt.append(q)
        i += 1; fronts.append(nxt)
    return fronts[:-1], rank


def crowding(F, idx):
    d = np.zeros(len(idx))
    if len(idx) <= 2:
        d[:] = np.inf
        return d
    for m in range(F.shape[1]):
        order = np.argsort(F[idx, m])
        d[order[0]] = np.inf; d[order[-1]] = np.inf
        span = F[idx[order[-1]], m] - F[idx[order[0]], m]
        if span <= 0:
            continue
        for k in range(1, len(order) - 1):
            d[order[k]] += (F[idx[order[k + 1]], m] - F[idx[order[k - 1]], m]) / span
    return d


def tournament(rank, crowd):
    i, j = np.random.randint(len(rank), size=2)
    if rank[i] < rank[j]:
        return i
    if rank[j] < rank[i]:
        return j
    return i if crowd[i] >= crowd[j] else j


def sbx(p1, p2, eta=15.0):
    u = np.random.rand(len(p1))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def mutate(x, eta=20.0, pm=None):
    n = len(x)
    pm = 1.0 / n if pm is None else pm
    mask = np.random.rand(n) < pm
    u = np.random.rand(n)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1,
                     1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    y = x.copy()
    y[mask] += delta[mask]
    return np.clip(y, 0, 1)


def run(eval_fn, n_var=6, pop=100, gen=100, seed=20261120):
    """eval_fn(x)->(f1,f2,violation)，f 为最小化目标。"""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    X = rng.random((pop, n_var))
    cache = {}

    def ev(x):
        key = tuple(np.round(x, 8))
        if key not in cache:
            cache[key] = eval_fn(x)
        return cache[key]

    F = np.zeros((pop, 2)); CV = np.zeros(pop)
    for i, x in enumerate(X):
        f1, f2, cv = ev(x); F[i] = (f1, f2); CV[i] = cv

    for g in range(gen):
        fronts, rank = fast_nds(F, CV)
        crowd = np.zeros(pop)
        for fr in fronts:
            crowd[fr] = crowding(F, fr)

        kids = np.zeros_like(X)
        for k in range(0, pop, 2):
            i = tournament(rank, crowd)
            j = tournament(rank, crowd)
            c1, c2 = sbx(X[i], X[j])
            kids[k] = mutate(c1)
            if k + 1 < pop:
                kids[k + 1] = mutate(c2)

        kF = np.zeros((pop, 2)); kCV = np.zeros(pop)
        for i, x in enumerate(kids):
            f1, f2, cv = ev(x); kF[i] = (f1, f2); kCV[i] = cv

        Xall = np.vstack([X, kids]); Fall = np.vstack([F, kF]); CVall = np.hstack([CV, kCV])
        fronts, rank = fast_nds(Fall, CVall)
        crowd = np.zeros(len(Xall))
        for fr in fronts:
            crowd[fr] = crowding(Fall, fr)

        sel = []
        for fr in fronts:
            if len(sel) + len(fr) <= pop:
                sel.extend(fr)
            else:
                order = sorted(fr, key=lambda i: -crowd[i])
                sel.extend(order[:pop - len(sel)])
                break
        sel = np.array(sel[:pop])
        X, F, CV = Xall[sel], Fall[sel], CVall[sel]

    return X, F, CV, cache
