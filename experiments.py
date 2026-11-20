#!/usr/bin/env python3
"""
This program produces Table 5 of the paper "Not All Feasibility Repairs Are Equal".

The table reports, for each repair, the number of coordinates in which the repaired dual
differs from the optimal dual. The paper calls this number the l0 distance after the repair,
and it is the quantity that the runtime bound of Chen et al. depends on. The repairs are the
sorted, forward and random scans and the walk, and they are run on assignment instances of
realistic size.

This file also holds the shared building blocks that experiments_itcs.py, measure_chen.py
and verify_dichotomy.py import: the optimal dual, the scan repair, the walk, the scan orders,
the count of differing coordinates and the three instance families.

The program requires numpy and scipy. Run it with the command python3 experiments.py.
"""
import numpy as np, time
from scipy.optimize import linear_sum_assignment, linprog

def optimal_dual(W):
    """Compute an optimal dual of maximum-weight bipartite matching for the weight matrix W.

    The dual has one value for every row and one value for every column. The function solves
    the linear program that minimises the sum of all dual values under two conditions: the
    two dual values of every edge add up to at least the edge weight, and no dual value is
    negative. It returns the row values and the column values as two arrays.
    """
    na, nb = W.shape
    n = na + nb
    A, b = [], []
    for a in range(na):
        for bb in range(nb):
            row = np.zeros(n); row[a] = -1.0; row[na+bb] = -1.0
            A.append(row); b.append(-float(W[a, bb]))
    res = linprog(np.ones(n), A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(0, None)]*n, method="highs")
    return res.x[:na], res.x[na:]

def repair(W, u, v, order):
    """Repair a prediction with the scan repair of Definition 1.

    The prediction has row values u and column values v. Its negative values are first
    replaced by zero. The pairs of a row and a column are then visited in the given order,
    and both ends of each pair that is still violated are raised by its residual. A pair is
    violated when its two dual values add up to less than its weight, and its residual is
    the weight minus that sum. The function returns the repaired row values and column
    values.
    """
    u, v = np.maximum(u, 0), np.maximum(v, 0)
    for a, bb in order:
        r = W[a, bb] - u[a] - v[bb]
        if r > 0: u[a] += r; v[bb] += r
    return u, v

def walk_repair(W, u, v):
    """Repair a prediction with the one-sided walk.

    The walk is our transplant of Algorithm 1 of Dinitz et al. to covering constraints. It
    is applied to the correction problem over the violated edges: give every vertex a
    correction so that the two corrections of every violated edge add up to at least the
    residual of that edge. The walk visits a vertex, sets its correction to the largest
    residual among its remaining violated edges, deletes the vertex, and steps to the
    neighbour that attains that largest residual.

    The choices that Dinitz et al. leave arbitrary are made in a fixed way here. The outer
    loop starts at the vertex of lowest index that still has violated edges, and row
    vertices come before column vertices. Ties for the largest residual are broken by
    lowest index. The residuals are fixed at their initial values, which are the right-hand
    side of the correction linear program.

    The prediction has row values u and column values v. Its negative values are replaced
    by zero before the residuals are computed, and the corrections are added to this
    clamped prediction. The function returns the repaired row values and column values.
    """
    u, v = np.maximum(u, 0), np.maximum(v, 0)
    na, nb = W.shape
    r = {}
    for a in range(na):
        for b in range(nb):
            res = W[a, b] - u[a] - v[b]
            if res > 0: r[(a, na + b)] = res
    adj = {}
    for (i, j), res in r.items():
        adj.setdefault(i, {})[j] = res; adj.setdefault(j, {})[i] = res
    delta = np.zeros(na + nb)
    while any(adj.get(i) for i in range(na + nb)):
        i = min(k for k in range(na + nb) if adj.get(k))
        while adj.get(i):
            j = max(adj[i], key=lambda t: (adj[i][t], -t))
            delta[i] = adj[i][j]
            for t in list(adj[i]): del adj[t][i]
            adj[i] = {}
            i = j
    return u + delta[:na], v + delta[na:]

def orders(W, rng=None):
    """Build the three scan orders over all pairs of a row and a column.

    The sorted order goes by weight, largest first, with ties in row-major order. The
    forward order is row-major: row by row and, within a row, column by column. The random
    order is a random permutation drawn from rng. Pass the random generator of the caller
    so that every instance gets a fresh random order. Without one a fixed seed is used,
    which is fine for callers that only want the sorted order. The function returns a
    dictionary with the keys sorted, forward and random.
    """
    na, nb = W.shape
    P = [(a, b) for a in range(na) for b in range(nb)]
    idx = np.argsort([-W[a, b] for a, b in P], kind="stable")
    if rng is None:
        rng = np.random.default_rng(0)
    return {"sorted": [P[i] for i in idx],
            "forward": P,
            "random": [P[i] for i in rng.permutation(len(P))]}

def l0(u, v, u0, v0, tol=1e-6):
    """Count the coordinates in which two duals differ.

    The first dual has row values u and column values v. The second has row values u0 and
    column values v0. Two values count as different when they are more than tol apart. The
    paper calls this count the l0 distance.
    """
    return int((np.abs(u-u0) > tol).sum() + (np.abs(v-v0) > tol).sum())

def instance(kind, n, rng):
    """Draw an n by n matrix of integer weights from one of the three families of the tables.

    The family is chosen by kind, which is uniform, clustered or euclidean. The random
    numbers come from the generator rng.
    """
    if kind == "uniform":
        return rng.integers(0, 100, size=(n, n)).astype(float)
    if kind == "clustered":                       # A few blocks of high values.
        W = rng.integers(0, 20, size=(n, n)).astype(float)
        for _ in range(max(1, n//8)):
            i, j = rng.integers(0, n, 2); s = max(2, n//6)
            W[i:i+s, j:j+s] += rng.integers(50, 100)
        return W
    # The weight is a constant minus the scaled distance, so nearby pairs are valuable.
    if kind == "euclidean":
        p = rng.random((n, 2)); q = rng.random((n, 2))
        D = np.linalg.norm(p[:, None, :]-q[None, :, :], axis=2)
        return np.round(100*(1 - D/D.max()))
    raise ValueError(kind)

def run(kinds=("uniform","clustered","euclidean"), sizes=(20,40,60), corrupt=0.10, reps=5, seed=1):
    rng = np.random.default_rng(seed)
    draws = []                 # The values of each instance, for the paired summary.
    cells = {}                 # The lists of values of each operator, keyed by family and size.
    hdr = f"{'instance':>11} {'n':>4} {'pre l0':>13} | "
    hdr += " ".join(f"{n_:>15}" for n_ in ("sorted","forward","random","walk"))
    print(hdr)
    for kind in kinds:
        for n in sizes:
            cell = {"pre":[], "sorted":[], "forward":[], "random":[], "walk":[]}
            for _ in range(reps):
                W = instance(kind, n, rng)
                us, vs = optimal_dual(W)
                k = max(1, int(corrupt*(2*n)))          # The number of coordinates to corrupt.
                u0, v0 = us.copy(), vs.copy()
                which = rng.choice(2*n, size=k, replace=False)
                for i in which:
                    d = rng.choice([-1, 1])*rng.integers(1, 20)
                    if i < n: u0[i] = max(0, u0[i]+d)
                    else:     v0[i-n] = max(0, v0[i-n]+d)
                cell["pre"].append(l0(u0, v0, us, vs))
                for name, o in orders(W, rng).items():
                    ru, rv = repair(W, u0, v0, o)
                    cell[name].append(l0(ru, rv, us, vs))
                wu, wv = walk_repair(W, u0, v0)
                assert (wu[:, None] + wv[None, :] >= W - 1e-9).all(), "walk output infeasible"
                cell["walk"].append(l0(wu, wv, us, vs))
            m = {k2: float(np.mean(v2)) for k2, v2 in cell.items()}
            # The population standard deviation, whose divisor is the number of repetitions.
            s = {k2: float(np.std(v2)) for k2, v2 in cell.items()}
            line = f"{kind:>11} {n:>4} {m['pre']:>7.1f}+-{s['pre']:<4.1f} | "
            line += " ".join(f"{m[k2]:>8.1f}+-{s[k2]:<5.1f}"
                             for k2 in ("sorted","forward","random","walk"))
            exc = {k2: 100.0*(m[k2]-m['pre'])/m['pre'] for k2 in ("sorted","forward","random","walk")}
            print(line + "   excess%  " + " ".join(f"{k2}:{exc[k2]:+.1f}"
                                                   for k2 in ("sorted","forward","random","walk")))
            cells[(kind, n)] = cell
            draws.extend(zip(cell["sorted"], cell["forward"], cell["random"],
                             cell["walk"], cell["pre"]))
    # The paired summary. Every operator sees the same instance and the same corrupted
    # prediction, so the operators are compared draw by draw.
    wf = sum(s_ < f for s_, f, _, _, _ in draws); tf = sum(s_ == f for s_, f, _, _, _ in draws)
    wr = sum(s_ < r_ for s_, _, r_, _, _ in draws); tr = sum(s_ == r_ for s_, _, r_, _, _ in draws)
    ww = sum(s_ < w_ for s_, _, _, w_, _ in draws); tw = sum(s_ == w_ for s_, _, _, w_, _ in draws)
    print(f"paired ({len(draws)} draws): sorted < forward in {wf} (ties {tf}); "
          f"sorted < random in {wr} (ties {tr}); sorted < walk in {ww} (ties {tw})")
    # Draw by draw, does the walk end with fewer coordinates that differ from the optimal dual
    # than the prediction itself had before the repair? The means of a cell can hide draws
    # where it does not.
    wb = sum(w_ < p for _, _, _, w_, p in draws)
    we = sum(w_ == p for _, _, _, w_, p in draws)
    wa = sum(w_ > p for _, _, _, w_, p in draws)
    print(f"walk vs pre ({len(draws)} draws): below {wb}; equal {we}; above {wa}")
    return cells, draws

if __name__ == "__main__":
    # The pinned configuration behind Table 5. A single random generator feeds every draw in
    # turn, so the sizes and the families must be run together, in this order, to reproduce
    # the table.
    t = time.time()
    run(kinds=("uniform", "clustered", "euclidean"), sizes=(50, 100, 150, 200),
        corrupt=0.10, reps=10, seed=7)
    print(f"\n({time.time()-t:.1f}s)")
