"""Shared helpers for the two scripts in this folder.

The scripts are about oblivious repair rules, which see only part of an instance. The class
View holds what such a rule sees, namely the clamped prediction and the violated edges. It
also lists the hidden weights that the rule cannot see, and the outputs that are feasible
for every completion of those weights. A completion is one choice of all the hidden weights,
and an output is a dual that the rule may return.

The function face_and_pre computes the optimal face of one completion and the distance of
the prediction to that face.

The functions solve_mixed_lp and rational_certify compute the value of the mixed game by
linear programming and then check it in exact rational arithmetic. In the mixed game the
repair side picks its output at random, with probabilities of its own choosing, and an
adversary picks the completion.

The face computations come from verify_dichotomy.py in the parent folder.
"""
import sys, os, itertools
# Add the parent folder to the import path, so that verify_dichotomy.py can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
import verify_dichotomy as vd


# Views.
class View:
    """What an oblivious repair rule sees of an instance, and what stays hidden from it.

    The instance has na rows and nb columns. Vertices are numbered with the rows first, so
    column b is vertex na plus b. p is the clamped prediction, one integer per vertex. vio is
    a dictionary that maps each violated edge, given as the pair of its row vertex and its
    column vertex, to the weight of that edge. Every other pair has a hidden weight between 0
    and the sum of the two predicted values at its endpoints, because a larger weight would
    make the pair violated. The pairs where that bound is positive are listed in self.hidden,
    and the rest have weight 0.
    """
    def __init__(self, na, nb, p, vio):
        self.na, self.nb = na, nb; self.n = na + nb
        self.p = tuple(int(x) for x in p)
        self.vio = dict(vio)
        for (i, j), w in self.vio.items():
            assert i < na <= j < na + nb and w > self.p[i] + self.p[j], "not violated"
        self.hidden = [(a, na + b) for a in range(na) for b in range(nb)
                       if (a, na + b) not in self.vio and self.p[a] + self.p[na + b] > 0]
        self.maxw = max([w for w in self.vio.values()] + [self.p[i] + self.p[j] for i, j in self.hidden] + [0])

    def n_completions(self):
        """Return the number of completions."""
        r = 1
        for i, j in self.hidden: r *= self.p[i] + self.p[j] + 1
        return r

    def completions(self):
        """Return an iterator over every choice of the hidden weights, in lexicographic order.

        A choice is a tuple with one whole number for each pair of self.hidden, in the same
        order. The number runs from 0 up to and including the bound of that pair.
        """
        return itertools.product(*(range(self.p[i] + self.p[j] + 1) for i, j in self.hidden))

    def W_of(self, h):
        """Return the weight matrix of the completion h.

        h holds one weight for each pair of self.hidden, in the same order. The violated edges
        get their own weights, and every pair that is neither violated nor hidden gets 0.
        """
        W = np.zeros((self.na, self.nb), dtype=int)
        for (i, j), w in self.vio.items(): W[i, j - self.na] = w
        for (i, j), w in zip(self.hidden, h): W[i, j - self.na] = w
        return W

    def feasible_z(self):
        """List every integral output that is feasible for every completion.

        An output z has one whole number per vertex, each from 0 to maxw, the largest weight
        that any pair can have in this view. It is feasible for a completion when the two
        values of every pair add up to at least the weight of the pair. It is enough to check
        the completion with every hidden weight at its maximum.
        """
        top = {}
        for (i, j), w in self.vio.items(): top[(i, j)] = w
        for i, j in self.hidden: top[(i, j)] = self.p[i] + self.p[j]
        out = []
        for z in itertools.product(range(self.maxw + 1), repeat=self.n):
            if all(z[i] + z[j] >= w for (i, j), w in top.items()):
                out.append(z)
        return out


def face_and_pre(view, h):
    """Return the integral optimal face of completion h and the distance of the prediction to it.

    The face is given by its integral points, one per row of the returned array. The distance
    is the smallest number of coordinates in which the prediction differs from one of those
    points (the paper calls it the l0 distance).
    """
    W = view.W_of(h); OPT = vd.matching_value(W)
    pts = vd.face_points_bip(W, OPT)
    pre = vd.face_l0(np.array(view.p, float), pts)
    return pts, pre


# The mixed game: solving it by LP, and certifying the solution exactly.
def solve_mixed_lp(R):
    """Solve the mixed game with payoff matrix R by linear programming.

    R is a float array with one row for each of the nz outputs and one column for each of the
    nc completions. An entry is the payoff of that output against that completion: either the
    distance after the repair divided by the distance before it, or the distance after minus
    the distance before. The repair side looks for a probability distribution pi over the
    outputs whose largest expected payoff over the completions is as small as possible. The
    function returns that smallest value, the distribution pi, and the dual solution q of the
    linear program. The program is solved in floating point with linprog.
    """
    nz, nc = R.shape
    # The variables are the nz probabilities of pi, followed by one more variable t. The
    # program minimises t such that the expected payoff of pi is at most t against every
    # completion, the probabilities add up to 1, and no probability is negative.
    c = np.zeros(nz + 1); c[-1] = 1.0
    A_ub = np.hstack([R.T, -np.ones((nc, 1))]); b_ub = np.zeros(nc)
    A_eq = np.zeros((1, nz + 1)); A_eq[0, :nz] = 1.0
    bounds = [(0, None)] * nz + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    assert res.status == 0, res.message
    pi = res.x[:nz]; t = res.x[-1]
    # q is the dual solution on the nc inequality rows. It is at least zero and sums to 1.
    q = -res.ineqlin.marginals
    return t, pi, q


def rational_certify(Rfrac, pi_float, q_float, tol=1e-7):
    """Turn a floating-point solution of the mixed game into an exact rational certificate.

    Rfrac is the payoff matrix in exact rationals, a list of lists of Fraction with one row
    per output and one column per completion. pi_float and q_float are the floating-point
    optimal strategies of the repair side and of the adversary, that is, the primal and the
    dual solution of the linear program. Only their supports are used, the entries larger
    than tol. Exact rational strategies pi and q and an exact value v are rebuilt by solving
    two systems of equations on the supports, one for each strategy, each with the value as
    an extra unknown. The two values must be equal. Then it is verified in exact arithmetic
    that the largest expected payoff of pi over all completions equals v, and that v equals
    the smallest expected payoff of q over all outputs. The function returns v and the exact
    strategies pi and q, each as a dictionary from an index in its support to a Fraction. It
    returns None if the reconstruction or the verification fails.
    """
    nz = len(Rfrac); nc = len(Rfrac[0])
    Sz = [z for z in range(nz) if pi_float[z] > tol]
    Sc = [c for c in range(nc) if q_float[c] > tol]
    def solve_square(rows, rhs):
        # Gauss-Jordan elimination in exact fractions. Free unknowns are set to 0, and None
        # is returned if the system has no solution. Any solution will do, because every
        # bound is checked exactly below.
        m = len(rows); k = len(rows[0])
        M = [list(r) + [b] for r, b in zip(rows, rhs)]
        piv = []
        r = 0
        for col in range(k):
            pr = None
            for i in range(r, m):
                if M[i][col] != 0: pr = i; break
            if pr is None: continue
            M[r], M[pr] = M[pr], M[r]
            pv = M[r][col]
            M[r] = [x / pv for x in M[r]]
            for i in range(m):
                if i != r and M[i][col] != 0:
                    f = M[i][col]
                    M[i] = [a - f * b for a, b in zip(M[i], M[r])]
            piv.append(col); r += 1
            if r == m: break
        # The rows below the pivot rows are all zero on the left. A nonzero right side in one
        # of them means that the system has no solution.
        if any(M[i][k] != 0 for i in range(r, m)): return None
        sol = [Fraction(0)] * k
        for i, col in enumerate(piv): sol[col] = M[i][k]
        return sol
    # The first system gives the strategy pi of the repair side. For every completion in the
    # support Sc, the expected payoff of pi equals v, and the probabilities of pi add up to 1.
    # The unknowns are the probabilities of pi, one for each output in the support Sz, and
    # the value v.
    rows = []; rhs = []
    for c in Sc:
        rows.append([Rfrac[z][c] for z in Sz] + [Fraction(-1)]); rhs.append(Fraction(0))
    rows.append([Fraction(1)] * len(Sz) + [Fraction(0)]); rhs.append(Fraction(1))
    sol = solve_square(rows, rhs)
    if sol is None: return None
    pi = {z: sol[i] for i, z in enumerate(Sz)}; v = sol[-1]
    if any(x < 0 for x in pi.values()): return None
    # The second system is the dual one and gives the strategy q of the adversary. For every
    # output in the support Sz, the expected payoff of q equals a second unknown value v2, and
    # the probabilities of q add up to 1. The unknowns are the probabilities of q, one for each
    # completion in the support Sc, and v2. None is returned unless v2 equals v.
    rows = []; rhs = []
    for z in Sz:
        rows.append([Rfrac[z][c] for c in Sc] + [Fraction(-1)]); rhs.append(Fraction(0))
    rows.append([Fraction(1)] * len(Sc) + [Fraction(0)]); rhs.append(Fraction(1))
    sol = solve_square(rows, rhs)
    if sol is None: return None
    q = {c: sol[i] for i, c in enumerate(Sc)}; v2 = sol[-1]
    if any(x < 0 for x in q.values()): return None
    if v2 != v: return None
    # Verify the primal strategy pi: in every column, that is, against every completion, its
    # expected payoff is at most v.
    for c in range(nc):
        s = sum(pi[z] * Rfrac[z][c] for z in Sz)
        if s > v: return None
    # Verify the dual strategy q: in every row, that is, for every output, its expected payoff
    # is at least v.
    for z in range(nz):
        s = sum(q[c] * Rfrac[z][c] for c in Sc)
        if s < v: return None
    return v, pi, q
