#!/usr/bin/env python3
"""Check the dichotomy results of the paper on repairs of a predicted covering dual.

A repair turns a prediction, which is a guessed dual that may be infeasible, into a feasible dual.
The dichotomy separates the repairs that have an additive bound on the optimal face from the repairs
that provably have none. An additive bound says that the number of coordinates in which the repaired
dual differs from the optimal face is at most the same number for the prediction plus a constant.

The setting is a covering system. Its vertices are split into rows and columns, and it has one
constraint for every pair of a row and a column, so it is bipartite. Every constraint has an integer
weight of at least zero. A dual gives one value, a price, to every vertex. It is feasible when every
value is at least zero and the two dual values of every pair add up to at least the weight of the
pair. OPT is the smallest sum of the values over all feasible duals. The optimal face is the set of
feasible duals whose values sum to OPT, that is, the set of all optimal duals.

The face distance of a point is the smallest number of coordinates in which it differs from a point
of the optimal face. The paper calls it the l0 distance to the face, and the code also calls it the
Hamming distance. For a bipartite system and a point with integer coordinates, the face distance is
attained at a face point with integer coordinates, because the constraint matrix is totally
unimodular. The same holds for the distance to the face measured by the sum of the absolute
differences, which the paper calls the l1 distance. Both distances are therefore computed exactly by
enumerating the integer points of the face, all of whose coordinates lie between zero and the
largest weight.

Section 2 of this program also uses systems that are not bipartite. For them the same enumeration
runs over the points whose coordinates are multiples of one half. That is exact there, for the
following reason. An agreement set is a set of coordinates
on which the face point must equal the given point. Once an agreement set is fixed, the system that
requires every value to be at least zero and the two values of every constraint to add up to at
least its weight has vertices whose coordinates are multiples of one half. The enumeration is also
cross-checked against linear programs that go through the agreement sets over the fractional face,
which is the optimal face with real coordinates allowed.

The distance to the face measured by the largest absolute difference (the l-infinity distance) and
the one measured by the largest increase plus the largest decrease (the two-sided l-infinity
distance) are computed by linear programs over the fractional face.

The clamped prediction is the prediction with negative values replaced by zero. The violated graph
has one edge for every constraint that the clamped prediction violates. Its vertex cover number is
the size of a minimum vertex cover, the fewest vertices that touch every violated constraint. On a
bipartite system it equals the size of a maximum matching of that graph. The two-sided greedy repair
scans the constraints in a given order and, at each violated one, raises both endpoints by its
residual, which is the weight minus the sum of the two dual values. A constraint fires when the scan
finds it violated and raises both of its endpoints.

In the code and in the output, yhat is the prediction, p the clamped prediction, tau the vertex
cover number of the violated graph, pre the face distance of the prediction and post the face
distance of the repaired dual. There are two exceptions. In section 8, pre and the other columns
count the coordinates that differ from one reference optimal dual, as the table does. In section 20,
p is the parameter of the even family and of the odd family.

A point is admissible for a prediction when it is feasible and, for every feasible point, its
distance to that point is at most the distance of the clamped prediction to that point, with
distance measured by the sum of the absolute differences. The class is the set of repairs whose
outputs are always admissible.

Some sections study repairs that see only a view of the instance. An oblivious view shows the
clamped prediction and the violated constraints. A one-hop view also shows every other pair that
touches a vertex of a violated constraint. The weights of the remaining pairs are hidden, and a
completion is one choice of them. The game value of a view, as computed here, is the smallest ratio
of post to pre that a single output with integer coordinates, feasible for every completion, can
guarantee over all completions.

The program runs in numbered sections. Each section prints the numbers that the paper quotes and
asserts every claim that it checks.

Section 1 checks the gadget G1 of Theorem 16, which says that no repair in the class has an additive
bound. The gadget has 1 row and 3 columns with weights 1, 2 and 2, and its prediction is 0 on the
row and 0, 2 and 2 on the columns. The section also checks the 2-fold tensor of G1 (two copies
joined by zero weights), a second gadget with weights 6, 6 and 5, and padded copies of G1
(Corollary 17 of the paper). Section 1b searches exhaustively for such witnesses on systems with 1
row and 2 or 3 columns, with weights and predictions from 0 to 4.

Section 2 checks Theorem 19, which bounds post by pre plus tau, and therefore by 2 times pre, for
the cover-directed scan. At each vertex of a minimum vertex cover of the violated graph in turn,
this scan fires the constraint at that vertex that is currently the most violated. The section also
checks the maximal-matching variant. Its random sweeps go over every minimum cover, every processing
order and every tie-break, on bipartite systems and on systems that are not bipartite. It also runs
the family of Theorem 8 and the balanced family. Section 2b checks the witness
on which the additive term, the size of the cover (which equals tau), is attained, with post equal
to seven quarters of pre. Section 2c checks that the mirror of the scan in the packing orientation,
which is the setting of Chen et al., obeys the same bound.

Section 3 checks the one-sided walk over every way to make the choices that it leaves open: the
start vertex of each outer iteration and every tie for the largest residual. The walk is our
transplant of Algorithm 1 of Dinitz et al. to covering systems, exactly as walk_repair in
experiments.py defines it. The section also checks the family behind its lower bound and a witness
on which no tie occurs, yet the choice of start vertices alone changes the face distance of the
output.

Section 4 checks the lemma that the optimal face of the balanced family is a triangle. It also
checks the reversal on that family, where n is the number of rows. Measured by the largest absolute
difference, the output of the descending scan is at distance 1 from the face and the output of the
ascending scan at distance n minus 1. Their face distances are n and 1. The largest increase plus
the largest decrease gives the same two values as the largest absolute difference, and the
prediction is at face distance 1.

Section 5 checks the transplant of the Min-Trick to covering systems, which the output calls T.

Section 6 checks the tight-row lower bound for solve_warm in solver.py: the number of dual
adjustments is at least U, the number of rows that have no tight edge at the start. An edge is tight
when its constraint holds with equality. The section also checks what the parity of epsilon does to
the uniform shift. This shift raises every coordinate by half the largest residual, rounded up, and
epsilon is that largest residual. On instances with integral duals, the shift leaves no tight edge
when epsilon is odd, and when epsilon is even it leaves tight exactly the constraints whose residual
was epsilon. Last, the section checks the discontinuity of the epsilon lift, where every value of an
integral optimal dual is raised by one thousandth: the solver makes no dual adjustment from the
optimal dual itself, and one per row after the lift.

Section 7 checks Proposition 18. Measure distance by the sum of the
absolute differences and take the face point nearest to the prediction. Its distance to any face
point is at most 2 times the distance of the prediction to that face point. Among the repairs with
which Corollary 3.3 of Chen et al. composes, it is the member with additive constant 0. Section 7b
finds, on 600 random instances, a face point that is non-expansive toward every face point: it is no
farther from any face point than the prediction is. This is an observation only.

Section 8 reproduces the run of experiments.py behind its Table 2, with the same seeded random
stream. It also runs the cover-directed scan on that stream, with minimum covers computed from a
maximum matching by the construction of Koenig.

Section 9 reproduces the run of experiments_itcs.py behind its Table 3, with the same seeded random
stream, for the row-reduction start, the sorted scan and the walk. The row-reduction start ignores
the prediction: every row gets its largest weight and every column gets zero. The code and the
output name it cold, but it is not the cold start of sections 12, 17 and 20, which runs the sorted
scan from the all-zero prediction. Section 9 also reports the dual adjustments of the solver when it
starts from the cover-directed scan on the same draws.

Section 10 checks that the constant of the class is three halves. This constant is the best factor
between post and pre that a repair in the class can guarantee. On the path family with parameter k,
every admissible output is at face distance 3 times k plus 1, from a prediction at face distance 2
times k plus 1. This is certified for k up to 4 from 2 to the power k feasible points alone, and
cross-checked for k up to 3 against the exact admissibility oracle. Each of these points is the
prediction with one coordinate raised by 1 for every row: either the row itself or the column with
the same index in the first group of columns. The upper bound says that some
admissible output is at face distance at most r plus half of r, rounded down, where r is the face
distance of the prediction. It is checked exhaustively on systems with 1 row and 2 or 3 columns,
with weights and predictions from 0 to 4, and on random instances near the face, and the two-round
construction is checked for admissibility.

Section 11 checks the chain family, on which the paper shows that the factor 2 of Theorem 19 is
tight in the limit for the cover-directed scan. For k from 2 to 8, the minimum cover on the column
side, taken in index order, meets no tie and ends with post divided by pre equal to (2 times k plus
1) divided by (k plus 1), which approaches 2 as k grows. For k up to 4 the best admissible output
stays at pre.

Sections 12 to 17 cover the cold-start envelope, the trap family and the ladder family, the
oblivious view V0, the case of an even face distance, and the bound of five halves of the smaller
side, rounded down. See the section headers. The cold-start envelope is the largest face distance
that the sorted scan can reach from the all-zero prediction, in terms of the smaller side of the
instance.

Section 18 checks that the view V2, with one violated pair and weights of at most 3, has an
oblivious game value of exactly four thirds. Every output with integer coordinates that is feasible
for all completions is forced to a ratio of at least four thirds by three completions, and the
two-sided firing has a ratio of at most four thirds on all 9216 completions.

Section 19 checks star views. The two-sided firing on a single violated edge gives post at most pre
plus 1, post at most pre when pre is at most 2, and a ratio of at most four thirds. The raise of the
centre on stars with 2 leaves gives post at most pre when pre is at most 3, and a ratio of at most
five quarters. Both are checked on random views with every completion enumerated. On the view V0 the
raise of the centre attains five quarters.

Section 20 checks the cold envelope of five halves of the smaller side, rounded down, minus 1, which
the paper proves exact for a smaller side of at least 3. It uses the even family and the odd family,
which the output labels E and O, with smaller sides from 4 to 10. Each has a unique optimal dual and
the same cold output under every weakly descending order, and its face distances are exact by linear
programming. The section adds a random sweep of the bound as an upper bound over every tie-break, on
instances whose smaller side is 2 or 3, and a validation of the MILP that keeps the dual continuous.

Section 21 checks that the mixed game value of V0, where randomisation is allowed, is exactly
fifteen sixteenths, by an exact certificate for both players.

Section 22 checks the matching-first rule, which fires the violated pairs of a maximum-weight
matching and then keeps firing the largest residual. Over every maximum-weight matching and every
tie-break, post is at most pre plus half of pre, rounded down. The section also checks a witness
with 2 rows and 5 columns at exactly three halves, and the failure of the rule when its matching is
taken only near the violated graph, over the pairs that touch one of its vertices.

Section 23 checks every order on the star of Theorem 8, which has d columns. Write d0 for the face
distance of an output and dinf for its distance to the face measured by the largest absolute
difference. The outputs correspond to the compositions of d. The product of d0 and dinf is at least
d minus 1. Over the outputs, the smallest value of the larger of d0 and dinf is the square root of
(d minus 1), rounded up. Over all orders, the mean of d0 is the harmonic number of d. The section
then checks the balanced family, which mirrors the star: on both, d0 is the number of constraints
that fire, and dinf follows from the gaps between the fired columns by the same rule. Last, it
checks on the star that every admissible point with integer coordinates that it examines has a
product of d0 and dinf of at least d minus 2, and that explicit admissible points attain d minus 2.

Section 24 checks one-hop views. The local-matching rule fails at a ratio of 2 on a witness with 3
rows and 2 columns whose one-hop view has a game value of one half.

Theorem numbers follow the NewInML paper. Two section titles keep an older table numbering:
"Table 2" in section 8 is Table 5 of the paper, and "Table 3" in section 9 is its Table 6.

The random seeds are fixed, there are no command-line arguments, and the exit code is 0 only if
every assertion holds. The program requires numpy and scipy. The files solver.py, experiments.py and
experiments_itcs.py must sit next to this file. They are imported and never modified.

Run it with the command python3 verify_dichotomy.py.
"""
import itertools
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The warm-started solver of the paper, used in sections 6 and 9 of this program.
from solver import solve_warm  # noqa: E402

TOL = 1e-9
T_START = time.time()


def section(num, title):
    print("\n" + "=" * 78)
    print(f"[{num}] {title}")
    print("=" * 78)


def elapsed():
    return f"({time.time() - T_START:.1f}s elapsed)"


# Helpers for covering systems.
def bip_cons(W):
    """Return the constraints of a bipartite system as a list of triples.

    Each triple holds the index of a row vertex, the index of a column vertex and the weight of that
    pair, taken from the matrix W. The rows are numbered first, starting at 0, and the columns
    follow them, so column b has index b plus the number of rows.
    """
    W = np.asarray(W)
    na, nb = W.shape
    return [(a, na + b, int(W[a, b])) for a in range(na) for b in range(nb)]


def clamp(yhat):
    return np.maximum(np.asarray(yhat, dtype=float), 0.0)


def l1(x, y):
    return float(np.abs(np.asarray(x, float) - np.asarray(y, float)).sum())


def is_feasible(y, cons, tol=TOL):
    y = np.asarray(y, float)
    return bool((y >= -tol).all() and all(y[i] + y[j] >= w - tol for i, j, w in cons))


def violated(y, cons, tol=TOL):
    """Return the constraints that y violates, together with their residuals.

    The result is a dictionary that maps the pair of endpoints of each violated constraint to its
    residual, which is the weight minus the sum of the two dual values. The residuals are plain
    floats, so that they print the same under every NumPy version.
    """
    return {(i, j): float(w - y[i] - y[j]) for i, j, w in cons if y[i] + y[j] < w - tol}


def greedy(yhat, cons, order):
    """Run the two-sided greedy repair in a given order of the constraints.

    The prediction yhat is clamped first. The argument order lists indices into cons. The
    constraints are scanned in that order and, at each violated one, both endpoints are raised by
    the full residual. Returns the repaired dual and the list of the indices of the constraints that
    fired.
    """
    y = clamp(yhat)
    fired = []
    for c in order:
        i, j, w = cons[c]
        r = w - y[i] - y[j]
        if r > TOL:
            y[i] += r
            y[j] += r
            fired.append(c)
    return y, fired


def matching_value(W):
    W = np.asarray(W, float)
    r, c = linear_sum_assignment(-W)
    return float(W[r, c].sum())


# Linear programs.
def lp_rows(n, cons):
    """Build the covering constraints as upper-bound rows for linprog.

    Returns the matrix and the right-hand side that linprog takes as A_ub and b_ub. Each row says
    that minus the sum of the two dual values of one constraint is at most minus its weight, which
    is the same as the two values adding up to at least the weight.
    """
    A = np.zeros((len(cons), n))
    b = np.zeros(len(cons))
    for k, (i, j, w) in enumerate(cons):
        A[k, i] = -1
        A[k, j] = -1
        b[k] = -w
    return A, b


def opt_lp(n, cons):
    A, b = lp_rows(n, cons)
    res = linprog(np.ones(n), A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def face_lp(n, cons, OPT, c, sense="min"):
    """Minimise or maximise a linear function over the fractional optimal face.

    The fractional face is the set of feasible duals, with real coordinates allowed, whose values
    sum to OPT. The vector c holds the coefficients of the function, and sense is "min" or "max".
    Returns the optimal value.
    """
    A, b = lp_rows(n, cons)
    cc = np.asarray(c, float) * (1 if sense == "min" else -1)
    res = linprog(cc, A_ub=A, b_ub=b, A_eq=np.ones((1, n)), b_eq=[OPT],
                  bounds=[(0, None)] * n, method="highs")
    assert res.status == 0, res.message
    return float(res.fun) * (1 if sense == "min" else -1)


def face_extremes(n, cons, OPT):
    out = []
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1
        out.append((face_lp(n, cons, OPT, c, "min"), face_lp(n, cons, OPT, c, "max")))
    return out


def face_dist_lp(n, cons, OPT, q, kind):
    """Return the distance from q to the fractional optimal face, by one linear program.

    The argument kind selects the distance by one of the keys l1, linf and pm. They stand for the
    sum of the absolute differences, the largest absolute difference, and the largest increase plus
    the largest decrease. The last one is the norm of Sakaue and Oki (arXiv:2205.09961).
    """
    q = np.asarray(q, float)
    A, b = lp_rows(n, cons)
    k = {"l1": n, "linf": 1, "pm": 2}[kind]
    cost = np.r_[np.zeros(n), np.ones(k)]
    rows = [np.r_[r, np.zeros(k)] for r in A]
    rhs = list(b)
    for i in range(n):
        # In coordinate i, row r1 says that the face point minus q is at most an auxiliary variable,
        # and row r2 says that q minus the face point is at most an auxiliary variable. The branches
        # below choose the variables.
        r1 = np.zeros(n + k); r1[i] = 1
        r2 = np.zeros(n + k); r2[i] = -1
        if kind == "l1":
            r1[n + i] = -1; r2[n + i] = -1
        elif kind == "linf":
            r1[n] = -1; r2[n] = -1
        else:
            # Row r2 uses the variable at index n, which bounds q minus the face point. Row r1 uses
            # the variable at index n plus 1, which bounds the face point minus q.
            r1[n + 1] = -1; r2[n] = -1
        rows += [r1, r2]
        rhs += [q[i], -q[i]]
    A_eq = np.zeros((1, n + k)); A_eq[0, :n] = 1
    res = linprog(cost, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=A_eq, b_eq=[OPT],
                  bounds=[(0, None)] * (n + k), method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


# Enumeration of the points of the optimal face.
def compositions(s, k):
    if k == 1:
        yield (s,)
        return
    for a in range(s + 1):
        for rest in compositions(s - a, k - 1):
            yield (a,) + rest


def face_points_bip(W, OPT):
    """Return all integer points of the optimal face of a bipartite system.

    The row values are enumerated first, keeping only the vectors whose sum is at most OPT. For each
    of them, every column value must be at least the smallest value that covers its column, and what
    is left of OPT after the row values and these smallest column values is distributed over the
    columns in every possible way. No face coordinate exceeds the largest weight, because lowering a
    larger one keeps the dual feasible and reduces the sum. The enumeration is therefore complete.
    The result is an array with one row per point.
    """
    W = np.asarray(W, dtype=np.int64)
    na, nb = W.shape
    maxw = int(W.max()) if W.size else 0
    if abs(OPT - round(OPT)) > 1e-6:
        return np.zeros((0, na + nb))
    OPT = int(round(OPT))
    U = np.zeros((1, 0), dtype=np.int64)
    for _ in range(na):
        U = np.hstack([np.repeat(U, maxw + 1, axis=0),
                       np.tile(np.arange(maxw + 1), len(U))[:, None]])
        U = U[U.sum(axis=1) <= OPT]
    Vmin = np.stack([np.maximum(0, (W[:, k][None, :] - U).max(axis=1)) for k in range(nb)], axis=1)
    slack = OPT - U.sum(axis=1) - Vmin.sum(axis=1)
    keep = slack >= 0
    U, Vmin, slack = U[keep], Vmin[keep], slack[keep]
    pts = []
    for s in np.unique(slack):
        idx = np.where(slack == s)[0]
        comps = np.array(list(compositions(int(s), nb)), dtype=np.int64).reshape(-1, nb)
        Ub = np.repeat(U[idx], len(comps), axis=0)
        Vb = np.repeat(Vmin[idx], len(comps), axis=0) + np.tile(comps, (len(idx), 1))
        P = np.hstack([Ub, Vb])
        pts.append(P[(Vb <= maxw).all(axis=1)])
    return np.vstack(pts).astype(float) if pts else np.zeros((0, na + nb))


def face_points_grid(n, cons, OPT, maxw, half=False):
    """Return all grid points of the optimal face of a general covering system.

    The grid holds the points whose coordinates lie between zero and the largest weight maxw and are
    integers when half is False, or multiples of one half when half is True. The coordinates are
    assigned one at a time, and a partial point is dropped when two assigned coordinates violate a
    constraint between them, or when the sum of the assigned coordinates plus the smallest values
    that the constraints force on the unassigned ones is more than OPT.
    """
    s = 2 if half else 1
    T = s * OPT
    if abs(T - round(T)) > 1e-6:
        return np.zeros((0, n))
    T = int(round(T))
    H = s * maxw
    deg = np.zeros(n, int)
    for i, j, _ in cons:
        deg[i] += 1; deg[j] += 1
    order = [int(v) for v in np.argsort(-deg, kind="stable")]
    pos = {v: k for k, v in enumerate(order)}
    nbr = [[] for _ in range(n)]
    for i, j, w in cons:
        nbr[pos[i]].append((pos[j], s * w))
        nbr[pos[j]].append((pos[i], s * w))
    P = np.zeros((1, 0), dtype=np.int64)
    for i in range(n):
        if i == n - 1:
            last = T - P.sum(axis=1)
            ok = (last >= 0) & (last <= H)
            P = np.hstack([P[ok], last[ok][:, None]])
        else:
            P = np.hstack([np.repeat(P, H + 1, axis=0), np.tile(np.arange(H + 1), len(P))[:, None]])
        ok = np.ones(len(P), bool)
        for j, w2 in nbr[i]:
            if j < i:
                ok &= P[:, i] + P[:, j] >= w2
        lb = P.sum(axis=1)
        for k in range(i + 1, n):
            mk = np.zeros(len(P), dtype=np.int64)
            for j, w2 in nbr[k]:
                if j <= i:
                    mk = np.maximum(mk, w2 - P[:, j])
            lb += mk
        P = P[ok & (lb <= T)]
    out = np.zeros_like(P)
    for k, v in enumerate(order):
        out[:, v] = P[:, k]
    return out.astype(float) / s


def face_l0(y, pts):
    return int((np.abs(pts - np.asarray(y, float)) > 1e-6).sum(axis=1).min())


def face_l0_subset_lp(n, cons, OPT, y):
    """Return the exact face distance of y, measured against the fractional optimal face.

    The distance is n, the number of coordinates, minus the size of the largest set of coordinates
    on which some face point agrees with y. Candidate agreement sets are tried in decreasing order
    of size, and each one is tested by the feasibility of a linear program. Section 2 uses this
    function as an independent cross-check of the grid enumeration, and several later sections call
    it directly for exact face distances.
    """
    A, b = lp_rows(n, cons)
    y = np.asarray(y, float)
    for k in range(n, -1, -1):
        for S in itertools.combinations(range(n), k):
            if any(y[i] < -TOL for i in S):
                continue
            if any(i in S and j in S and y[i] + y[j] < w - TOL for i, j, w in cons):
                continue
            bounds = [(0, None)] * n
            for i in S:
                bounds[i] = (y[i], y[i])
            res = linprog(np.zeros(n), A_ub=A, b_ub=b, A_eq=np.ones((1, n)), b_eq=[OPT],
                          bounds=bounds, method="highs")
            if res.status == 0:
                return n - k
    return n


# Vertex covers and matchings.
def min_vertex_covers(edges):
    """Find every minimum vertex cover of a graph by brute force.

    Returns the cover number tau, which is the size of a minimum cover, and the list of all minimum
    covers. This is meant for small graphs only.
    """
    if not edges:
        return 0, [frozenset()]
    verts = sorted({v for e in edges for v in e})
    for k in range(len(verts) + 1):
        covers = [frozenset(S) for S in itertools.combinations(verts, k)
                  if all(i in S or j in S for i, j in edges)]
        if covers:
            return k, covers


def maximal_matchings(edges):
    """Return all maximal matchings of a small graph.

    Every maximal matching that a greedy procedure builds is one of them.
    """
    res = set()

    def rec(M, used, avail):
        ext = [e for e in avail if e[0] not in used and e[1] not in used]
        if not ext:
            res.add(frozenset(M))
            return
        for e in ext:
            rec(M + [e], used | set(e), [f for f in avail if f != e])
    rec([], set(), list(edges))
    return res


def cover_scan_all(p, cons, S):
    """Run the cover-directed scan on the vertex set S in every possible way.

    The scan starts from the point p and visits the vertices of S in turn. At each vertex it fires
    the constraint at that vertex that is currently the most violated, raising both endpoints by the
    residual, and it skips the vertex if none of its constraints is violated. This function
    enumerates every processing order of S and every way to break a tie for the largest residual. It
    does so by expanding the set of reachable states step by step, keeping each state once. Returns
    a list of pairs, each holding an output and the number of firings that produced it.
    """
    inc = {i: [] for i in S}
    for a, b, w in cons:
        if a in inc:
            inc[a].append((b, w))
        if b in inc:
            inc[b].append((a, w))
    states = {(frozenset(), tuple(p), 0)}
    for _ in range(len(S)):
        nxt = set()
        for done, yt, nf in states:
            for i in S:
                if i in done:
                    continue
                y = np.array(yt)
                cand = [(w - y[i] - y[j], j) for j, w in inc[i] if w - y[i] - y[j] > TOL]
                if not cand:
                    nxt.add((done | {i}, yt, nf))
                    continue
                rmax = max(r for r, _ in cand)
                for r, j in cand:
                    if abs(r - rmax) < TOL:
                        y2 = y.copy()
                        y2[i] += rmax
                        y2[j] += rmax
                        nxt.add((done | {i}, tuple(y2), nf + 1))
        states = nxt
    return [(np.array(yt), nf) for _, yt, nf in states]


def _cover_scan_orders(p, cons, S, rule="largest"):
    """Collect the outputs of the cover-directed scan over every order of S and every tie.

    Returns the set of outputs, each as a tuple of rounded values, and the number of ties met. When
    rule is "largest", the scan fires the largest residual once per vertex, which gives the same
    outputs as cover_scan_all. When rule is "smallest", it keeps firing the smallest residual at a
    vertex until that vertex is satisfied.
    """
    outs = set(); ties = [0]
    def run(z, rem):
        if not rem: outs.add(tuple(np.round(z, 9))); return
        v = rem[0]
        vio = [(w - z[i] - z[j], i, j) for i, j, w in cons if (i == v or j == v) and z[i] + z[j] < w - TOL]
        if not vio: run(z, rem[1:]); return
        r = max(x[0] for x in vio) if rule == "largest" else min(x[0] for x in vio)
        cand = [x for x in vio if abs(x[0] - r) < TOL]; ties[0] += len(cand) - 1
        for _, i, j in cand:
            zz = z.copy(); zz[i] += r; zz[j] += r
            if rule == "largest": run(zz, rem[1:])
            else: run(zz, rem)  # The smallest rule repeats at v until v is satisfied.
    for perm in itertools.permutations(sorted(S)): run(p.copy(), list(perm))
    return outs, ties[0]


# The one-sided walk.
def walk_all(n, edges, start_any=True, cap=5000):
    """Enumerate the outputs of the one-sided walk over every choice that it leaves open.

    The walk is our transplant of Algorithm 1 of Dinitz et al. The open choices are the start
    vertex of every outer iteration and every tie for the largest residual. When start_any is
    true, the start vertex can be any vertex that still has edges. Otherwise it is the one of
    lowest index. The residuals stay fixed at their initial values, as in walk_repair of
    experiments.py. The argument edges maps each violated pair to its residual, and n is the
    number of vertices.

    Returns three things. The first is the list of leaves, one per complete run, each a pair that
    holds the vector of raises and the list of raised vertices. The second tells whether the
    enumeration was cut short at cap leaves. The third tells whether any tie was seen.
    """
    adj0 = {}
    for (i, j), r in edges.items():
        adj0.setdefault(i, {})[j] = r
        adj0.setdefault(j, {})[i] = r
    leaves, capped, ties = [], [False], [False]

    def outer(adj, delta, raised):
        if len(leaves) >= cap:
            capped[0] = True
            return
        live = [k for k in range(n) if adj.get(k)]
        if not live:
            leaves.append((delta.copy(), list(raised)))
            return
        for s in (live if start_any else [min(live)]):
            inner(s, adj, delta, raised)

    def inner(i, adj, delta, raised):
        if len(leaves) >= cap:
            capped[0] = True
            return
        if not adj.get(i):
            outer(adj, delta, raised)
            return
        mx = max(adj[i].values())
        cand = sorted(t for t in adj[i] if adj[i][t] == mx)
        if len(cand) > 1:
            ties[0] = True
        for j in cand:
            d = delta.copy()
            d[i] = mx
            a2 = {k: dict(v) for k, v in adj.items()}
            for t in list(a2[i]):
                del a2[t][i]
            a2[i] = {}
            inner(j, a2, d, raised + [i])

    outer(adj0, np.zeros(n), [])
    return leaves, capped[0], ties[0]


def walk_default(p, edges, n):
    """Run the one-sided walk with the conventions of the paper.

    These are the conventions of walk_repair in experiments.py: each outer iteration starts at the
    lowest index that still has edges, and a tie for the largest residual goes to the lowest index.
    Returns the point p plus the raises.
    """
    adj = {}
    for (i, j), r in edges.items():
        adj.setdefault(i, {})[j] = r
        adj.setdefault(j, {})[i] = r
    delta = np.zeros(n)
    while any(adj.get(i) for i in range(n)):
        i = min(k for k in range(n) if adj.get(k))
        while adj.get(i):
            j = max(adj[i], key=lambda t: (adj[i][t], -t))
            delta[i] = adj[i][j]
            for t in list(adj[i]):
                del adj[t][i]
            adj[i] = {}
            i = j
    return p + delta


# Scan orders.
def monotone_order_outputs(W, yhat, descending=True):
    """Return the outputs of the greedy repair over every weakly descending scan order.

    When descending is False, the orders are the weakly ascending ones. The constraints are grouped
    into classes of equal weight, and the classes are processed in sequence. Inside a class every
    permutation is covered, by expanding the set of reachable states and keeping once each
    combination of a set of processed constraints and a current point. Returns the set of outputs.
    """
    cons = bip_cons(W)
    classes = {}
    for k, (_, _, w) in enumerate(cons):
        classes.setdefault(w, []).append(k)
    states = {tuple(clamp(yhat))}
    for w in sorted(classes, reverse=descending):
        cls = classes[w]
        frontier = {(frozenset(), s) for s in states}
        for _ in range(len(cls)):
            nxt = set()
            for done, s in frontier:
                for c in cls:
                    if c in done:
                        continue
                    y = np.array(s)
                    i, j, ww = cons[c]
                    r = ww - y[i] - y[j]
                    if r > TOL:
                        y[i] += r
                        y[j] += r
                    nxt.add((done | {c}, tuple(y)))
            frontier = nxt
        states = {s for _, s in frontier}
    return states


def n_monotone_orders(W):
    return math.prod(math.factorial(int(c)) for c in np.unique(np.asarray(W), return_counts=True)[1])


def sorted_order(W):
    na, nb = W.shape
    P = [(a, b) for a in range(na) for b in range(nb)]
    return [int(i) for i in np.argsort([-W[a, b] for a, b in P], kind="stable")]


# Families of instances.
def balanced_W(n):
    """Return the weight matrix of the balanced family with n rows and n columns.

    Number the rows and the columns from 1. In the first row, the weight in column k is 2 times n,
    minus k. In every other row, the weight in column k is 2 times (n minus k).
    """
    W = np.zeros((n, n), int)
    for k in range(1, n + 1):
        W[0, k - 1] = 2 * n - k
        for i in range(2, n + 1):
            W[i - 1, k - 1] = 2 * (n - k)
    return W


def f_star(n):
    u = np.zeros(n); u[0] = n
    v = np.array([2.0 * (n - k) for k in range(1, n + 1)])
    return np.r_[u, v]


def g_face(n, m, gam):
    """Return the point with parameters m and gam of the optimal face of the balanced family.

    This is the parametrisation of the face given by the lemma that section 4 of this program
    checks (Lemma 32 of the paper). The two parameters are at least zero and their sum is at most 1.
    """
    u = np.full(n, float(gam)); u[0] = n - m
    v = np.array([2.0 * (n - k) - gam for k in range(1, n)] + [float(m)])
    return np.r_[u, v]


def order_family_W(d):
    """Return the weight matrix of the family of Theorem 8.

    There is 1 row and there are d columns. With the columns numbered from 0, the weight in column k
    is k plus 1.
    """
    return np.array([[k + 1 for k in range(d)]], dtype=int)


def _path_family(k):
    """Build the path family with parameter k, from Theorem 38.

    There are k rows, numbered from 0. The columns come in two groups: a first group of k columns
    and a second group of k plus 1 columns, each numbered from 0. Row i has weight 1 on column i of
    the first group, and weight 2 on columns i and i plus 1 of the second group. The prediction is 2
    on every column of the second group and 0 elsewhere.

    Returns the weight matrix, the prediction and the single optimal dual, which is 2 on every row.
    """
    W = np.zeros((k, 2 * k + 1), dtype=int)
    for i in range(k):
        W[i, i] = 1; W[i, k + i] = 2; W[i, k + i + 1] = 2
    yhat = np.r_[np.zeros(2 * k), np.full(k + 1, 2.0)]
    return W, yhat, np.r_[np.full(k, 2.0), np.zeros(2 * k + 1)]


def _chain_family(k):
    """Build the chain family with parameter k, from Proposition 41.

    There are k rows and k plus 1 columns, numbered from 0. Row i has a value of 3 plus 2 times i.
    That value is its weight on column i and on the last column. The row after row i, which is row 0
    when row i is the last row, has its own value minus 1 as its weight on column i. The prediction
    is the value of the last row on the last column and 0 elsewhere.

    Returns the weight matrix, the prediction and the single optimal dual, which gives every row its
    value.
    """
    Av = [3 + 2 * i for i in range(k)]
    W = np.zeros((k, k + 1), dtype=int)
    for i in range(k):
        W[i, i] = Av[i]; W[i, k] = Av[i]; W[(i + 1) % k, i] = Av[(i + 1) % k] - 1
    yhat = np.zeros(2 * k + 1); yhat[-1] = Av[-1]
    return W, yhat, np.r_[np.array(Av, float), np.zeros(k + 1)]


# The cold-start witnesses used in sections 12, 17 and 20.
COLD_4X5 = np.array([[12, 11, 0, 7, 10], [11, 2, 8, 11, 5], [10, 11, 8, 12, 10], [9, 0, 7, 5, 1]])
COLD_6X8 = np.array([[14, 12, 5, 6, 3, 8, 3, 14], [8, 15, 1, 8, 15, 13, 7, 10], [4, 11, 7, 1, 3, 12, 14, 15],
                     [10, 7, 4, 5, 1, 3, 4, 8], [15, 7, 2, 2, 14, 11, 14, 5], [15, 5, 4, 6, 2, 7, 1, 8]])


def hamming_to_balanced_face(n, q):
    """Return the exact Hamming distance from q to the optimal face of the balanced family.

    By the lemma that section 4 of this program checks (Lemma 32 of the paper), the face is the
    set of points that g_face returns as its parameters m and gam range over a triangle. For each
    coordinate, the parameter values at which the face point agrees with q are the solutions of
    one linear equation. Coordinates with the same equation are grouped, and every subset of the
    groups is tested for whether its equations can all hold at one point of the triangle, by a
    linear program in 2 variables. The minimum over all agreement sets is therefore exact.
    """
    base = g_face(n, 0, 0)
    dm = g_face(n, 1, 0) - base
    dg = g_face(n, 0, 1) - base
    groups, always = {}, 0
    for i in range(2 * n):
        key = (round(float(dm[i]), 9), round(float(dg[i]), 9), round(float(q[i] - base[i]), 9))
        if key[0] == 0 and key[1] == 0:
            if key[2] == 0:
                always += 1          # This coordinate agrees on the whole face.
            continue                 # Otherwise it never agrees.
        groups.setdefault(key, []).append(i)
    keys = list(groups)
    best = 2 * n - always
    for r in range(1, len(keys) + 1):
        for E in itertools.combinations(keys, r):
            A_eq = np.array([[k[0], k[1]] for k in E]); b_eq = [k[2] for k in E]
            res = linprog(np.zeros(2), A_ub=np.array([[1.0, 1.0]]), b_ub=[1.0], A_eq=A_eq,
                          b_eq=b_eq, bounds=[(0, 1), (0, 1)], method="highs")
            if res.status == 0:
                best = min(best, 2 * n - always - sum(len(groups[k]) for k in E))
    return best


def mintrick(W, yhat):
    """Apply the Min-Trick as transplanted to covering systems.

    The row values are the predicted row values, clamped at zero. Each column value is then the
    largest, over the rows, of the weight of the pair minus the row value, or zero if that is
    negative.
    """
    W = np.asarray(W, float)
    na = W.shape[0]
    u = np.maximum(np.asarray(yhat, float)[:na], 0)
    v = np.maximum(0, (W - u[:, None]).max(axis=0))
    return np.r_[u, v]


# Section 1 of this program: the gadget behind Theorem 16, which says that no repair in the
# class has an additive bound.
def check_theorem_a():
    section(1, "THEOREM 16: no l1-non-expansive repair is l0-additive -- the gadget G1")
    W1 = np.array([[1, 2, 2]]); cons = bip_cons(W1); n = 4
    yhat = np.array([0, 0, 2, 2.]); f = np.array([2, 0, 0, 0.])
    y1 = np.array([0, 1, 2, 2.]); y2 = np.array([1, 0, 2, 2.])
    OPT = opt_lp(n, cons)
    assert abs(OPT - 2) < 1e-9 and abs(matching_value(W1) - 2) < 1e-9
    ext = face_extremes(n, cons, OPT)
    assert all(abs(lo - fi) < 1e-9 and abs(hi - fi) < 1e-9 for (lo, hi), fi in zip(ext, f))
    pts = face_points_bip(W1, OPT)
    assert len(pts) == 1 and np.allclose(pts[0], f)
    pre = face_l0(yhat, pts)
    assert pre == 3 and not is_feasible(yhat, cons)
    print("G1: A = {a}, B = {b1,b2,b3}, w = (1,2,2), yhat = (0|0,2,2)")
    print(f"    OPT = {OPT:.0f} (LP) = {matching_value(W1):.0f} (max-weight matching)")
    print(f"    fractional face = the single point (2|0,0,0): LP (min, max) of each coordinate = "
          f"{[(round(lo, 9) + 0.0, round(hi, 9) + 0.0) for lo, hi in ext]}")
    print(f"    d0(yhat, F) = {pre};  violated at yhat: {violated(yhat, cons)}")
    assert is_feasible(y1, cons) and is_feasible(y2, cons)
    assert l1(yhat, y1) == 1 and l1(yhat, y2) == 1 and l1(y1, y2) == 2
    print(f"    y1 = (0|1,2,2) and y2 = (1|0,2,2) feasible; ||yhat-y1||_1 = {l1(yhat, y1):.0f}, "
          f"||yhat-y2||_1 = {l1(yhat, y2):.0f}, ||y1-y2||_1 = {l1(y1, y2):.0f}")
    # Four linear programs, one per coordinate. Each shows that no feasible point with real
    # coordinates that is within distance 1 of y1, in the sum of the absolute differences, agrees
    # with the face point in that coordinate. The face point is 2 on the row and 0 on the columns.
    A, b = lp_rows(n, cons)
    for i in range(n):
        rows = [np.r_[r, np.zeros(n)] for r in A]; rhs = list(b)
        for jj in range(n):
            r1 = np.zeros(2 * n); r1[jj] = 1; r1[n + jj] = -1; rows.append(r1); rhs.append(y1[jj])
            r2 = np.zeros(2 * n); r2[jj] = -1; r2[n + jj] = -1; rows.append(r2); rhs.append(-y1[jj])
        rows.append(np.r_[np.zeros(n), np.ones(n)]); rhs.append(1.0)
        A_eq = np.zeros((1, 2 * n)); A_eq[0, i] = 1
        res = linprog(np.zeros(2 * n), A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=A_eq,
                      b_eq=[f[i]], bounds=[(0, None)] * (2 * n), method="highs")
        assert res.status == 2, (i, res.status, res.message)
    print("    LP: {z in D, ||z-y1||_1 <= 1 (t-variables), z_i = f_i} is INFEASIBLE for each of the "
          "4 coordinates i,\n        so every real z in D within l1 distance 1 of y1 agrees with "
          "(2|0,0,0) in no coordinate")
    outs = {tuple(greedy(yhat, cons, list(o))[0]) for o in itertools.permutations(range(3))}
    assert outs == {(1., 1., 2., 2.)}
    post = face_l0(np.array([1, 1, 2, 2.]), pts)
    assert post == 4
    print(f"    all 6 scan orders output (1|1,2,2) with d0 = {post}:  3 -> 4")

    # The tensor with 2 copies of the gadget.
    W2 = np.zeros((2, 6), int); W2[0, :3] = [1, 2, 2]; W2[1, 3:] = [1, 2, 2]
    cons2 = bip_cons(W2); n2 = 8
    yhat2 = np.array([0, 0, 0, 2, 2, 0, 2, 2.]); f2 = np.array([2, 2, 0, 0, 0, 0, 0, 0.])
    OPT2 = opt_lp(n2, cons2)
    assert abs(OPT2 - 4) < 1e-9
    ext2 = face_extremes(n2, cons2, OPT2)
    assert all(abs(lo - fi) < 1e-9 and abs(hi - fi) < 1e-9 for (lo, hi), fi in zip(ext2, f2))
    pts2 = face_points_bip(W2, OPT2)
    assert len(pts2) == 1 and np.allclose(pts2[0], f2)
    pre2 = face_l0(yhat2, pts2)
    assert pre2 == 6
    grid = np.array(list(itertools.product(range(4), repeat=8)), dtype=np.int64)
    ok = np.ones(len(grid), bool)
    for i, j, w in cons2:
        ok &= grid[:, i] + grid[:, j] >= w
    P = grid[ok]
    dy = np.abs(P - yhat2).sum(axis=1)
    # A feasible point refutes a candidate when the candidate is farther from it than the prediction
    # is, with distance measured by the sum of the absolute differences. The refuting points closest
    # to the prediction are tried first.
    strongest = np.argsort(dy, kind="stable")
    alive = np.ones(len(P), bool)
    for c0 in range(0, len(P), 64):
        ref = strongest[c0:c0 + 64]
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        dz = np.abs(P[idx][:, None, :] - P[ref][None, :, :]).sum(axis=2)
        alive[idx[(dz > dy[ref][None, :]).any(axis=1)]] = False
    S = P[alive]
    assert len(S) == 1 and tuple(S[0]) == (1, 1, 1, 2, 2, 1, 2, 2)
    assert face_l0(S[0], pts2) == 8
    # A cross-check that uses separability. The excess of a point is the largest value, over all
    # feasible points, of its distance to that feasible point minus the distance of the prediction
    # to it, with distance measured by the sum of the absolute differences. Here the excess of a
    # point is the excess of its first block plus the excess of its second block, where the excess
    # of a block is taken over the feasible set of that block.
    grid1 = np.array(list(itertools.product(range(4), repeat=4)), dtype=np.int64)
    ok1 = np.ones(len(grid1), bool)
    for i, j, w in cons:
        ok1 &= grid1[:, i] + grid1[:, j] >= w
    P1 = grid1[ok1]
    dy1 = np.abs(P1 - yhat).sum(axis=1)
    E1 = (np.abs(P1[:, None, :] - P1[None, :, :]).sum(axis=2) - dy1[None, :]).max(axis=1)
    surv_sep = int(((E1[:, None] + E1[None, :]) <= 0).sum())
    assert surv_sep == 1 and E1.min() == 0
    print(f"tensor k = 2 (block-diagonal 2 x 6, zero cross weights), yhat = (0,0|0,2,2,0,2,2): OPT = {OPT2:.0f}, "
          f"face = single point (2,2|0^6), d0(yhat) = {pre2}")
    print(f"    integral D-points in [0,3]^8: {len(P)}; after discarding every z refuted by some integral "
          f"y in D (||z-y||_1 > ||yhat-y||_1): {len(S)} survivor = {tuple(int(v) for v in S[0])}, d0 = 8")
    print(f"    (separable cross-check: {len(P1)} block points, block excess min = {E1.min():.0f}, "
          f"survivors by e_1 + e_2 <= 0: {surv_sep})")
    # Every one of the 12 factorial orders gives this output too. The zero weights across the blocks
    # never fire and the two blocks share no coordinate, so any order runs the three constraints of
    # each block in some order, and all 6 such orders were checked above for G1. The lines below
    # sample that.
    outs2 = monotone_order_outputs(W2, yhat2, descending=True)
    assert outs2 == {(1., 1., 1., 2., 2., 1., 2., 2.)}
    rng = np.random.default_rng(1)
    rand_outs = set()
    for _ in range(500):
        y, fired = greedy(yhat2, cons2, list(rng.permutation(len(cons2))))
        rand_outs.add((tuple(y), len(fired)))
    assert rand_outs == {((1., 1., 1., 2., 2., 1., 2., 2.), 2)}
    print(f"    all {n_monotone_orders(W2)} weakly-descending tie-breaks and 500 random orders give that "
          f"output (2 firings): d0 6 -> 8, excess k = 2")

    # A second gadget, with a single violated constraint.
    W3 =np.array([[6, 6, 5]]); cons3 = bip_cons(W3)
    yh3 = np.array([4, 4, 6, 0.]); f3 = np.array([6, 0, 0, 0.])
    OPT3 = opt_lp(4, cons3)
    assert abs(OPT3 - 6) < 1e-9
    ext3 = face_extremes(4, cons3, OPT3)
    assert all(abs(lo - fi) < 1e-9 and abs(hi - fi) < 1e-9 for (lo, hi), fi in zip(ext3, f3))
    pts3 = face_points_bip(W3, OPT3)
    assert len(pts3) == 1
    v3 = violated(yh3, cons3)
    assert v3 == {(0, 3): 1.0}
    outs3 = {tuple(greedy(yh3, cons3, list(o))[0]) for o in itertools.permutations(range(3))}
    assert outs3 == {(5., 4., 6., 1.)}
    assert face_l0(yh3, pts3) == 3 and face_l0(np.array([5, 4, 6, 1.]), pts3) == 4
    print(f"older gadget w = (6,6,5), yhat = (4|4,6,0): OPT = {OPT3:.0f}, face = single point (6|0,0,0); "
          f"violated at yhat: {v3} (one constraint);")
    print(f"    output (5|4,6,1) under all 6 orders; d0 3 -> 4")
    # Corollary 17 of the paper. Pad k copies of G1 with M rows and N columns of zero
    # weight, and predict 0 there. Every weight outside the copies is 0, so the feasible set stays a
    # product. OPT and the single face point are unchanged, with a 0 in each new coordinate, which
    # the prediction already matches. The excess gains one term for each new coordinate, equal to
    # the value of that coordinate, so an excess of at most zero forces every new coordinate to 0.
    # Every admissible point is therefore still at face distance exactly 4 times k, now with k plus
    # M rows and 3 times k plus N columns. This is checked for five small choices of k, M and N. The
    # admissibility test built from _phi_D goes through every sign vector, 2 to the power of the
    # number of vertices in all, which limits the number of vertices to about 12. The decomposition
    # argument in the paper covers general M and N.
    for (kk, MM, NN) in [(1, 0, 3), (1, 1, 1), (2, 0, 1), (2, 2, 2), (1, 4, 4)]:
        na, nb = kk + MM, 3 * kk + NN
        Wp = np.zeros((na, nb), dtype=int)
        for g in range(kk):
            Wp[g, 3 * g:3 * g + 3] = [1, 2, 2]
        consp = bip_cons(Wp); np_ = na + nb
        yhp = np.zeros(np_); fp = np.zeros(np_)
        for g in range(kk):
            yhp[na + 3 * g + 1] = 2; yhp[na + 3 * g + 2] = 2; fp[g] = 2
        OPTp = opt_lp(np_, consp)
        assert abs(OPTp - 2 * kk) < 1e-9 and abs(matching_value(Wp) - 2 * kk) < 1e-9
        extp = face_extremes(np_, consp, OPTp)
        assert all(abs(lo - hi) < 1e-9 and abs(lo - fp[i_]) < 1e-9 for i_, (lo, hi) in enumerate(extp))
        ptsp = face_points_bip(Wp, OPTp)
        assert len(ptsp) == 1 and np.allclose(ptsp[0], fp)
        assert face_l0(yhp, ptsp) == 3 * kk
        Aq, bq = lp_rows(np_, consp)
        rws = [(np.array(sg, float), _phi_D(np_, consp, yhp, sg))
               for sg in itertools.product([-1, 1], repeat=np_)]
        A_ub = np.vstack([Aq] + [sg[None, :] for sg, _ in rws])
        b_ub = np.concatenate([bq, [-ph for _, ph in rws]])
        gad = list(range(kk)) + [na + j for j in range(3 * kk)]
        pad = list(range(kk, na)) + [na + j for j in range(3 * kk, nb)]
        for i_ in gad:                      # No gadget coordinate can agree with the face.
            bd = [(0, None)] * np_; bd[i_] = (fp[i_], fp[i_])
            assert linprog(np.zeros(np_), A_ub=A_ub, b_ub=b_ub, bounds=bd,
                           method="highs").status == 2, (kk, MM, NN, i_)
        for i_ in pad:                      # Every padded coordinate is pinned to 0.
            cq = np.zeros(np_); cq[i_] = 1
            lo_ = linprog(cq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * np_, method="highs")
            hi_ = linprog(-cq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * np_, method="highs")
            assert lo_.status == 0 and hi_.status == 0
            assert abs(lo_.fun) < 1e-7 and abs(hi_.fun) < 1e-7, (kk, MM, NN, i_)
        print(f"    padded k={kk} M={MM} N={NN}: |A|={na} |B|={nb} |V|={np_}, OPT={OPTp:.0f}, face a single point, "
              f"pre = {3*kk} -> every admissible z at exactly {4*kk} < |A|+|B| = {np_}")

    print("all assertions in [1] hold", elapsed())


# Section 2 of this program: the two bounds of Theorem 19. For the cover-directed scan over a
# minimum vertex cover, post is at most pre plus tau. For its maximal-matching variant, post is at
# most pre plus 2 times tau, and therefore at most 3 times pre.
def check_theorem_b():
    section(2, "THEOREM 19: the cover-directed scan R_S is l0-additive up to tau, sharply")
    rng = np.random.default_rng(2026)
    N = 3000
    st = dict(inst=0, pre0=0, ratio_max=0.0, ratio_arg=None, exc_max=0.0, exc_arg=None,
              fired_eq_tau=0, outputs=0, covers=0, changed_max_over_2S=0.0,
              mm_ratio_max=0.0, mm_exc_max=0.0, mm_outputs=0, mm_matchings=0,
              ratio_hist={}, half_checked=0, lp_checked=0)
    for it in range(N):
        na, nb = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        W = rng.integers(0, 6, size=(na, nb))
        yhat = rng.integers(-2, 7, size=na + nb).astype(float)
        cons = bip_cons(W); n = na + nb
        OPT = matching_value(W)
        if it % 50 == 0:
            assert abs(OPT - opt_lp(n, cons)) < 1e-7
            st["lp_checked"] += 1
        pts = face_points_bip(W, OPT)
        assert len(pts) > 0
        # On every 20th instance, check that the enumeration over integer points agrees with the one
        # over multiples of one half, as total unimodularity implies.
        if it % 20 == 0:
            pts_h = face_points_grid(n, cons, OPT, int(W.max()), half=True)
            assert face_l0(yhat, pts) == face_l0(yhat, pts_h)
            assert {tuple(x) for x in pts} == {tuple(x) for x in pts_h if np.allclose(x, np.round(x))}
            st["half_checked"] += 1
        pre = face_l0(yhat, pts)
        p = clamp(yhat)
        E = violated(p, cons); edges = list(E)
        tau, covers = min_vertex_covers(edges)
        assert tau <= pre, "tau > d0(yhat, F)"
        if pre == 0:
            st["pre0"] += 1
        st["inst"] += 1
        worst = -1
        for S in covers:
            st["covers"] += 1
            S = sorted(S)
            for y, nf in cover_scan_all(p, cons, S):
                st["outputs"] += 1
                assert is_feasible(y, cons), "R_S output infeasible"
                changed = int((np.abs(y - p) > 1e-9).sum())
                assert changed <= 2 * len(S) and nf <= len(S)
                st["changed_max_over_2S"] = max(st["changed_max_over_2S"], changed / (2 * len(S)) if S else 0)
                post = face_l0(y, pts)
                assert post <= pre + nf, "post > pre + #fired"
                assert post <= pre + len(S) == pre + tau, "post > pre + tau (sharp bound)"
                assert post <= pre + min(pre, na, nb) and post <= 2 * pre, "weak bound"
                worst = max(worst, post)
                if nf == tau:
                    st["fired_eq_tau"] += 1
        if pre > 0:
            r = worst / pre
            st["ratio_hist"][round(r, 3)] = st["ratio_hist"].get(round(r, 3), 0) + 1
            if r > st["ratio_max"]:
                st["ratio_max"] = r
                st["ratio_arg"] = (W.tolist(), yhat.tolist(), pre, tau, worst)
        if tau > 0:
            e = (worst - pre) / tau
            if e > st["exc_max"]:
                st["exc_max"] = e
                st["exc_arg"] = (W.tolist(), yhat.tolist(), pre, tau, worst)
        # The maximal-matching variant scans the endpoints of a maximal matching of the violated
        # graph. It is run for every maximal matching, every processing order of those endpoints
        # and every way of breaking a tie for the largest residual.
        worst_mm = -1
        for M in maximal_matchings(edges):
            st["mm_matchings"] += 1
            S = sorted({v for e in M for v in e})
            assert len(S) <= 2 * tau
            for y, nf in cover_scan_all(p, cons, S):
                st["mm_outputs"] += 1
                assert is_feasible(y, cons)
                post = face_l0(y, pts)
                assert post <= pre + nf <= pre + len(S) <= pre + 2 * tau <= 3 * pre, "mm bound"
                worst_mm = max(worst_mm, post)
        if pre > 0:
            st["mm_ratio_max"] = max(st["mm_ratio_max"], worst_mm / pre)
        if tau > 0:
            st["mm_exc_max"] = max(st["mm_exc_max"], (worst_mm - pre) / tau)
    print(f"bipartite sweep (seed 2026): {st['inst']} instances, shapes 1x1..4x4, w in 0..5, yhat in -2..6; "
          f"{st['pre0']} with pre = 0")
    print(f"    exactness checks: OPT(LP) = OPT(matching) on {st['lp_checked']} instances; integral = half-integral "
          f"face enumeration on {st['half_checked']} instances")
    print(f"    cover-directed scan R_S: {st['covers']} minimum covers, {st['outputs']} outputs over ALL processing "
          f"orders and argmax tie-breaks:")
    print(f"       feasible; #changed <= 2|S| (max #changed / 2|S| = {st['changed_max_over_2S']:.3f}); tau <= d0(yhat,F); "
          f"post <= pre + #fired <= pre + tau (sharp), hence <= pre + min(pre,|A|,|B|) <= 2 pre")
    print(f"       max post/pre = {st['ratio_max']:.4f}  at W = {st['ratio_arg'][0]}, yhat = {st['ratio_arg'][1]}: "
          f"pre {st['ratio_arg'][2]}, tau {st['ratio_arg'][3]}, worst post {st['ratio_arg'][4]}")
    print(f"       max (post - pre)/tau = {st['exc_max']:.4f}  at W = {st['exc_arg'][0]}, yhat = {st['exc_arg'][1]}: "
          f"pre {st['exc_arg'][2]}, tau {st['exc_arg'][3]}, worst post {st['exc_arg'][4]}")
    print(f"       post/pre histogram (worst case per instance): "
          f"{sorted(st['ratio_hist'].items())}")
    print(f"    maximal-matching variant: {st['mm_matchings']} maximal matchings, {st['mm_outputs']} outputs: "
          f"post <= pre + #fired <= pre + |S| <= pre + 2 tau <= 3 pre;")
    print(f"       max post/pre = {st['mm_ratio_max']:.4f}, max (post - pre)/tau = {st['mm_exc_max']:.4f}")
    assert st["ratio_max"] <= 4 / 3 + 1e-9 and st["exc_max"] <= 1 + 1e-9
    assert st["mm_ratio_max"] <= 2 + 1e-9

    # Covering systems on general graphs, which need not be bipartite.
    rng = np.random.default_rng(77)
    gs = dict(inst=0, opt_frac=0, no_integral=0, mm_ratio_max=0.0, mm_exc_max=0.0, cov_ratio_max=0.0,
              cov_exc_max=0.0, xcheck=0, xcheck_frac=0, outputs=0)
    NG = 500
    while gs["inst"] < NG:
        n = int(rng.integers(3, 8))
        cons = [(i, j, int(rng.integers(0, 6))) for i in range(n) for j in range(i + 1, n)
                if rng.random() < 0.55]
        if not cons:
            continue
        maxw = max(w for _, _, w in cons)
        yhat = rng.integers(-2, 7, size=n).astype(float)
        p = clamp(yhat)
        OPT = opt_lp(n, cons)
        frac = abs(OPT - round(OPT)) > 1e-6
        pts = face_points_grid(n, cons, OPT, maxw, half=True)
        assert len(pts) > 0, "no half-integral face point"
        if frac:
            gs["opt_frac"] += 1
        elif len(face_points_grid(n, cons, OPT, maxw, half=False)) == 0:
            gs["no_integral"] += 1
        pre = face_l0(yhat, pts)
        if (frac and gs["xcheck_frac"] < 15) or (not frac and gs["xcheck"] < 15):
            assert pre == face_l0_subset_lp(n, cons, OPT, yhat), "grid vs subset-LP face-l0 mismatch"
            gs["xcheck_frac" if frac else "xcheck"] += 1
        E = violated(p, cons); edges = list(E)
        tau, covers = min_vertex_covers(edges)
        assert tau <= pre
        gs["inst"] += 1
        worst_mm = -1
        for M in maximal_matchings(edges):
            S = sorted({v for e in M for v in e})
            assert len(S) <= 2 * tau
            for y, nf in cover_scan_all(p, cons, S):
                gs["outputs"] += 1
                assert is_feasible(y, cons)
                post = face_l0(y, pts)
                assert post <= pre + nf <= pre + len(S) <= pre + 2 * tau <= 3 * pre, "general mm bound"
                worst_mm = max(worst_mm, post)
        worst_c = -1
        for S in covers:
            S = sorted(S)
            for y, nf in cover_scan_all(p, cons, S):
                assert is_feasible(y, cons)
                post = face_l0(y, pts)
                assert post <= pre + nf <= pre + len(S) == pre + tau, "general R_S bound"
                worst_c = max(worst_c, post)
        if pre > 0:
            gs["mm_ratio_max"] = max(gs["mm_ratio_max"], worst_mm / pre)
            gs["cov_ratio_max"] = max(gs["cov_ratio_max"], worst_c / pre)
        if tau > 0:
            gs["mm_exc_max"] = max(gs["mm_exc_max"], (worst_mm - pre) / tau)
            gs["cov_exc_max"] = max(gs["cov_exc_max"], (worst_c - pre) / tau)
    print(f"non-bipartite sweep (seed 77): {gs['inst']} random graphs on 3..7 vertices (edge prob 0.55, w in 0..5, "
          f"yhat in -2..6), {gs['outputs']} maximal-matching outputs")
    print(f"    OPT fractional on {gs['opt_frac']} instances; integral OPT but no integral face point on "
          f"{gs['no_integral']}; face-l0 by exact half-integral enumeration, cross-checked against subset-LP "
          f"enumeration over the fractional face on {gs['xcheck']} integral-OPT and {gs['xcheck_frac']} fractional-OPT instances")
    print(f"    maximal-matching variant: post <= pre + #fired <= pre + |S| <= pre + 2 tau <= 3 pre on every output; "
          f"max post/pre = {gs['mm_ratio_max']:.4f}, max (post - pre)/tau = {gs['mm_exc_max']:.4f}")
    print(f"    cover-directed scan (all minimum covers): post <= pre + tau on every output; max post/pre = "
          f"{gs['cov_ratio_max']:.4f}, max (post - pre)/tau = {gs['cov_exc_max']:.4f}")

    # The families.
    print("thm:order family (|A| = 1, |B| = d, w_{0k} = k + 1, yhat = 0):")
    for d in range(2, 8):
        W = order_family_W(d); cons = bip_cons(W); n = d + 1
        OPT = matching_value(W); pts = face_points_bip(W, OPT)
        yhat = np.zeros(n); p = clamp(yhat)
        pre = face_l0(yhat, pts)
        tau, covers = min_vertex_covers(list(violated(p, cons)))
        posts = set()
        for S in covers:
            for y, nf in cover_scan_all(p, cons, sorted(S)):
                posts.add(face_l0(y, pts))
        fwd, _ = greedy(yhat, cons, list(range(len(cons))))
        post_fwd = face_l0(fwd, pts)
        assert abs(OPT - d) < 1e-9 and pre == 1 and tau == 1 and posts == {1} and post_fwd == d
        print(f"    d = {d}: OPT = {OPT:.0f}, pre = {pre}, tau = {tau}, covers = {[sorted(S) for S in covers]}; "
              f"R_S -> face-l0 {sorted(posts)}; forward scan -> {post_fwd}")
    print("balanced family B_n (yhat = f - n e_{a1}):")
    for n in range(2, 7):
        W = balanced_W(n); cons = bip_cons(W)
        OPT = matching_value(W); assert abs(OPT - n * n) < 1e-9
        pts = face_points_bip(W, OPT)
        f = f_star(n); yhat = f.copy(); yhat[0] = 0; p = clamp(yhat)
        pre = face_l0(yhat, pts)
        tau, covers = min_vertex_covers(list(violated(p, cons)))
        posts = set()
        for S in covers:
            for y, nf in cover_scan_all(p, cons, sorted(S)):
                posts.add(face_l0(y, pts))
        desc_outs = monotone_order_outputs(W, yhat, descending=True)
        desc_posts = {face_l0(np.array(y), pts) for y in desc_outs}
        assert pre == 1 and tau == 1 and posts == {1} and desc_posts == {n}
        print(f"    n = {n}: OPT = {OPT:.0f}, pre = {pre}, tau = {tau}; R_S -> face-l0 {sorted(posts)}; all "
              f"{n_monotone_orders(W)} weakly-descending orders -> {sorted(desc_posts)} ({len(desc_outs)} distinct output(s))")
    W1 = np.array([[1, 2, 2]]); cons = bip_cons(W1); yhat = np.array([0, 0, 2, 2.]); p = clamp(yhat)
    pts = face_points_bip(W1, matching_value(W1))
    tau, covers = min_vertex_covers(list(violated(p, cons)))
    posts = {face_l0(y, pts) for S in covers for y, _ in cover_scan_all(p, cons, sorted(S))}
    assert posts == {4} and face_l0(yhat, pts) == 3
    print(f"G1: R_S with covers {[sorted(S) for S in covers]} -> face-l0 {sorted(posts)}: 3 -> 4 (= pre + tau, the bound is tight)")
    print("all assertions in [2] hold", elapsed())


# Section 3 of this program: the one-sided walk repair, our transplant of Algorithm 1 of
# Dinitz et al., run with every choice of start vertices and every way of breaking its ties.
def check_walk():
    section(3, "THE WALK: covering transplant of Algorithm 1 over ALL tie-break resolutions")
    rng = np.random.default_rng(20260902)
    N = 3000
    st = dict(inst=0, leaves=0, max_leaves=0, capped=0, pre0=0, ties=0,
              raised_over_tau=0.0, exc_max=0.0, exc_arg=None, ratio_max=0.0, ratio_arg=None, hist={})
    for it in range(N):
        na, nb = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        W = rng.integers(0, 6, size=(na, nb))
        yhat = rng.integers(-2, 7, size=na + nb).astype(float)
        cons = bip_cons(W); n = na + nb
        OPT = matching_value(W)
        pts = face_points_bip(W, OPT)
        pre = face_l0(yhat, pts)
        p = clamp(yhat)
        E = violated(p, cons)
        tau, _ = min_vertex_covers(list(E))
        assert tau <= pre
        leaves, capped, ties = walk_all(n, E, start_any=True, cap=5000)
        st["inst"] += 1; st["leaves"] += len(leaves); st["capped"] += capped; st["ties"] += ties
        st["max_leaves"] = max(st["max_leaves"], len(leaves))
        if pre == 0:
            st["pre0"] += 1
        worst = -1
        for delta, raised in leaves:
            y = p + delta
            assert is_feasible(y, cons), "walk output infeasible"
            assert len(raised) == int((delta > 0).sum()) <= 2 * tau, "#raised > 2 tau"
            post = face_l0(y, pts)
            assert post <= pre + tau, "post > pre + tau (sharp walk bound)"
            assert post <= pre + 2 * min(tau, na, nb) and post <= 2 * pre <= 3 * pre, "weak walk bound"
            worst = max(worst, post)
            if tau > 0:
                st["raised_over_tau"] = max(st["raised_over_tau"], len(raised) / tau)
            st["hist"][post - pre] = st["hist"].get(post - pre, 0) + 1
        if pre > 0 and worst / pre > st["ratio_max"]:
            st["ratio_max"] = worst / pre; st["ratio_arg"] = (W.tolist(), yhat.tolist(), pre, tau, worst)
        if tau > 0 and (worst - pre) / tau > st["exc_max"]:
            st["exc_max"] = (worst - pre) / tau; st["exc_arg"] = (W.tolist(), yhat.tolist(), pre, tau, worst)
    assert st["capped"] == 0
    print(f"random sweep (seed 20260902): {st['inst']} instances, shapes 1x1..4x4, w in 0..5, yhat in -2..6; "
          f"{st['leaves']} tie-break leaves (any start vertex with edges, every argmax tie), max {st['max_leaves']} "
          f"per instance, {st['capped']} capped; {st['ties']} instances had an argmax tie; {st['pre0']} with pre = 0")
    print(f"    every leaf: feasible; #raised <= 2 tau (max #raised/tau = {st['raised_over_tau']:.3f}); "
          f"post <= pre + tau (sharp), hence <= 2 pre")
    print(f"    max post/pre = {st['ratio_max']:.4f} at W = {st['ratio_arg'][0]}, yhat = {st['ratio_arg'][1]}: "
          f"pre {st['ratio_arg'][2]}, tau {st['ratio_arg'][3]}, worst post {st['ratio_arg'][4]}")
    print(f"    max (post - pre)/tau = {st['exc_max']:.4f} at W = {st['exc_arg'][0]}, yhat = {st['exc_arg'][1]}: "
          f"pre {st['exc_arg'][2]}, tau {st['exc_arg'][3]}, worst post {st['exc_arg'][4]}")
    print(f"    histogram of post - pre over leaves: {sorted(st['hist'].items())}")
    assert st["exc_max"] <= 1 + 1e-9 and st["raised_over_tau"] <= 2 + 1e-9

    # The lower-bound family for the walk. Distances here count the coordinates that differ
    # from the nearest optimal dual. On one copy of the instance every run of the walk ends at
    # distance 6, from a prediction at distance 5. On two disjoint copies every run ends at
    # distance 12, from a prediction at distance 10.
    W4 = np.array([[1, 2, 3, 1], [1, 2, 1, 2]]); cons = bip_cons(W4); n = 6
    y4 = np.array([0, 0, 0, 3, 3, 3.]); p = clamp(y4)
    OPT = opt_lp(n, cons); assert abs(OPT - 5) < 1e-9
    ext = face_extremes(n, cons, OPT)
    want = [(2, 3), (2, 2), (0, 0), (0, 0), (0, 1), (0, 0)]
    assert all(abs(lo - a) < 1e-9 and abs(hi - b) < 1e-9 for (lo, hi), (a, b) in zip(ext, want))
    c = np.zeros(n); c[0] = 1; c[4] = 1
    assert abs(face_lp(n, cons, OPT, c, "min") - 3) < 1e-9 and abs(face_lp(n, cons, OPT, c, "max") - 3) < 1e-9
    e1, e2 = np.array([2, 2, 0, 0, 1, 0.]), np.array([3, 2, 0, 0, 0, 0.])
    assert is_feasible(e1, cons) and is_feasible(e2, cons)
    pts = face_points_bip(W4, OPT)
    assert {tuple(x) for x in pts} == {tuple(e1), tuple(e2)}
    pre = face_l0(y4, pts); E = violated(p, cons); tau, _ = min_vertex_covers(list(E))
    leaves, capped, ties = walk_all(n, E, cap=10 ** 6)
    posts = {face_l0(p + d, pts) for d, _ in leaves}
    dflt = walk_default(p, E, n)
    assert pre == 5 and tau == 1 and posts == {6} and face_l0(dflt, pts) == 6 and not capped
    print(f"family W = [[1,2,3,1],[1,2,1,2]], yhat = (0,0|0,3,3,3): OPT = {OPT:.0f}; face = segment between "
          f"(2,2|0,0,1,0) and (3,2|0,0,0,0)")
    print(f"    (LP coordinate extremes {[(round(a, 6) + 0.0, round(b, 6) + 0.0) for a, b in ext]}, a1 + b3 = 3 on the face, both "
          f"endpoints feasible, integral face points = exactly these two)")
    print(f"    pre = {pre}, tau = {tau}, violated {E}; walk post = {sorted(posts)} under every one of {len(leaves)} "
          f"tie-break leaves; paper-convention output {dflt.tolist()}")
    W8 = np.zeros((4, 8), int); W8[:2, :4] = W4; W8[2:, 4:] = W4
    cons8 = bip_cons(W8); n8 = 12
    y8 = np.array([0, 0, 0, 0] + [0, 3, 3, 3, 0, 3, 3, 3], float); p8 = clamp(y8)
    OPT8 = opt_lp(n8, cons8); pts8 = face_points_bip(W8, OPT8)
    pre8 = face_l0(y8, pts8); E8 = violated(p8, cons8); tau8, _ = min_vertex_covers(list(E8))
    leaves8, capped8, _ = walk_all(n8, E8, cap=10 ** 6)
    posts8 = {face_l0(p8 + d, pts8) for d, _ in leaves8}
    assert abs(OPT8 - 10) < 1e-9 and len(pts8) == 4 and pre8 == 10 and tau8 == 2 and posts8 == {12} and not capped8
    print(f"    two disjoint copies: OPT = {OPT8:.0f}, {len(pts8)} integral face points, pre = {pre8}, tau = {tau8}; "
          f"walk post = {sorted(posts8)} under every one of {len(leaves8)} tie-break leaves (10 -> 12: no additive constant)")

    # The witness that the result of the walk depends on its free choices. The prediction is
    # at distance 5 from the optimal face, counted in differing coordinates as above. No tie
    # for a largest residual occurs, yet the choice of start vertices alone puts the output of
    # the walk at distance 2, 3, 4 or 6. The conventions of the paper, which always take the
    # lowest index, give 6, and the sorted scan gives 4 under every order of weakly decreasing
    # weight.
    W0 = np.array([[2, 0, 3, 0], [4, 0, 0, 5], [0, 5, 4, 2]]); cons0 = bip_cons(W0); n0 = 7
    y0 = np.array([2, 1, 0, 4, 6, 0, 1.]); p0 = clamp(y0)
    OPT0 = opt_lp(n0, cons0); pts0 = face_points_bip(W0, OPT0)
    pre0 = face_l0(y0, pts0); E0 = violated(p0, cons0); tau0, _ = min_vertex_covers(list(E0))
    sorted_posts = {face_l0(np.array(y), pts0) for y in monotone_order_outputs(W0, y0, descending=True)}
    dflt0 = walk_default(p0, E0, n0)
    leaves0, capped0, ties0 = walk_all(n0, E0, cap=10 ** 6)
    reach = {}
    for d, _ in leaves0:
        post = face_l0(p0 + d, pts0)
        reach[post] = reach.get(post, 0) + 1
    assert abs(OPT0 - 13) < 1e-9 and pre0 == 5 and tau0 == 2 and sorted_posts == {4}
    assert face_l0(dflt0, pts0) == 6 and not ties0 and set(reach) == {2, 3, 4, 6}
    print(f"witness W0 = {W0.tolist()}, yhat = (2,1,0|4,6,0,1): OPT = {OPT0:.0f}, {len(pts0)} integral face points, "
          f"pre = {pre0}, tau = {tau0}, violated {E0}")
    print(f"    sorted scan -> face-l0 {sorted(sorted_posts)} under every weakly-descending order "
          f"({n_monotone_orders(W0)} orders); walk under the paper's conventions -> {face_l0(dflt0, pts0)} "
          f"(output {dflt0.tolist()})")
    print(f"    no argmax ties exist; over all {len(leaves0)} start choices the walk reaches face-l0 "
          f"{sorted(reach)} (multiplicities {dict(sorted(reach.items()))}): exactly {{2,3,4,6}}")
    print("all assertions in [3] hold", elapsed())


# Section 4 of this program: the lemma that describes the optimal face of the balanced
# family as a triangle, and the two scan outputs on that family, compared by their distances
# to the face in the largest absolute difference and in the number of differing coordinates.
def check_balanced_face():
    section(4, "BALANCED FACE LEMMA: F(B_n) is the triangle {g(m,gamma)}, and the l_inf reversal")
    rng = np.random.default_rng(4)
    hdr = ("n", "OPT", "#func", "ranges", "int pts", "linf q", "pm q", "linf q'", "pm q'", "ham q", "ham q'", "ham yhat")
    print("{:>3} {:>5} {:>5} {:>7} {:>7} | {:>6} {:>6} {:>7} {:>6} | {:>5} {:>6} {:>8} | tie-breaks".format(*hdr))
    for n in range(2, 13):
        W = balanced_W(n); cons = bip_cons(W); N = 2 * n
        OPT = opt_lp(N, cons)
        assert abs(OPT - n * n) < 1e-9 and abs(matching_value(W) - n * n) < 1e-9
        verts = [g_face(n, 0, 0), g_face(n, 1, 0), g_face(n, 0, 1)]
        for V in verts:
            assert is_feasible(V, cons) and abs(V.sum() - n * n) < 1e-9
        assert np.allclose(verts[0], f_star(n))
        funcs = []
        for i in range(3, n + 1):
            c = np.zeros(N); c[i - 1] = 1; c[1] = -1; funcs.append((c, 0.0, (0, 0)))
        for k in range(1, n):
            c = np.zeros(N); c[n + k - 1] = 1; c[1] = 1; funcs.append((c, -2.0 * (n - k), (0, 0)))
        c = np.zeros(N); c[N - 1] = 1; c[0] = 1; funcs.append((c, -float(n), (0, 0)))
        c = np.zeros(N); c[1] = 1; funcs.append((c, 0.0, (0, 1)))
        c = np.zeros(N); c[N - 1] = 1; funcs.append((c, 0.0, (0, 1)))
        c = np.zeros(N); c[1] = 1; c[N - 1] = 1; funcs.append((c, 0.0, (0, 1)))
        for c, shift, want in funcs:
            lo = face_lp(N, cons, OPT, c, "min") + shift
            hi = face_lp(N, cons, OPT, c, "max") + shift
            assert abs(lo - want[0]) < 1e-7 and abs(hi - want[1]) < 1e-7, (n, lo, hi, want)
        # A stronger check: for 100 random linear objectives, the largest value over the optimal
        # face, computed by LP, equals the largest value over the three vertices.
        for _ in range(100):
            c = rng.normal(size=N)
            assert abs(face_lp(N, cons, OPT, c, "max") - max(c @ V for V in verts)) < 1e-6
        npts = None
        if n <= 6:
            pts = face_points_bip(W, OPT)
            npts = len(pts)
            assert npts == 3 and all(any(np.allclose(x, V) for V in verts) for x in pts)
        # The two scan outputs, for the prediction that is the optimal dual f with the value of
        # the first row set to zero.
        f = f_star(n); yhat = f.copy(); yhat[0] = 0
        # The expected output of the scan in decreasing weight order, written out directly: f
        # with every column value raised by 1. Its first row value is n, its other row values
        # are 0, and column k, counted from 1, has the value 2 times (n minus k), plus 1, which
        # is 1 for the last column.
        q = f.copy(); q[n:] += 1
        # The expected output of the scan in increasing weight order, written out directly: f
        # with the value of the last column raised by n.
        qp = f.copy(); qp[N - 1] += n
        if n <= 6:
            outs_d = monotone_order_outputs(W, yhat, descending=True)
            outs_i = monotone_order_outputs(W, yhat, descending=False)
            assert outs_d == {tuple(q)} and outs_i == {tuple(qp)}
            tb = f"all {n_monotone_orders(W)} desc / all {n_monotone_orders(W)} asc orders"
        else:
            P = [(a, b) for a in range(n) for b in range(n)]
            sd = {tuple(greedy(yhat, cons, sorted_order(W))[0])}
            si = {tuple(greedy(yhat, cons, sorted_order(W)[::-1])[0])}
            for _ in range(30):
                perm = list(rng.permutation(len(P)))
                sd.add(tuple(greedy(yhat, cons, sorted(perm, key=lambda k: -W[P[k]]))[0]))
                si.add(tuple(greedy(yhat, cons, sorted(perm, key=lambda k: W[P[k]]))[0]))
            assert sd == {tuple(q)} and si == {tuple(qp)}
            tb = "sorted + 30 random tie-breaks each"
        assert is_feasible(q, cons) and is_feasible(qp, cons)
        li_q = face_dist_lp(N, cons, OPT, q, "linf"); pm_q = face_dist_lp(N, cons, OPT, q, "pm")
        li_qp = face_dist_lp(N, cons, OPT, qp, "linf"); pm_qp = face_dist_lp(N, cons, OPT, qp, "pm")
        assert abs(li_q - 1) < 1e-7 and abs(pm_q - 1) < 1e-7
        assert abs(li_qp - (n - 1)) < 1e-7 and abs(pm_qp - (n - 1)) < 1e-7
        h_q = hamming_to_balanced_face(n, q); h_qp = hamming_to_balanced_face(n, qp); h_y = hamming_to_balanced_face(n, yhat)
        assert h_q == n and h_qp == 1 and h_y == 1
        if n <= 6:
            assert face_l0(q, pts) == n and face_l0(qp, pts) == 1 and face_l0(yhat, pts) == 1
        print(f"{n:>3} {OPT:>5.0f} {len(funcs):>5} {'ok':>7} {str(npts) if npts else '-':>7} | {li_q:>6.3f} {pm_q:>6.3f} "
              f"{li_qp:>7.3f} {pm_qp:>6.3f} | {h_q:>5} {h_qp:>6} {h_y:>8} | {tb}")
    print("lemma (n = 2..12): the three vertices g(0,0), g(1,0), g(0,1) are feasible with sum n^2; over the face the")
    print("    functionals u_{a_i}-u_{a_2} (i>=3), v_{b_k}+u_{a_2}-2(n-k) (k<n), v_{b_n}+u_{a_1}-n have range [0,0] and")
    print("    u_{a_2}, v_{b_n}, u_{a_2}+v_{b_n} have range [0,1] (LP); 100 random directions per n agree with the")
    print("    triangle; exactly 3 integral face points for n <= 6.")
    print("reversal (n = 2..12): descending output q has l_inf = l_inf^pm = 1 and Hamming n; increasing output q' has")
    print("    l_inf = l_inf^pm = n-1 and Hamming 1 (Hamming by integral enumeration for n <= 6 and by the lemma for all n).")
    print("all assertions in [4] hold", elapsed())


# Section 5 of this program: the Min-Trick repair, which keeps the clamped row values and
# gives each column the smallest value, not below zero, that satisfies all of its constraints.
def check_mintrick():
    section(5, "MIN-TRICK covering transplant T: u = max(yhat_A,0), v_b = max_a (w_ab - u_a)^+")
    rng = np.random.default_rng(5)
    bad = 0
    for _ in range(500):
        na, nb = rng.integers(1, 6, size=2)
        W = rng.integers(0, 10, size=(na, nb))
        yhat = rng.normal(size=na + nb) * 5
        if not is_feasible(mintrick(W, yhat), bip_cons(W)):
            bad += 1
    assert bad == 0
    print(f"feasible on 500/500 random instances (|A|,|B| in 1..5, w in 0..9, Gaussian predictions)")
    W = np.array([[1]]); y = np.array([0, 5.])
    T = mintrick(W, y)
    assert is_feasible(y, bip_cons(W)) and np.allclose(T, [0, 1])
    print(f"not a retraction: w = 1, feasible y = (0|5) is mapped to T(y) = ({T[0]:.0f}|{T[1]:.0f})")
    for d in range(1, 21):
        W = np.zeros((1, d), int); yhat = np.ones(d + 1); yf = np.ones(d + 1); yf[0] = 0
        assert is_feasible(yf, bip_cons(W))
        num, den = l1(mintrick(W, yhat), yf), l1(yhat, yf)
        assert num == d + 1 and den == 1
    print(f"l1 expansion: |A| = 1, |B| = d, w = 0, yhat = (1|1..1), feasible y = (0|1..1): "
          f"||T(yhat)-y||_1 / ||yhat-y||_1 = (d+1)/1 for d = 1..20 (d = 20: 21)")
    for d in range(2, 9):
        W = order_family_W(d); cons = bip_cons(W); n = d + 1
        OPT = matching_value(W); pts = face_points_bip(W, OPT)
        T = mintrick(W, np.zeros(n))
        assert np.allclose(T, np.r_[0, np.arange(1, d + 1)])
        h = face_l0(T, pts)
        assert h == d + 1 and face_l0(np.zeros(n), pts) == 1
        print(f"    thm:order family d = {d}: T(0) = {T.astype(int).tolist()}, face-l0 = {h} = |V| (pre = 1)")
    for n in range(2, 9):
        W = balanced_W(n); cons = bip_cons(W)
        f = f_star(n); yhat = f.copy(); yhat[0] = 0
        T = mintrick(W, yhat)
        assert np.allclose(T, np.r_[np.zeros(n), np.arange(2 * n - 1, n - 1, -1)]) and is_feasible(T, cons)
        h_lemma = hamming_to_balanced_face(n, T)
        h_enum = face_l0(T, face_points_bip(W, n * n)) if n <= 6 else None
        assert h_lemma == n + 1 and (h_enum is None or h_enum == n + 1)
        print(f"    B_n n = {n}: T(yhat) = {T.astype(int).tolist()}, face-l0 = {h_lemma} = n+1 "
              f"(lemma face{'; enumeration ' + str(h_enum) if h_enum is not None else ''}; pre = 1)")
    print("all assertions in [5] hold", elapsed())


# Section 6 of this program: the check that solve_warm of solver.py makes at least as many
# dual adjustments as its start has rows without a tight edge, that is, an edge whose
# constraint holds with equality. Two related checks follow, on the uniform shift and on a
# tiny lift of an optimal dual.
def square_instance(kind, n, rng):
    if kind == "uniform":
        return rng.integers(0, 100, size=(n, n)).astype(float)
    if kind == "clustered":
        W = rng.integers(0, 20, size=(n, n)).astype(float)
        for _ in range(max(1, n // 8)):
            i, j = rng.integers(0, n, 2); s = max(2, n // 6)
            W[i:i + s, j:j + s] += rng.integers(50, 100)
        return W
    if kind == "sparse":
        W = np.zeros((n, n))
        idx = rng.integers(0, n, size=(2 * n, 2))
        W[idx[:, 0], idx[:, 1]] = rng.integers(1, 50, size=2 * n)
        return W
    raise ValueError(kind)


def rep_onesided(W, u, v):
    """Repair on one side only by raising each row value just enough to cover its pairs.

    Negative values of u and v are first replaced by zero. The column values are then left
    unchanged. The result is the new row values and a copy of the column values.
    """
    u = np.maximum(u, 0).astype(float); v = np.maximum(v, 0).astype(float)
    return np.maximum(u, (W - v[None, :]).max(axis=1)), v.copy()


def check_tight_rows():
    section(6, "TIGHT-ROW LOWER BOUND for solver.py::solve_warm: adjustments >= U(u0, v0)")
    # The module experiments supplies the optimal dual and the sorted scan.
    import experiments as E
    # The module experiments_itcs supplies the two shifts and the count U of rows without a
    # tight edge, exactly as the tables use them.
    import experiments_itcs as EI
    rng = np.random.default_rng(6)
    kinds = ("uniform", "clustered", "sparse")
    sizes = (6, 10, 15, 20, 30)
    starts = ("sorted", "shift", "local_shift", "one_sided", "cold", "opt_lift")
    runs = []
    t = 0
    # The instance family and the type of start advance together (6 is a multiple of 3), so
    # each type of start meets only one family: sorted and one_sided meet uniform, shift and
    # cold meet clustered, and local_shift and opt_lift meet sparse.
    while len(runs) < 360:
        kind = kinds[t % 3]; n = sizes[(t // 3) % 5]; sname = starts[t % 6]
        frac = ((t // 18) % 2 == 1) or sname == "opt_lift"
        t += 1
        W = square_instance(kind, n, rng)
        OPT = matching_value(W)
        us, vs = E.optimal_dual(W)
        u0, v0 = us.copy(), vs.copy()
        k = int(rng.integers(1, 2 * n + 1))
        for i in rng.choice(2 * n, size=k, replace=False):
            d = rng.choice([-1, 1]) * (rng.random() * 20 if frac else rng.integers(1, 20))
            if i < n:
                u0[i] = max(0, u0[i] + d)
            else:
                v0[i - n] = max(0, v0[i - n] + d)
        if sname == "sorted":
            ru, rv = E.repair(W, u0, v0, E.orders(W)["sorted"])
        elif sname == "shift":
            ru, rv = EI.shift_repair(W, u0, v0)
        elif sname == "local_shift":
            ru, rv = EI.local_shift_repair(W, u0, v0)
        elif sname == "one_sided":
            ru, rv = rep_onesided(W, u0, v0)
        elif sname == "cold":
            ru, rv = W.max(axis=1).astype(float), np.zeros(n)
        else:
            ru, rv = us + rng.random(n) * rng.random() * 10, vs + rng.random(n) * rng.random() * 10
        assert float(np.max(W - (ru[:, None] + rv[None, :]))) <= 1e-9, "infeasible start"
        val, adj, ph = solve_warm(W, ru, rv)
        assert abs(val - OPT) < 1e-6 and ph == n
        U = EI.U_of(W, ru, rv)
        assert adj >= U, ("adjustments < U", kind, n, sname, adj, U)
        runs.append((kind, n, sname, frac, adj, U))
    adj = np.array([r[4] for r in runs]); U = np.array([r[5] for r in runs])
    nfrac = sum(1 for r in runs if r[3])
    print(f"{len(runs)} random feasible starts (families uniform/clustered/sparse, n in {sizes}; {nfrac} fractional, "
          f"{len(runs) - nfrac} integral): adjustments >= U on all {len(runs)}; equality on {(adj == U).sum()} "
          f"({100 * (adj == U).mean():.1f}%); U = 0 on {(U == 0).sum()} runs")
    m = U > 0
    print(f"    adj/U over the {m.sum()} runs with U > 0: min {(adj[m] / U[m]).min():.3f}, median "
          f"{np.median(adj[m] / U[m]):.3f}, max {(adj[m] / U[m]).max():.3f}")
    for sname in starts:
        sub = [r for r in runs if r[2] == sname]
        a = np.array([r[4] for r in sub]); u = np.array([r[5] for r in sub])
        print(f"    {sname:>12}: {len(sub)} runs, adj == U on {(a == u).sum()}, mean adj {a.mean():.1f}, mean U {u.mean():.1f}")

    # A check, on random instances with integral duals, of how the parity of the largest
    # residual (eps in the code) decides what the uniform shift leaves tight. The shift raises
    # every coordinate by half of eps and rounds up. The assertions are that an odd eps leaves
    # no tight edge at all, so that every row is without one, and that an even eps leaves
    # tight exactly the constraints whose residual was eps.
    rng = np.random.default_rng(66)
    odd = even = 0; t = 0
    while odd + even < 120:
        kind = kinds[t % 3]; n = (6, 10, 20)[(t // 3) % 3]; t += 1
        W = square_instance(kind, n, rng)
        us, vs = E.optimal_dual(W)
        ur, vr = np.round(us), np.round(vs)
        if np.abs(us - ur).max() > 1e-6 or np.abs(vs - vr).max() > 1e-6:
            continue
        u0, v0 = ur.copy(), vr.copy()
        for i in rng.choice(2 * n, size=int(rng.integers(1, n + 1)), replace=False):
            d = rng.choice([-1, 1]) * rng.integers(1, 20)
            if i < n:
                u0[i] = max(0, u0[i] + d)
            else:
                v0[i - n] = max(0, v0[i - n] + d)
        S0 = W - (u0[:, None] + v0[None, :]); eps = int(round(S0.max()))
        if eps <= 0:
            continue
        su, sv = EI.shift_repair(W, u0, v0)
        tight = np.abs((su[:, None] + sv[None, :]) - W) <= 1e-9
        Us = EI.U_of(W, su, sv)
        if eps % 2 == 1:
            assert tight.sum() == 0 and Us == n, ("odd eps", eps, tight.sum(), Us, n)
            odd += 1
        else:
            assert np.array_equal(tight, S0 == eps) and Us == n - int((S0 == eps).any(axis=1).sum())
            even += 1
    print(f"epsilon parity of the uniform shift ceil(x + eps/2) on {odd + even} instances with integral duals "
          f"({odd} with eps odd, {even} with eps even):")
    print("    eps odd  => no tight edge at all and U = n;   eps even => the tight edges are exactly the maximally "
          "violated constraints (and U = n - #rows carrying one)")

    # The discontinuity under a tiny lift, which the output calls the epsilon-lift, checked on
    # nine random instances. The assertions are that the solver makes no dual adjustment from
    # an integral optimal dual, and one adjustment per row from the same dual with every value
    # raised by one thousandth, although the dual gap (the dual objective minus the optimal
    # value) is then only one thousandth times the number of vertices.
    print("epsilon-lift (generic weights 0..999):")
    for n in (10, 20, 40):
        for s in range(3):
            rng = np.random.default_rng(100 * n + s)
            W = rng.integers(0, 1000, size=(n, n)).astype(float)
            OPT = matching_value(W)
            us, vs = E.optimal_dual(W)
            ur, vr = np.round(us), np.round(vs)
            assert np.abs(us - ur).max() < 1e-6 and np.abs(vs - vr).max() < 1e-6
            assert (W - (ur[:, None] + vr[None, :])).max() <= 0 and abs(ur.sum() + vr.sum() - OPT) < 1e-9
            v0, a0, _ = solve_warm(W, ur, vr)
            eps = 1e-3
            v1, a1, _ = solve_warm(W, ur + eps, vr + eps)
            gap = (ur + eps).sum() + (vr + eps).sum() - OPT
            assert a0 == 0 and a1 == n and abs(gap - 2 * n * eps) < 1e-9
            assert abs(v0 - OPT) < 1e-9 and abs(v1 - OPT) < 1e-9
            print(f"    n = {n:>2} seed {s}: from an optimal dual (u*, v*): {a0} adjustments; from (u* + 1e-3, v* + 1e-3): "
                  f"{a1} = n adjustments, initial gap {gap:.4f} = 2n*1e-3, U = {EI.U_of(W, ur + eps, vr + eps)}")
    print("all assertions in [6] hold", elapsed())


# Section 7 of this program: the check, on random instances, that the integral face point
# nearest to the prediction in the sum of absolute differences is at most twice as far from
# every integral face point as the prediction is.
def check_l1_nearest():
    section(7, "the l1-nearest face point f1 satisfies ||f1 - y*||_1 <= 2 ||yhat - y*||_1 for every face point y*")
    rng = np.random.default_rng(7)
    tested = 0; ratio_max = 0.0; arg = None; shapes = {}; lp_checked = 0
    while tested < 300:
        na, nb = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        W = rng.integers(0, 5, size=(na, nb)); cons = bip_cons(W); n = na + nb
        OPT = matching_value(W); pts = face_points_bip(W, OPT)
        if len(pts) < 2:
            continue
        M = max(int(W.max()), 1)
        yhat = rng.integers(0, M + 2, size=n).astype(float)
        if is_feasible(yhat, cons) and abs(yhat.sum() - OPT) < 1e-9:
            continue
        tested += 1; shapes[(na, nb)] = shapes.get((na, nb), 0) + 1
        d = np.abs(pts - yhat).sum(axis=1)
        f1 = pts[int(np.argmin(d))]
        # On every tenth instance, check that the smallest distance found by enumerating the
        # integral face points equals the LP distance to the whole face, fractional points
        # included.
        if tested % 10 == 0:
            assert abs(d.min() - face_dist_lp(n, cons, OPT, yhat, "l1")) < 1e-7
            lp_checked += 1
        for ys in pts:
            den = l1(yhat, ys); num = l1(f1, ys)
            assert den > 0 and num <= 2 * den + 1e-9
            if num / den > ratio_max:
                ratio_max = num / den; arg = (W.tolist(), yhat.tolist(), f1.tolist(), ys.tolist())
    assert abs(ratio_max - 2.0) < 1e-9
    print(f"{tested} random instances with non-singleton face (shapes {dict(sorted(shapes.items()))}, w in 0..4), "
          f"yhat off the face; face-l1 enumeration = LP on {lp_checked} of them")
    print(f"    max over instances and integral face points y* of ||f1-y*||_1 / ||yhat-y*||_1 = {ratio_max:.4f} "
          f"(bound 2 attained, never exceeded)")
    print(f"    attained at W = {arg[0]}, yhat = {arg[1]}, f1 = {arg[2]}, y* = {arg[3]}")
    print("all assertions in [7] hold", elapsed())


# Section 8 of this program: the reproduction of the pinned run of experiments.py, with the
# cover-directed scan added on the same draws.
def koenig_cover(na, nb, edges):
    """Find a minimum vertex cover of a bipartite graph.

    The graph has na row vertices and nb column vertices, and edges lists its edges as pairs
    of a row index and a column index. A maximum matching is computed with scipy. Koenig's
    construction then starts from the unmatched rows and follows paths that alternate between
    edges outside the matching and edges of the matching. The cover is made of the rows that
    have at least one edge and that these paths do not reach, together with the columns that
    the paths reach. The function
    asserts that the result covers every edge and has the same size as the matching. It
    returns the cover and that size. In the cover a row keeps its index and a column appears
    as na plus its index.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if not edges:
        return set(), 0
    rows = np.array([e[0] for e in edges]); cols = np.array([e[1] for e in edges])
    M = csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(na, nb))
    # The entry of mcol for a row is its matched column, or minus 1 if the row is unmatched.
    mcol = maximum_bipartite_matching(M, perm_type="column")
    msize = int((mcol >= 0).sum())
    mrow = -np.ones(nb, dtype=int)
    for a, b in enumerate(mcol):
        if b >= 0:
            mrow[b] = a
    adjA = [[] for _ in range(na)]
    for a, b in edges:
        adjA[a].append(b)
    ZA = np.zeros(na, bool); ZB = np.zeros(nb, bool)
    stack = [a for a in range(na) if mcol[a] < 0 and adjA[a]]
    for a in stack:
        ZA[a] = True
    while stack:
        a = stack.pop()
        for b in adjA[a]:
            if mcol[a] == b:
                continue
            if not ZB[b]:
                ZB[b] = True
                a2 = mrow[b]
                if a2 >= 0 and not ZA[a2]:
                    ZA[a2] = True; stack.append(a2)
    cover = {a for a in range(na) if adjA[a] and not ZA[a]} | {na + b for b in range(nb) if ZB[b]}
    assert all(a in cover or (na + b) in cover for a, b in edges), "not a cover"
    assert len(cover) == msize, (len(cover), msize)
    return cover, msize


def cover_scan_stream(W, u, v, order_mode="index", rng=None):
    """Run the cover-directed scan with a minimum vertex cover computed by koenig_cover.

    W is the weight matrix, and u and v are the predicted row and column values, which are
    first clamped at zero. The cover is a minimum vertex cover of the constraints that the
    clamped prediction violates. For each vertex of the cover in turn, the scan takes the
    constraint at that vertex with the largest residual and, if it is violated, raises both of
    its endpoints by the residual. A tie for the largest residual goes to the lowest index.
    The argument order_mode selects the order in which the cover is processed: increasing
    vertex index by default (rows before columns), the reverse of that when order_mode is
    reverse, and a random order drawn from rng when it is random. The function asserts that
    the output is feasible. It returns the repaired u and v, the size of the cover (the vertex
    cover number of the violated graph) and the number of constraints fired.
    """
    na, nb = W.shape
    u = np.maximum(u, 0).copy(); v = np.maximum(v, 0).copy()
    R = W - u[:, None] - v[None, :]
    A_idx, B_idx = np.nonzero(R > 1e-9)
    cover, tau = koenig_cover(na, nb, list(zip(A_idx.tolist(), B_idx.tolist())))
    S = sorted(cover)
    if order_mode == "reverse":
        S = S[::-1]
    elif order_mode == "random":
        S = [S[k] for k in rng.permutation(len(S))]
    nf = 0
    for i in S:
        if i < na:
            res = W[i] - u[i] - v; j = int(np.argmax(res))
            if res[j] > 1e-9:
                u[i] += res[j]; v[j] += res[j]; nf += 1
        else:
            b = i - na; res = W[:, b] - u - v[b]; j = int(np.argmax(res))
            if res[j] > 1e-9:
                u[j] += res[j]; v[b] += res[j]; nf += 1
    assert (u[:, None] + v[None, :] >= W - 1e-9).all(), "cover-directed output infeasible"
    return u, v, tau, nf


def check_table_stream():
    section(8, "TABLE 2 STREAM: reproduction of experiments.py's pinned run + the cover-directed scan")
    t8 = time.time()
    import experiments as E
    # Each key is an instance family and a size. Each value gives the mean and the standard
    # deviation printed in Table 5 of the paper for pre, sorted, forward, random and walk, in
    # that order. The title of this section and its output call that table Table 2.
    TABLE = {
        ("uniform", 100): [(19.9, 0.3), (21.7, 1.6), (29.4, 2.9), (28.9, 2.5), (17.9, 1.3)],
        ("uniform", 200): [(39.8, 0.6), (43.4, 2.3), (54.9, 4.9), (54.7, 5.9), (32.9, 3.1)],
        ("clustered", 100): [(19.9, 0.3), (22.1, 2.3), (28.1, 2.5), (28.7, 2.7), (17.9, 1.2)],
        ("clustered", 200): [(39.7, 0.6), (47.9, 3.6), (67.2, 6.2), (64.2, 8.5), (35.3, 2.8)],
        ("euclidean", 100): [(20.0, 0.0), (23.2, 1.8), (29.4, 4.9), (27.9, 2.3), (17.9, 1.4)],
        ("euclidean", 200): [(39.8, 0.4), (48.2, 3.2), (63.8, 6.3), (66.2, 5.3), (35.5, 1.9)],
    }
    kinds, sizes, corrupt, reps, seed = ("uniform", "clustered", "euclidean"), (50, 100, 150, 200), 0.10, 10, 7
    # The generator of the pinned stream of random draws. It is used in exactly the same order
    # as in the run function of experiments.py.
    rng = np.random.default_rng(seed)
    # A second generator, used only for the random order of the cover, so that it never draws
    # from the pinned stream.
    rng_aux = np.random.default_rng(12345)
    cols = ("pre", "sorted", "forward", "random", "walk", "cov", "cov_rev", "cov_rand", "tau", "fired")
    print(f"{'instance':>10} {'n':>4} | " + " ".join(f"{c:>11}" for c in cols) + "   (mean+-std over 10 draws; "
          f"l0 against the perturbed reference optimum, as in the table)")
    checked = 0
    for kind in kinds:
        for n in sizes:
            cell = {c: [] for c in cols}
            for _ in range(reps):
                W = E.instance(kind, n, rng)
                us, vs = E.optimal_dual(W)
                k = max(1, int(corrupt * (2 * n)))
                u0, v0 = us.copy(), vs.copy()
                which = rng.choice(2 * n, size=k, replace=False)
                for i in which:
                    d = rng.choice([-1, 1]) * rng.integers(1, 20)
                    if i < n:
                        u0[i] = max(0, u0[i] + d)
                    else:
                        v0[i - n] = max(0, v0[i - n] + d)
                cell["pre"].append(E.l0(u0, v0, us, vs))
                for name, o in E.orders(W, rng).items():
                    ru, rv = E.repair(W, u0, v0, o)
                    cell[name].append(E.l0(ru, rv, us, vs))
                wu, wv = E.walk_repair(W, u0, v0)
                cell["walk"].append(E.l0(wu, wv, us, vs))
                cu, cv, tau, nf = cover_scan_stream(W, u0, v0)
                cell["cov"].append(E.l0(cu, cv, us, vs)); cell["tau"].append(tau); cell["fired"].append(nf)
                cu, cv, _, _ = cover_scan_stream(W, u0, v0, "reverse")
                cell["cov_rev"].append(E.l0(cu, cv, us, vs))
                cu, cv, _, _ = cover_scan_stream(W, u0, v0, "random", rng_aux)
                cell["cov_rand"].append(E.l0(cu, cv, us, vs))
            m = {c: float(np.mean(cell[c])) for c in cols}
            s = {c: float(np.std(cell[c])) for c in cols}
            if (kind, n) in TABLE:
                # The means are exact tenths, because each one averages ten whole numbers.
                for c, (tm, ts) in zip(cols[:5], TABLE[(kind, n)]):
                    assert abs(m[c] - tm) < 0.051 and f"{s[c]:.1f}" == f"{ts:.1f}", (kind, n, c, m[c], s[c], tm, ts)
                checked += 1
            print(f"{kind:>10} {n:>4} | " + " ".join(f"{m[c]:>6.1f}+-{s[c]:<4.1f}" for c in cols)
                  + ("   = Table 2" if (kind, n) in TABLE else ""))
    assert checked == 6
    print(f"the 6 Table 2 rows (n = 100, 200) are reproduced to the printed 0.1 in all 5 columns (means and stds);")
    print("cov / cov_rev / cov_rand: cover-directed scan on a Koenig minimum cover in index / reversed / random order;")
    print("tau = |minimum cover of G(p)|, fired = number of constraints the scan fires.")
    print(f"all assertions in [8] hold  (section [8] took {time.time() - t8:.1f}s) {elapsed()}")


# Section 1b of this program: the exhaustive search for witnesses of Theorem 16 on instances
# with 1 row and 2 or 3 columns, with weights and predictions from 0 to 4.
def _phi_D(n, cons, yhat, sgn):
    """Compute the value that the admissibility test needs for one sign vector.

    The argument sgn is a vector of signs, each plus 1 or minus 1. The result is the largest
    value, over all feasible duals y, of minus the dot product of sgn and y, minus the sum of
    the absolute differences between yhat and y. It is computed with one LP whose variables
    are y and a vector t that is at least the absolute difference between yhat and y in every
    coordinate.
    """
    A, b = lp_rows(n, cons)
    rows = [np.r_[r, np.zeros(n)] for r in A]; rhs = list(b)
    for i in range(n):
        r1 = np.zeros(2 * n); r1[i] = -1; r1[n + i] = -1; rows.append(r1); rhs.append(-yhat[i])
        r2 = np.zeros(2 * n); r2[i] = 1; r2[n + i] = -1; rows.append(r2); rhs.append(yhat[i])
    # Minimise the dot product of sgn and y plus the sum of t, which is minus the quantity to
    # maximise.
    c = np.r_[np.asarray(sgn, float), np.ones(n)]
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(0, None)] * (2 * n), method="highs")
    assert res.status == 0, res.message
    return -float(res.fun)


def _real_post(n, cons, yhat, f):
    """Return the fewest coordinates in which a real admissible point differs from f.

    Here f is the only point of the optimal face. A vector z is admissible when it is feasible
    and, for every sign vector s, the dot product of s and z plus the value of _phi_D for s is
    at most zero. Together these inequalities say that z is at most as far as yhat from every
    feasible dual, in the sum of absolute differences. The points z are real vectors, not
    only integral ones. The search goes through sets of coordinates from the largest to the
    smallest and tests by LP feasibility whether some admissible z agrees with f on the whole
    set. The result is n minus the size of the first set that works, and n if no set with at
    least one coordinate works.
    """
    A, b = lp_rows(n, cons)
    rows = [(np.array(sg, float), _phi_D(n, cons, yhat, sg)) for sg in itertools.product([-1, 1], repeat=n)]
    A_ub = np.vstack([A] + [sg[None, :] for sg, _ in rows]); b_ub = np.concatenate([b, [-ph for _, ph in rows]])
    for k in range(n, -1, -1):
        for S in itertools.combinations(range(n), k):
            if k == 0:
                return n
            bounds = [(0, None)] * n
            for i in S:
                bounds[i] = (f[i], f[i])
            res = linprog(np.zeros(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.status == 0:
                return n - k
    return n


def check_witness_search():
    section("1b", "EXHAUSTIVE SEARCH for Theorem-16 witnesses on 1x2 and 1x3 systems (w, yhat in 0..4)")
    BOX = 4
    wits = []; n_inst = n_single = n_pairs = 0
    for nB in (2, 3):
        n = 1 + nB
        P = np.array([pt for pt in itertools.product(range(BOX + 1), repeat=n)], float)
        for w in itertools.product(range(5), repeat=nB):
            W = np.array([list(w)]); cons = bip_cons(W); OPT = float(max(w)); n_inst += 1
            pts = face_points_bip(W, OPT)
            if len(pts) != 1:
                continue
            n_single += 1; f = pts[0]
            feas = np.array([is_feasible(pt, cons) for pt in P]); PD = P[feas]
            DZ = np.abs(PD[:, None, :] - PD[None, :, :]).sum(axis=2)
            dP = (np.abs(PD - f) > 1e-9).sum(axis=1)
            for yh in itertools.product(range(5), repeat=n):
                yh = np.array(yh, float)
                if is_feasible(yh, cons):
                    continue
                n_pairs += 1
                pre = int((np.abs(yh - f) > 1e-9).sum())
                dy = np.abs(PD - yh).sum(axis=1)
                # Keep the integral points z that no integral point y refutes, with both taken
                # among the feasible points of the box. A point y refutes z when z is farther
                # from y than the prediction is, in the sum of absolute differences.
                adm = (DZ <= dy[None, :] + 1e-9).all(axis=1)
                assert adm.any()
                post = int(dP[adm].min())
                if post > pre:
                    wits.append((nB, w, tuple(int(v) for v in yh), pre, post, f))
    print(f"stage 1 (integral candidates, integral refuters, box [0,{BOX}]): {n_inst} weight vectors, {n_single} with a "
          f"singleton face, {n_pairs} infeasible predictions, {len(wits)} integral witnesses")
    real = []
    for nB, w, yh, pre, post, f in wits:
        rp = _real_post(1 + nB, bip_cons(np.array([list(w)])), np.array(yh, float), f)
        real.append((nB, w, yh, pre, post, rp))
    true_w = [r for r in real if r[5] > r[3]]
    assert len(true_w) == len(wits), "an integral witness failed the exact real-z check"
    assert all(r[0] == 3 for r in true_w), "a 1x2 witness exists"
    assert all((r[3], r[5]) == (3, 4) for r in true_w), "a witness with (pre, post) != (3, 4)"
    assert all(r[5] == 1 + r[0] for r in true_w), "a non-saturating witness exists"
    ratio = max(r[5] / r[3] for r in true_w)
    g1 = any(r[1] == (1, 2, 2) and r[2] == (0, 0, 2, 2) for r in true_w)
    minimal = sorted(true_w, key=lambda r: (max(r[1]), sum(r[1]), sum(r[2])))[:3]
    assert g1 and any(r[1] == (1, 2, 2) and r[2] == (0, 0, 2, 2) for r in minimal), "G1 is not among the minimal witnesses"
    print(f"stage 2 (exact over REAL z: e(z) <= 0 as 2^n sign-vector LPs, agreement sets by LP feasibility): "
          f"{len(true_w)} witnesses confirmed, {len(real) - len(true_w)} integral-only")
    print(f"    every witness is 1x3 with (pre, post) = (3, 4) = (3, |V|): saturating, ratio {ratio:.4f}; none on 1x2; "
          f"G1 present: {g1}")
    print(f"    minimal witnesses (by max weight, total weight, total prediction): "
          + "; ".join(f"w={r[1]} yhat=({r[2][0]}|{r[2][1]},{r[2][2]},{r[2][3]})" for r in minimal))
    print("all assertions in [1b] hold", elapsed())


# Section 2b of this program: a witness on which the additive term of Theorem 19 is attained.
# The worst output of the cover-directed scan differs from the optimal face in 7 coordinates,
# which is the 4 of the prediction plus the vertex cover number 3 of the violated graph.
def check_coverscan_witness():
    section("2b", "the additive term of Theorem 19 is attained: the hill-climb witness with ratio 7/4")
    W = np.array([[3, 5, 3, 6], [0, 5, 6, 5], [6, 6, 6, 2]]); yhat = np.array([-2, 2, 4, 0, 6, 1, 1.])
    cons = bip_cons(W); n = 7; OPT = matching_value(W); pts = face_points_bip(W, OPT)
    p = clamp(yhat); pre = face_l0(yhat, pts)
    edges = sorted(violated(p, cons).keys()); tau, covers = min_vertex_covers(edges)
    assert pre == 4 and tau == 3, (pre, tau)
    worst = 0; arg = None
    for S in covers:
        for out in _cover_scan_orders(p, cons, S)[0]:
            assert is_feasible(out, cons)
            d = face_l0(out, pts)
            assert d <= pre + tau
            if d > worst:
                worst, arg = d, (S, out)
    assert worst == 7
    print(f"W = {W.tolist()}, yhat = {yhat.tolist()}: OPT = {OPT:.0f}, {len(pts)} integral face point(s), pre = {pre}, "
          f"tau = {tau}, minimum covers {covers}")
    print(f"    worst output over every minimum cover, processing order and tie-break: face-l0 {worst} = pre + tau "
          f"(cover {sorted(arg[0])}, output {[float(x) for x in arg[1]]}): ratio {worst / pre:.4f} = 7/4")
    print("all assertions in [2b] hold", elapsed())


# Section 7b of this program: non-expansion toward the optimal face alone, which means not
# increasing the distance to any point of the face, with other feasible points left aside. On
# random instances, exact LP tests look for a face point that is at most as far as the
# prediction from every face point.
def _phi_F(n, cons, OPT, yhat, sgn):
    """Compute the same value as _phi_D with y restricted to the optimal face.

    The result is the largest value, over all points y of the optimal face, fractional points
    included, of minus the dot product of sgn and y, minus the sum of the absolute differences
    between yhat and y. A point of the face is a feasible dual whose values add up to OPT.
    """
    A, b = lp_rows(n, cons)
    rows = [np.r_[r, np.zeros(n)] for r in A]; rhs = list(b)
    for i in range(n):
        r1 = np.zeros(2 * n); r1[i] = -1; r1[n + i] = -1; rows.append(r1); rhs.append(-yhat[i])
        r2 = np.zeros(2 * n); r2[i] = 1; r2[n + i] = -1; rows.append(r2); rhs.append(yhat[i])
    A_eq = np.zeros((1, 2 * n)); A_eq[0, :n] = 1
    c = np.r_[np.asarray(sgn, float), np.ones(n)]
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=A_eq, b_eq=[OPT],
                  bounds=[(0, None)] * (2 * n), method="highs")
    assert res.status == 0, res.message
    return -float(res.fun)


def _face_point_admissible_toward_face(n, cons, OPT, yhat):
    """Tell whether some face point is at most as far as yhat from every point of the face.

    Distances are sums of absolute differences, and the face is the whole optimal face,
    fractional points included. The test is exact: one LP looks for a point z of the face such
    that, for every sign vector s, the dot product of s and z plus the value of _phi_F for s
    is at most zero.
    """
    A, b = lp_rows(n, cons)
    rows = [(np.array(sg, float), _phi_F(n, cons, OPT, yhat, sg)) for sg in itertools.product([-1, 1], repeat=n)]
    A_ub = np.vstack([A] + [sg[None, :] for sg, _ in rows]); b_ub = np.concatenate([b, [-ph for _, ph in rows]])
    res = linprog(np.zeros(n), A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[OPT],
                  bounds=[(0, None)] * n, method="highs")
    return res.status == 0


def check_face_only_admissibility():
    section("7b", "non-expansion toward the optimal face ALONE: a face point admissible toward every face point (exact LP tests)")
    rng = np.random.default_rng(11)
    ok = tested = 0; shapes = {}
    for batch, (shape_choices, wmax, lo_extra, hi_extra, count) in enumerate(
            [(((2, 2), (2, 3), (3, 2), (3, 3)), 4, 0, 2, 300),
             (((2, 3), (3, 3), (3, 4), (4, 3)), 6, -2, 4, 300)]):
        got = 0
        while got < count:
            na, nb = shape_choices[int(rng.integers(len(shape_choices)))]
            W = rng.integers(0, wmax + 1, size=(na, nb)); cons = bip_cons(W); n = na + nb
            OPT = matching_value(W); pts = face_points_bip(W, OPT)
            if len(pts) < 2:
                continue
            M = max(int(W.max()), 1)
            yhat = rng.integers(lo_extra, M + hi_extra, size=n).astype(float)
            if is_feasible(yhat, cons) and abs(yhat.sum() - OPT) < 1e-9:
                continue
            got += 1; tested += 1; shapes[(na, nb)] = shapes.get((na, nb), 0) + 1
            if _face_point_admissible_toward_face(n, cons, OPT, yhat):
                ok += 1
    assert ok == tested
    print(f"{tested} random instances with a non-singleton face ({dict(sorted(shapes.items()))}; two batches: w in 0..4 with "
          f"yhat in 0..max w + 1, and w in 0..6 with yhat in -2..max w + 3, clamping exercised)")
    print(f"    a face point z with ||z - y*||_1 <= ||yhat - y*||_1 for every y* in the FRACTIONAL face exists on {ok}/{tested}")
    print("    (an observation, not a theorem: the paper reports it as such)")
    print("all assertions in [7b] hold", elapsed())


# Section 2c of this program: the packing setting of Chen et al. On random instances, and for
# every minimum cover, processing order and tie, the mirror of the cover-directed scan adds at
# most the size of the cover to the number of coordinates that differ from the optimal face.
def _pack_face_l0(n, cons, OPT, z):
    """Return the exact smallest number of coordinates in which z differs from a face point.

    The system is a packing system: the two dual values of every edge add up to at most the
    edge cost, and the optimal face is the set of feasible duals whose values add up to OPT.
    The dual values have no sign constraint. The face is unbounded and may contain fractional
    points. The search goes through sets of coordinates from the largest to the smallest and
    tests by LP feasibility whether some face point agrees with z on the whole set. The
    result is n minus the size of the first set that works.
    """
    A = np.zeros((len(cons), n)); b = np.zeros(len(cons))
    for k, (i, j, c) in enumerate(cons):
        A[k, i] = 1; A[k, j] = 1; b[k] = c
    z = np.asarray(z, float)
    for k in range(n, -1, -1):
        for S in itertools.combinations(range(n), k):
            if any(i in S and j in S and z[i] + z[j] > c + TOL for i, j, c in cons):
                continue
            bounds = [(None, None)] * n
            for i in S:
                bounds[i] = (z[i], z[i])
            res = linprog(np.zeros(n), A_ub=A, b_ub=b, A_eq=np.ones((1, n)), b_eq=[OPT],
                          bounds=bounds, method="highs")
            if res.status == 0:
                return n - k
    return n


def _pack_scan_outputs(p, cons, S):
    """Enumerate the outputs of the packing mirror of the cover-directed scan.

    The scan starts from p and takes the vertices of the cover S one at a time. At each vertex
    it picks the constraint there that is currently most violated, if any, and lowers both of
    its endpoints by the full excess, where the covering scan raises them. The excess of a
    constraint is the sum of its two dual values minus its cost. Every order of S and every
    tie among the largest excesses is followed. The result is a set of pairs, each made of an
    output and the number of constraints fired on the way to it.
    """
    outs = set()
    def run(y, remaining, fired):
        if not remaining:
            outs.add((tuple(np.round(y, 9)), fired)); return
        i = remaining[0]
        vio = [(y[a] + y[b] - c, a, b) for a, b, c in cons if (a == i or b == i) and y[a] + y[b] > c + TOL]
        if not vio:
            run(y, remaining[1:], fired); return
        emax = max(v[0] for v in vio)
        for e, a, b in vio:
            if abs(e - emax) < TOL:
                z = y.copy(); z[a] -= e; z[b] -= e
                run(z, remaining[1:], fired + 1)
    for perm in itertools.permutations(sorted(S)):
        run(p.copy(), list(perm), 0)
    return outs


def check_packing_mirror():
    section("2c", "PACKING ORIENTATION (Chen et al.'s setting): the cover-directed mirror scan obeys post <= pre + |S|")
    rng = np.random.default_rng(5)
    inst = outs = 0; ratio_max = 0.0; exc_max = 0.0; frac_opt = 0
    while inst < 200:
        n_side = int(rng.integers(2, 4)); C = rng.integers(0, 6, size=(n_side, n_side))
        n = 2 * n_side
        cons = [(a, n_side + b, int(C[a, b])) for a in range(n_side) for b in range(n_side)]
        # The values may have any sign, because the packing setting has no clamp.
        yhat = rng.integers(-2, 8, size=n).astype(float)
        edges = sorted((a, b) for a, b, c in cons if yhat[a] + yhat[b] > c + TOL)
        if not edges:
            continue
        inst += 1
        # OPT is the cost of a minimum-cost perfect matching.
        r, cidx = linear_sum_assignment(C); OPT = float(C[r, cidx].sum())
        A = np.zeros((len(cons), n)); bb = np.zeros(len(cons))
        for k, (i, j, c) in enumerate(cons):
            A[k, i] = 1; A[k, j] = 1; bb[k] = c
        res = linprog(-np.ones(n), A_ub=A, b_ub=bb, bounds=[(None, None)] * n, method="highs")
        # The LP maximum equals the value of the assignment.
        assert res.status == 0 and abs(-res.fun - OPT) < 1e-7
        pre = _pack_face_l0(n, cons, OPT, yhat)
        tau, covers = min_vertex_covers(edges)
        for S in covers:
            for out, fired in _pack_scan_outputs(yhat, cons, S):
                out = np.array(out); outs += 1
                # The output is feasible.
                assert all(out[a] + out[b] <= c + TOL for a, b, c in cons)
                # The output is integral.
                assert np.allclose(out, np.round(out))
                post = _pack_face_l0(n, cons, OPT, out)
                assert post <= pre + fired <= pre + len(S) == pre + tau, (C.tolist(), yhat.tolist(), pre, post, tau)
                if pre > 0:
                    ratio_max = max(ratio_max, post / pre)
                exc_max = max(exc_max, (post - pre) / tau)
    print(f"{inst} random square packing instances (n = 2, 3 per side, c in 0..5, yhat in -2..7 of any sign, no clamp), "
          f"{outs} outputs over ALL minimum covers, processing orders and ties among largest excesses:")
    print(f"    every output feasible and integral; face-l0 to the UNBOUNDED fractional face by subset LPs; "
          f"post <= pre + #fired <= pre + tau on every output; max post/pre = {ratio_max:.4f}, max (post-pre)/tau = {exc_max:.4f}")
    print("all assertions in [2c] hold", elapsed())


# Section 9 of this program: the cover-directed scan as a warm start for the solver, on the
# same draws as Table 6 of the paper.
def check_solver_stream():
    section(9, "TABLE 3 STREAM: the cover-directed scan as a warm start for the solver (same draws as tab:solver)")
    import experiments as E
    import experiments_itcs as EI
    t9 = time.time()
    # The means and standard deviations of Table 6 for the columns cold, sorted and walk,
    # kept here to check the reproduction. Here cold is the start that Table 6 lists as the
    # row-reduction start. It ignores the prediction: each row gets its largest weight and each
    # column gets zero.
    TABLE = {
        ("uniform", 100): ((48.6, 4.9), (52.7, 6.4), (42.9, 5.6)),
        ("uniform", 200): ((42.7, 3.4), (72.6, 8.6), (56.3, 10.4)),
        ("clustered", 100): ((131.1, 29.4), (56.9, 10.6), (41.9, 8.7)),
        ("clustered", 200): ((367.2, 83.1), (120.3, 13.5), (95.6, 10.7)),
        ("euclidean", 100): ((143.1, 14.2), (65.7, 10.0), (50.1, 7.9)),
        ("euclidean", 200): ((223.8, 26.2), (115.5, 15.3), (95.0, 10.5))}
    rng = np.random.default_rng(17)
    print(f"{'instance':>10} {'n':>4} | {'cold':>12} {'sorted':>12} {'walk':>12} {'cov':>12} {'shift':>12}   (dual adjustments, mean+-std over 10 draws)")
    checked = 0; between = 0; cells = 0; beats_sorted = beats_walk = draws = 0
    for kind in ("uniform", "clustered", "euclidean"):
        for n in (50, 100, 150, 200):
            acc = {k: [] for k in ("cold", "sorted", "walk", "cov", "shift")}
            for _ in range(10):
                W = E.instance(kind, n, rng)
                us, vs = E.optimal_dual(W)
                u0, v0 = EI.corrupt(us, vs, n, n, rng)
                # This call draws from the stream exactly as the measure function of
                # experiments_itcs.py does.
                reps = EI.all_repairs(W, u0, v0, rng)
                cu, cv, tau, nf = cover_scan_stream(W, u0, v0)
                for name, (ru, rv) in (("sorted", reps["sorted"]), ("walk", reps["walk"]),
                                       ("shift", reps["shift"]), ("cov", (cu, cv)),
                                       ("cold", (W.max(axis=1).astype(float), np.zeros(n)))):
                    _, adj, ph = solve_warm(W, ru, rv)
                    assert ph == n
                    acc[name].append(adj)
                draws += 1
                beats_sorted += acc["cov"][-1] < acc["sorted"][-1]
                beats_walk += acc["cov"][-1] < acc["walk"][-1]
            m = {k: float(np.mean(v)) for k, v in acc.items()}; sd = {k: float(np.std(v)) for k, v in acc.items()}
            if (kind, n) in TABLE:
                for k, (tm, ts) in zip(("cold", "sorted", "walk"), TABLE[(kind, n)]):
                    assert abs(m[k] - tm) < 0.051 and f"{sd[k]:.1f}" == f"{ts:.1f}", (kind, n, k, m[k], sd[k], tm, ts)
                checked += 1
            cells += 1
            between += min(m["sorted"], m["walk"]) <= m["cov"] <= max(m["sorted"], m["walk"])
            print(f"{kind:>10} {n:>4} | " + " ".join(f"{m[k]:>7.1f}+-{sd[k]:<4.1f}" for k in ("cold", "sorted", "walk", "cov", "shift"))
                  + ("   = Table 3" if (kind, n) in TABLE else ""))
    assert checked == 6
    print(f"the six Table 3 cells at n = 100, 200 are reproduced (cold, sorted, walk) to the printed 0.1;")
    print(f"cover-directed ('cov') cell mean lies between the sorted scan's and the walk's in {between}/{cells} cells; "
          f"per draw it beats sorted in {beats_sorted}/{draws} and beats the walk in {beats_walk}/{draws}")
    print(f"all assertions in [9] hold  (section [9] took {time.time() - t9:.1f}s)", elapsed())


# Section 10 of this program: checks that three halves is the best factor that a repair in the
# class can guarantee between the face distance of its output and that of the prediction. The
# face distance counts the coordinates that differ from the nearest optimal dual. The path family
# gives the lower bound. The upper bound says that some admissible output is at face distance at
# most r plus half of r, rounded down, where r is the face distance of the prediction.
def _admissible_polytope_rows(n, cons, p):
    """Build the sign-vector inequalities that describe the admissible points for p.

    The argument p is the clamped prediction. The result has one pair for every sign vector s
    of length n: s itself and the value of _phi_D for s. A point z is admissible exactly when
    it is feasible and, for every pair, the dot product of s and z plus that value is at most
    zero. The largest of these sums is the largest value, over all feasible duals y, of the
    distance from z to y minus the distance from p to y, with distances taken as sums of
    absolute differences.
    """
    return [(np.array(sg, float), _phi_D(n, cons, p, sg)) for sg in itertools.product([-1, 1], repeat=n)]


def _min_admissible_d0(n, cons, OPT, p, rows):
    """Return the fewest coordinates in which an admissible point can differ from the face.

    The minimum is exact. It runs over all real admissible vectors z, not only integral ones,
    and over all points of the optimal face. The argument rows is the list built by
    _admissible_polytope_rows. The search goes through sets of coordinates from the largest
    to the smallest. For each set, one LP looks for an admissible z and a face point y that
    agree on the whole set. The result is n minus the size of the first set for which the LP
    is feasible.
    """
    A, b = lp_rows(n, cons)
    Az = np.vstack([A] + [sg[None, :] for sg, _ in rows]); bz = np.concatenate([b, [-ph for _, ph in rows]])
    A_ub = np.block([[Az, np.zeros((Az.shape[0], n))], [np.zeros((A.shape[0], n)), A]])
    b_ub = np.concatenate([bz, b])
    for k in range(n, -1, -1):
        for S in itertools.combinations(range(n), k):
            A_eq = np.zeros((1 + k, 2 * n)); b_eq = np.zeros(1 + k)
            A_eq[0, n:] = 1; b_eq[0] = OPT
            for r, i in enumerate(S):
                A_eq[1 + r, i] = 1; A_eq[1 + r, n + i] = -1
            res = linprog(np.zeros(2 * n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=[(0, None)] * (2 * n), method="highs")
            if res.status == 0:
                return n - k
    return n


def _two_round_construction(W, yhat, f):
    """Build the admissible output used in the proof of the upper bound of three halves.

    Here f is an optimal dual. A coordinate is low when the clamped prediction is below f
    there. Starting from the clamped prediction, round one takes each low coordinate in turn
    and fires its edge in a maximum-weight matching, if it has one and that edge is violated.
    Round two takes each low coordinate that is still below f and fires its violated
    constraint with the largest residual, if it has one. Firing a constraint raises both of
    its endpoints by its residual. The result is the dual after both rounds.
    """
    W = np.asarray(W, float); na, nb = W.shape; n = na + nb
    cons = bip_cons(W)
    p = clamp(yhat); z = p.copy()
    low = [i for i in range(n) if p[i] < f[i] - TOL]
    r, c = linear_sum_assignment(-W); partner = {}
    for a, bb in zip(r, c):
        partner[a] = na + bb; partner[na + bb] = a
    def fire(i, j):
        w = W[i, j - na] if i < na else W[j, i - na]
        res = w - z[i] - z[j]
        if res > TOL:
            z[i] += res; z[j] += res
    for v in low:
        if v in partner:
            fire(v, partner[v])
    for v in low:
        if z[v] < f[v] - TOL:
            best = None
            for i, j, w in cons:
                if (i == v or j == v) and z[i] + z[j] < w - TOL and (best is None or w - z[i] - z[j] > best[0]):
                    best = (w - z[i] - z[j], i, j)
            if best:
                fire(best[1], best[2])
    return z


def check_three_halves():
    section(10, "THE CLASS CONSTANT IS 3/2: the path family P_k (lower bound) and r + floor(r/2) (upper bound)")
    # Part (b) of Theorem 38, the lower bound: the path family for k from 1 to 4. The certificate
    # uses only a set of feasible points, 2 to the power k of them, and so does not depend on the
    # full admissibility test.
    for k in (1, 2, 3, 4):
        Wk, yhat, f = _path_family(k)
        n = 3 * k + 1; cons = bip_cons(Wk); OPT = matching_value(Wk); A, b = lp_rows(n, cons)
        assert OPT == 2 * k
        ext = face_extremes(n, cons, OPT)
        assert all(abs(lo - fi) < 1e-9 and abs(hi - fi) < 1e-9 for (lo, hi), fi in zip(ext, f)), "face not the single point f"
        pre = int((np.abs(yhat - f) > TOL).sum()); assert pre == 2 * k + 1
        # There is one point for every way of choosing, for each row, either the row itself or
        # the column with the same index: the point is the prediction with the chosen
        # coordinates raised by 1. The polytope tested is the set of feasible z whose distance
        # to every one of these points, in the sum of absolute differences, is at most k. The
        # checks are that every z in it has the value 2 on every column after the first k, and
        # that each row has the same value as the column with the same index, a value between
        # one half and 1.
        Y = []
        for x in itertools.product([0, 1], repeat=k):
            y = yhat.copy()
            for i, xi in enumerate(x):
                y[i if xi == 0 else k + i] += 1
            assert is_feasible(y, cons) and abs(l1(y, yhat) - k) < 1e-9
            Y.append(y)
        m = len(Y); nv = n + m * n
        rows = [np.r_[r_, np.zeros(m * n)] for r_ in A]; rhs = list(b)
        for q, y in enumerate(Y):
            base = n + q * n
            for i in range(n):
                r1 = np.zeros(nv); r1[i] = 1; r1[base + i] = -1; rows.append(r1); rhs.append(y[i])
                r2 = np.zeros(nv); r2[i] = -1; r2[base + i] = -1; rows.append(r2); rhs.append(-y[i])
            r3 = np.zeros(nv); r3[base:base + n] = 1; rows.append(r3); rhs.append(k)
        A_ub = np.array(rows); b_ub = np.array(rhs); bounds = [(0, None)] * nv
        def rng_of(cvec):
            cc = np.r_[cvec, np.zeros(m * n)]
            lo = linprog(cc, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            hi = linprog(-cc, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            assert lo.status == 0 and hi.status == 0
            return lo.fun, -hi.fun
        for j in range(k + 1):
            cvec = np.zeros(n); cvec[2 * k + j] = 1; lo, hi = rng_of(cvec); assert abs(lo - 2) < 1e-7 and abs(hi - 2) < 1e-7
        for i in range(k):
            cvec = np.zeros(n); cvec[i] = 1; cvec[k + i] = -1; lo, hi = rng_of(cvec); assert abs(lo) < 1e-7 and abs(hi) < 1e-7
            cvec = np.zeros(n); cvec[i] = 1; lo, hi = rng_of(cvec); assert abs(lo - 0.5) < 1e-7 and abs(hi - 1) < 1e-7
        line = f"P_{k}: |A| = {k}, |B| = {2*k+1}, OPT = {OPT:.0f}, face = single point f, pre = {pre}; every z in D within l1 distance {k} of all {m} points y_x has c_j = 2, a_i = b_i in [1/2, 1]: post = {3*k+1} = pre + floor(pre/2), ratio {(3*k+1)/(2*k+1):.4f}"
        # Cross-check with the full admissibility test over every sign vector, which the output
        # calls the oracle.
        if k <= 3:
            p = clamp(yhat); rows_e = _admissible_polytope_rows(n, cons, p)
            post = _min_admissible_d0(n, cons, OPT, p, rows_e)
            assert post == 3 * k + 1
            line += "  [e(z)-oracle agrees]"
        print("    " + line)
    # Part (a) of Theorem 38, the upper bound. It is checked exhaustively on the instances with
    # 1 row and 2 or 3 columns, for every sorted list of weights from 0 to 4 and every prediction
    # with values from 0 to 4, and then on random instances whose prediction is near the optimal
    # face.
    rng = np.random.default_rng(32)
    viol = 0; tested = 0; worst = (0, None); constr_fail = 0
    def run_case(W, yhat):
        nonlocal viol, tested, worst, constr_fail
        n, cons = len(W) + len(W[0]), bip_cons(W); p = clamp(yhat)
        if is_feasible(p, cons):
            return
        OPT = matching_value(W); pts = face_points_bip(W, OPT)
        r = face_l0(yhat, pts)
        if r == 0:
            return
        rows_e = _admissible_polytope_rows(n, cons, p)
        post = _min_admissible_d0(n, cons, OPT, p, rows_e)
        tested += 1
        if post > r + r // 2:
            viol += 1
        if post / r > worst[0]:
            worst = (post / r, (np.asarray(W).tolist(), list(map(float, yhat)), r, post))
        # The two-round construction, with f the integral face point that differs from the
        # prediction in the fewest coordinates.
        d = (np.abs(pts - yhat) > 1e-6).sum(axis=1); f = pts[int(np.argmin(d))]
        z = _two_round_construction(W, yhat, f)
        Cset = int((np.abs(p - f) > TOL).sum()); lo = int((p < f - TOL).sum()); hi = int((p > f + TOL).sum())
        ok = is_feasible(z, cons) and max(float(sg @ z) + ph for sg, ph in rows_e) <= 1e-7 \
            and int((np.abs(z - f) > 1e-7).sum()) <= Cset + hi
        constr_fail += (not ok)
    for nb in (2, 3):
        for w in itertools.combinations_with_replacement(range(5), nb):
            for yh in itertools.product(range(5), repeat=1 + nb):
                run_case(np.array([list(w)]), np.array(yh, float))
    ex = tested
    while tested < ex + 250:
        na, nb = int(rng.integers(1, 4)), int(rng.integers(2, 5))
        if na + nb > 7:
            continue
        W = rng.integers(0, 7, size=(na, nb)); OPT = matching_value(W); pts = face_points_bip(W, OPT)
        if not len(pts):
            continue
        f = pts[int(rng.integers(len(pts)))]; n = na + nb
        yhat = f.copy(); idx = rng.permutation(n)
        for i in idx[:int(rng.integers(1, 3))]:
            if f[i] > 0: yhat[i] = max(0.0, f[i] - float(rng.integers(1, int(f[i]) + 1)))
        for i in idx[2:2 + int(rng.integers(0, 3))]:
            yhat[i] = f[i] + float(rng.integers(1, 4))
        run_case(W, yhat)
    assert viol == 0 and constr_fail == 0
    print(f"    upper bound: on {ex} exhaustive 1x2/1x3 instances (weights and predictions 0..4) and {tested - ex} random near-face "
          f"instances (n <= 7), min admissible post <= r + floor(r/2) always; the two-round construction was feasible, admissible "
          f"and within |supp(p-f)| + |high| of f on every instance; largest ratio seen {worst[0]:.4f} at W = {worst[1][0]}, yhat = {worst[1][1]}")
    print("all assertions in [10] hold", elapsed())


# Section 11 of this program: the chain family for Theorem 19. For k from 2 to 8, the
# cover-directed scan over the column cover, taken in index order, ends at distance 2 times k
# plus 1 from the optimal face, while the prediction is at distance k plus 1. Distances count
# the coordinates that differ from the nearest optimal dual.
def check_chain_family():
    section(11, "THEOREM 19 IS TIGHT FOR THE COVER-DIRECTED SCAN: the chain family C_k, ratio (2k+1)/(k+1) -> 2")
    for k in (2, 3, 4, 5, 6, 7, 8):
        Wk, yhat, f = _chain_family(k)
        n = 2 * k + 1; cons = bip_cons(Wk); OPT = matching_value(Wk)
        assert is_feasible(f, cons) and abs(f.sum() - OPT) < 1e-9
        ext = face_extremes(n, cons, OPT)
        assert all(abs(lo - fi) < 1e-7 and abs(hi - fi) < 1e-7 for (lo, hi), fi in zip(ext, f)), "face not the single point f"
        pre = int((np.abs(yhat - f) > TOL).sum()); assert pre == k + 1
        p = clamp(yhat)
        edges = sorted(violated(p, cons).keys()); assert len(edges) == 2 * k
        tau, covers = min_vertex_covers(edges)
        assert tau == k and frozenset(range(k, 2 * k)) in covers
        z = p.copy(); ties = 0
        # Run the cover-directed scan on the column cover in index order. The cover is the first k
        # columns, and each one fires its violated constraint with the largest residual.
        for col in range(k, 2 * k):
            vio = [(w - z[i] - z[j], i, j) for i, j, w in cons if j == col and z[i] + z[j] < w - TOL]
            r = max(v[0] for v in vio); cand = [v for v in vio if abs(v[0] - r) < TOL]; ties += len(cand) - 1
            _, i, j = cand[0]; z[i] += r; z[j] += r
        assert ties == 0 and is_feasible(z, cons)
        post = int((np.abs(z - f) > 1e-9).sum()); assert post == 2 * k + 1
        line = f"C_{k}: |A| = {k}, |B| = {k+1}, OPT = {OPT:.0f}, face = single point, pre = {pre}, tau = {tau}; column cover in index order, no ties: post = {post} = pre + tau, ratio {post/pre:.4f} = {2*k+1}/{k+1}"
        if k <= 4:
            rows_e = _admissible_polytope_rows(n, cons, p)
            best = _min_admissible_d0(n, cons, OPT, p, rows_e); assert best == pre
            line += f"; best admissible output at {best} = pre"
        print("    " + line)
    print("all assertions in [11] hold", elapsed())


# Section 12 of this program: the cold-start envelope. A cold start runs the sorted scan from the
# all-zero prediction. The envelope is the largest distance to the optimal face at which such a run
# can end, as a function of the smaller side. The witnesses below end beyond twice the smaller
# side.
def _cold_outputs_all(W):
    """Return every distinct output of the sorted scan started from the all-zero prediction.

    Every weakly descending order of the pairs is covered, that is, every way to order pairs of
    equal weight among themselves. Within a class of pairs of equal weight, the pairs that fire are
    exactly a maximal matching on the vertices that are still at zero, because a firing in a cold
    start needs both endpoints at zero. Pairs of weight zero never fire. The result is a sorted
    list of tuples, one for each distinct output.
    """
    na, nb = W.shape
    groups = {}
    for a in range(na):
        for b in range(nb):
            if W[a, b] > 0: groups.setdefault(int(W[a, b]), []).append((a, na + b))
    outs = set()
    def rec(ki, keys, z, free):
        if ki == len(keys): outs.add(tuple(z)); return
        E = [(a, b) for a, b in groups[keys[ki]] if a in free and b in free]
        for Mt in maximal_matchings(E) if E else [()]:
            z2 = list(z); fr = set(free)
            for a, b in Mt: z2[a] = z2[b] = keys[ki]; fr -= {a, b}
            rec(ki + 1, keys, z2, fr)
    rec(0, sorted(groups, reverse=True), [0] * (na + nb), set(range(na + nb)))
    return sorted(outs)


def _cold_lex(W):
    """Run the paper's canonical cold scan and return its output.

    The scan starts from the all-zero prediction. The pairs of positive weight are visited in order
    of decreasing weight, with ties broken by row index and then by column index. A pair fires when
    both of its endpoints are still at zero, and both endpoints then take the weight of the pair.
    The comments and the output call this tie-break lexicographic.
    """
    na, nb = W.shape; z = np.zeros(na + nb)
    for a, b in sorted(((a, b) for a in range(na) for b in range(nb) if W[a, b] > 0), key=lambda e: (-W[e[0], e[1]], e[0], e[1])):
        if z[a] == 0 and z[na + b] == 0: z[a] = z[na + b] = W[a, b]
    return z


def _direct_sum(W1, W2):
    S = np.zeros((W1.shape[0] + W2.shape[0], W1.shape[1] + W2.shape[1]), dtype=int)
    S[:W1.shape[0], :W1.shape[1]] = W1; S[W1.shape[0]:, W1.shape[1]:] = W2; return S


def check_cold_envelope():
    section(12, "THE COLD-START ENVELOPE IS NOT 2 min(|A|,|B|): witnesses, the 3 min bound, direct sums")
    import experiments as E          # Supplies an optimal dual for part (a).
    # The parts (a), (b) and (c) below are the parts of Proposition 26 of the paper.
    # Part (b): the 4 by 5 witness, whose cold output does not depend on the tie-break. Its optimal
    # face is a single point, the cold scan has one output under every weakly descending order, and
    # that output differs from the optimal dual in every coordinate, so its distance is the number
    # of rows plus the number of columns.
    Wr = COLD_4X5
    fr =np.array([10, 9, 10, 7, 2, 1, 0, 2, 0], float)
    na, nb = Wr.shape; n = na + nb; cons = bip_cons(Wr); OPT = matching_value(Wr)
    assert abs(OPT - 41) < 1e-9 and is_feasible(fr, cons) and abs(fr.sum() - OPT) < 1e-9
    ext = face_extremes(n, cons, OPT)
    assert all(abs(lo - fi) < 1e-7 and abs(hi - fi) < 1e-7 for (lo, hi), fi in zip(ext, fr)), "face not the single point f"
    outs = _cold_outputs_all(Wr); assert len(outs) == 1, len(outs)
    q = np.array(outs[0], float); assert np.array_equal(q, [12, 8, 12, 1, 12, 0, 8, 12, 1]), q
    assert np.array_equal(_cold_lex(Wr), q)
    fired = [(a, b) for a in range(na) for b in range(nb) if q[a] == q[na + b] == Wr[a, b] > 0]
    d = int((np.abs(q - fr) > 1e-9).sum()); assert d == 9 == na + nb and face_l0_subset_lp(n, cons, OPT, q) == 9
    n_orders = n_monotone_orders(Wr[Wr > 0])
    print(f"    robust 4x5 witness: OPT = {OPT:.0f}, face = single point, {n_orders} weakly-descending orders of the positive weights,"
          f" {len(outs)} distinct output, fired matching {fired} (weight {sum(Wr[a,b] for a,b in fired)}), d0 = {d} = |A|+|B| > 2min = {2*min(na,nb)}")
    # Part (c): direct sums. The optimal face, the scan and the distance all split over the blocks
    # of a direct sum. On the direct sum of two copies of the witness, the code checks three
    # consequences: the optimal face is the single point made of two copies of the optimal dual,
    # the cold scan has one output under every weakly descending order, and that output differs
    # from this point in all 18 coordinates.
    S2 = _direct_sum(Wr, Wr); n2 = S2.shape[0] + S2.shape[1]; cons2 = bip_cons(S2); OPT2 = matching_value(S2)
    f2 = np.r_[fr[:na], fr[:na], fr[na:], fr[na:]]
    ext2 = face_extremes(n2, cons2, OPT2)
    assert abs(OPT2 - 82) < 1e-9 and all(abs(lo - fi) < 1e-7 and abs(hi - fi) < 1e-7 for (lo, hi), fi in zip(ext2, f2))
    outs2 = _cold_outputs_all(S2); assert len(outs2) == 1
    d2 = int((np.abs(np.array(outs2[0]) - f2) > 1e-9).sum()); assert d2 == 18
    print(f"    direct sum of two copies (8x10): face = single point, one output under every order, d0 = {d2} = |A|+|B| against 2min = 16")
    # The 5 by 8 witness with a unique optimal dual. Under the lexicographic tie-break its cold
    # output differs from that dual in 11 coordinates, which is 2.2 times the smaller side. Its
    # double, the direct sum of two copies, is checked after it.
    W1 = np.array([[3, 0, 0, 0, 4, 7, 3, 0], [1, 0, 0, 0, 0, 6, 5, 1], [1, 0, 0, 0, 5, 0, 7, 3], [0, 0, 0, 1, 0, 5, 5, 5], [0, 3, 0, 1, 0, 7, 0, 6]])
    f1 = np.array([3, 2, 4, 2, 3, 0, 0, 0, 0, 1, 4, 3, 3], float)
    na1, nb1 = W1.shape; n1 = na1 + nb1; cons1 = bip_cons(W1); OPT1 = matching_value(W1)
    assert abs(OPT1 - 25) < 1e-9 and is_feasible(f1, cons1) and abs(f1.sum() - OPT1) < 1e-9
    ext1 = face_extremes(n1, cons1, OPT1)
    assert all(abs(lo - fi) < 1e-7 and abs(hi - fi) < 1e-7 for (lo, hi), fi in zip(ext1, f1)), "face not the single point f1"
    q1 = _cold_lex(W1); d1 = int((np.abs(q1 - f1) > 1e-9).sum()); assert d1 == 11 and face_l0_subset_lp(n1, cons1, OPT1, q1) == 11
    ds1 = sorted(int((np.abs(np.array(o) - f1) > 1e-9).sum()) for o in _cold_outputs_all(W1))
    print(f"    5x8 witness (weights <= 7): OPT = {OPT1:.0f}, face = single point, lexicographic output d0 = {d1} = 2.2 min; over all orders d0 in {ds1}")
    S1 = _direct_sum(W1, W1); q11 = _cold_lex(S1); f11 = np.r_[f1[:na1], f1[:na1], f1[na1:], f1[na1:]]
    ext11 = face_extremes(2 * n1, bip_cons(S1), matching_value(S1))
    assert all(abs(lo - fi) < 1e-7 and abs(hi - fi) < 1e-7 for (lo, hi), fi in zip(ext11, f11)) and int((np.abs(q11 - f11) > 1e-9).sum()) == 22
    print(f"    its direct sum (10x16): d0 = 22 = 2.2 min(|A|,|B|)")
    # The witness with the smallest weights: a 4 by 5 instance with weights at most 6, under the
    # lexicographic tie-break. Its cold output differs from the unique optimal dual in 9
    # coordinates, which is more than twice the smaller side.
    Wl = np.array([[5, 0, 0, 0, 6], [0, 5, 0, 2, 5], [5, 6, 4, 0, 0], [0, 4, 1, 2, 0]]); fl = np.array([4, 3, 4, 2, 1, 2, 0, 0, 2], float)
    consl = bip_cons(Wl); OPTl = matching_value(Wl); assert abs(OPTl - 18) < 1e-9 and is_feasible(fl, consl) and abs(fl.sum() - OPTl) < 1e-9
    assert all(abs(lo - fi) < 1e-7 and abs(hi - fi) < 1e-7 for (lo, hi), fi in zip(face_extremes(9, consl, OPTl), fl))
    ql = _cold_lex(Wl); dl = int((np.abs(ql - fl) > 1e-9).sum()); assert dl == 9
    dsl = sorted(set(int((np.abs(np.array(o) - fl) > 1e-9).sum()) for o in _cold_outputs_all(Wl)))
    print(f"    4x5 witness with weights <= 6: face = single point, lexicographic output d0 = {dl} > 8; over all orders d0 in {dsl}")
    # The 6 by 8 instance. Under the lexicographic tie-break the distance of its cold output
    # exceeds twice the smaller side by 2.
    W6 = COLD_6X8
    q6 = _cold_lex(W6); d6 = face_l0_subset_lp(14, bip_cons(W6), matching_value(W6), q6); assert d6 == 14
    print(f"    6x8 instance: lexicographic output at d0 = {d6} = |A|+|B| = 2min + 2, ratio d0/min = {d6/6:.4f} = 7/3")
    # The direct sum of copies of this instance holds the record in part (c) of Proposition 26.
    # For the sum of two copies, the subset LP of face_l0_subset_lp is not solved again on 28
    # coordinates. Instead the code checks the two structural facts that the decomposition argument
    # in the proof of part (c) uses. First, the optimal value is additive over the blocks, which
    # are separated by zeros. Second, the lexicographic cold scan of the sum is the concatenation
    # of the lexicographic cold outputs of the blocks, because inside a block the global index
    # order gives the tie-break of that block. With the decomposition of the face from part (c),
    # this gives a distance of 14 times k against a smaller side of 6 times k, that is, a ratio of
    # seven thirds, for every number k of copies.
    S6 = _direct_sum(W6, W6)
    assert abs(matching_value(S6) - 2 * matching_value(W6)) < 1e-9
    q66 = _cold_lex(S6); assert np.allclose(q66, np.r_[q6[:6], q6[:6], q6[6:], q6[6:]])
    print(f"    its direct sum (12x16): OPT additive, cold output = the concatenation, so d0 = 28 = |A|+|B| and the ratio 7/3 persists (record for (c))")
    # Part (a): the bound, checked on the four witness instances above and on 400 random instances.
    # The number of coordinates in which a cold output differs from an optimal dual is at most the
    # number of positive coordinates of that dual plus the number of fired pairs, and this sum is
    # at most 3 times the smaller side. A matrix with at most 20 cells is checked over every
    # tie-break, and a larger one on the lexicographic output only.
    rng = np.random.default_rng(12); worst = 0.0; N = 0
    for W in [Wr, W1, Wl, W6] + [rng.integers(0, 13, size=(rng.integers(2, 6), rng.integers(2, 7))) for _ in range(400)]:
        a_, b_ = W.shape; q = _cold_lex(W); f = np.concatenate(E.optimal_dual(W)); supp = int((f > 1e-9).sum())
        for o in (_cold_outputs_all(W) if a_ * b_ <= 20 else [q]):
            # The fired vertices are the nonzero ones, two for each fired pair.
            o = np.array(o, float); nz = int((o > TOL).sum()); assert nz % 2 == 0; mo = nz // 2
            hm = int((np.abs(o - f) > 1e-9).sum()); assert hm <= supp + mo <= 3 * min(a_, b_), (W, hm, supp, mo); worst = max(worst, hm / min(a_, b_)); N += 1
    # Proposition 26 of the paper gives the bound of 3 times the smaller side in part (a) and the
    # ratio of seven thirds to the smaller side in part (c). The assertion below ties that ratio
    # to this run: the worst ratio found here must be at least seven thirds.
    assert worst >= 7 / 3 - 1e-9, worst
    print(f"    bound ||q - f||_0 <= |supp f| + |M_f| <= 3 min(|A|,|B|) holds on {N} cold outputs (all tie-breaks at <= 20 cells); worst ||q - f||_0 / min = {worst:.3f} >= 7/3, the interval quoted in prop:coldrefuted(c)")
    print("all assertions in [12] hold", elapsed())


# Section 13 of this program: the trap family, a trap for rules that fire the constraint with the
# largest residual, and the rule zoo, a collection of other repair rules.
def _trap_family(k):
    """Build the trap instance with k rows, which the output calls T_k.

    There are k rows and k plus 1 columns, all counted from zero. Row i has the value D, equal to 2
    times (k minus i) plus 3. The weight of row i with column i is that value, and so is the weight
    of row i with the last column. The weight of row i with column i plus 1 is that value minus 1,
    when column i plus 1 is not the last column. Every other weight is zero. The prediction is the
    value D of row 0 on column 0 and on the last column, and zero elsewhere.

    The function returns the weight matrix, the prediction and f, the single optimal dual, which
    has the values D on the rows and zero on the columns.
    """
    D = [2 * (k - i) + 3 for i in range(k)]
    W = np.zeros((k, k + 1), dtype=int)
    for i in range(k):
        W[i, i] = D[i]; W[i, k] = D[i]
        if i + 1 < k: W[i, i + 1] = D[i] - 1
    yhat = np.zeros(2 * k + 1); yhat[k] = D[0]; yhat[2 * k] = D[0]
    return W, yhat, np.r_[np.array(D, float), np.zeros(k + 1)]


def _side_scan(p, cons, side):
    """Scan the vertices of one side in order, firing at most once at each vertex.

    The scan starts from the dual p, and cons is the list of constraints. At each vertex of side,
    the violated constraint at that vertex with the largest residual fires, if there is one. A tie
    goes to the pair with the larger indices, because the code takes the maximum of tuples that
    hold the residual followed by the two indices. The function returns the resulting dual and the
    number of firings.
    """
    z = p.copy(); fires = 0
    for v in side:
        vio = [(w - z[i] - z[j], i, j) for i, j, w in cons if (i == v or j == v) and z[i] + z[j] < w - TOL]
        if vio:
            r, i, j = max(vio); z[i] += r; z[j] += r; fires += 1
    return z, fires


def _global_rule(p, cons, rule="largest"):
    """Keep firing the violated constraint with the largest residual until none is left.

    The firing starts from the dual p and looks at the whole system each time. With rule set to
    smallest, the violated constraint with the smallest residual fires instead. A tie goes to the
    pair with the larger indices under the largest rule and to the pair with the smaller indices
    under the smallest rule. The function returns the repaired dual.
    """
    z = p.copy()
    while True:
        vio = [(w - z[i] - z[j], i, j) for i, j, w in cons if z[i] + z[j] < w - TOL]
        if not vio: return z
        r, i, j = (max if rule == "largest" else min)(vio); z[i] += r; z[j] += r


def _fire_matching_then_largest(p, cons, M):
    """Apply the matching-first rule and return the repaired dual.

    The rule starts from the dual p. Every violated pair of the matching M fires first. Then the
    violated constraint with the largest residual in the whole system fires, again and again, until
    no constraint is violated. A pair of M is written as a row index followed by the number of rows
    plus a column index.
    """
    z = p.copy()
    for i, j, w in cons:
        if (i, j) in M and z[i] + z[j] < w - TOL: r = w - z[i] - z[j]; z[i] += r; z[j] += r
    return _global_rule(z, cons, "largest")


def check_trap_family():
    section(13, "A TRAP FOR LARGEST-RESIDUAL RULES: T_k, every order of its unique cover, and the rule zoo")
    for k in (2, 3, 4, 5, 6):
        W, yhat, f = _trap_family(k); n = 2 * k + 1; cons = bip_cons(W); OPT = matching_value(W)
        assert is_feasible(f, cons) and abs(f.sum() - OPT) < 1e-9
        ext = face_extremes(n, cons, OPT)
        assert all(abs(lo - fi) < 1e-7 and abs(hi - fi) < 1e-7 for (lo, hi), fi in zip(ext, f)), "face not the single point f"
        pre = int((np.abs(yhat - f) > TOL).sum()); assert pre == k + 2
        p = clamp(yhat); edges = sorted(violated(p, cons).keys())
        assert edges == sorted([(i, k + i) for i in range(1, k)] + [(i, k + i + 1) for i in range(k - 1)])
        tau, covers = min_vertex_covers(edges); S = frozenset(range(k + 1, 2 * k))
        assert tau == k - 1 and [frozenset(c) for c in covers] == [S]
        outs, ties = _cover_scan_orders(p, cons, S)
        ds = {int((np.abs(np.array(o) - f) > 1e-9).sum()) for o in outs}
        assert ties == 0 and ds == {2 * k + 1}, (ties, ds)
        z4 = _global_rule(p, cons); d4 = int((np.abs(z4 - f) > 1e-9).sum()); assert d4 == 2 * k + 1
        zr, _ = _side_scan(p, cons, range(k))  # Scan the row side in index order.
        dr = int((np.abs(zr - f) > 1e-9).sum()); assert dr == 2 * k + 1
        line = (f"T_{k}: OPT = {OPT:.0f}, face = single point, pre = {pre}, tau = {tau}, unique cover {sorted(S)}; largest-residual scan over all"
                f" {math.factorial(k-1)} orders: no tie, d0 = {2*k+1} = pre + tau (ratio {(2*k+1)/(k+2):.3f}); global largest-first d0 = {d4}; row side in index order d0 = {dr}")
        if k <= 3:
            best = _min_admissible_d0(n, cons, OPT, p, _admissible_polytope_rows(n, cons, p)); assert best == pre; line += f"; best admissible = {best} = pre"
        print("    " + line)
    # On the star of Theorem 8 of the paper, which has one row and
    # columns of weights 1, 2 and so on, the rules that fire the constraint with the smallest
    # residual first are the same as the forward scan, which visits the constraints in index order.
    # The cover-directed rule and the global rule are both run for 3, 5 and 8 columns, and the
    # distance goes from 1 before the repair to the number of columns after it.
    for d in (3, 5, 8):
        W = np.array([[i + 1 for i in range(d)]]); yhat = np.zeros(d + 1); cons = bip_cons(W); OPT = matching_value(W); n = d + 1
        p = clamp(yhat); outs, _ = _cover_scan_orders(p, cons, {0}, rule="smallest"); zg = _global_rule(p, cons, "smallest")
        pre = face_l0_subset_lp(n, cons, OPT, p); dc = {face_l0_subset_lp(n, cons, OPT, np.array(o)) for o in outs}; dg = face_l0_subset_lp(n, cons, OPT, zg)
        assert pre == 1 and dc == {d} and dg == d
        print(f"    star with weights 1..{d}: pre = {pre}; cover-directed smallest-residual d0 = {d}, global smallest-residual d0 = {d}")
    # Observable quantities on the chain family, for k equal to 4 and 6. The column cover is the
    # bad one, because it ends farther from the optimal dual. Yet it has the smaller raise by three
    # measures: the sum of the raises, the square root of the sum of their squares, and the largest
    # raise.
    for k in (4, 6):
        Wk, yhat, f = _chain_family(k); cons = bip_cons(Wk); p = clamp(yhat)
        res = {}
        for name, side in (("rows", range(k)), ("columns", range(k, 2 * k))):
            z, fires = _side_scan(p, cons, side)
            d = int((np.abs(z - f) > 1e-9).sum()); raise_ = z - p
            res[name] = (d, fires, int((raise_ > TOL).sum()), raise_.sum(), np.sqrt((raise_ ** 2).sum()), raise_.max())
        (dr, fr_, mr, s1r, s2r, sir), (dc, fc, mc, s1c, s2c, sic) = res["rows"], res["columns"]
        assert dr == k + 1 and dc == 2 * k + 1 and fr_ == fc == k and mr == mc == 2 * k and s1c < s1r and s2c < s2r and sic < sir
        print(f"    C_{k}: row cover d0 = {dr}, column cover d0 = {dc}; firings {fr_} vs {fc}, moved coordinates {mr} vs {mc} (ties);"
              f" total raise {s1r:.0f} vs {s1c:.0f}, l2 {s2r:.2f} vs {s2c:.2f}, l_inf {sir:.0f} vs {sic:.0f}: every strict observable picks the column cover")
    # The pendant variant of the chain family, which the output prints with the prefix PC. Here the
    # cover with fewer firings and fewer moved coordinates is also the wrong one.
    for k in (4, 6):
        Av = [3 + 2 * i for i in range(k)]; R = Av[-1] + 2; W = np.zeros((k + 1, k + 2), dtype=int)
        W[:k, :k + 1] = _chain_family(k)[0]
        # The pendant row is attached at the first column. A hidden edge, which the prediction
        # already satisfies, joins it to the new last column.
        W[k, 0] = R - 1; W[k, k + 1] = R
        n = 2 * k + 3; yhat = np.zeros(n); yhat[k + 1 + k] = Av[-1]; yhat[n - 1] = R
        cons = bip_cons(W); OPT = matching_value(W); p = clamp(yhat)
        pts = [np.r_[np.array(Av, float), [R - 1], np.zeros(k + 1), [1]], np.r_[np.array(Av, float), [R], np.zeros(k + 1), [0]]]
        assert all(is_feasible(t, cons) and abs(t.sum() - OPT) < 1e-9 for t in pts)
        # The optimal face is exactly the segment between the two points in pts.
        ext = face_extremes(n, cons, OPT)
        assert all(abs(lo - min(a, b)) < 1e-7 and abs(hi - max(a, b)) < 1e-7 for (lo, hi), a, b in zip(ext, pts[0], pts[1]))
        # Every point used below is integral, and the inside of the segment is fractional in the
        # two coordinates that move along it, so the nearest face point is always one of the two
        # ends of the segment.
        pre = face_l0(p, pts)
        res = {}
        # The row side must include the pendant row.
        for name, side in (("rows", range(k + 1)), ("columns", range(k + 1, 2 * k + 1))):
            z, fires = _side_scan(p, cons, side)
            assert is_feasible(z, cons)
            res[name] = (face_l0(z, pts), fires, int((z - p > TOL).sum()))
        (dr, fr_, mr), (dc, fc, mc) = res["rows"], res["columns"]
        assert fc < fr_ and mc < mr and dc > dr, res
        print(f"    PC_{k}: pre = {pre}; row side (with the pendant row) d0 = {dr} with {fr_} firings / {mr} moved, column side d0 = {dc} with {fc} firings / {mc} moved: fewest firings, fewest moved and the smaller side all pick the column side")
    # A rule that first fires a maximal matching, built greedily in order of decreasing weight, and
    # then keeps firing the constraint with the largest residual. On this 2 by 2 instance its
    # ratio, the distance after the repair divided by the distance before it, is exactly 2.
    W = np.array([[3, 2], [2, 0]]); yhat = np.array([0, 0, 2, 0], float); cons = bip_cons(W); OPT = matching_value(W); p = clamp(yhat)
    Mg = set(); used = set()
    for a, b in sorted(((a, b) for a in range(2) for b in range(2) if W[a, b] > 0), key=lambda e: (-W[e[0], e[1]], e)):
        if a not in used and 2 + b not in used: Mg.add((a, 2 + b)); used |= {a, 2 + b}
    z = _fire_matching_then_largest(p, cons, Mg); pre = face_l0_subset_lp(4, cons, OPT, p); post = face_l0_subset_lp(4, cons, OPT, z)
    assert Mg == {(0, 2)} and pre == 1 and post == 2
    print(f"    greedy-weight matching first on w = [[3,2],[2,0]], yhat = (0,0|2,0): matching {sorted(Mg)}, pre = {pre}, post = {post}: ratio 2")
    # A rule that first fires a maximum-weight matching and then keeps firing the constraint with
    # the largest residual. In a random search over 1500 draws its ratio is never above three
    # halves.
    rng = np.random.default_rng(13); worst = (0.0, None); N = 0
    for _ in range(1500):
        na, nb = int(rng.integers(1, 5)), int(rng.integers(1, 5)); W = rng.integers(0, 6, size=(na, nb)); yhat = rng.integers(-2, 7, size=na + nb).astype(float)
        cons = bip_cons(W); OPT = matching_value(W); p = clamp(yhat); n = na + nb
        if not violated(p, cons): continue
        pts = face_points_bip(W, OPT); pre = face_l0(p, pts)
        if pre == 0: continue
        ri, ci = linear_sum_assignment(W, maximize=True); Mx = {(int(a), na + int(b)) for a, b in zip(ri, ci) if W[a, b] > 0}
        z = _fire_matching_then_largest(p, cons, Mx); post = face_l0(z, pts); N += 1
        if post / pre > worst[0]: worst = (post / pre, (W.tolist(), yhat.tolist(), pre, post))
    assert worst[0] <= 1.5 + 1e-9, worst
    print(f"    maximum-weight matching first, then largest residual: {N} random instances, worst ratio {worst[0]:.3f} (pre {worst[1][2]} -> post {worst[1][3]}) -- never above 3/2")
    # The matching-first rule of section 22 of this program, printed as R_M, on the hard families,
    # which are the path, trap and chain families, with exact distances to the optimal face. The
    # rule reads only the weights, the prediction and the matching, and uses no optimal dual.
    fams = [(f"P_{kk}",) + _path_family(kk)[:2] for kk in (1, 2, 3, 4, 5)]
    fams += [(f"T_{kk}",) + _trap_family(kk)[:2] for kk in (2, 3, 4, 5, 6)]
    fams += [(f"C_{kk}",) + _chain_family(kk)[:2] for kk in (2, 3, 4, 5, 6)]
    tight = []
    for nm, Wf, yf in fams:
        naf, nbf = Wf.shape; Vf = naf + nbf
        consf = bip_cons(Wf); OPTf = matching_value(Wf)
        rr, cc = linear_sum_assignment(Wf, maximize=True)
        Mf = {(int(i_), naf + int(j_)) for i_, j_ in zip(rr, cc) if Wf[i_, j_] > 0}
        zf = _fire_matching_then_largest(clamp(yf), consf, Mf)
        assert is_feasible(zf, consf), nm
        pre_f = face_l0_subset_lp(Vf, consf, OPTf, np.maximum(yf, 0))
        post_f = face_l0_subset_lp(Vf, consf, OPTf, zf)
        assert post_f <= pre_f + pre_f // 2, (nm, pre_f, post_f)
        if post_f == pre_f + pre_f // 2 and pre_f > 0: tight.append(nm)
    print(f"    max-weight-matching rule R_M (no optimal dual) obeys pre + floor(pre/2) on all "
          f"{len(fams)} family members; attained with equality on {', '.join(tight)}")
    print("all assertions in [13] hold", elapsed())


# Section 14 of this program: the ladder family for the walk. For m from 2 to 7, the designed run
# of the walk ends at distance 2 times m minus 2 from the optimal face, while the prediction is at
# distance m.
def _ladder_family(m):
    """Build the ladder instance with m rows and m columns.

    This is the family of Proposition 42 of the paper. Rows,
    columns and indices are counted from 1 in this description. Every index from 1 to m has a
    value, kept as the entry of the list d at that index. The last entry is m plus 1. Going down
    from there, the entry at index i is the next entry, plus i, plus 1. The weight of row i with
    column i is the entry at index i. The weight of row i with the next column is the next entry.
    The weight of the next row with column i is the entry at index i reduced by i, a number that
    the proposition calls s. Every other weight is zero.

    The function returns the weight matrix and the list d, whose entry 0 is not used.
    """
    d = [0] * (m + 1)
    d[m] = m + 1
    for i in range(m - 1, 0, -1):
        d[i] = d[i + 1] + i + 1
    W = np.zeros((m, m), dtype=int)
    for i in range(1, m + 1):
        W[i - 1, i - 1] = d[i]
    for i in range(1, m):
        W[i - 1, i] = d[i + 1]
        W[i, i - 1] = d[i] - i
    return W, d


def check_ladder_family():
    section(14, "THE LADDER FAMILY: the walk's factor 2 is tight in the limit")
    for m in range(2, 8):
        W, d = _ladder_family(m)
        n = 2 * m
        cons = bip_cons(W)
        OPT = matching_value(W)
        assert OPT == sum(d[1:]), (m, OPT, sum(d[1:]))
        yhat = np.zeros(n)
        pre = face_l0_subset_lp(n, cons, OPT, yhat)
        assert pre == m, (m, pre)
        E = violated(yhat, cons)
        assert len(E) == 3 * m - 2, (m, len(E))
        tau, _ = min_vertex_covers(list(E))
        assert tau == m, (m, tau)
        # The designed resolution of the choices that the walk leaves open: start at the first row
        # and make a single pass that alternates between rows and columns, from the first row and
        # the first column to the last row and the last column. Its output is written down directly
        # here.
        z = np.zeros(n)
        for i in range(1, m + 1):
            z[i - 1] = d[i]                      # Row i is raised by the entry of d at index i.
        for i in range(1, m):
            # Column i is raised by the number called s, the entry of d at index i reduced by i.
            # The last column is never raised.
            z[m + i - 1] = d[i] - i
        assert is_feasible(z, cons), m
        post = face_l0_subset_lp(n, cons, OPT, z)
        assert post == 2 * m - 2, (m, post)
        line = (f"    m={m}: OPT={OPT}, pre={pre}, tau={tau}, designed output at "
                f"face distance {post} = 2m-2, ratio {post}/{pre}")
        if m <= 6:
            leaves, capped, ties = walk_all(n, E, start_any=True, cap=20000)
            assert capped == 0
            dists = set()
            for delta, raised in leaves:
                y = yhat + delta
                assert is_feasible(y, cons)
                dists.add(face_l0_subset_lp(n, cons, OPT, y))
            assert max(dists) == 2 * m - 2, (m, sorted(dists))
            assert 2 * m - 2 <= pre + tau, m
            line += f"; ALL {len(leaves)} resolutions <= {max(dists)} (worst attained)"
        print(line)
    print("    ratio (2m-2)/m increases to 2: the walk's constant is exactly 2")
    print("all assertions in [14] hold", elapsed())


# Section 15 of this program: the oblivious view V0. The check enumerates every completion of the
# view and every integral output, with coordinates up to the weight scale, that is feasible for all
# of them. The smallest worst-case ratio that it finds is five quarters.
def _oblivious_view_minimax(M):
    """Run the finite check of the oblivious view at weight scale M.

    An oblivious repair sees only a view: the clamped prediction and the violated constraints with
    their residuals. The weights of the other pairs are hidden from it, and a completion is one
    choice of all hidden weights. The check is a game: the repair picks one output from the view,
    an adversary then picks the completion, and the repair wants a small ratio. With M equal to 5
    this is the view that the program calls V0, and with M equal to 3 it is the scaled copy of that
    view.

    The view has two rows, u1 and u2, and three columns, v1, v2 and c. The clamped prediction is
    zero on both rows, M on v1, zero on v2 and M on c. The violated constraints are u1 with v2 and
    u2 with v2, with residuals M minus 1 and M minus 2. So those two weights are M minus 1 and M
    minus 2, and the vertex cover number of the violated graph is 1. The hidden weights are those
    of u1 with v1, u2 with v1, u1 with c and u2 with c. Each of them ranges from 0 to M, because
    anything larger would be violated at the prediction.

    The function returns five values. They are the number of completions, the number of integral
    outputs with coordinates from 0 to M that are feasible for every completion, the smallest value
    over those outputs of the worst ratio over all completions, an output that attains this value,
    and the smallest value over those outputs of the worst additive excess over all completions.
    The ratio is the distance to the optimal face after the repair divided by the distance before
    it, and the additive excess is the first of these distances minus the second. Distances count
    the coordinates that differ from the nearest integral optimal dual. The third value is what the
    later sections call the game value of the view.
    """
    p = (0, 0, M, 0, M)
    r1, r2 = M - 1, M - 2

    def faces(w11, w21, w1c, w2c):
        best, pts = None, []
        for y in itertools.product(range(M + 1), repeat=5):
            u1, u2, v1, v2, c = y
            if (u1 + v1 >= w11 and u2 + v1 >= w21 and u1 + c >= w1c and u2 + c >= w2c
                    and u1 + v2 >= r1 and u2 + v2 >= r2):
                s = sum(y)
                if best is None or s < best: best, pts = s, [y]
                elif s == best: pts.append(y)
        return pts

    def ham(a, b): return sum(x != y for x, y in zip(a, b))

    comps = []
    for h in itertools.product(range(M + 1), repeat=4):
        F = faces(*h)
        pre = min(ham(p, f) for f in F)
        assert pre >= 1
        comps.append((F, pre))

    best = None; n_feas = 0; add_min = None
    # Outputs with a coordinate above M are not needed, as explained in _feasible_for_all.
    for z in itertools.product(range(M + 1), repeat=5):
        u1, u2, v1, v2, c = z
        # An output is feasible for every completion exactly when it is feasible with every hidden
        # weight set to M.
        if not (u1 + v1 >= M and u2 + v1 >= M and u1 + c >= M and u2 + c >= M
                and u1 + v2 >= r1 and u2 + v2 >= r2):
            continue
        n_feas += 1
        wr, wa = 0.0, -10
        for F, pre in comps:
            post = min(ham(z, f) for f in F)
            wr = max(wr, post / pre); wa = max(wa, post - pre)
        if best is None or wr < best[0]: best = (wr, z)
        add_min = wa if add_min is None else min(add_min, wa)
    return len(comps), n_feas, best[0], best[1], add_min


def check_oblivious_lb():
    section(15, "OBLIVIOUS RULES: no view-based repair beats 5/4, or removes the additive term")
    for M in (5, 3):
        nc, nz, mm, argz, addm = _oblivious_view_minimax(M)
        print(f"    view at scale M={M}: {nc} completions, {nz} feasible integral outputs; "
              f"minimax ratio {mm:.6f} attained by z={list(argz)}; "
              f"min over outputs of worst additive excess = {addm} (tau = 1)")
        assert abs(mm - 1.25) < 1e-12, (M, mm)
        assert addm >= 1, (M, addm)
    print("    every deterministic oblivious repair with integral outputs is forced to "
          "ratio >= 5/4 and to post >= pre + tau on some completion")
    print("all assertions in [15] hold", elapsed())


# Section 16 of this program: an even distance to the optimal face before the repair. With r that
# distance, the bound checked is r, plus half of r rounded up, minus 1.
def check_even_parity():
    section(16, "EVEN FACE DISTANCE: the r + ceil(r/2) - 1 bound and its sharpness")
    # Part (a): the bound on the best admissible output. An admissible output is a feasible dual
    # that is at most as far as the clamped prediction from every feasible dual, in the sum of
    # absolute differences. When r is at least 1, the smallest distance to the optimal face over
    # all admissible outputs must be at most r, plus half of r rounded up, minus 1. This is checked
    # on every instance with 1 row and 2 columns whose weights and predictions run from 0 to 3, and
    # on 400 random draws with 1 or 2 rows and 2 or 3 columns. The all-zero weight matrix and the
    # predictions that are already on the optimal face are skipped.
    n_checked = 0; tight_even = 0
    for w in itertools.product(range(4), repeat=2):
        if max(w) == 0: continue
        W = np.array([list(w)]); cons = bip_cons(W); OPT = matching_value(W)
        for yh in itertools.product(range(4), repeat=3):
            yhat = np.array(yh, float)
            pre = face_l0_subset_lp(3, cons, OPT, np.maximum(yhat, 0))
            if pre == 0: continue
            p = np.maximum(yhat, 0)
            best = _min_admissible_d0(3, cons, OPT, p, _admissible_polytope_rows(3, cons, p))
            bound = pre + (pre + 1) // 2 - 1
            assert best <= bound, (w, yh, pre, best, bound)
            n_checked += 1
            if pre % 2 == 0 and best == bound: tight_even += 1
    rng = np.random.default_rng(20260904)
    for _ in range(400):
        na = int(rng.integers(1, 3)); nb = int(rng.integers(2, 4))
        W = rng.integers(0, 5, size=(na, nb))
        if W.max() == 0: continue
        n = na + nb; cons = bip_cons(W); OPT = matching_value(W)
        yhat = rng.integers(-1, 6, size=n).astype(float)
        p = np.maximum(yhat, 0)
        pre = face_l0_subset_lp(n, cons, OPT, p)
        if pre == 0: continue
        best = _min_admissible_d0(n, cons, OPT, p, _admissible_polytope_rows(n, cons, p))
        bound = pre + (pre + 1) // 2 - 1
        assert best <= bound, (W.tolist(), yhat.tolist(), pre, best, bound)
        n_checked += 1
        if pre % 2 == 0 and best == bound: tight_even += 1
    print(f"    exhaustive 1x2 (w<=3, yhat<=3) + 400 random 1x2..2x3 draws: {n_checked} predictions "
          f"at pre>=1; 0 violations of best <= pre + ceil(pre/2) - 1; tight on {tight_even} even-pre cases")
    # Part (b): the extended family for the lower bound, for k equal to 1 and 2. It is the path
    # family of _path_family with one more column, joined to the last row. There are k rows, then k
    # columns in a group b, then k plus 2 columns in a group c, and each group is counted from
    # zero. The distance before the repair is r, equal to 2 times k plus 2, and every admissible
    # output is at distance 3 times k plus 2.
    for k in (1, 2):
        na, nb = k, 2 * k + 2
        W = np.zeros((na, nb))
        for i in range(k):
            W[i, i] = 1                    # Row i with column i of group b.
            W[i, k + i] = 2                # Row i with column i of group c.
            W[i, k + i + 1] = 2            # Row i with column i plus 1 of group c.
        # Row k minus 1 with column k plus 1 of group c, counted from zero like the lines above.
        W[k - 1, 2 * k + 1] = 2
        n = na + nb; cons = bip_cons(W); OPT = matching_value(W)
        assert abs(OPT - 2 * k) < 1e-9
        ext = face_extremes(n, cons, OPT)
        f = np.array([2.0] * k + [0.0] * nb)
        assert all(abs(lo - fi) < 1e-7 and abs(hi - fi) < 1e-7 for (lo, hi), fi in zip(ext, f)), k
        yhat = np.array([0.0] * k + [0.0] * k + [2.0] * (k + 2))
        pre = face_l0_subset_lp(n, cons, OPT, yhat)
        assert pre == 2 * k + 2, (k, pre)
        rows = _admissible_polytope_rows(n, cons, yhat)
        best = _min_admissible_d0(n, cons, OPT, yhat, rows)
        assert best == 3 * k + 2 == pre + pre // 2 - 1, (k, pre, best)
        print(f"    extended family k={k}: face = single point, r = {pre} (even), "
              f"min admissible d0 = {best} = r + r/2 - 1; ratio {best}/{pre}")
    # Part (c): the gadget with distance 2 before the repair. The smallest distance over admissible
    # outputs is also 2.
    W = np.array([[1, 1]]); cons = bip_cons(W); OPT = matching_value(W)
    yhat = np.array([0.0, 0.0, 1.0])
    pre = face_l0_subset_lp(3, cons, OPT, yhat)
    rows = _admissible_polytope_rows(3, cons, yhat)
    best = _min_admissible_d0(3, cons, OPT, yhat, rows)
    assert pre == 2 and best == 2, (pre, best)
    print(f"    r=2 gadget w=(1 1), yhat=(0|0,1): pre = 2, min admissible d0 = 2 = r + r/2 - 1")
    print("all assertions in [16] hold", elapsed())


# Section 17 of this program: the cold-start bound of five halves of the smaller side, rounded
# down.
def check_cold_five_halves():
    section(17, "THE COLD BOUND floor(5 min / 2): witnesses and a random sweep")
    Wr = COLD_4X5
    na, nb = Wr.shape; n = na + nb; cons = bip_cons(Wr); OPT = matching_value(Wr)
    q = np.array(_cold_outputs_all(Wr)[0], float)
    d = face_l0_subset_lp(n, cons, OPT, q); m = min(na, nb)
    assert d == 9 <= (5 * m) // 2, (d, m)
    print(f"    4x5 witness: cold face distance {d} <= floor(5m/2) = {(5*m)//2}")
    rng = np.random.default_rng(20260904)
    checked = 0; worst = 0.0
    for _ in range(300):
        na = int(rng.integers(2, 6)); nb = int(rng.integers(na, 8))
        W = rng.integers(0, 9, size=(na, nb))
        if W.max() == 0: continue
        n = na + nb; cons = bip_cons(W); OPT = matching_value(W)
        m = min(na, nb); cap = (5 * m) // 2
        for out in _cold_outputs_all(W):
            dd = face_l0_subset_lp(n, cons, OPT, np.array(out, float))
            assert dd <= cap, (W.tolist(), dd, cap)
            checked += 1; worst = max(worst, dd / m)
    print(f"    random sweep (seed 20260904): {checked} cold outputs over all tie-breaks, "
          f"0 violations of d <= floor(5m/2); worst d/m = {worst:.4f}")
    print("all assertions in [17] hold", elapsed())


# Sections 18, 19 and 21 of this program: oblivious views.
def _view_completions(na, nb, p, viol):
    """Generate every completion of an oblivious view.

    The view consists of the clamped prediction p and the dictionary viol. Each key of viol is a
    violated pair, written as a row index and a column index that both count from zero, and its
    value is the positive residual of that pair. The weight of a violated pair is therefore the sum
    of the two prediction values at its endpoints plus its residual. Every other pair is hidden,
    and its weight can be any whole number from zero to the sum of the two prediction values at its
    endpoints. A completion is one choice of all the hidden weights.

    For every completion the generator yields four things: the tuple h of hidden weights, in the
    order of the hidden pairs row by row, the completed weight matrix, the integral points of its
    optimal face as computed by face_points_bip, and pre, the number of coordinates in which the
    prediction differs from the nearest of those points.
    """
    p = np.asarray(p, float)
    hidden = [(a, b) for a in range(na) for b in range(nb) if (a, b) not in viol]
    ranges = [range(int(p[a] + p[na + b]) + 1) for a, b in hidden]
    base = np.zeros((na, nb), dtype=np.int64)
    for (a, b), r in viol.items():
        base[a, b] = int(p[a] + p[na + b] + r)
    for h in itertools.product(*ranges):
        W = base.copy()
        for (a, b), w in zip(hidden, h):
            W[a, b] = w
        OPT = matching_value(W)
        F = face_points_bip(W, OPT)
        pre = face_l0(p, F)
        assert pre >= 1
        yield h, W, F, pre


def _feasible_for_all(z, na, nb, p, viol):
    """Tell whether the output z is feasible for every completion of a view.

    An output is feasible for every completion exactly when it is feasible with every hidden weight
    at its maximum, the sum of the two prediction values at the endpoints of the pair. The checks
    that compute game values list outputs only up to the largest weight that each vertex can carry,
    because a larger coordinate can be lowered to that cap without breaking feasibility for any
    completion and without losing agreement with any face point.
    """
    z = np.asarray(z, float)
    if (z < -TOL).any():
        return False
    for a in range(na):
        for b in range(nb):
            w = p[a] + p[na + b] + viol.get((a, b), 0)
            if z[a] + z[na + b] < w - TOL:
                return False
    return True


def check_fourthirds_view():
    section(18, "THE VIEW V2: exact oblivious game value 4/3 (three-completion certificate)")
    na, nb = 3, 3
    p = np.array([0, 1, 1, 0, 2, 2], float)
    # The only violated pair, row 0 with column 0, has weight 1 and residual 1. Every other pair is
    # hidden.
    viol = {(0, 0): 1}
    # The largest weight that each vertex can carry, as explained in _feasible_for_all.
    caps = [2, 3, 3, 1, 3, 3]
    Z = [np.array(z, float) for z in itertools.product(*[range(c + 1) for c in caps])
         if _feasible_for_all(z, na, nb, p, viol)]
    # The certificate: three completions, each given by its nonzero hidden weights. Every output in
    # Z, the list of feasible integral outputs, has a ratio of at least four thirds on at least one
    # of them.
    certs = [{(0, 2): 1, (1, 1): 3, (1, 2): 1, (2, 0): 1, (2, 1): 3},
             {(1, 1): 3, (1, 2): 3, (2, 0): 1, (2, 1): 2, (2, 2): 2},
             {(0, 1): 2, (0, 2): 2, (1, 0): 1, (2, 2): 1}]
    cert = []
    for C in certs:
        W = np.zeros((3, 3), dtype=np.int64); W[0, 0] = 1
        for (a, b), w in C.items():
            assert w <= p[a] + p[na + b]
            W[a, b] = w
        OPT = matching_value(W); F = face_points_bip(W, OPT); pre = face_l0(p, F)
        cert.append((pre, F))
        print(f"    certificate completion {sorted(C.items())}: OPT {OPT:.0f}, pre {pre}, "
              f"face {[tuple(int(x) for x in f) for f in F]}")
    unforced = [z for z in Z if not any(Fraction(face_l0(z, F), pre) >= Fraction(4, 3)
                                        for pre, F in cert)]
    assert not unforced, unforced[:3]
    print(f"    {len(Z)} integral outputs feasible for every completion; all forced to "
          f"post/pre >= 4/3 by the three certificate completions")
    zs = np.array([1, 1, 1, 1, 2, 2], float)
    worst = Fraction(0); n = 0
    for h, W, F, pre in _view_completions(na, nb, p, viol):
        worst = max(worst, Fraction(face_l0(zs, F), pre)); n += 1
    assert n == 9216 and worst == Fraction(4, 3), (n, worst)
    print(f"    z* = (1,1,1 | 1,2,2): worst post/pre over all {n} completions = {worst}")
    print("    hence the game value of V2 is exactly 4/3: no deterministic oblivious repair "
          "with integral outputs beats 4/3")
    print("all assertions in [18] hold", elapsed())


def check_star_views():
    section(19, "STAR VIEWS: two-sided firing <= 4/3 on a single violated edge, centre raise <= 5/4 on k = 2 stars")
    rng =np.random.default_rng(20260930)
    # Part (i): a single violated edge. On 24 random views with 2 rows and 3 columns, prediction
    # values at most 2 and residual 1 or 2, the output is the clamped prediction with both
    # endpoints of the violated edge raised by the residual. Every completion of each view is
    # enumerated.
    n_views = 0; n_comp = 0; worst = Fraction(0)
    while n_views < 24:
        na, nb = 2, 3
        p = rng.integers(0, 3, size=na + nb).astype(float)
        a, b = int(rng.integers(0, na)), int(rng.integers(0, nb)); r = int(rng.integers(1, 3))
        viol = {(a, b): r}
        z = p.copy(); z[a] += r; z[na + b] += r
        for h, W, F, pre in _view_completions(na, nb, p, viol):
            post = face_l0(z, F); n_comp += 1
            assert post <= pre + 1 and (pre > 2 or post <= pre), (p, viol, h, pre, post)
            worst = max(worst, Fraction(post, pre))
        n_views += 1
    assert worst <= Fraction(4, 3)
    print(f"    single edge: {n_views} random 2x3 views, {n_comp} completions; post <= pre + 1 "
          f"always and post <= pre whenever pre <= 2; worst ratio {worst} <= 4/3")
    # Part (ii): stars with two violated edges on the grid of the view V0, laid out here as a
    # centre row, two leaf columns and two more rows whose pairs are all hidden. On 16 random
    # views with prediction values at most 3, the output raises only the centre, by the larger of
    # the two residuals.
    n_views = 0; n_comp = 0; worst = Fraction(0)
    while n_views < 16:
        na, nb = 3, 2
        p = rng.integers(0, 4, size=na + nb).astype(float)
        r1, r2 = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        viol = {(0, 0): r1, (0, 1): r2}
        z = p.copy(); z[0] += max(r1, r2)
        for h, W, F, pre in _view_completions(na, nb, p, viol):
            post = face_l0(z, F); n_comp += 1
            assert post <= pre + 1 and (pre > 3 or post <= pre), (p, viol, h, pre, post)
            worst = max(worst, Fraction(post, pre))
        n_views += 1
    assert worst <= Fraction(5, 4)
    print(f"    k = 2 star: {n_views} random 3x2 views, {n_comp} completions; post <= pre + 1 "
          f"always and post <= pre whenever pre <= 3; worst ratio {worst} <= 5/4")
    # The view V0 itself, whose centre is the column v2. Raising only the centre attains the ratio
    # five quarters.
    p0 = np.array([0, 0, 5, 0, 5], float); viol0 = {(0, 1): 4, (1, 1): 3}
    z0 = p0.copy(); z0[2 + 1] += 4
    w0 = max(Fraction(face_l0(z0, F), pre) for h, W, F, pre in _view_completions(2, 3, p0, viol0))
    assert w0 == Fraction(5, 4), w0
    print(f"    on V0 the centre raise has worst ratio exactly {w0}: the k >= 2 star bound 5/4 is attained")
    print("all assertions in [19] hold", elapsed())


# Section 20 of this program: the exact cold-start envelope. On the instances of the even family
# and the odd family that are run here, the distance of the cold output to the optimal face is
# five halves of the smaller side, rounded down, minus 1. On random instances the same number is
# checked as an upper bound on that distance.
def _cold_even_family(p):
    """Build the even extremal instance for the cold start, which the output calls E_p.

    The rows form two groups of p rows each, named R and S in this order. The columns form three
    groups of p columns each, named H, T and U in this order. So the smaller side is 2 times p and
    the number of columns is 3 times p. The function returns the weight matrix, the optimal dual
    and the expected distance of the cold output to the optimal face, which is 5 times p minus 1.
    Section 20 of this program verifies the dual and the distance for the sizes it runs.
    """
    t = [2 * i - 1 for i in range(1, p + 1)]; s = [2 * i for i in range(1, p + 1)]
    r = [2 * p + i for i in range(1, p + 1)]; h = [p + i for i in range(1, p + 1)]
    u = [j for j in range(1, p)] + [0]
    W = np.zeros((2 * p, 3 * p), dtype=int)
    for i in range(p):
        W[i, i] = r[i] + h[i]; W[i, p + i] = r[i]
        for j in range(p):
            W[i, 2 * p + j] = r[i] + u[j]; W[p + i, j] = s[i] + h[j]
        W[p + i, p + i] = t[i]
    W[p, p + 1] = s[0]
    return W, np.array(r + s + h + [0] * p + u, float), 5 * p - 1


def _cold_odd_family(p):
    """Build the odd extremal instance for the cold start, which the output calls O_p.

    It is the even instance with the base values t, s and r of the code larger by two, plus one
    more row and one more column, which the code stores as the last row and the last column. The
    paper calls them the row S with index 0 and the column T with index 0. So the smaller side
    is 2 times p plus 1. The function returns the weight matrix, the optimal dual and the
    expected distance of the cold output to the optimal face, which is 5 times p plus 1. Section
    20 of this program verifies the dual and the distance for the sizes it runs.
    """
    t = [2 * i + 1 for i in range(1, p + 1)]; s = [2 * i + 2 for i in range(1, p + 1)]
    r = [2 * p + i + 2 for i in range(1, p + 1)]; h = [p + i for i in range(1, p + 1)]
    u = [j for j in range(1, p)] + [0]
    W = np.zeros((2 * p + 1, 3 * p + 1), dtype=int)
    for i in range(p):
        W[i, i] = r[i] + h[i]; W[i, p + i] = r[i]
        for j in range(p):
            W[i, 2 * p + j] = r[i] + u[j]; W[p + i, j] = s[i] + h[j]
        W[p + i, p + i] = t[i]
    W[p, p + 1] = s[0]
    for j in range(p):
        W[2 * p, j] = 2 + h[j]
    W[2 * p, 3 * p] = 1; W[2 * p, p] = 2
    return W, np.array(r + s + [2] + h + [0] * p + u + [0], float), 5 * p + 1


def face_l0_milp(W, q):
    """Return the exact smallest number of coordinates in which q differs from a face point.

    The answer comes from a mixed-integer program with a continuous dual y and one binary indicator
    per coordinate, which marks a disagreement between y and q. The program minimises the sum of
    the indicators, subject to y being feasible, the sum of y being the optimal value, and the
    absolute difference between y and q in each coordinate being at most M times the indicator of
    that coordinate. The constant M is larger than that absolute difference for every face point y,
    whose coordinates lie between zero and the largest weight, so an indicator equal to one leaves
    its coordinate of y free.

    The result is exact because, for a fixed set of agreeing coordinates, the restricted polytope
    is totally unimodular, so its optimal face contains an integral point. The dual y is kept
    continuous on purpose: with y declared integral, the HiGHS solver returned wrong optima on
    these instances.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    W = np.asarray(W, dtype=np.int64); na, nb = W.shape; n = na + nb
    q = np.asarray(q, float); OPT = matching_value(W)
    M = float(max(np.abs(q).max(), np.abs(W.max() - q).max())) + 1.0
    cons = bip_cons(W)
    # Each row of A with its entry of b states one covering constraint with both sides negated:
    # minus the sum of the two dual values of the edge is at most minus its weight.
    A, b = lp_rows(n, cons)
    rows = [np.r_[r_, np.zeros(n)] for r_ in A]; lo = [-np.inf] * len(rows); hi = list(b)
    rows.append(np.r_[np.ones(n), np.zeros(n)]); lo.append(OPT); hi.append(OPT)
    for v in range(n):
        r1 = np.zeros(2 * n); r1[v] = 1; r1[n + v] = -M; rows.append(r1); lo.append(-np.inf); hi.append(q[v])
        r2 = np.zeros(2 * n); r2[v] = -1; r2[n + v] = -M; rows.append(r2); lo.append(-np.inf); hi.append(-q[v])
    c = np.r_[np.zeros(n), np.ones(n)]
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=np.r_[np.zeros(n), np.ones(n)],
               bounds=Bounds(np.zeros(2 * n), np.r_[np.full(n, np.inf), np.ones(n)]))
    assert res.status == 0, res.message
    return int(round(res.fun))


def check_cold_closed():
    section(20, "THE COLD ENVELOPE IS floor(5 min/2) - 1: the families E_p and O_p, and the bound")
    # Cross-check the MILP with a continuous dual against the exact enumeration of the integral
    # face points, on 60 random draws of a weight matrix and a point.
    rng = np.random.default_rng(20260930); mism = 0; nchk = 0
    for _ in range(60):
        na = int(rng.integers(2, 4)); nb = int(rng.integers(na, na + 3))
        W = rng.integers(0, 6, size=(na, nb))
        if W.max() == 0: continue
        OPT = matching_value(W); pts = face_points_bip(W, OPT)
        q = rng.integers(0, 6, size=na + nb).astype(float)
        if face_l0_milp(W, q) != face_l0(q, pts): mism += 1
        nchk += 1
    assert mism == 0, mism
    print(f"    continuous-dual MILP == integral face enumeration on {nchk} random (W, q) pairs")
    for name, fam, ps in (("E", _cold_even_family, (2, 3, 4, 5)), ("O", _cold_odd_family, (2, 3, 4))):
        for p in ps:
            W, f, target = fam(p); na, nb = W.shape; n = na + nb; m = min(na, nb)
            cons = bip_cons(W); OPT = matching_value(W)
            assert is_feasible(f, cons) and abs(f.sum() - OPT) < 1e-9
            ext = face_extremes(n, cons, OPT)
            assert all(abs(hi - lo) < 1e-7 for lo, hi in ext), "optimal dual not unique"
            outs = _cold_outputs_all(W)
            assert len(outs) == 1, "cold output depends on the order"
            q = np.array(outs[0], float)
            lb = sum(1 for v in range(n) if q[v] < ext[v][0] - 1e-7 or q[v] > ext[v][1] + 1e-7)
            ub = int((np.abs(q - f) > 1e-9).sum())
            assert lb == ub == target == (5 * m) // 2 - 1, (name, p, lb, ub, target)
            print(f"    {name}_{p}: {na}x{nb}, m = {m}, unique optimal dual, one cold output under "
                  f"every weakly-descending order, face distance exactly {ub} = floor(5m/2) - 1")
    # The cold output of the 6 by 8 instance of section 12 of this program depends on the
    # tie-break. Over all weakly descending orders its cold outputs lie at several different
    # distances from the optimal face.
    W6 = COLD_6X8
    ds = sorted(face_l0_milp(W6, np.array(o, float)) for o in _cold_outputs_all(W6))
    assert ds == [11, 11, 12, 12, 13, 14], ds
    print(f"    the 6x8 witness of section [12] is not tie-break robust: its cold outputs over all "
          f"weakly-descending orders sit at face distances {ds} (the lexicographic one at 14)")
    # The bound itself, five halves of the smaller side, rounded down, minus 1. It is checked on
    # 200 random draws, over every tie-break and with exact distances to the optimal face.
    rng = np.random.default_rng(20260931); tested = 0; at_bound = 0
    for _ in range(200):
        na = int(rng.integers(2, 4)); nb = int(rng.integers(na, na + 3)); m = na
        W = rng.integers(0, 7, size=(na, nb))
        if W.max() == 0: continue
        OPT = matching_value(W); pts = face_points_bip(W, OPT); cap = (5 * m) // 2 - 1 if m >= 2 else 1
        for o in _cold_outputs_all(W):
            d = face_l0(np.array(o, float), pts); tested += 1
            assert d <= cap, (W.tolist(), o, d, cap)
            at_bound += (d == cap)
    print(f"    random sweep: {tested} cold outputs over all tie-breaks, 0 violations of "
          f"d <= floor(5m/2) - 1, {at_bound} at the bound")
    print("all assertions in [20] hold", elapsed())


# Section 21 of this program: the mixed game value of the view V0, where the repair and the
# adversary may both randomise.
def check_v0_mixed():
    section(21, "RANDOMISED OBLIVIOUS RULES: the mixed game value of V0 is 15/16 (exact certificate)")
    M = 5
    na, nb = 2, 3; p = np.array([0, 0, M, 0, M], float); viol = {(0, 1): M - 1, (1, 1): M - 2}
    comps = list(_view_completions(na, nb, p, viol)); assert len(comps) == (M + 1) ** 4
    Z = [np.array(z, float) for z in itertools.product(range(M + 1), repeat=5)
         if _feasible_for_all(z, na, nb, p, viol)]
    assert len(Z) == 1528
    pi = {(0, 0, 5, 4, 5): Fraction(7, 16), (5, 5, 0, 0, 0): Fraction(7, 16), (4, 4, 1, 0, 1): Fraction(1, 8)}
    for z in pi:
        assert _feasible_for_all(z, na, nb, p, viol)
    val = Fraction(15, 16)
    # The mixed strategy of the repair, a probability distribution over three outputs, guarantees
    # an expected ratio of at most fifteen sixteenths on every completion.
    worst = max(sum(pr * Fraction(face_l0(np.array(z, float), F), pre) for z, pr in pi.items())
                for h, W, F, pre in comps)
    assert worst == val, worst
    # The mixed completion of the adversary, a probability distribution over three completions,
    # forces an expected ratio of at least fifteen sixteenths on every output in Z, the list of
    # feasible integral outputs. Each completion lists its hidden weights in this order: u1 with
    # v1, u1 with c, u2 with v1 and u2 with c. These are the pairs of row 0 with column 0, row 0
    # with column 2, row 1 with column 0 and row 1 with column 2 of the 2 by 3 grid, which is the
    # order that _view_completions uses.
    byh = {h: (F, pre) for h, W, F, pre in comps}
    qadv = {(5, 0, 4, 1): Fraction(1, 2), (0, 0, 0, 0): Fraction(3, 8), (0, 5, 0, 5): Fraction(1, 8)}
    hidden_order = [(a, b) for a in range(na) for b in range(nb) if (a, b) not in viol]
    assert hidden_order == [(0, 0), (0, 2), (1, 0), (1, 2)]
    best = min(sum(pr * Fraction(face_l0(z, byh[h][0]), byh[h][1]) for h, pr in qadv.items())
               for z in Z)
    assert best == val, best
    print(f"    {len(comps)} completions, {len(Z)} feasible integral outputs; the mixed output "
          f"(7/16, 7/16, 1/8) on three outputs has expected ratio <= {val} on every completion, and the "
          f"mixed completion (1/2, 3/8, 1/8) forces expected ratio >= {val} on every output")
    print("    so the mixed game value of V0 is exactly 15/16 (deterministic value 5/4, section [15])")
    print("all assertions in [21] hold", elapsed())


# Section 22 of this program: the matching-first rule, which fires a maximum-weight matching and
# then keeps firing the constraint with the largest residual.
def _max_weight_matchings_all(W):
    """Find all maximum-weight matchings of a small bipartite instance.

    The function returns the maximum weight and the list of matchings that reach it. Each matching
    is a set of pairs, and a pair holds a row index followed by the number of rows plus a column
    index. Only pairs of positive weight are used.
    """
    W = np.asarray(W); na, nb = W.shape
    best, out = None, []
    def rec(a, used, M, wsum):
        nonlocal best, out
        if a == na:
            if best is None or wsum > best + 1e-9: best, out = wsum, [frozenset(M)]
            elif abs(wsum - best) < 1e-9: out.append(frozenset(M))
            return
        rec(a + 1, used, M, wsum)
        for b in range(nb):
            if b not in used and W[a, b] > 0:
                rec(a + 1, used | {b}, M + [(a, na + b)], wsum + W[a, b])
    rec(0, frozenset(), [], 0.0)
    return best, out


def _mf_outputs(W, yhat, M):
    """Return every output of the matching-first rule, which the program prints as R_M.

    The rule first fires the violated edges of the matching M. Any order gives the same point,
    because the edges of a matching are disjoint. It then keeps firing the violated constraint with
    the largest residual in the whole system until no constraint is violated. Every way to break a
    tie between largest residuals is followed, by expanding the set of reachable states. The result
    is a list with one array for each distinct final point.
    """
    W = np.asarray(W, float); na, nb = W.shape; cons = bip_cons(W)
    z = clamp(yhat)
    for a, j in M:
        w = W[a, j - na]
        if z[a] + z[j] < w - TOL:
            r = w - z[a] - z[j]; z[a] += r; z[j] += r
    states = {tuple(z)}; finals = set()
    while states:
        nxt = set()
        for st in states:
            y = np.array(st); vio = [(w - y[i] - y[j], i, j) for i, j, w in cons if y[i] + y[j] < w - TOL]
            if not vio: finals.add(st); continue
            rmax = max(v[0] for v in vio)
            for r, i, j in vio:
                if abs(r - rmax) < TOL:
                    y2 = y.copy(); y2[i] += rmax; y2[j] += rmax; nxt.add(tuple(y2))
        states = nxt
    return [np.array(f) for f in finals]


def _local_vs_global(W, yhat):
    """Run the matching-first rule with a local and with a global maximum-weight matching.

    The local matching has maximum weight only over the pairs that touch a violated constraint,
    that is, the pairs with an endpoint on one. The global matching has maximum weight over all
    pairs. The function returns six values: pre, which is the distance of the clamped prediction to
    the optimal face, the set V of vertices that lie on a violated constraint, the list of local
    matchings, the list of global matchings, the set of distances to the optimal face that the rule
    reaches with a local matching, and the same set for a global matching.
    """
    W = np.asarray(W); na, nb = W.shape; p = clamp(yhat); cons = bip_cons(W)
    pts = face_points_bip(W, matching_value(W)); pre = face_l0(p, pts)
    V = {v for i, j in violated(p, cons) for v in (i, j)}
    H = W.copy()
    for a in range(na):
        for b in range(nb):
            if a not in V and na + b not in V: H[a, b] = 0
    _, MH = _max_weight_matchings_all(H); _, MG = _max_weight_matchings_all(W)
    lo = {face_l0(z, pts) for M in MH for z in _mf_outputs(W, p, M)}
    gl = {face_l0(z, pts) for M in MG for z in _mf_outputs(W, p, M)}
    return pre, V, MH, MG, lo, gl


def check_matching_first():
    section(22, "THE MATCHING-FIRST RULE HAS THE CLASS CONSTANT 3/2: bound, attainment witness, local-matching failure")
    rng = np.random.default_rng(20260930)
    def per_f_bound(W, yhat, z, M, pts):
        """Check a bound on the output z against every integral face point f.

        The number of coordinates in which z differs from f must be at most the number of
        coordinates where the clamped prediction is below f, plus the number where it is above f,
        plus the number of low coordinates whose partner in the matching M is high. A coordinate is
        low when the clamped prediction is below f there and high when it is above. The function
        returns True when the bound holds for every f in pts.
        """
        p = clamp(yhat); partner = {}
        for a, j in M: partner[a] = j; partner[j] = a
        ok = True
        for f in pts:
            Cm = {v for v in range(len(p)) if p[v] < f[v] - TOL}; Cp = {v for v in range(len(p)) if p[v] > f[v] + TOL}
            L = {v for v in Cm if partner.get(v) in Cp}
            ok &= int((np.abs(z - f) > TOL).sum()) <= len(Cm) + len(Cp) + len(L)
        return ok
    tested = 0; worst = 0.0; excess = 0
    fams = [_trap_family(k)[:2] for k in (2, 3, 4)] + [_chain_family(k)[:2] for k in (2, 3)]
    for _ in range(120):
        na = int(rng.integers(2, 4)); nb = int(rng.integers(na, na + 3))
        W = rng.integers(0, 7, size=(na, nb)); yh = rng.integers(-2, 7, size=na + nb).astype(float)
        if W.max() > 0: fams.append((W, yh))
    for W, yh in fams:
        W = np.asarray(W, dtype=np.int64); OPT = matching_value(W); pts = face_points_bip(W, OPT)
        if len(pts) == 0: continue
        pre = face_l0(clamp(yh), pts)
        if pre == 0: continue
        _, Ms = _max_weight_matchings_all(W)
        for M in Ms:
            for z in _mf_outputs(W, yh, M):
                post = face_l0(z, pts); tested += 1
                assert post <= pre + pre // 2, (W.tolist(), yh.tolist(), sorted(M), post, pre)
                assert per_f_bound(W, yh, z, M, pts), (W.tolist(), yh.tolist(), sorted(M))
                worst = max(worst, post / pre); excess = max(excess, post - pre - pre // 2)
    print(f"    {tested} outputs (every maximum-weight matching, every tie-break) on T_k, C_k and 120 random "
          f"instances: post <= pre + floor(pre/2) and the per-dual bound |C| + |L| hold; worst ratio {worst:.4f}, "
          f"worst excess over the bound {excess}")
    # The witness that attains the bound: the ratio is exactly three halves, with a unique
    # maximum-weight matching and no tie.
    W = np.array([[5, 4, 7, 7, 3], [3, 2, 6, 7, 1]]); yh = np.array([2, 0, 0, 0, 4, 7, 0], float)
    OPT = matching_value(W); pts = face_points_bip(W, OPT); pre = face_l0(clamp(yh), pts)
    _, Ms = _max_weight_matchings_all(W); assert len(Ms) == 1
    outs = _mf_outputs(W, yh, Ms[0]); assert len(outs) == 1
    post = face_l0(outs[0], pts); assert (pre, post) == (4, 6), (pre, post)
    two = _two_round_construction(W, yh, pts[np.argmin((np.abs(pts - clamp(yh)) > TOL).sum(axis=1))])
    print(f"    witness 2x5: pre {pre}, matching-first output {tuple(int(x) for x in outs[0])} at face distance {post} = 3 pre/2; "
          f"the two-round construction lands at {face_l0(two, pts)}")
    # The failure of the local matching, which has maximum weight only among the pairs that have an
    # endpoint in the violated graph.
    pre, _, _, _, lo, gl = _local_vs_global(np.array([[5, 4, 0], [6, 0, 20]]), np.array([0, 6, 0, 0, 14], float))
    assert pre == 1 and lo == {2} and gl == {1}, (pre, lo, gl)
    print(f"    local matching (pairs incident to the violated graph only): pre {pre}, post {lo}; global matching: post {gl}")
    print("all assertions in [22] hold", elapsed())


# Section 23 of this program: the star trade-off. On the stars of order_family_W with 2 to 7
# columns every scan order is run, and the number of coordinates that differ from the optimal face
# is weighed against the largest absolute difference from it.
def _compositions_of(d):
    """Generate, as tuples, all the ways to write d as an ordered sum of positive parts."""
    if d == 0: yield (); return
    for first in range(1, d + 1):
        for rest in _compositions_of(d - first): yield (first,) + rest


def _D(a):
    """Return the largest of 1, every part of a except the last, and the last part minus 1.

    This is the distance to the optimal face of the star output whose fired gaps are a, measured as
    the largest absolute difference in one coordinate. The same number is that distance measured as
    the largest increase plus the largest decrease. Here a lists the gaps between consecutive fired
    columns, and its first part is the position of the first fired column, counted from 1. The code
    stores the two distances in dinf and dpm.
    """
    return max([1] + list(a[:-1]) + [a[-1] - 1])


def check_star_tradeoff():
    section(23, "EVERY ORDER ON THE STAR: compositions, the hyperbola d0 * dinf >= d-1, ceil(sqrt(d-1)), H_d, the admissible class")
    for d in range(2, 8):
        W = order_family_W(d); cons = bip_cons(W); n = d + 1; OPT = matching_value(W); pts = face_points_bip(W, OPT)
        seen = {}; Hd = Fraction(0); cnt = 0
        for perm in itertools.permutations(range(d)):
            y, fired = greedy(np.zeros(n), cons, list(perm))
            S = sorted(fired); assert d - 1 in S
            ltr = [k for i, k in enumerate(perm) if all(k > perm[j] for j in range(i))]
            assert sorted(ltr) == S
            a = tuple([S[0] + 1] + [S[i] - S[i - 1] for i in range(1, len(S))])
            d0 = face_l0(y, pts); dinf = face_dist_lp(n, cons, OPT, y, "linf"); dpm = face_dist_lp(n, cons, OPT, y, "pm")
            assert d0 == len(S) and abs(dinf - _D(a)) < 1e-7 and abs(dpm - _D(a)) < 1e-7, (d, perm, a, d0, dinf, dpm)
            assert y[0] == d and all(abs(y[1 + S[i]] - a[i]) < 1e-9 for i in range(len(S)))
            seen[tuple(y)] = (d0, _D(a)); Hd += Fraction(d0); cnt += 1
        assert len(seen) == 2 ** (d - 1)
        assert Hd / cnt == sum(Fraction(1, i) for i in range(1, d + 1))
        mm = min(max(v) for v in seen.values()); assert mm == math.ceil(math.sqrt(d - 1) - 1e-12)
        assert all(v[0] * v[1] >= d - 1 for v in seen.values())
        print(f"    d = {d}: all {cnt} orders give {len(seen)} outputs; d0 = |S|, dinf = dinf^pm = D(a); E[d0] = {Hd/cnt} = H_d; "
              f"min max(d0, dinf) = {mm} = ceil(sqrt(d-1))")
    for d in range(2, 19):
        vals = {}
        for a in _compositions_of(d):
            m = len(a); D = _D(a); assert m * D >= d - 1
            vals[m] = min(vals.get(m, 10 ** 9), D)
        assert all(vals[m] == max(1, -(-(d - 1) // m)) for m in range(1, d + 1))
        assert min(max(m, v) for m, v in vals.items()) == math.ceil(math.sqrt(d - 1) - 1e-12)
    print("    compositions d = 2..18: per-m frontier max(1, ceil((d-1)/m)) and min-max ceil(sqrt(d-1)) hold")
    for d in range(2, 401):
        t = math.ceil(math.sqrt(d - 1) - 1e-12); q = d - t - 1
        a = (d,) if q == 0 else tuple([t] * (-(-q // t) - 1) + [q - (-(-q // t) - 1) * t] + [t + 1])
        assert sum(a) == d and max(len(a), _D(a)) == t, (d, a)
    print("    witness composition attains ceil(sqrt(d-1)) for d = 2..400")
    # The balanced mirror of the star, the family that the output calls B_n, for n from 2 to 5. For
    # n equal to 2, every order of the n times n constraints is run. For n at least 3, 400 random
    # orders are run.
    rng = np.random.default_rng(20260930)
    for n in range(2, 6):
        W = balanced_W(n); cons = bip_cons(W); N = 2 * n; OPT = matching_value(W)
        yh = f_star(n).copy(); yh[0] -= n
        orders = itertools.permutations(range(n * n)) if n == 2 else (rng.permutation(n * n) for _ in range(400))
        seen = set()
        for perm in orders:
            y, fired = greedy(yh, cons, list(perm))
            cols = sorted(cons[c][1] - n + 1 for c in fired); assert all(cons[c][0] == 0 for c in fired) and n in cols
            a = tuple([cols[0]] + [cols[i] - cols[i - 1] for i in range(1, len(cols))])
            d0 = hamming_to_balanced_face(n, y); dinf = face_dist_lp(N, cons, OPT, y, "linf"); dpm = face_dist_lp(N, cons, OPT, y, "pm")
            assert d0 == len(cols) and abs(dinf - _D(a)) < 1e-7 and abs(dpm - _D(a)) < 1e-7, (n, perm, a, d0, dinf, dpm)
            seen.add(tuple(cols))
        if n == 2: assert len(seen) == 2 ** (n - 1)
        print(f"    B_{n}: {'all' if n == 2 else '400 random'} orders, only row a_1 fires, d0 = |S| and dinf = dinf^pm = D(a) ({len(seen)} fired sets seen)")
    # The admissible outputs on the star, for 3 to 5 columns. Three things are checked. First, a
    # direct characterisation of admissibility, adm_char, agrees with the test adm_lp, which uses
    # the rows of _admissible_polytope_rows, on every integral point that is examined. Second, on
    # every admissible one of these points the product of two distances is at least d minus 2. The
    # two distances are the number of coordinates that differ from the optimal face and the largest
    # absolute difference from it, and the section title calls a bound of this kind on their
    # product the hyperbola. Third, for every k from 1 to d an explicit admissible point has the
    # first distance equal to k and the second equal to (d minus 2) divided by k, so that its
    # product is exactly d minus 2.
    for d in range(3, 6):
        W = order_family_W(d); cons = bip_cons(W); n = d + 1; OPT = matching_value(W); p = np.zeros(n)
        rows = _admissible_polytope_rows(n, cons, p)
        def adm_lp(z): return is_feasible(z, cons) and all(float(sg @ z) + ph <= 1e-7 for sg, ph in rows)
        def adm_char(z):
            if not is_feasible(z, cons) or abs(z[0] - z[1:].sum()) > 1e-9: return False
            return all(z[1:1 + k + 1].sum() <= k + 1 + 1e-9 for k in range(d))
        box = itertools.product(range(d + 1), repeat=n); nchk = 0; worst_prod = None
        for z in box:
            z = np.array(z, float)
            # A point farther than d from the dual with d on the row and zero on every column, in
            # the sum of absolute differences, cannot be admissible, so it is skipped.
            if abs(z[0] - d) + z[1:].sum() > d: continue
            assert adm_lp(z) == adm_char(z), z; nchk += 1
            if adm_char(z):
                d0 = face_l0(z, face_points_bip(W, OPT)); dinf = face_dist_lp(n, cons, OPT, z, "linf")
                assert d0 * dinf >= d - 2 - 1e-7
                worst_prod = d0 * dinf if worst_prod is None else min(worst_prod, d0 * dinf)
        for k in range(1, d + 1):
            eps = (d - 2) / k; z = np.zeros(n); z[0] = d - 1
            for i in range(d - k, d - 1): z[1 + i] = eps
            z[d] = eps + 1
            assert adm_char(z) and adm_lp(z)
            # A fractional face point differs from this point at the row and at the last column, so
            # the integral face points are enough here.
            assert face_l0(z, face_points_bip(W, OPT)) == k
            dinf = face_dist_lp(n, cons, OPT, z, "linf"); assert abs(dinf - eps) < 1e-7
        print(f"    d = {d}: characterisation == LP oracle on {nchk} integral points; d0 * dinf >= d-2 on all admissible "
              f"ones (min product {worst_prod}); z_k admissible with dinf = (d-2)/k for k = 1..{d}")
    print("all assertions in [23] hold", elapsed())


# Section 24 of this program: one-hop views. On one witness the matching-first rule with a local
# matching doubles the distance to the optimal face, while the one-hop view of the same instance
# has game value one half.
def check_one_hop():
    section(24, "ONE-HOP VIEWS: the cheap local-matching rule fails at ratio 2 while its one-hop view has value 1/2")
    W = np.array([[2, 6], [5, 2], [6, 4]]); yh = np.array([-1, -2, -1, 4, 6], float); p = clamp(yh); na, nb = W.shape
    pre, V, MH, MG, lo, gl = _local_vs_global(W, yh)
    assert pre == 1 and lo == {2} and gl == {1} and len(MH) == 1 and len(MG) == 1, (pre, lo, gl)
    print(f"    3x2 witness: pre {pre}; local matching of the revealed pairs then largest residual: post {lo}; global matching: post {gl}")
    # The one-hop view of this instance shows every pair that has an endpoint on a violated
    # constraint. It hides only the pair of row 0 with column 1. The weight of that pair can be
    # any whole number from 0 to 6, the sum of the two clamped prediction values at its endpoints,
    # so the view has 7 completions. Its game value is one half.
    hidden = [(a, b) for a in range(na) for b in range(nb) if a not in V and na + b not in V]
    assert hidden == [(0, 1)]
    base = W.copy(); comps = []
    for h in range(int(p[0] + p[na + 1]) + 1):
        Wc = base.copy(); Wc[0, 1] = h; O = matching_value(Wc); F = face_points_bip(Wc, O); comps.append((F, face_l0(p, F)))
    Wmax = W.copy(); Wmax[0, 1] = int(p[0] + p[na + 1]); cmax = bip_cons(Wmax)
    # Coordinates above 6 are not needed, because 6 is the largest weight, as explained in
    # _feasible_for_all.
    Z = [np.array(z, float) for z in itertools.product(range(7), repeat=5)
         if is_feasible(np.array(z, float), cmax)]
    val = min(max(Fraction(face_l0(z, F), pr) for F, pr in comps) for z in Z)
    assert val == Fraction(1, 2), val
    print(f"    its one-hop view ({len(comps)} completions, {len(Z)} feasible outputs) has game value {val}: the failure is the rule's, not the view's")
    print("all assertions in [24] hold", elapsed())


# The main function runs the sections in the order of their numbers, which is not the order of
# their definitions in this file.
def main():
    check_theorem_a()
    check_witness_search()
    check_theorem_b()
    check_coverscan_witness()
    check_packing_mirror()
    check_walk()
    check_balanced_face()
    check_mintrick()
    check_tight_rows()
    check_l1_nearest()
    check_face_only_admissibility()
    check_table_stream()
    check_solver_stream()
    check_three_halves()
    check_chain_family()
    check_cold_envelope()
    check_trap_family()
    check_ladder_family()
    check_oblivious_lb()
    check_even_parity()
    check_cold_five_halves()
    check_fourthirds_view()
    check_star_views()
    check_cold_closed()
    check_v0_mixed()
    check_matching_first()
    check_star_tradeoff()
    check_one_hop()
    print("\nall assertions passed")
    print(f"total wall-clock time: {time.time() - T_START:.1f}s")


if __name__ == "__main__":
    main()
