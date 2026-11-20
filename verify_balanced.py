#!/usr/bin/env python3
"""Check the balanced lower bound of Theorem 13, which shows that sorting is not additive.

The problem is maximum-weight bipartite matching, with a weight for every pair of a row and a
column. Its dual has one value per vertex. A dual is feasible when no value is negative and
the two dual values of every edge add up to at least the edge weight. The optimal face is the
set of all optimal duals. A prediction is a guessed dual that may be infeasible. The repair
scans the edges in a given order and, at each violated one, raises both dual values by the
full residual. The distance of a point to the optimal face is the number of coordinates in
which it differs from the nearest point of the face (the paper calls this the l0 or Hamming
distance). Sorting, the scan by decreasing weight, would be additive if the distance after
the repair were always at most the distance before it plus a constant.

The balanced family that the paper uses for Theorem 13 has, for each n, an instance with n
rows and n columns. For n from 2 to 8 the script checks that the optimal value is n squared,
that the prediction is at distance 1 from the optimal face, and that the repaired dual is at
distance n under every weakly descending scan order, that is, every order in which the
weights never increase (which orders are tried is described below). So the distance grows by
the size of the smaller side minus 1, starting from a prediction that is wrong in a single
coordinate. The first table prints the two distances as pre and post and their difference as
excess.

For n from 2 to 7 the script also checks the reversal reported in Remark 33. Measured to the
same face by the largest absolute difference (the l-infinity distance), the output of the
scan by decreasing weight is at distance 1 and the output of the scan by increasing weight is
at distance n minus 1. So when n is at least 3, the largest absolute difference ranks the two
orders the opposite way to the number of differing coordinates. When n equals 2 both outputs
are at distance 1, so the two orders tie in this measure. The same two values are checked in
the norm of Sakaue and Oki, which adds the largest increase to the largest decrease (the
second table labels it pm). It is taken in covering coordinates, that is, on the dual values
as this script holds them. Remark 33 adds that in the difference coordinates that Sakaue and
Oki use themselves, the output of the scan by increasing weight is at distance n instead.

Every distance is an exact minimum over the full optimal face, fractional points included.
The face is the set of points with no negative value in which the two values of every edge
add up to at least the edge weight and all the values together add up to the optimal value.
The number of differing coordinates is minimised over that set by a mixed-integer program,
with one indicator variable per coordinate that is tied to it by a large constant (the big-M
method), so the face is nowhere assumed to be integral. (The paper notes separately that
enumerating the integral points would be enough. This script does not rely on that.) The
other distances are minimised over the same set by linear programs.

When n is at most 4, every weakly descending order is tried (there are 20,736 of them when n
equals 4). For larger n a random sample of 300 of them is tried. Step 3 of the proof of the
theorem shows that the output does not depend on the order, so the sampling only spot-checks
that proof.

The script requires numpy and scipy. Run it with: python3 verify_balanced.py
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment, milp, linprog, LinearConstraint, Bounds


def balanced_family(n):
    """Build the balanced instance of Theorem 13 with n rows and n columns.

    The first row is the distinguished one, and the other n minus 1 rows are identical copies
    of a second row. The function returns the weight matrix, then the row values and the
    column values of an optimal dual, then the row values and the column values of the
    prediction. The prediction is that optimal dual with its one nonzero row value set to
    zero.
    """
    W = np.zeros((n, n))
    for k in range(1, n + 1):
        # Rows and columns are counted from 1 in this loop. The first row has weight
        # (2 times n) minus k in column k.
        W[0, k - 1] = 2 * n - k
        for i in range(2, n + 1):
            # Every other row has weight 2 times (n minus k) in column k.
            W[i - 1, k - 1] = 2 * (n - k)
    u_star = np.zeros(n); u_star[0] = n
    v_star = np.array([2.0 * (n - k) for k in range(1, n + 1)])
    y_hat_u = u_star.copy(); y_hat_u[0] = 0.0        # Set the one nonzero row value to zero.
    return W, u_star, v_star, y_hat_u, v_star.copy()


def opt_value(W):
    r, c = linear_sum_assignment(-W)
    return float(W[r, c].sum())


def greedy_repair(W, u0, v0, order):
    """Run the two-sided greedy repair of Definition 1 in the given scan order.

    u0 and v0 are the row values and the column values of the prediction. They are first
    clamped, with negative values replaced by zero. The edges are then scanned in the given
    order, which is a list of (row, column) pairs. At each violated edge both of its dual
    values are raised by the full residual, the weight minus the sum of the two dual values.
    The function returns the repaired row values, the repaired column values and the number
    of raises, that is, the number of constraints that fired.
    """
    u = np.maximum(u0, 0).astype(float).copy()
    v = np.maximum(v0, 0).astype(float).copy()
    fires = 0
    for a, b in order:
        r = W[a, b] - u[a] - v[b]
        if r > 0:
            u[a] += r; v[b] += r; fires += 1
    return u, v, fires


def face_hamming(y, W, opt):
    """Return the exact smallest number of coordinates in which y differs from a face point.

    y is a dual as one vector, row values first and then column values. W is the weight matrix
    and opt is the optimal value. The minimum is taken over the full optimal face, fractional
    points included. It is found by a mixed-integer program whose variables are a face point
    and one binary indicator per coordinate. An indicator at zero forces the face point to
    agree with y in that coordinate, and the program minimises the number of indicators at 1.
    """
    na, nb = W.shape
    N = na + nb
    BIG = float(max(W.max(), y.max())) + 10.0
    cost = np.concatenate([np.zeros(N), np.ones(N)])
    A, lo, hi = [], [], []
    for a in range(na):
        for b in range(nb):
            row = np.zeros(2 * N); row[a] = 1; row[na + b] = 1
            A.append(row); lo.append(float(W[a, b])); hi.append(np.inf)
    row = np.zeros(2 * N); row[:N] = 1
    A.append(row); lo.append(opt); hi.append(opt)
    for i in range(N):
        r1 = np.zeros(2 * N); r1[i] = 1; r1[N + i] = -BIG
        A.append(r1); lo.append(-np.inf); hi.append(float(y[i]))
        r2 = np.zeros(2 * N); r2[i] = 1; r2[N + i] = BIG
        A.append(r2); lo.append(float(y[i])); hi.append(np.inf)
    res = milp(c=cost,
               constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.concatenate([np.zeros(N), np.ones(N)]),
               bounds=Bounds(np.zeros(2 * N),
                             np.concatenate([np.full(N, BIG), np.ones(N)])))
    assert res.success, "MILP failed"
    return int(round(res.fun))


def _face_rows(W, opt, nextra):
    """Build the constraint rows that every linear program over the optimal face shares.

    The variables of such a program are the N values of a face point, rows of W first and then
    columns, followed by nextra extra variables that belong to the caller. Four things are
    returned. The first two are the inequality rows and their right sides. They say that the
    two values of every edge add up to at least the edge weight, with all signs flipped
    because linprog wants every row to be at most its right side. The third is the
    coefficient row of the equation that makes all the values add up to the optimal value.
    The fourth is N. The caller supplies the right side of that equation, and it keeps every
    variable at least zero through the bounds it passes to linprog.
    """
    na, nb = W.shape; N = na + nb
    A, b = [], []
    for a in range(na):
        for b_ in range(nb):
            r = np.zeros(N + nextra); r[a] = -1; r[na + b_] = -1
            A.append(r); b.append(-float(W[a, b_]))
    Aeq = np.zeros((1, N + nextra)); Aeq[0, :N] = 1
    return A, b, Aeq, N


def face_linf(y, W, opt):
    """Return the exact distance from y to the optimal face as a largest absolute difference.

    The distance between two points is here the largest absolute difference in any coordinate.
    The distance to the face is the smallest such value over the full optimal face, fractional
    points included. It is found by a linear program.
    """
    A, b, Aeq, N = _face_rows(W, opt, 1)
    for i in range(N):
        r = np.zeros(N + 1); r[i] = 1;  r[N] = -1; A.append(r); b.append(float(y[i]))
        r = np.zeros(N + 1); r[i] = -1; r[N] = -1; A.append(r); b.append(-float(y[i]))
    c = np.zeros(N + 1); c[N] = 1
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[opt],
                  bounds=[(0, None)] * (N + 1), method="highs")
    assert res.success, "l_inf LP failed"
    return float(res.fun)


def face_pm(y, W, opt):
    """Return the exact smallest two-sided distance from y to the optimal face.

    The two-sided norm of a vector adds its largest positive entry to its largest negative
    entry taken without the sign, and a side with no entries counts as zero. Here the vector
    is y minus the face point, in covering coordinates. So the distance is the largest amount
    by which y is above the face point plus the largest amount by which it is below. This is
    the norm of Sakaue and Oki (arXiv:2205.09961, Section 3). They state that it is at least
    the largest absolute entry and at most twice the largest absolute entry. The minimum over
    the full optimal face is found by a linear program.
    """
    A, b, Aeq, N = _face_rows(W, opt, 2)
    for i in range(N):
        r = np.zeros(N + 2); r[i] = 1;  r[N] = -1;     A.append(r); b.append(float(y[i]))
        r = np.zeros(N + 2); r[i] = -1; r[N + 1] = -1; A.append(r); b.append(-float(y[i]))
    c = np.zeros(N + 2); c[N] = 1; c[N + 1] = 1
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[opt],
                  bounds=[(0, None)] * (N + 2), method="highs")
    assert res.success, "l_inf^pm LP failed"
    return float(res.fun)


def face_l1(y, W, opt):
    """Return the exact distance from y to the optimal face as a sum of absolute differences.

    The paper calls this the l1 distance. It is the smallest such sum over the full optimal
    face, fractional points included, and it is found by a linear program. Remark 33 reports
    this quantity for the balanced family: it equals n for both the descending and the
    ascending scan, so it cannot tell them apart.
    """
    A, b, Aeq, N = _face_rows(W, opt, 0)
    # The variables are the N values of the face point, followed by N helper variables, one
    # per coordinate. Each helper is at least the absolute difference between y and the face
    # point in its coordinate.
    A = [np.concatenate([r, np.zeros(N)]) for r in A]
    Aeq = np.concatenate([Aeq, np.zeros((1, N))], axis=1)
    for i in range(N):
        r = np.zeros(2 * N); r[i] = 1;  r[N + i] = -1; A.append(r); b.append(float(y[i]))
        r = np.zeros(2 * N); r[i] = -1; r[N + i] = -1; A.append(r); b.append(-float(y[i]))
    c = np.concatenate([np.zeros(N), np.ones(N)])
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[opt],
                  bounds=[(0, None)] * (2 * N), method="highs")
    assert res.success, "l1 LP failed"
    return float(res.fun)


def descending_orders(W, limit=None, rng=None):
    """List weakly descending scan orders of the edges of W.

    An order is weakly descending when the weights never increase along it, so edges of equal
    weight may come in any order among themselves. When limit is None, every such order is
    returned. Otherwise limit random ones are returned, each made with the generator rng by
    shuffling every group of equal-weight edges. An order is a list of (row, column) pairs.
    """
    na, nb = W.shape
    groups = {}
    for a in range(na):
        for b in range(nb):
            groups.setdefault(W[a, b], []).append((a, b))
    keys = sorted(groups, reverse=True)
    if limit is None:
        outs = [[]]
        for k in keys:
            outs = [o + list(pm) for o in outs for pm in itertools.permutations(groups[k])]
        return outs
    outs = []
    for _ in range(limit):
        o = []
        for k in keys:
            g = groups[k][:]; rng.shuffle(g); o += g
        outs.append(o)
    return outs


def main(nmax=8, sample=300, seed=3):
    rng = np.random.default_rng(seed)
    print(f"{'n':>3} {'OPT':>6} {'n^2':>6} {'pre':>5} {'post':>5} {'excess':>7} "
          f"{'n-1':>4} {'orders':>14} {'fires':>7}")
    for n in range(2, nmax + 1):
        W, _, _, u0, v0 = balanced_family(n)
        opt = opt_value(W)
        assert abs(opt - n * n) < 1e-9, f"OPT != n^2 at n={n}"
        pre = face_hamming(np.concatenate([u0, v0]), W, opt)
        assert pre == 1, f"pre != 1 at n={n}: {pre}"

        if n <= 4:
            orders = descending_orders(W); tag = f"{len(orders)} (all)"
        else:
            orders = descending_orders(W, limit=sample, rng=rng); tag = f"{sample} sampled"

        posts, fires = set(), set()
        for o in orders:
            ru, rv, f = greedy_repair(W, u0, v0, o)
            assert (ru[:, None] + rv[None, :] >= W - 1e-9).all(), "output infeasible"
            posts.add(face_hamming(np.concatenate([ru, rv]), W, opt))
            fires.add(f)
        assert posts == {n}, f"post distance not uniformly n at n={n}: {sorted(posts)}"
        assert fires == {n}, f"firing count not uniformly n at n={n}: {sorted(fires)}"
        print(f"{n:>3} {opt:>6.0f} {n*n:>6d} {pre:>5d} {n:>5d} {n-1:>7d} "
              f"{n-1:>4d} {tag:>14} {n:>7d}")

    # The companion claim. The weakly increasing scan ends at distance 1 from the face, that is,
    # one differing coordinate. And every quantity that sums over the coordinates is identical
    # for the two orders: the dual gap, the total displacement from the clamped prediction and
    # the sum of absolute differences to the face.
    print("\nascending vs descending: the l1 quantities are blind, l_inf separates BACKWARDS")
    print(f"{'n':>3} {'l0 desc':>8} {'l0 asc':>7} {'linf desc':>10} {'linf asc':>9} "
          f"{'pm desc':>8} {'pm asc':>7} {'gap both':>9} {'l1 disp both':>13} "
          f"{'l1-to-face both':>16}")
    for n in range(2, min(nmax, 7) + 1):
        W, _, _, u0, v0 = balanced_family(n)
        opt = opt_value(W)
        P = [(a, b) for a in range(n) for b in range(n)]
        desc = sorted(P, key=lambda t: (-W[t[0], t[1]], t))
        asc = sorted(P, key=lambda t: (W[t[0], t[1]], t))
        out = {}
        pre_vec = np.concatenate([np.maximum(u0, 0), np.maximum(v0, 0)])
        for name, o in (("desc", desc), ("asc", asc)):
            ru, rv, _ = greedy_repair(W, u0, v0, o)
            y = np.concatenate([ru, rv])
            out[name] = (face_hamming(y, W, opt),
                         float(y.sum() - opt),
                         float(np.abs(y - pre_vec).sum()),
                         face_linf(y, W, opt),
                         face_pm(y, W, opt),
                         face_l1(y, W, opt))
        assert out["desc"][1] == out["asc"][1], "dual gaps differ"
        assert out["desc"][2] == out["asc"][2], "realised l1 displacements differ"
        # Remark 33: the sum of absolute differences to the face is n for both orders.
        assert abs(out["desc"][5] - out["asc"][5]) < 1e-7, "l1 distances to the face differ"
        assert abs(out["desc"][5] - n) < 1e-7, f"l1 distance to the face != n at n={n}"
        assert out["asc"][0] == 1, f"ascending scan not at distance 1 at n={n}"
        # The reversal: the order that is catastrophic in the number of differing coordinates
        # is the optimal one in the largest absolute difference.
        assert abs(out["desc"][3] - 1.0) < 1e-7, f"desc l_inf != 1 at n={n}"
        assert abs(out["asc"][3] - (n - 1)) < 1e-7, f"asc l_inf != n-1 at n={n}"
        for n_ in ("desc", "asc"):
            assert abs(out[n_][3] - out[n_][4]) < 1e-7, "l_inf and l_inf^pm differ"
        print(f"{n:>3} {out['desc'][0]:>8d} {out['asc'][0]:>7d} "
              f"{out['desc'][3]:>10.0f} {out['asc'][3]:>9.0f} "
              f"{out['desc'][4]:>8.0f} {out['asc'][4]:>7.0f} "
              f"{out['desc'][1]:>9.0f} {out['desc'][2]:>13.0f} "
              f"{out['desc'][5]:>16.0f}")
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
