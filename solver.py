#!/usr/bin/env python3
"""
This file holds a warm-startable primal-dual solver for the dual of maximum-weight bipartite
matching, with an iteration counter.

The experiments use the solver to test, end to end, whether the number of coordinates in
which a repaired dual differs from the reference optimum, one fixed optimal dual of the
instance, predicts the work that a warm-started solver does. The paper calls that number the
l0 distance after the repair.

The algorithm is the standard Hungarian, or successive-shortest-path, method. It runs on
potentials, which are dual values, with one value per row in u and one value per column in v.
The two values of every edge add up to at least the edge weight throughout the run. The dual
of the paper also requires every value to be at least zero. The solver does not check that
requirement, and it may lower a row value below zero. Each phase starts from an unmatched
row, makes a dual adjustment whenever no tight edge leaves its search tree, and ends by
augmenting the matching. We count the dual adjustments.
This is the quantity that Lemma 11 of Dinitz et al. bounds by the dual gap, which is the sum
of all dual values minus the optimal value: on integer weights and integer starting duals,
each adjustment lowers the sum of all dual values by at least 1. Corollary 6 of the paper
uses this bound.

Correctness is checked against linear_sum_assignment from scipy in two functions. The
function self_test is a fast smoke test. The function verify_reference runs a designed
population that includes the degenerate and sparse weight matrices on which
successive-shortest-path implementations tend to break. Each matrix is solved from several
feasible starting duals: the cold row reduction and column reduction, the repaired zero
prediction, three repaired warm starts, and the all-zero dual where it is feasible.

Run python3 solver.py for the smoke test followed by the full verification. Add the option
quick, typed with two hyphens in front of it, to run the smoke test only.
"""
import numpy as np


def solve_warm(W, u0, v0):
    """Find a maximum-weight perfect matching on the n by n matrix W from feasible starting duals.

    The starting duals are u0 for the rows and v0 for the columns. The function returns
    three values: the value of the matching, the number of dual adjustments and the number
    of phases.

    The potentials are kept feasible throughout: the two potentials of every edge add up to
    at least the edge weight. An edge is tight when its constraint holds with equality. The
    algorithm augments along tight edges and, when it is stuck, shifts the potentials by the
    smallest slack that creates a new tight edge. The slack of an edge is the sum of its two
    potentials minus its weight.
    """
    n = W.shape[0]
    u = np.array(u0, dtype=float).copy()
    v = np.array(v0, dtype=float).copy()
    # The start must already be feasible, because this solver does not repair.
    slack0 = W - (u[:, None] + v[None, :])
    if slack0.max() > 1e-9:
        raise ValueError("solve_warm requires feasible duals; max violation %.6g" % slack0.max())

    match_row = -np.ones(n, dtype=int)   # The column matched to each row, or minus 1 if none.
    match_col = -np.ones(n, dtype=int)   # The row matched to each column, or minus 1 if none.
    adjustments = 0
    phases = 0

    for root in range(n):
        if match_row[root] != -1:
            continue
        phases += 1
        minv = np.full(n, np.inf)     # The smallest slack from the tree to each column.
        way = np.full(n, -1, dtype=int)   # For each column, the row that attains minv.
        used_col = np.zeros(n, dtype=bool)
        i0 = root
        rows_in_tree = [root]
        while True:
            slack = (u[i0] + v) - W[i0]        # The slack is at least zero by feasibility.
            better = (~used_col) & (slack < minv)
            minv = np.where(better, slack, minv)
            way = np.where(better, i0, way)
            cand = np.where(used_col, np.inf, minv)
            j1 = int(np.argmin(cand))
            delta = float(cand[j1])
            if delta > 1e-9:
                adjustments += 1
                u[rows_in_tree] -= delta
                v[used_col] += delta
                minv = np.where(used_col, minv, minv - delta)
            used_col[j1] = True
            if match_col[j1] == -1:
                break
            i0 = int(match_col[j1])
            rows_in_tree.append(i0)
        # Augment along the alternating path that ends at j1, in the standard e-maxx form.
        j = j1
        while j != -1:
            i = int(way[j])
            prev = int(match_row[i])
            match_row[i] = j
            match_col[j] = i
            j = prev
    val = float(sum(W[a, match_row[a]] for a in range(n)))
    return val, adjustments, phases


def self_test(seeds=(0, 1, 2, 3, 4), sizes=(6, 10, 20, 40)):
    """Check on random matrices that the warm solver finds the true optimum.

    The warm solver must find the true optimum from any feasible start. This smoke test
    starts it from the row-reduction dual and compares the value it reaches with the optimum
    that scipy finds. The function returns the number of instances that matched.
    """
    from scipy.optimize import linear_sum_assignment
    ok = 0
    for s in seeds:
        rng = np.random.default_rng(s)
        for n in sizes:
            W = rng.integers(0, 50, size=(n, n)).astype(float)
            r, c = linear_sum_assignment(-W)
            true = W[r, c].sum()
            # A trivial feasible dual: every row gets its largest weight and every column gets
            # zero. This is the row-reduction start of the Hungarian method.
            u0 = W.max(axis=1).astype(float); v0 = np.zeros(n)
            val = solve_warm(W, u0, v0)[0]
            assert abs(val - true) < 1e-6, f"n={n} seed={s}: got {val}, want {true}"
            ok += 1
    return ok


# Verification.
# Everything below calls solve_warm and compares the value it returns with an exact
# reference, which is scipy.optimize.linear_sum_assignment on the same matrix. The function
# verify_reference also checks the number of phases and a bound on the number of adjustments.
#
# The population is chosen on purpose. A successive-shortest-path implementation that is
# correct on dense uniform weights can still be wrong in two situations. In the first, the
# weights are heavily degenerate, so many columns tie for the smallest slack and the dual
# adjustment is zero or shared. In the second, the weights are sparse, so the subgraph of
# tight edges is nearly disconnected and augmenting paths are long. An implementation can
# also be correct from a cold start and wrong from a warm one, which is the case that the
# tables of the paper rely on. All three cases are covered.


def _instance_bank():
    """Build the bank of test matrices, each paired with the name of its family.

    All matrices are square, because solve_warm is a routine for square matrices.
    """
    bank = []

    # Dense and non-degenerate matrices, the regime that self_test already covered.
    for s in range(6):
        rng = np.random.default_rng(100 + s)
        for n in (4, 6, 10, 20, 40, 60):
            bank.append(("dense_uniform", rng.integers(0, 50, size=(n, n)).astype(float)))
    for s in range(4):
        rng = np.random.default_rng(200 + s)
        for n in (6, 20, 50):
            bank.append(("dense_wide_range", rng.integers(0, 1000, size=(n, n)).astype(float)))
    for s in range(5):                      # The clustered family of the paper's tables.
        rng = np.random.default_rng(300 + s)
        for n in (10, 30, 60):
            W = rng.integers(0, 20, size=(n, n)).astype(float)
            for _ in range(max(1, n // 8)):
                i, j = rng.integers(0, n, 2); sz = max(2, n // 6)
                W[i:i + sz, j:j + sz] += rng.integers(50, 100)
            bank.append(("dense_clustered", W))
    for s in range(5):                      # The euclidean family of the paper's tables.
        rng = np.random.default_rng(350 + s)
        for n in (10, 30, 60):
            p, q = rng.random((n, 2)), rng.random((n, 2))
            D = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
            bank.append(("dense_euclidean", np.round(100 * (1 - D / D.max()))))

    # Heavily degenerate matrices, with many equal weights and large blocks of ties.
    for s in range(6):                      # Only the weights 0, 1 and 2, so ties are everywhere.
        rng = np.random.default_rng(400 + s)
        for n in (5, 10, 20, 40):
            bank.append(("degenerate_small_alphabet",
                         rng.integers(0, 3, size=(n, n)).astype(float)))
    for s in range(2):                      # Weights of 0 or 1, at several densities.
        rng = np.random.default_rng(500 + s)
        for n in (5, 10, 20, 40):
            for p in (0.1, 0.25, 0.5, 0.75, 0.9):
                bank.append(("degenerate_binary01",
                             (rng.random((n, n)) < p).astype(float)))
    for c in (0.0, 1.0, 7.0, 50.0):         # All weights equal: every perfect matching is optimal.
        for n in (3, 5, 10, 20, 40):
            bank.append(("degenerate_all_equal", np.full((n, n), c)))
    for s in range(4):                      # Large tie blocks, constant on k by k blocks.
        rng = np.random.default_rng(600 + s)
        for n in (8, 12, 20, 40):
            for k in (2, 4):
                nb = (n + k - 1) // k
                B = rng.integers(0, 4, size=(nb, nb)).astype(float)
                bank.append(("degenerate_tie_blocks",
                             np.kron(B, np.ones((k, k)))[:n, :n].copy()))
    for s in range(4):                      # Constant rows, and constant columns.
        rng = np.random.default_rng(700 + s)
        for n in (5, 10, 20, 40):
            r = rng.integers(0, 6, size=n).astype(float)
            bank.append(("degenerate_row_constant", np.repeat(r[:, None], n, axis=1)))
            bank.append(("degenerate_col_constant", np.repeat(r[None, :], n, axis=0)))
    # Rank-one sums, where each weight is a row term plus a column term. The whole optimal
    # face, the set of all optimal duals, is degenerate.
    for s in range(4):
        rng = np.random.default_rng(800 + s)
        for n in (5, 10, 20, 40):
            r = rng.integers(0, 5, size=n).astype(float)
            c = rng.integers(0, 5, size=n).astype(float)
            bank.append(("degenerate_rank_one", (r[:, None] + c[None, :]).astype(float)))

    # Sparse matrices, mostly zero and structurally close to disconnected.
    for s in range(3):                      # Up to k times n scattered positive entries, rest zero.
        rng = np.random.default_rng(900 + s)
        for n in (10, 20, 40, 60):
            for k in (1, 2, 5):
                W = np.zeros((n, n))
                idx = rng.integers(0, n, size=(k * n, 2))
                W[idx[:, 0], idx[:, 1]] = rng.integers(1, 50, size=k * n).astype(float)
                bank.append(("sparse_scattered", W))
    for s in range(4):                      # Block diagonal, a nearly disconnected bipartite graph.
        rng = np.random.default_rng(1000 + s)
        for n in (12, 20, 40):
            for nblk in (2, 4):
                W = np.zeros((n, n)); step = n // nblk
                for t in range(nblk):
                    lo, hi = t * step, (t + 1) * step if t < nblk - 1 else n
                    W[lo:hi, lo:hi] = rng.integers(1, 30, size=(hi - lo, hi - lo))
                bank.append(("sparse_block_diagonal", W))
    for s in range(4):                      # Exactly one positive entry per row and per column.
        rng = np.random.default_rng(1100 + s)
        for n in (5, 10, 20, 40):
            W = np.zeros((n, n))
            perm = rng.permutation(n)
            W[np.arange(n), perm] = rng.integers(1, 40, size=n).astype(float)
            bank.append(("sparse_permutation", W))
    for s in range(4):                      # A positive row and a positive column, the rest zero.
        rng = np.random.default_rng(1200 + s)
        for n in (6, 12, 24, 48):
            W = np.zeros((n, n))
            W[int(rng.integers(0, n)), :] = rng.integers(1, 40, size=n).astype(float)
            W[:, int(rng.integers(0, n))] = rng.integers(1, 40, size=n).astype(float)
            bank.append(("sparse_hot_cross", W))
    for n in (3, 5, 10, 20, 40):            # The all-zero extreme, where zero duals are feasible.
        bank.append(("sparse_all_zero", np.zeros((n, n))))

    return bank


# Repair operators that are used only to manufacture warm starts for the verification.
# They are self-contained restatements, so that solver.py verifies without importing the
# experiment scripts. The function _starting_duals asserts that the three repairs of its
# corrupted prediction are feasible before it hands them to solve_warm. It does not assert
# the repaired zero start. In every case solve_warm checks the start itself.

def _repair_sorted(W, u, v):
    """Repair a prediction with the sorted scan of Definition 9.

    Negative values of the prediction are first replaced by zero. The edges are then visited
    from the largest weight to the smallest. An edge is violated when its two dual values add
    up to less than its weight, and its residual is the weight minus that sum. Both endpoints
    of each edge that is still violated are raised by its residual.
    """
    u = np.maximum(u, 0).astype(float).copy(); v = np.maximum(v, 0).astype(float).copy()
    na, nb = W.shape
    P = [(a, b) for a in range(na) for b in range(nb)]
    for i in np.argsort([-W[a, b] for a, b in P], kind="stable"):
        a, b = P[i]
        r = W[a, b] - u[a] - v[b]
        if r > 0:
            u[a] += r; v[b] += r
    return u, v


def _repair_one_sided(W, u, v):
    """Fix every violation by raising row values only.

    Negative values of the prediction are first replaced by zero. Each row value then
    becomes the larger of two numbers: the row value itself, and the largest value, over all
    columns, of the weight minus the column value.
    """
    u = np.maximum(u, 0).astype(float).copy(); v = np.maximum(v, 0).astype(float).copy()
    return np.maximum(u, (W - v[None, :]).max(axis=1)), v


def _repair_shift(W, u, v):
    """Repair a prediction by lifting every coordinate by half the largest violation.

    This is the projection with respect to the largest absolute difference, which the paper
    calls the l-infinity distance, followed by rounding up. Negative values of the
    prediction are first replaced by zero. If nothing is violated, the clamped prediction
    is returned as it is.
    """
    u = np.maximum(u, 0).astype(float); v = np.maximum(v, 0).astype(float)
    eps = float(np.max(W - (u[:, None] + v[None, :])))
    if eps <= 0:
        return u.copy(), v.copy()
    return np.ceil(u + eps / 2), np.ceil(v + eps / 2)


def _starting_duals(W, rng):
    """Build the feasible starts that verify_reference runs the solver from.

    They are the all-zero dual where it is feasible, the two textbook cold starts (the row
    reduction and the column reduction), the repaired zero prediction, and one corrupted
    prediction repaired in three ways. These are not all the starts of the paper, which also
    runs this solver from duals repaired by operators that this file does not restate. The
    function returns a list of triples, each made of the name of the start, its row values
    and its column values.
    """
    n = W.shape[0]
    starts = []
    if W.max() <= 0:                     # The zero start, only where it is feasible.
        starts.append(("zero", np.zeros(n), np.zeros(n)))
    starts += [("cold_row_reduction", W.max(axis=1).astype(float), np.zeros(n)),
               ("cold_col_reduction", np.zeros(n), W.max(axis=0).astype(float))]
    # The repaired zero prediction. This is the cold start of Proposition 15, and the
    # cold-start comparison in experiments_itcs.py also uses it.
    zu, zv = _repair_sorted(W, np.zeros(n), np.zeros(n))
    starts.append(("repaired_zero", zu, zv))
    # A corrupted prediction: the row-reduction dual, perturbed up and down and clipped at
    # zero. It is then repaired in three different ways, as in the warm-start pipeline of
    # the tables.
    span = max(1, int(0.25 * (float(W.max()) + 1)))
    pu = np.maximum(W.max(axis=1).astype(float) + rng.integers(-span, span + 1, size=n), 0)
    pv = np.maximum(rng.integers(-span, span + 1, size=n).astype(float), 0)
    for name, f in (("repaired_sorted", _repair_sorted),
                    ("repaired_one_sided", _repair_one_sided),
                    ("repaired_shift", _repair_shift)):
        ru, rv = f(W, pu, pv)
        assert float(np.max(W - (ru[:, None] + rv[None, :]))) <= 1e-9, \
            f"{name} produced an infeasible start"
        starts.append((name, np.asarray(ru, float), np.asarray(rv, float)))
    return starts


def verify_reference(verbose=True):
    """Compare the optimum that solve_warm reaches with scipy.optimize.linear_sum_assignment.

    The comparison runs on a designed population of weight matrices, each solved from
    several feasible starting duals.

    An instance is one pair of a weight matrix and a starting dual, that is, one solver run.
    Identical pairs are counted once. Degenerate families do produce them: on a matrix whose
    weights are all equal, the row reduction and the column reduction can coincide. The
    reported count is therefore a count of distinct verified problems, not of calls.

    Besides the optimum, two structural invariants of the counter that Tables 6 and 8 read
    are checked on every run. The first is that the number of phases equals the number of
    rows, with one augmentation per row. The second applies to integral weights and integral
    starting duals: each dual adjustment lowers the sum of all dual values by at least 1,
    and the final duals are optimal, so the number of adjustments is at most the sum of all
    starting dual values minus the optimal value.
    """
    from scipy.optimize import linear_sum_assignment
    seen = set()
    stats, mismatches, bound_violations = {}, [], []
    total = matched = integral_checked = nontrivial = 0
    for idx, (family, W) in enumerate(_instance_bank()):
        n = W.shape[0]
        r, c = linear_sum_assignment(-W)
        true = float(W[r, c].sum())
        # The generator is seeded by the position in the bank, so the whole suite, and with it
        # the instance count reported in the paper, is reproducible bit for bit.
        rng = np.random.default_rng(7_000_000 + idx)
        for start_name, u0, v0 in _starting_duals(W, rng):
            key = (W.tobytes(), np.asarray(u0, float).tobytes(), np.asarray(v0, float).tobytes())
            if key in seen:
                continue
            seen.add(key)
            total += 1
            d = stats.setdefault(family, {"n": 0, "ok": 0, "starts": {}})
            d["n"] += 1
            d["starts"][start_name] = d["starts"].get(start_name, 0) + 1
            val, adj, phases = solve_warm(W, u0, v0)
            if abs(val - true) < 1e-6:
                matched += 1; d["ok"] += 1
            else:
                mismatches.append((family, start_name, n, val, true))
            if adj > 0:
                nontrivial += 1
            if phases != n:
                bound_violations.append((family, start_name, n, "phases", phases, n))
            integral = all(np.allclose(x, np.round(x))
                           for x in (W, np.asarray(u0, float), np.asarray(v0, float)))
            if integral:
                integral_checked += 1
                gap = float(np.sum(u0) + np.sum(v0)) - true
                if adj > gap + 1e-6:
                    bound_violations.append((family, start_name, n, "adj>gap", adj, gap))
    if verbose:
        print(f"\nverify_reference: {matched}/{total} distinct (matrix, starting dual) "
              f"instances matched scipy's exact optimum")
        print(f"{'family':>28} {'instances':>10} {'matched':>8}")
        for family in sorted(stats):
            d = stats[family]
            print(f"{family:>28} {d['n']:>10} {d['ok']:>8}")
        starts_tot = {}
        for d in stats.values():
            for k, m in d["starts"].items():
                starts_tot[k] = starts_tot.get(k, 0) + m
        print(f"{'by starting dual':>28} {'instances':>10}")
        for k in sorted(starts_tot):
            print(f"{k:>28} {starts_tot[k]:>10}")
        print(f"distinct weight matrices: {len({k[0] for k in seen})}")
        if mismatches:
            print(f"\n!!! {len(mismatches)} MISMATCHES vs the exact reference:")
            for m in mismatches[:20]:
                print("   ", m)
        if bound_violations:
            print(f"\n!!! {len(bound_violations)} structural-invariant violations:")
            for b in bound_violations[:20]:
                print("   ", b)
        if not mismatches and not bound_violations:
            print(f"no mismatches; phases == n on all {total} runs and "
                  f"adjustments <= sum(u0)+sum(v0) - OPT on all {integral_checked} "
                  f"integral-data runs")
        print(f"runs that actually performed >=1 dual adjustment: {nontrivial}/{total}")
    return {"total": total, "matched": matched, "families": stats,
            "matrices": len({k[0] for k in seen}),
            "integral_checked": integral_checked, "nontrivial": nontrivial,
            "mismatches": mismatches, "invariant_violations": bound_violations}


if __name__ == "__main__":
    import sys
    n = self_test()
    print(f"self_test: {n}/{n} instances matched scipy's optimum exactly")
    if "--quick" not in sys.argv:
        res = verify_reference()
        if res["mismatches"] or res["invariant_violations"]:
            sys.exit(1)
