#!/usr/bin/env python3
"""Measure Chen et al.'s Algorithm 1 on the pinned draws of Table 6 of the paper.

The program runs those draws again, with seed 17, sizes 50, 100, 150 and 200, and 10
repetitions. On exactly the same repaired duals, it records the costs of Chen et al.'s
Algorithm 1 alongside the number of dual adjustments counted by solver.py.

The order of the calls to the random generator is copied from E1_solver in
experiments_itcs.py, so that the draws are identical to the paper's bit for bit. Reproducing
the published means of Table 6 is the check that this is so.
"""
import sys, time, numpy as np
from scipy.optimize import linear_sum_assignment
from experiments import optimal_dual, instance, l0
from experiments_itcs import (all_repairs, corrupt, OPS, OPS_NOLOCAL, QUANTS, _spear)
from solver import solve_warm
from chen_alg1 import chen_alg1

# The maximum-flow step in chen_alg1.py searches depth first by recursion, and its paths can
# be longer than the default recursion limit of Python allows.
sys.setrecursionlimit(100000)

COSTS  = ["adj", "chen_iters", "chen_iters_nost", "chen_defic", "chen_work", "chen_work_nost", "chen_dijkstra_plus_ff"]

def collect(kinds=("uniform", "clustered", "euclidean"), sizes=(50, 100, 150, 200),
            reps=10, seed=17):
    rng = np.random.default_rng(seed)
    raw = []
    for kind in kinds:
        for n in sizes:
            t0 = time.time()
            for _ in range(reps):
                W = instance(kind, n, rng)              # First use of the random generator.
                us, vs = optimal_dual(W)
                us, vs = np.rint(us), np.rint(vs)
                OPTW = int(W[linear_sum_assignment(-W)].sum())
                assert abs(us.sum() + vs.sum() - OPTW) < 1e-6
                u0, v0 = corrupt(us, vs, n, n, rng)     # Second use of the random generator.
                reps_ops = all_repairs(W, u0, v0, rng)  # Third use, for the random scan order.
                star = np.concatenate([us, vs])
                pre = np.concatenate([np.maximum(u0, 0), np.maximum(v0, 0)])
                OPT = us.sum() + vs.sum()
                rec = {"kind": kind, "n": n}
                for name, (ru, rv) in reps_ops.items():
                    y = np.concatenate([ru, rv])
                    assert (ru[:, None] + rv[None, :] >= W - 1e-9).all()
                    d = {"l0": l0(ru, rv, us, vs),
                         "l1": float(np.abs(y - star).sum()),
                         "l1disp": float(np.abs(y - pre).sum()),
                         "gap": float(ru.sum() + rv.sum() - OPT)}
                    d["adj"] = solve_warm(W, ru, rv)[1]
                    c = chen_alg1(W, np.rint(ru), np.rint(rv), st_residual=True, check=False)
                    c2 = chen_alg1(W, np.rint(ru), np.rint(rv), st_residual=False, check=False)
                    assert c["weight"] == OPTW == c2["weight"], "Algorithm 1 suboptimal"
                    d["chen_iters"] = c["iters"]
                    d["chen_iters_nost"] = c2["iters"]
                    d["chen_defic"] = c["deficiency"]
                    d["chen_work"] = c["work"]
                    d["chen_work_nost"] = c2["work"]
                    # The next line adds the two while-loop terms of Theorem 3.2 of Chen et al.,
                    # in units of the number of edges. In the first term, every iteration costs
                    # the number of edges plus n times the logarithm of n, taken here to base 2.
                    # The second term is the number of edges times ff_units. The number of edges
                    # is n times n, because every row is joined to every column. Divided by the
                    # number of edges, an iteration costs 1 plus the logarithm of n divided by
                    # n, and a unit of ff_units costs 1.
                    d["chen_dijkstra_plus_ff"] = c["iters"] * (1 + np.log2(n) / n) + c["ff_units"]
                    assert c["deficiency"] <= d["l0"], "Lemma B.8 violated"
                    assert c["iters"] <= d["l0"], "Lemma B.9 violated"
                    rec[name] = d
                cu, cv = W.max(axis=1).astype(float), np.zeros(n)
                cc = chen_alg1(W, np.rint(cu), np.rint(cv), st_residual=True, check=False)
                rec["_cold"] = {"adj": solve_warm(W, cu, cv)[1],
                                "chen_iters": cc["iters"], "chen_defic": cc["deficiency"],
                                "chen_work": cc["work"]}
                raw.append(rec)
            print(f"  {kind:>10} n={n:<4} {time.time()-t0:6.1f}s", flush=True)
    return raw

def table(raw, key):
    cells = sorted({(r["kind"], r["n"]) for r in raw}, key=lambda c: (c[0], c[1]))
    print(f"\n=== mean {key} by starting point ===")
    print(f"{'instance':>11}{'n':>5}{'cold':>9}" + "".join(f"{o:>13}" for o in OPS))
    for c in cells:
        sub = [r for r in raw if (r["kind"], r["n"]) == c]
        line = f"{c[0]:>11}{c[1]:>5}{np.mean([r['_cold'][key] for r in sub]):>9.1f}"
        for o in OPS:
            line += f"{np.mean([r[o][key] for r in sub]):>8.1f}" + \
                    f"+-{np.std([r[o][key] for r in sub]):<3.1f}"
        print(line)

def corr(raw, ops, cost, label):
    print(f"\n--- operator-only Spearman vs {cost}   [{label}] ---")
    print(f"{'quantity':>9} {'mean rho':>9} {'n used':>7} {'n NaN':>6} {'pooled':>8}")
    res = {}
    for q in QUANTS:
        per = [_spear([r[o][q] for o in ops], [r[o][cost] for o in ops]) for r in raw]
        per = np.array(per, float)
        used = per[~np.isnan(per)]
        pooled = _spear([r[o][q] for r in raw for o in ops],
                        [r[o][cost] for r in raw for o in ops])
        res[q] = (float(np.mean(used)) if len(used) else np.nan, len(used),
                  int(np.isnan(per).sum()), pooled, per)
        print(f"{q:>9} {res[q][0]:>+9.3f} {res[q][1]:>7} {res[q][2]:>6} {pooled:>+8.3f}")
    return res

def reversals(raw, ops, cost, q="l0"):
    """Report how often the quantity and the cost order two operators in opposite ways.

    The records in raw come from collect, and ops lists the operators to compare. The argument
    q is the key of the quantity in the record of each operator. Its default is l0, the number
    of coordinates in which the repaired dual differs from the optimal dual of the instance.
    The argument cost is the key of the cost, such as one of the entries of COSTS. The pairs
    are pairs of operators within one instance, and only the pairs that the quantity orders
    strictly are counted. A pair is a genuine reversal when the cost orders it strictly the
    other way. The function prints the fraction of pairs on which the two orders agree, the
    fraction of genuine reversals, and the fraction of pairs that the quantity orders strictly
    but the cost ties. It returns the four counts: agreeing, reversed, tied and total.
    """
    rev = tie = agree = tot = 0
    for r in raw:
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                a, b = r[ops[i]], r[ops[j]]
                if a[q] == b[q]:
                    continue
                tot += 1
                s1 = 1 if a[q] < b[q] else -1
                if a[cost] == b[cost]:
                    tie += 1
                else:
                    s2 = 1 if a[cost] < b[cost] else -1
                    if s1 == s2: agree += 1
                    else: rev += 1
    print(f"pairs with strict {q} order, {cost}: agree {agree}/{tot} = {agree/tot:.3f}, "
          f"reversed {rev}/{tot} = {rev/tot:.3f}, cost-tied {tie}/{tot} = {tie/tot:.3f}")
    return agree, rev, tie, tot

if __name__ == "__main__":
    t = time.time()
    raw = collect()
    table(raw, "adj")
    table(raw, "chen_iters")
    table(raw, "chen_defic")
    table(raw, "chen_work")
    allc = {}
    for ops, lab in ((OPS_NOLOCAL, "5-op menu"), (OPS, "6-op menu")):
        for cost in COSTS:
            allc[(lab, cost)] = corr(raw, ops, cost, lab)
    for ops, lab in ((OPS_NOLOCAL, "5-op"), (OPS, "6-op")):
        for cost in COSTS:
            print(f"[{lab}] ", end="")
            reversals(raw, ops, cost)
    print(f"\n({time.time()-t:.0f}s)")
