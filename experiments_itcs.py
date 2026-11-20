#!/usr/bin/env python3
"""Run the solver and statistics experiments of the paper "Not All Feasibility Repairs Are Equal".

The program experiments.py makes Table 5 (tab:exp) and holds the shared building blocks. This
file makes Tables 6 to 8 and the numbers quoted around them. A name in parentheses after a
table number, such as tab:exp, is the label of that table in the source of the paper. This
file prints the rows of its three tables under their labels.

A few words are used throughout. A draw is one random instance together with its prediction,
and a cell is one combination of instance family and size (or shape, in E2). The clamped
prediction is the prediction with its negative values replaced by zero, and every repair
starts from it. The reference optimum of an instance is the optimal dual that optimal_dual
computes for it. Solver work is counted in dual adjustments, the steps in which the solver
changes dual values.

Experiment E1 makes Table 6 (tab:solver). It counts the dual adjustments of a warm-started
solver from each repaired start, on square instances whose prediction is the optimal dual
with a tenth of its coordinates corrupted.

Experiment E2 makes Table 7 (tab:rect). It counts the coordinates in which each repaired dual
differs from the reference optimum, on unbalanced shapes, which here have far fewer rows than
columns. On those shapes the bound of Theorem 11, which grows with the smaller of the number
of rows and the number of columns, is small. The experiment comes with a paired analysis,
draw by draw, and with the same design at three other seeds.

Experiment E3 makes Table 8 (tab:drift). It is like E1, but the prediction is the optimal
dual of the previous period, used after the weights have drifted.

Experiment E4 compares the repaired zero start with the row-reduction start. The first is the
all-zero prediction after repair. In the second, every row value is the largest weight in its
row and every column value is zero. The tables printed by E1 and E3 show the row-reduction
start in a column named cold.

Three more analyses use the draws of E1. The first asks which quantity measured on the
repaired dual best predicts solver work. It uses Spearman rank correlations, with a sign
test, a sign-flip test, a t test and a bootstrap. The second compares the sorted scan with
the forward scan, draw by draw. The third reports the count U of Proposition 43, the number
of rows with no tight edge.

The repair operators are the following. The sorted, forward and random scans are those of
Definition 1. A scan visits the constraints in a given order and, at each one that is still
violated, raises both endpoints by the full residual, which is the weight minus the sum of
the two dual values. The sorted scan goes by decreasing weight, the forward scan in index
order and the random scan in a random order. The walk is our transplant of the one-sided
walk of Dinitz et al. to the covering setting, that is, to constraints in which the two dual
values of every edge must add up to at least the edge weight. The uniform shift is our
transplant, to the same setting, of the projection of Sakaue and Oki, rounded up. That
projection is taken in the distance that adds the largest increase to the largest decrease
(the two-sided l-infinity distance). The uniform shift lifts every coordinate by half the
largest residual and rounds up, so it moves every coordinate. The endpoint shift, named
local_shift in the code, applies the same lift only to the endpoints of violated pairs. The
two shifts have the same largest absolute displacement (the l-infinity displacement), so
comparing them separates being optimal in that measure from touching every coordinate.

The measured quantities are stored under short keys. Under the keys l0 and l1 are the two
distances that the paper calls by those names, both taken to the reference optimum (us and vs
in the code): the number of coordinates that differ, and the sum of the absolute differences.
Under l1disp is the realised move from the clamped prediction. Under gap is the dual gap, the
sum of the dual values minus the optimal value. Rank correlations use average ranks for ties.

To run the program, type python3 experiments_itcs.py at the command line. It takes about two
minutes, and the defaults are the pinned settings.
"""
import itertools, time
from fractions import Fraction
from math import comb
import numpy as np
from scipy.stats import rankdata, wilcoxon, t as tdist
from experiments import optimal_dual, repair, walk_repair, orders, instance, l0
from solver import solve_warm


# The count U of rows that have no tight edge.

def U_of(W, u, v):
    """Count the rows that have no tight edge, the quantity U of Proposition 43 (prop:tightrows).

    W is the weight matrix, u holds the dual values of the rows and v those of the columns.
    The slack of an edge is the sum of its two dual values minus its weight, and the edge is
    tight when its slack is zero. The count uses the same floating-point expression and the
    same tolerance as the solver. When solve_warm starts its search from a row, it adds the
    dual value of the row to the dual values of the columns, subtracts the weights of the
    row, and counts an adjustment when the smallest of these slacks exceeds one billionth
    (1e-9 in the code). A row is
    counted here under that same condition. The function does not use the random generator.
    """
    return int(sum(1 for a in range(W.shape[0])
                   if ((float(u[a]) + v) - W[a]).min() > 1e-9))


# The repair operators.

def shift_repair(W, u, v):
    """Repair a prediction with the uniform shift and return the repaired dual.

    The uniform shift is the projection of Sakaue and Oki in the covering setting. That
    projection is taken in the distance that adds the largest increase to the largest
    decrease. The function raises every coordinate of the clamped prediction by half the
    largest violation and then rounds up to integers. The largest violation, eps in the code,
    is the largest residual over all edges, where the residual of an edge is its weight minus
    the sum of its two dual values. The largest absolute change that the shift makes to a
    coordinate is within 1 of the smallest that any feasible dual can achieve. When anything
    is violated the shift moves all the coordinates, one for each row and one for each column.
    When nothing is violated the clamped prediction is returned as it is.
    """
    u = np.maximum(u, 0).astype(float); v = np.maximum(v, 0).astype(float)
    eps = float(np.max(W - (u[:, None] + v[None, :])))
    if eps <= 0:
        return u, v
    return np.ceil(u + eps / 2), np.ceil(v + eps / 2)


def local_shift_repair(W, u, v):
    """Repair a prediction with the endpoint shift and return the repaired dual.

    This is the same lift as in the uniform shift, by half the largest violation, but it is
    applied only to the endpoints of the constraints that are actually violated. The lifted
    values are then rounded up, as for the uniform shift.

    The result is feasible for the following reason. Every coordinate only increases, so a
    satisfied constraint stays satisfied. A violated constraint has both of its endpoints
    lifted by half the largest violation, and its own violation is at most the largest
    violation by definition, so it becomes satisfied. Rounding up only helps.

    The realised displacement of each coordinate is as follows. A touched coordinate moves to
    its old value plus half the largest violation, rounded up, so it moves by at least half
    the largest violation and by less than half the largest violation plus 1. The uniform
    shift moves every coordinate in exactly this way. Here the other coordinates do not move.
    So the largest absolute displacement of both operators is within 1 of the optimum, the
    smallest that any feasible dual can achieve, which is half the largest violation. The
    function measure asserts this.
    """
    u = np.maximum(u, 0).astype(float); v = np.maximum(v, 0).astype(float)
    S = W - (u[:, None] + v[None, :])
    eps = float(np.max(S))
    if eps <= 0:
        return u, v
    viol = S > 1e-9
    ra = viol.any(axis=1)          # Rows that are endpoints of violated constraints.
    rb = viol.any(axis=0)          # Columns that are endpoints of violated constraints.
    u[ra] = np.ceil(u[ra] + eps / 2)
    v[rb] = np.ceil(v[rb] + eps / 2)
    return u, v


def all_repairs(W, u0, v0, rng=None):
    """Apply every operator of OPS to one prediction.

    The result maps the name of each operator to its repaired dual. The random scan order is
    drawn from rng.
    """
    out = {}
    for name, o in orders(W, rng).items():
        out[name] = repair(W, u0, v0, o)
    out["walk"] = walk_repair(W, u0, v0)
    out["shift"] = shift_repair(W, u0, v0)
    out["local_shift"] = local_shift_repair(W, u0, v0)
    return out


# Measurement of the repairs.

OPS = ["sorted", "forward", "random", "walk", "shift", "local_shift"]
OPS_NOLOCAL = ["sorted", "forward", "random", "walk", "shift"]
QUANTS = ["l0", "l1", "l1disp", "gap"]


def measure(W, u0, v0, us, vs, with_solver, rng=None):
    """Measure every repair of one prediction, and the solver work too if with_solver is true.

    W is the weight matrix. The arguments u0 and v0 are the row values and the column values
    of the prediction, and us and vs are those of the reference optimum. The random generator
    rng is passed on to all_repairs.

    The result is a dictionary. It has one entry for each operator, which holds the quantities
    measured on the repaired dual under the following keys.
    The key l0 holds the number of coordinates that differ from the reference optimum.
    The key l1 holds the sum of the absolute differences from the reference optimum.
    The key l1disp holds the sum of the absolute differences from the clamped prediction,
    which is the realised displacement.
    The key gap holds the dual gap, the sum of the repaired dual values minus the optimal value.
    The key linf holds the largest absolute difference from the clamped prediction.
    The key U holds the number of rows with no tight edge, computed by U_of.
    The key adj holds the number of dual adjustments that solve_warm makes from the repaired
    start. It is present only when with_solver is true.

    The result has two more entries. The entry _pre holds the number of coordinates in which
    the clamped prediction differs from the reference optimum, its largest violation (eps),
    and the shape of the instance. The entry _cold, present only when with_solver is true,
    holds the number of dual adjustments and the value of U for the row-reduction start.
    """
    OPT = us.sum() + vs.sum()
    star = np.concatenate([us, vs])
    pre = np.concatenate([np.maximum(u0, 0), np.maximum(v0, 0)])
    pre_l0 = l0(np.maximum(u0, 0), np.maximum(v0, 0), us, vs)
    na, nb = W.shape
    eps = float(np.max(W - (np.maximum(u0, 0)[:, None] + np.maximum(v0, 0)[None, :])))
    reps_all = all_repairs(W, u0, v0, rng)
    rows = {}
    for name, (ru, rv) in reps_all.items():
        y = np.concatenate([ru, rv])
        viol = float(np.max(W - (ru[:, None] + rv[None, :])))
        assert viol <= 1e-9, f"{name} produced an infeasible dual (violation {viol})"
        rec = {"l0": l0(ru, rv, us, vs),
               "l1": float(np.abs(y - star).sum()),
               "l1disp": float(np.abs(y - pre).sum()),
               "gap": float(ru.sum() + rv.sum() - OPT),
               "linf": float(np.abs(y - pre).max()),
               "U": U_of(W, ru, rv)}
        if with_solver:
            _, adj, _ = solve_warm(W, ru, rv)
            rec["adj"] = adj
        if name == "shift" and eps > 1e-9:
            # Observation 25. The rounded-up shift moves every coordinate, so the
            # number of coordinates that differ from the fixed reference after the repair is
            # at least the number of rows plus the number of columns, minus the number that
            # differed before the repair.
            assert rec["l0"] >= (na + nb) - pre_l0 - 1e-9, \
                f"shift observation violated: post {rec['l0']} < {na+nb} - {pre_l0}"
        rows[name] = rec
    if eps > 1e-9:
        # The largest absolute displacement of local_shift is optimal up to the same rounding
        # as that of the uniform shift.
        lu, ls_ = rows["local_shift"]["linf"], rows["shift"]["linf"]
        assert lu <= ls_ + 1e-9, f"local_shift l_inf {lu} > uniform shift l_inf {ls_}"
        assert lu >= eps / 2 - 1e-9, f"local_shift l_inf {lu} below the optimum {eps/2}"
        assert lu < eps / 2 + 1 + 1e-9, f"local_shift l_inf {lu} beyond rounding of {eps/2}"
        assert ls_ < eps / 2 + 1 + 1e-9, f"shift l_inf {ls_} beyond rounding of {eps/2}"
    rows["_pre"] = {"l0": pre_l0, "eps": eps, "na": na, "nb": nb}
    if with_solver:
        cu, cv = W.max(axis=1).astype(float), np.zeros(W.shape[1])
        _, adj_cold, _ = solve_warm(W, cu, cv)
        rows["_cold"] = {"adj": adj_cold, "U": U_of(W, cu, cv)}
    return rows


def corrupt(us, vs, na, nb, rng, frac=0.10):
    """Build a prediction by corrupting the exact optimum, with the recipe of Tables 5 to 7.

    This function builds the predictions of Tables 6 and 7. The program experiments.py applies
    the same recipe in its own loop for Table 5.

    The exact optimum is passed as its row values and its column values, followed by the
    number of rows and the number of columns. A fraction frac of its coordinates, chosen at
    random, is moved up or down by a whole number from 1 to 19, and a moved value that falls
    below 0 is set to 0. The function returns the row values and the column values of the
    prediction.
    """
    u0, v0 = us.copy(), vs.copy()
    k = max(1, int(frac * (na + nb)))
    for i in rng.choice(na + nb, size=k, replace=False):
        d = rng.choice([-1, 1]) * rng.integers(1, 20)
        if i < na: u0[i] = max(0, u0[i] + d)
        else:      v0[i - na] = max(0, v0[i - na] + d)
    return u0, v0


def drifted(kind, n, rng, drift=0.10):
    """Build a realistic prediction from an instance whose weights drift.

    The function solves the instance of the previous period, drifts the weights, and uses the
    optimal dual of the previous period as the prediction for this period. Each weight moves
    by a whole number from minus 10 to 10 (the bound is drift times 100) and is then clipped
    at 0. The weights of the uniform and euclidean families lie between 0 and 100, so for them
    that is up to a tenth of their range. The weights of the clustered family can be larger,
    because each of its blocks adds between 50 and 99 to weights between 0 and 19, and blocks
    can overlap. The function returns the drifted weight matrix, followed by the row values
    and the column values of the prediction.
    """
    W_prev = instance(kind, n, rng)
    up, vp = optimal_dual(W_prev)
    noise = rng.integers(-int(drift * 100), int(drift * 100) + 1, size=W_prev.shape)
    W = np.maximum(W_prev + noise, 0).astype(float)
    return W, up, vp


# The experiments.

def E1_solver(kinds=("uniform", "clustered", "euclidean"), sizes=(50, 100, 150, 200),
              reps=10, seed=17):
    """Run experiment E1 and print Table 6 (tab:solver).

    The question is whether the number of coordinates in which a repaired dual differs from
    the reference optimum predicts solver work. The instances are square and the prediction
    is a corrupted optimum. The function returns the list of all draws and the list of
    per-cell means with their standard deviations.
    """
    rng = np.random.default_rng(seed)
    raw = []; cells = []
    print("\n=== E1: warm-started solver, dual adjustments (square, corrupted prediction) ===")
    print(f"{'instance':>11} {'n':>5} {'cold':>16} |" + "".join(f" {o:>16}" for o in OPS))
    for kind in kinds:
        for n in sizes:
            acc = {o: [] for o in OPS}; cold = []
            for _ in range(reps):
                W = instance(kind, n, rng)
                us, vs = optimal_dual(W)
                u0, v0 = corrupt(us, vs, n, n, rng)
                r = measure(W, u0, v0, us, vs, with_solver=True, rng=rng)
                cold.append(r["_cold"]["adj"])
                for o in OPS: acc[o].append(r[o]["adj"])
                raw.append({o: r[o] for o in OPS} | {"pre": r["_pre"]["l0"],
                            "cold": r["_cold"]["adj"], "kind": kind, "n": n,
                            "eps": r["_pre"]["eps"], "na": r["_pre"]["na"],
                            "nb": r["_pre"]["nb"], "coldU": r["_cold"]["U"]})
            line = f"{kind:>11} {n:>5} {np.mean(cold):>9.1f}+/-{np.std(cold):<5.1f} |"
            for o in OPS:
                line += f" {np.mean(acc[o]):>9.1f}+/-{np.std(acc[o]):<5.1f}"
            print(line)
            cells.append((kind, n, np.mean(cold), np.std(cold),
                          {o: (np.mean(acc[o]), np.std(acc[o])) for o in OPS}))
    print("\nLATEX rows (tab:solver, as in the paper but without the bold marks):")
    for kind, n, cold_m, cold_s, per in cells:
        cs = " & ".join(f"${per[o][0]:.1f}\\std{{{per[o][1]:.1f}}}$" for o in OPS)
        print(f"  {kind} & {n} & ${cold_m:.1f}\\std{{{cold_s:.1f}}}$ & {cs} \\\\")

    # Recompute the claims that the paper makes in its paragraph on Table 6.
    print("\n--- tab:solver prose checks ---")
    D = {(k, n): (cm, cs, per) for k, n, cm, cs, per in cells}
    shift_most = [(k, n) for (k, n), (cm, cs, per) in D.items()
                  if per["shift"][0] < max(per[o][0] for o in OPS if o != "shift")]
    print(f"cells where 'shift' is NOT the most expensive repair (6-op menu): {shift_most or 'none'}")
    shift_most5 = [(k, n) for (k, n), (cm, cs, per) in D.items()
                   if per["shift"][0] < max(per[o][0] for o in OPS_NOLOCAL if o != "shift")]
    print(f"cells where 'shift' is NOT the most expensive repair (5-op menu): {shift_most5 or 'none'}")
    ls_gt_shift = [(k, n) for (k, n), (cm, cs, per) in D.items()
                   if per["local_shift"][0] > per["shift"][0]]
    print(f"cells where local_shift costs MORE than the uniform shift: "
          f"{len(ls_gt_shift)}/{len(D)} {ls_gt_shift}")
    shift_gt_cold = [(k, n) for (k, n), (cm, cs, per) in D.items() if per["shift"][0] > cm]
    print(f"shift exceeds cold in {len(shift_gt_cold)}/{len(D)} cells: {shift_gt_cold}")
    for (k, n), (cm, cs, per) in sorted(D.items()):
        print(f"   {k:>10} n={n:<4} shift/cold = {per['shift'][0]/cm:.2f}x "
              f"(shift {per['shift'][0]:.1f}, cold {cm:.1f})")
    walk_first = [(k, n) for (k, n), (cm, cs, per) in D.items()
                  if per["walk"][0] > min(per[o][0] for o in OPS)]
    print(f"cells where walk is NOT first among repairs: {walk_first or 'none (walk first everywhere)'}")
    walk_first5 = [(k, n) for (k, n), (cm, cs, per) in D.items()
                   if per["walk"][0] > min(per[o][0] for o in OPS_NOLOCAL)]
    print(f"  ... excluding local_shift from the menu: {walk_first5 or 'none'}")
    print("cold vs repairs, per cell:")
    for (k, n), (cm, cs, per) in sorted(D.items()):
        beats_scans = all(cm < per[o][0] for o in ("sorted", "forward", "random"))
        beats_all = all(cm < per[o][0] for o in OPS)
        beats_all5 = all(cm < per[o][0] for o in OPS_NOLOCAL)
        print(f"   {k:>10} n={n:<4} cold {cm:>7.1f}  beats all scans: {beats_scans}   "
              f"beats every repair (6-op): {beats_all}   (5-op): {beats_all5}")
    return raw, cells


def E1_quantities(raw):
    """Print the per-cell means of the quantities measured on the repaired duals.

    The purpose is to compare the two shifts. The argument raw is the list of draws returned
    by E1_solver. The function returns the means in a dictionary keyed by cell and operator.
    """
    cells = sorted({_cell_key(r) for r in raw})
    print(f"\n=== E1 quantities per cell (mean over {len(raw) // len(cells)} instances) ===")
    print(f"{'cell':>18} {'op':>12} {'l0':>9} {'l1(opt)':>11} {'l1disp':>11} {'gap':>10} {'adj':>9}")
    tab = {}
    for c in cells:
        sub = [r for r in raw if _cell_key(r) == c]
        for o in OPS:
            m = {q: float(np.mean([r[o][q] for r in sub])) for q in QUANTS + ["adj"]}
            tab[(c, o)] = m
            print(f"{str(c):>18} {o:>12} {m['l0']:>9.1f} {m['l1']:>11.1f} "
                  f"{m['l1disp']:>11.1f} {m['gap']:>10.1f} {m['adj']:>9.1f}")
        pre = float(np.mean([r["pre"] for r in sub]))
        print(f"{str(c):>18} {'(pre l0)':>12} {pre:>9.1f}")
    print("\n--- l_inf displacement: local_shift is l_inf-optimal to the same rounding ---")
    for c in cells:
        sub = [r for r in raw if _cell_key(r) == c]
        li_u = float(np.mean([r["shift"]["linf"] for r in sub]))
        li_l = float(np.mean([r["local_shift"]["linf"] for r in sub]))
        mx = max(abs(r["shift"]["linf"] - r["local_shift"]["linf"]) for r in sub)
        print(f"{str(c):>18} uniform-shift linf {li_u:>7.2f}   local_shift linf {li_l:>7.2f}   "
              f"max per-draw |difference| {mx:.2f}")
    print("\n--- the local_shift confound, stated numerically ---")
    print(f"{'cell':>18} {'l0 loc/uni':>11} {'gap loc/uni':>12} {'adj loc':>9} {'adj uni':>9} {'adj loc>uni?':>13}")
    for c in cells:
        ml = tab[(c, "local_shift")]; mu = tab[(c, "shift")]
        print(f"{str(c):>18} {ml['l0']/mu['l0']:>11.2f} {ml['gap']/mu['gap']:>12.2f} "
              f"{ml['adj']:>9.1f} {mu['adj']:>9.1f} {str(ml['adj'] > mu['adj']):>13}")
    return tab


E2_SHAPES = ((5, 500), (10, 400), (20, 300), (3, 600))
E2_KINDS = ("uniform", "clustered")


def _rect_draws(shapes=E2_SHAPES, kinds=E2_KINDS, reps=10, seed=23):
    """Generate the draws of experiment E2 in their pinned order.

    The loops run over the kinds, then the shapes, then the repeats, and one random generator
    seeded with seed feeds every draw in turn. Each draw takes a square instance, cuts from it
    a corner with the wanted numbers of rows and columns, corrupts the optimal dual of that
    corner, and measures every repair without running the solver. A draw is stored as a tuple
    that holds, in this order, the kind, the number of rows, the number of columns, the weight
    matrix, the row values and the column values of the prediction, the row values and the
    column values of the reference optimum, and the measurements.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for kind in kinds:
        for (na, nb) in shapes:
            for _ in range(reps):
                W = instance(kind, max(na, nb), rng)[:na, :nb].copy()
                us, vs = optimal_dual(W)
                u0, v0 = corrupt(us, vs, na, nb, rng)
                r = measure(W, u0, v0, us, vs, with_solver=False, rng=rng)
                draws.append((kind, na, nb, W, u0, v0, us, vs, r))
    return draws


def E2_rectangular(draws=None, shapes=E2_SHAPES, kinds=E2_KINDS, reps=10, seed=23):
    """Run experiment E2 and print Table 7 (tab:rect).

    The table gives the number of coordinates in which each repaired dual differs from the
    reference optimum, on unbalanced shapes. On those shapes the general warm-start bound,
    which is the bound of Theorem 11, is not vacuous, that is, it is below the number of
    coordinates. If no draws are passed in, they are generated
    by _rect_draws. The function returns the draws, so that E2_paired can reuse them.
    """
    if draws is None:
        draws = _rect_draws(shapes, kinds, reps, seed)
    cells = []
    print("\n=== E2: rectangular instances, min(|A|,|B|) << max  (l0 displacement) ===")
    print(f"{'instance':>11} {'shape':>10} {'dim':>5} {'2min-1':>7} {'pre':>6} |"
          + "".join(f" {o:>15}" for o in OPS))
    for kind in kinds:
        for (na, nb) in shapes:
            rs = [d[-1] for d in draws if d[:3] == (kind, na, nb)]
            pres = [r["_pre"]["l0"] for r in rs]
            already = sum(1 for r in rs if r["_pre"]["eps"] <= 1e-9)
            acc = {o: [r[o]["l0"] for r in rs] for o in OPS}
            line = (f"{kind:>11} {f'{na}x{nb}':>10} {na+nb:>5} {2*min(na,nb)-1:>7} "
                    f"{np.mean(pres):>6.1f} |")
            for o in OPS:
                line += f" {np.mean(acc[o]):>8.1f}+/-{np.std(acc[o]):<5.1f}"
            print(line)
            print(f"{'':>11} {'':>10} {'':>5} {'':>7} {'':>6} | draws whose corrupted "
                  f"prediction was ALREADY FEASIBLE (every repair = identity): {already}/{len(rs)}")
            cells.append((kind, na, nb, np.mean(pres), already, len(rs),
                          {o: (np.mean(acc[o]), np.std(acc[o])) for o in OPS}))
    print("\nLATEX rows (tab:rect, as in the paper):")
    for kind, na, nb, pre_m, already, n_draws, per in cells:
        cs = " & ".join(f"${per[o][0]:.1f}\\std{{{per[o][1]:.1f}}}$"
                        for o in ("sorted", "forward", "random", "walk"))
        print(f"  {kind} & ${na}\\times{nb}$ & ${na+nb}$ & ${2*min(na,nb)-1}$ & "
              f"${already}/{n_draws}$ & ${pre_m:.1f}$ & {cs} \\\\")
    print("\n--- tab:rect prose checks ---")
    sh = [per["shift"][0] for *_, per in cells]
    shs = [per["shift"][1] for *_, per in cells]
    ls = [per["local_shift"][0] for *_, per in cells]
    lss = [per["local_shift"][1] for *_, per in cells]
    scanmax = max(per[o][0] for *_, per in cells for o in ("sorted", "forward", "random", "walk"))
    print(f"shift mean l0 range over rows: {min(sh):.1f} .. {max(sh):.1f}  "
          f"(std range {min(shs):.1f} .. {max(shs):.1f})")
    print(f"local_shift mean l0 range over rows: {min(ls):.1f} .. {max(ls):.1f}  "
          f"(std range {min(lss):.1f} .. {max(lss):.1f})")
    print(f"max scan/walk mean l0 over all rows: {scanmax:.1f}")
    print("within-one-std-of-pre check (|mean_op - pre_mean| <= std_op):")
    bad = []
    for kind, na, nb, pre_m, _, _, per in cells:
        for o in ("sorted", "forward", "random", "walk"):
            d = abs(per[o][0] - pre_m)
            if d > per[o][1] + 1e-12:
                bad.append((kind, f"{na}x{nb}", o, round(d, 2), round(per[o][1], 2)))
    print(f"   violations: {bad or 'none -- every scan within one std of pre'}")
    print("mutual within-one-std check (each pair of scans within the larger std):")
    bad2 = []
    for kind, na, nb, _, _, _, per in cells:
        for o1, o2 in itertools.combinations(("sorted", "forward", "random", "walk"), 2):
            d = abs(per[o1][0] - per[o2][0])
            if d > max(per[o1][1], per[o2][1]) + 1e-12:
                bad2.append((kind, f"{na}x{nb}", o1, o2, round(d, 2)))
    print(f"   violations: {bad2 or 'none'}")
    return draws


def E2_paired(draws=None, shapes=E2_SHAPES, kinds=E2_KINDS, reps=10, seed=23, label="pinned"):
    """Print the paired analysis, draw by draw, behind what the paper says about Table 7.

    Pass the draws of E2_rectangular to reuse them. Otherwise they are drawn again from the
    seed.

    The function prints three things. The first is the count of no-ops, the draws whose
    prediction is already feasible, so that every repair is the identity. The second is the
    paired comparisons on the other draws, called the active draws, with their exact sign
    tests. When every nonzero difference has the same sign, these coincide with the exact
    signed-rank tests. The third is how many coordinates the sorted scan moves, and how many
    it restores and destroys against the reference optimum. A coordinate is wrong when it
    differs from the reference optimum. It is restored when it is wrong before the repair and
    correct after it, and destroyed when it is correct before the repair and wrong after it.
    """
    if draws is None:
        draws = _rect_draws(shapes, kinds, reps, seed)
    total = len(draws)
    noop = [(k, na, nb) for k, na, nb, *_, r in draws if r["_pre"]["eps"] <= 1e-9]
    print(f"\n--- E2 per-draw paired analysis ({label} stream, seed {seed}) ---")
    print(f"  no-ops (prediction already feasible, every repair the identity): "
          f"{len(noop)}/{total}")
    for kind in kinds:
        for (na, nb) in shapes:
            c = sum(1 for k, a, b in noop if (k, a, b) == (kind, na, nb))
            n_cell = sum(1 for k, a, b, *_ in draws if (k, a, b) == (kind, na, nb))
            print(f"    {kind:>10} {na}x{nb}: {c}/{n_cell}")
    act = [(k, na, nb, W, u0, v0, us, vs, r) for k, na, nb, W, u0, v0, us, vs, r in draws
           if r["_pre"]["eps"] > 1e-9]
    pre = np.array([r["_pre"]["l0"] for *_, r in act], dtype=float)
    val = {o: np.array([r[o]["l0"] for *_, r in act], dtype=float)
           for o in ("sorted", "forward", "random", "walk")}
    def signtest(worse, better):
        w, l = int((worse > better).sum()), int((worse < better).sum())
        return w, l, w + l, _sign_test_wl(w, l)
    print(f"  active draws: {len(act)}")
    print(f"  sorted == pre exactly on {int((val['sorted'] == pre).sum())}/{len(act)} "
          f"(above pre: {int((val['sorted'] > pre).sum())}, below: "
          f"{int((val['sorted'] < pre).sum())})")
    for o in ("forward", "random"):
        w, l, nz, p = signtest(val[o], val["sorted"])
        print(f"  {o} strictly worse than sorted on {w}/{len(act)}, better on {l}; "
              f"exact one-sided sign test p = 2^-{nz} = {p:.1e}" if l == 0 else
              f"  {o} worse than sorted on {w}/{len(act)}, better on {l}; sign test p = {p:.1e}")
    w, l, nz, p = signtest(pre, val["walk"])
    print(f"  walk strictly below pre on {w}/{len(act)}, above on {l}; "
          f"exact one-sided sign test p = 2^-{nz} = {p:.1e}" if l == 0 else
          f"  walk below pre on {w}/{len(act)}, above on {l}; sign test p = {p:.1e}")
    ws, ls, _, _ = signtest(val["sorted"], val["walk"])
    print(f"  walk beats sorted {ws}-{ls} on the active draws")
    # What the sorted scan does to the coordinates, measured against the reference optimum,
    # which the variables us and vs hold.
    moved, restored, destroyed, wrongdiff = [], 0, 0, 0
    for k, na, nb, W, u0, v0, us, vs, r in act:
        o = orders(W, None)["sorted"]
        ru, rv = repair(W, u0, v0, o)
        y = np.concatenate([ru, rv]); star = np.concatenate([us, vs])
        p0 = np.concatenate([np.maximum(u0, 0), np.maximum(v0, 0)])
        moved.append(int((np.abs(y - p0) > 1e-9).sum()))
        pre_wrong = np.abs(p0 - star) > 1e-6
        post_wrong = np.abs(y - star) > 1e-6
        restored += int((pre_wrong & ~post_wrong).sum())
        destroyed += int((~pre_wrong & post_wrong).sum())
        if (pre_wrong != post_wrong).any(): wrongdiff += 1
    print(f"  sorted moved between {min(moved)} and {max(moved)} coordinates per active draw; "
          f"its wrong set differs from the prediction's on {wrongdiff}/{len(act)}; "
          f"restored {restored} correct coordinates and destroyed {destroyed} across them")
    return draws


def E2_seed_variants(seeds=(101, 303, 404)):
    """Run the design of experiment E2 again at other seeds.

    This backs the remark the paper makes about Table 7. At the pinned seed, the paired
    comparisons that E2_paired prints show no exception: no draw on which a comparison goes
    the other way from the rest. The paper says that this clean sweep is a property of the
    draws at the pinned seed rather than a law. The function prints the paired analysis for
    each other seed, so that the exceptions in each direction can be counted.
    """
    for s in seeds:
        print(f"\n=== E2 per-draw analysis at seed {s} (alternate-seed disclosure) ===")
        E2_paired(seed=s, label=f"seed {s}")


def E3_drift(kinds=("uniform", "clustered", "euclidean"), sizes=(60, 120), reps=10, seed=29):
    """Run experiment E3 and print Table 8 (tab:drift).

    The prediction is the optimal dual of the previous period, used after the weights have
    drifted. The function returns one tuple for each cell. The tuple holds the family, the
    size, the mean number of coordinates in which the clamped prediction differs from the
    reference optimum, the mean and the standard deviation of the adjustment count from the
    row-reduction start, and a dictionary that maps each operator of OPS to the mean and the
    standard deviation of its adjustment count.
    """
    rng = np.random.default_rng(seed)
    cells = []
    print("\n=== E3: drifted prediction (last period's dual), with solver ===")
    print(f"{'instance':>11} {'n':>5} {'pre l0':>7} {'cold':>16} |" + "".join(f" {o:>16}" for o in OPS))
    for kind in kinds:
        for n in sizes:
            acc = {o: [] for o in OPS}; cold = []; pres = []
            for _ in range(reps):
                W, u0, v0 = drifted(kind, n, rng)
                us, vs = optimal_dual(W)
                r = measure(W, u0, v0, us, vs, with_solver=True, rng=rng)
                pres.append(r["_pre"]["l0"]); cold.append(r["_cold"]["adj"])
                for o in OPS: acc[o].append(r[o]["adj"])
            line = (f"{kind:>11} {n:>5} {np.mean(pres):>7.1f} "
                    f"{np.mean(cold):>9.1f}+/-{np.std(cold):<5.1f} |")
            for o in OPS:
                line += f" {np.mean(acc[o]):>9.1f}+/-{np.std(acc[o]):<5.1f}"
            print(line)
            cells.append((kind, n, np.mean(pres), np.mean(cold), np.std(cold),
                          {o: (np.mean(acc[o]), np.std(acc[o])) for o in OPS}))
    print("\nLATEX rows (tab:drift, as in the paper):")
    for kind, n, pre_m, cold_m, cold_s, per in cells:
        cs = " & ".join(f"${per[o][0]:.1f}\\std{{{per[o][1]:.1f}}}$" for o in OPS_NOLOCAL)
        print(f"  {kind} & {n} & ${pre_m:.1f}$ & ${cold_m:.1f}\\std{{{cold_s:.1f}}}$ & {cs} \\\\")
    print("\n--- tab:drift prose checks ---")
    for kind, n, pre_m, cold_m, cold_s, per in cells:
        best6 = min(per[o][0] for o in OPS); best5 = min(per[o][0] for o in OPS_NOLOCAL)
        print(f"   {kind:>10} n={n:<4} pre {pre_m:.1f}/{2*n}  cold {cold_m:.1f}+/-{cold_s:.1f}  "
              f"walk {per['walk'][0]:.1f}  walk first(6op): {per['walk'][0] <= best6 + 1e-12}  "
              f"(5op): {per['walk'][0] <= best5 + 1e-12}  "
              f"cold beats every repair(6op): {cold_m < best6}  (5op): {cold_m < best5}")
    cw6 = sum(1 for *_, cold_m, _, per in cells if cold_m < min(per[o][0] for o in OPS))
    print(f"cold wins outright (6-op menu) in {cw6} of {len(cells)} cells")
    cw5 = sum(1 for *_, cold_m, _, per in cells if cold_m < min(per[o][0] for o in OPS_NOLOCAL))
    print(f"cold wins outright (5-op menu) in {cw5} of {len(cells)} cells")
    print("cold one-std overlap checks (is a 'cold win' larger than the noise?):")
    for kind, n, pre_m, cold_m, cold_s, per in cells:
        sm, ss = per["sorted"]
        bo = min(OPS, key=lambda o: per[o][0]); bm, bs = per[bo]
        print(f"   {kind:>10} n={n:<4} cold {cold_m:.1f}+/-{cold_s:.1f} | vs sorted "
              f"{sm:.1f}+/-{ss:.1f} within one std: {abs(cold_m-sm) <= max(cold_s, ss)}"
              f" | vs best repair {bo} {bm:.1f}+/-{bs:.1f} within one std: "
              f"{abs(cold_m-bm) <= max(cold_s, bs)}")
    return cells


# Rank correlations.

def _spear(x, y):
    """Compute the Spearman rank correlation of x and y, with average ranks for ties.

    The result is nan, meaning not a number, when there are fewer than 3 points or when all
    the values of x or all the values of y are equal.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _cell_key(r):
    return (r["kind"], r["n"])


def correlation_stats(raw, ops, quants=QUANTS):
    """Compute the Spearman correlation of each quantity with solver adjustments, in three ways.

    The argument raw is the list of draws, ops names the operators to include and quants
    names the quantities. The result is a dictionary with one entry for each way, and each of
    these entries is keyed by quantity.
    The entry operator_only holds a list with one value for each draw: the correlation
    between the quantity and the adjustment count across the operators applied to that one
    instance. This is the statistic that the text of the paper describes.
    The entry pooled_within maps each cell to the correlation over all the points of that
    cell, where a point is one pair of instance and operator.
    The entry pooled holds a single correlation over every such point.
    The entries n_draws and n_ops hold the number of draws and the number of operators.
    """
    cells = sorted({_cell_key(r) for r in raw})
    out = {"operator_only": {}, "pooled_within": {}, "pooled": {},
           "n_draws": len(raw), "n_ops": len(ops)}
    for q in quants:
        out["operator_only"][q] = [_spear([r[o][q] for o in ops], [r[o]["adj"] for o in ops])
                                   for r in raw]
        pw = {}
        for c in cells:
            sub = [r for r in raw if _cell_key(r) == c]
            pw[c] = _spear([r[o][q] for r in sub for o in ops],
                           [r[o]["adj"] for r in sub for o in ops])
        out["pooled_within"][q] = pw
        out["pooled"][q] = _spear([r[o][q] for r in raw for o in ops],
                                  [r[o]["adj"] for r in raw for o in ops])
    return out


def report_correlations(raw, ops, label, quants=QUANTS):
    st = correlation_stats(raw, ops, quants)
    print(f"\n=== Which repair-side quantity predicts solver adjustments?  [{label}] ===")
    print(f"    operator menu = {ops}   ({st['n_draws']} draws, {st['n_ops']} operators)")
    print(f"{'quantity':>8} {'operator-only':>15} {'(n used)':>9} {'pooled-within-cell':>20} "
          f"{'pooled':>9}")
    for q in quants:
        od = [v for v in st["operator_only"][q] if not np.isnan(v)]
        pw = [v for v in st["pooled_within"][q].values() if not np.isnan(v)]
        print(f"{q:>8} {np.mean(od):>+15.4f} {len(od):>9} {np.mean(pw):>+20.4f} "
              f"{st['pooled'][q]:>+9.4f}")
    return st


# Significance tests.

def _sign_test_wl(wins, losses):
    """Return the p-value of the exact one-sided sign test.

    It is the chance that a fair coin, flipped once for every win and once for every loss,
    comes up heads at least as many times as there are wins. Ties are left out before calling.
    """
    n = wins + losses
    return float(Fraction(sum(comb(n, k) for k in range(wins, n + 1)), 2 ** n)) if n else 1.0


def _sign_flip_p(d):
    """Run the exact one-sided sign-flip permutation test that the mean of d is above 0.

    The function enumerates every pattern of signs for the entries of d. There are 2 to the
    power of the length of d such patterns. For each pattern it multiplies every entry of d by
    its sign and takes the mean of the products. It returns the p-value, the share of the
    patterns for which that mean is at least the observed mean of d.
    """
    d = np.asarray(d, float); k = len(d)
    obs = d.mean(); cnt = 0
    for bits in range(1 << k):
        s = np.array([1.0 if (bits >> i) & 1 else -1.0 for i in range(k)])
        if (s * d).mean() >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << k)


def _sign_test_p(d):
    """Run the exact one-sided sign test that the median of d is above 0.

    Zeros are dropped. The function returns the p-value, the number of positive entries and
    the number of nonzero entries.
    """
    d = np.asarray(d, float)
    nz = d[np.abs(d) > 1e-12]
    n = len(nz); k = int((nz > 0).sum())
    return _sign_test_wl(k, n - k), k, n


def _paired_t_p(d):
    """Run the one-sided t test that the mean of d is above 0.

    The function returns the p-value and the t statistic.
    """
    d = np.asarray(d, float); n = len(d)
    sd = d.std(ddof=1); m = d.mean()
    if sd == 0:
        if m > 0:
            return 0.0, np.inf
        return 1.0, (-np.inf if m < 0 else np.nan)
    tstat = m / (sd / np.sqrt(n))
    return float(tdist.sf(tstat, n - 1)), float(tstat)


def significance(raw, ops, qa="l0", qb="gap", label="", B=20000, seed=101):
    """Test whether the quantity qa predicts solver work better than the quantity qb.

    For each draw the function takes the operator-only Spearman correlation of qa minus that
    of qb, and it averages these differences within each cell. It tests the per-cell means
    over the cells with a paired t test, a sign test and an exact sign-flip permutation test.
    A bootstrap resamples the draws inside each of the fixed cells. It does so B times, with a
    random generator of its own that is seeded with seed. The same three tests are then run on
    the pooled-within-cell statistic.
    """
    cells = sorted({_cell_key(r) for r in raw})
    st = correlation_stats(raw, ops, [qa, qb])
    # The difference for each draw, grouped by cell. A draw is dropped when either of its two
    # correlations is nan.
    per_cell_draws = {}
    for c in cells:
        vals = []
        for r, a, b in zip(raw, st["operator_only"][qa], st["operator_only"][qb]):
            if _cell_key(r) == c and not np.isnan(a) and not np.isnan(b):
                vals.append(a - b)
        per_cell_draws[c] = np.array(vals, float)
    dcell = np.array([per_cell_draws[c].mean() for c in cells], float)
    print(f"\n--- significance: {qa} vs {qb}  [{label}] ---")
    print(f"per-cell mean difference of the OPERATOR-ONLY Spearman ({qa} - {qb}):")
    for c, d in zip(cells, dcell):
        print(f"    {str(c):>18} {d:+.4f}   (n draws used {len(per_cell_draws[c])})")
    print(f"  mean over {len(cells)} cells {dcell.mean():+.4f}  sd {dcell.std(ddof=1):.4f}  "
          f"{qa} wins in {(dcell > 0).sum()}/{len(cells)} cells")
    pt, tst = _paired_t_p(dcell)
    ps, k, nn = _sign_test_p(dcell)
    pp = _sign_flip_p(dcell)
    print(f"  paired t (one-sided)      t={tst:.3f}  p={pt:.4f}")
    print(f"  sign test (one-sided)     {k}/{nn} positive  p={ps:.4f}")
    print(f"  exact sign-flip perm      p={pp:.4f}  (2^{len(cells)} enumerations)")
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        acc = 0.0
        for c in cells:
            v = per_cell_draws[c]
            acc += v[rng.integers(0, len(v), len(v))].mean()
        boots[b] = acc / len(cells)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pboot = float((boots <= 0).mean())
    print(f"  bootstrap over instances within the fixed cells (B={B}): "
          f"mean {dcell.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(diff<=0)={pboot:.4f}")
    # The same tests on the pooled-within-cell statistic.
    dpw = np.array([st["pooled_within"][qa][c] - st["pooled_within"][qb][c] for c in cells], float)
    dpw = dpw[~np.isnan(dpw)]
    pt2, tst2 = _paired_t_p(dpw); ps2, k2, n2 = _sign_test_p(dpw); pp2 = _sign_flip_p(dpw)
    print(f"  [pooled-within-cell statistic] per-cell diffs mean {dpw.mean():+.4f} "
          f"sd {dpw.std(ddof=1):.4f}  {qa} wins {(dpw > 0).sum()}/{len(dpw)}   "
          f"t p={pt2:.4f}  sign p={ps2:.4f}  perm p={pp2:.4f}")
    return {"cells": cells, "dcell": dcell, "t_p": pt, "sign_p": ps, "perm_p": pp,
            "boot_ci": (lo, hi), "boot_p": pboot, "dpw": dpw}


# The report on U.

def report_U(raw, tol=1e-9):
    """Print the six statistics about U that the paper quotes after Proposition 43.

    They are recomputed on the very draws of Table 6. The output numbers them from 1 to 6,
    after a sanity check numbered 0. The function only reads the draws and does not use the
    random generator.
    """
    print("===== U INSTRUMENTATION BEGIN =====")
    cells = sorted({_cell_key(r) for r in raw})
    N = len(raw)
    print(f"\n=== U (tight-row lower bound) on the {N} draws of tab:solver ===")

    # Item 0 is a sanity check that U is a valid lower bound on the adjustment count everywhere.
    bad = [(r["kind"], r["n"], o) for r in raw for o in OPS if r[o]["U"] > r[o]["adj"]]
    print(f"[0] prop:tightrows holds (U <= adj) on all {N}x{len(OPS)} "
          f"(instance, repair) pairs: {not bad}   violations: {bad or 'none'}")
    print(f"    cold start: U = 0 on {sum(1 for r in raw if r['coldU'] == 0)}/{N} draws "
          f"(max cold U = {max(r['coldU'] for r in raw)})")

    # Item 1 is the parity of the largest violation, eps.
    integral = [r for r in raw if abs(r["eps"] - round(r["eps"])) <= 1e-6]
    nonint = N - len(integral)
    pos = [r for r in raw if r["eps"] > tol]
    odd = [r for r in integral if r["eps"] > tol and int(round(r["eps"])) % 2 == 1]
    even = [r for r in integral if r["eps"] > tol and int(round(r["eps"])) % 2 == 0]
    print(f"[1] eps: {len(pos)}/{N} draws have eps > 0; {nonint} draws have NON-INTEGRAL eps")
    print(f"    eps ODD  in {len(odd)}/{N} draws;  eps EVEN in {len(even)}/{N};  "
          f"eps <= 0 in {N - len(pos)}")

    # Item 2 asks whether, on the draws where eps is odd, the value of U for the uniform shift
    # is exactly the number of rows.
    ok = [r for r in odd if r["shift"]["U"] == r["na"]]
    print(f"[2] on the {len(odd)} odd-eps draws, uniform shift has U == |A| exactly in "
          f"{len(ok)}/{len(odd)}")
    if len(ok) != len(odd):
        print(f"    exceptions: {[(r['kind'], r['n'], r['shift']['U'], r['na']) for r in odd if r['shift']['U'] != r['na']][:10]}")
    okall = [r for r in raw if r["shift"]["U"] == r["na"]]
    print(f"    (over ALL {N} draws, shift has U == |A| in {len(okall)})")

    # Item 3 is U as a share of the adjustment count, for each cell.
    print("[3] U / adj per cell   (ratio of cell means, and mean of per-draw ratios)")
    print(f"{'cell':>20} " + "".join(f"{o:>14}" for o in OPS))
    share = {o: [] for o in OPS}; share_pd = {o: [] for o in OPS}
    for c in cells:
        sub = [r for r in raw if _cell_key(r) == c]
        line = f"{str(c):>20} "
        for o in OPS:
            mU = float(np.mean([r[o]["U"] for r in sub]))
            mA = float(np.mean([r[o]["adj"] for r in sub]))
            pd = float(np.mean([r[o]["U"] / r[o]["adj"] for r in sub if r[o]["adj"] > 0]))
            share[o].append(mU / mA); share_pd[o].append(pd)
            line += f"  {100*mU/mA:>5.1f}% /{100*pd:>5.1f}%"
        print(line)
    for o in OPS:
        print(f"    {o:>12}: share of adjustments (cell-mean ratio) "
              f"{100*min(share[o]):.1f}%--{100*max(share[o]):.1f}%   "
              f"(per-draw mean {100*min(share_pd[o]):.1f}%--{100*max(share_pd[o]):.1f}%)")
    grp = ["sorted", "forward", "random", "walk"]
    lo = min(min(share[o]) for o in grp); hi = max(max(share[o]) for o in grp)
    lo2 = min(min(share_pd[o]) for o in grp); hi2 = max(max(share_pd[o]) for o in grp)
    print(f"    scans+walk together: {100*lo:.1f}%--{100*hi:.1f}%  "
          f"(per-draw mean {100*lo2:.1f}%--{100*hi2:.1f}%)")

    # Item 4 is the value of U for the endpoint shift, in units of the number of rows.
    print("[4] endpoint (local) shift, mean U / |A| per cell:")
    rat = []; rat_pd = []
    for c in cells:
        sub = [r for r in raw if _cell_key(r) == c]
        mU = float(np.mean([r["local_shift"]["U"] for r in sub]))
        na = sub[0]["na"]
        pd = float(np.mean([r["local_shift"]["U"] / r["na"] for r in sub]))
        rat.append(mU / na); rat_pd.append(pd)
        print(f"    {str(c):>20} mean U {mU:>7.2f} / |A| {na:>4} = {mU/na:.3f}   "
              f"(mean of per-draw U/|A| = {pd:.3f})")
    print(f"    range over cells: {min(rat):.2f}|A| .. {max(rat):.2f}|A|   "
          f"(per-draw-mean version {min(rat_pd):.2f} .. {max(rat_pd):.2f})")
    allpd = [r["local_shift"]["U"] / r["na"] for r in raw]
    print(f"    range over the {N} individual draws: {min(allpd):.2f}|A| .. {max(allpd):.2f}|A|")

    # Item 5 compares the value of U for the endpoint shift with that for the uniform shift,
    # draw by draw.
    lt = sum(1 for r in raw if r["local_shift"]["U"] < r["shift"]["U"])
    eq = sum(1 for r in raw if r["local_shift"]["U"] == r["shift"]["U"])
    gt = sum(1 for r in raw if r["local_shift"]["U"] > r["shift"]["U"])
    print(f"[5] U(local_shift) < U(shift) in {lt}/{N} draws  (equal {eq}, greater {gt})")

    # Item 6 is U as a quantity by which to rank the repairs.
    for ops, lab in ((OPS_NOLOCAL, "5 operators (paper's menu)"), (OPS, "6 operators")):
        st = correlation_stats(raw, ops, ["U", "l0", "gap", "l1"])
        print(f"[6] operator-only Spearman with adj, averaged over the {N} draws  [{lab}]:")
        for q in ("U", "l0", "gap", "l1"):
            od = [x for x in st["operator_only"][q] if not np.isnan(x)]
            pwv = [x for x in st["pooled_within"][q].values() if not np.isnan(x)]
            print(f"    {q:>6}  operator-only {np.mean(od):+.4f} (n={len(od)})   "
                  f"pooled-within-cell {np.mean(pwv):+.4f}   pooled {st['pooled'][q]:+.4f}")
    print("\n===== U INSTRUMENTATION END =====")


# The paired comparison of scan orders.

def scan_order_paired(raw, a="sorted", b="forward"):
    """Compare the sorted scan with the forward scan on dual adjustments, draw by draw.

    Those two scans are the defaults of the arguments a and b. The comparison is paired over
    every draw of E1, and a win is a draw on which the first scan leads to fewer adjustments
    than the second. The function prints the wins, the ties, the mean margin in each cell, and
    the exact one-sided sign test that the paper quotes. The Wilcoxon line is for reference
    only; at this size it uses a normal approximation. The function returns the number of
    wins, the number of ties, the p-value of the Wilcoxon test, and the smallest and the
    largest of the per-cell mean margins.
    """
    xa = np.array([r[a]["adj"] for r in raw], dtype=float)
    xb = np.array([r[b]["adj"] for r in raw], dtype=float)
    wins = int((xa < xb).sum()); ties = int((xa == xb).sum()); losses = int((xa > xb).sum())
    stat, p = wilcoxon(xb, xa, alternative="greater", zero_method="wilcox")
    cells = sorted({(r["kind"], r["n"]) for r in raw})
    margins = []
    print(f"\n--- {a} vs {b}, paired over all draws (dual adjustments) ---")
    for k, n in cells:
        idx = [i for i, r in enumerate(raw) if r["kind"] == k and r["n"] == n]
        m = float(np.mean(xb[idx] - xa[idx])); margins.append(m)
        print(f"   {k:>10} n={n:<4} mean margin ({b} - {a}) = {m:+.1f}")
    print(f"  {a} < {b} in {wins} of {len(xa)} draws ({ties} ties, {losses} losses)")
    print(f"  per-cell mean margins from {min(margins):.1f} to {max(margins):.1f} adjustments")
    print(f"  one-sided Wilcoxon signed-rank ({b} > {a}): W={stat:.1f}  p={p:.3g}"
          f"  (normal approximation; past its validity at this magnitude)")
    nz = wins + losses
    print(f"  exact one-sided sign test on the {nz} nonzero differences: "
          f"p = {_sign_test_wl(wins, losses):.1e}")
    smallest = 0
    for k, n in cells:
        idx = [i for i, r in enumerate(raw) if r["kind"] == k and r["n"] == n]
        others = [np.mean([raw[i][o]["adj"] for i in idx]) for o in ("forward", "random")]
        if np.mean(xa[idx]) < min(others):
            smallest += 1
    print(f"  {a} has the smallest mean among the scan orders in {smallest} of {len(cells)} cells")
    return wins, ties, p, min(margins), max(margins)


# Experiment E4.

def E4_coldzero(kinds=("uniform", "clustered", "euclidean"), sizes=(100, 200), reps=10, seed=31):
    """Run experiment E4, on the repaired zero prediction as a start for the solver.

    Proposition 15 describes the repair of the all-zero prediction. The question here is how
    that repaired zero compares, as a solver start, with the classical row-reduction start.
    The zero prediction is repaired by the sorted scan and by the walk. The function returns,
    for every cell and for each of these two repairs, the mean number of adjustments from the
    repaired zero divided by the mean number from the row-reduction start.
    """
    rng = np.random.default_rng(seed)
    print("\n=== E4: repaired-zero start vs row-reduction start (dual adjustments) ===")
    print(f"{'instance':>11} {'n':>5} {'rowmax':>16} {'sorted(0)':>16} {'walk(0)':>16} "
          f"{'s/base':>7} {'w/base':>7}")
    ratios = []
    for kind in kinds:
        for n in sizes:
            base, srt, wlk = [], [], []
            for _ in range(reps):
                W = instance(kind, n, rng)
                base.append(solve_warm(W, W.max(axis=1).astype(float), np.zeros(n))[1])
                z = np.zeros(n)
                # Only the sorted scan and the walk are used here, but all_repairs also draws
                # the random scan order from rng, and the pinned numbers depend on that draw.
                reps_ops = all_repairs(W, z, z, rng)
                srt.append(solve_warm(W, *reps_ops["sorted"])[1])
                wlk.append(solve_warm(W, *reps_ops["walk"])[1])
            b, s_, w_ = np.mean(base), np.mean(srt), np.mean(wlk)
            ratios += [s_ / b, w_ / b]
            print(f"{kind:>11} {n:>5} {b:>9.1f}+/-{np.std(base):<5.1f} "
                  f"{s_:>9.1f}+/-{np.std(srt):<5.1f} {w_:>9.1f}+/-{np.std(wlk):<5.1f} "
                  f"{s_/b:>7.2f} {w_/b:>7.2f}")
    print(f"repaired-zero / row-reduction ratio range over all cells and both repairs: "
          f"{min(ratios):.2f}x .. {max(ratios):.2f}x")
    return ratios


if __name__ == "__main__":
    # The defaults of every function are the pinned settings behind the tables of the paper.
    t = time.time()
    raw, _ = E1_solver()
    E1_quantities(raw)
    st6 = report_correlations(raw, OPS, "6 operators, local_shift INCLUDED")
    st5 = report_correlations(raw, OPS_NOLOCAL, "5 operators, local_shift EXCLUDED (paper's menu)")
    print("\n--- size of the operator-menu confound (6-op minus 5-op) ---")
    for q in QUANTS:
        o6 = np.nanmean(st6["operator_only"][q]); o5 = np.nanmean(st5["operator_only"][q])
        p6 = np.nanmean(list(st6["pooled_within"][q].values()))
        p5 = np.nanmean(list(st5["pooled_within"][q].values()))
        print(f"  {q:>7}: operator-only {o5:+.3f} -> {o6:+.3f} ({o6-o5:+.3f}); "
              f"pooled-within {p5:+.3f} -> {p6:+.3f} ({p6-p5:+.3f}); "
              f"pooled {st5['pooled'][q]:+.3f} -> {st6['pooled'][q]:+.3f} "
              f"({st6['pooled'][q]-st5['pooled'][q]:+.3f})")
    for ops, lab in ((OPS_NOLOCAL, "5-op menu"), (OPS, "6-op menu")):
        significance(raw, ops, "l0", "gap", lab)
        significance(raw, ops, "l0", "l1", lab)
        significance(raw, ops, "l0", "l1disp", lab)
    scan_order_paired(raw, "sorted", "forward")
    E2_paired(E2_rectangular())
    E2_seed_variants()
    E3_drift()
    E4_coldzero()
    report_U(raw)
    print(f"\n({time.time()-t:.0f}s)")
