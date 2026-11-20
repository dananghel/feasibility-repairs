"""Compute exact, certified game values of the view of the trap instance with three rows.

The view has three rows, u0, u1 and u2, and four columns, v0, v1, v2 and c. The clamped
prediction is 9 on v0 and on c and zero on the other five vertices. Four edges are violated:
(u0, v1) with weight 8, (u1, v1) with weight 7, (u1, v2) with weight 6 and (u2, v2) with
weight 5. Six pairs have hidden weights, each a whole number from 0 to 9. They are (u0, v0),
(u0, c), (u1, v0), (u1, c), (u2, v0) and (u2, c), which makes one million completions. The
feasible integral outputs are the vectors of seven whole numbers from 0 to 9 that are
feasible for the completion with every hidden weight at 9. There are 831,881 of them.

Here pre is the distance of the prediction to the optimal face of a completion, and post is
the distance of an output to the same face. Both count the coordinates that differ from the
nearest point of the face. In the ratio game the payoff is post divided by pre.

The certification is rigorous and needs no exact LP. Any rational distribution pi over the
outputs gives an upper bound on the game value, namely the largest expected payoff of pi over
the completions. It is computed exactly in integers over all one million completions. Any
rational distribution q over the completions gives a lower bound, namely the smallest
expected payoff of q over the outputs. It is computed exactly in integers over all 831,881
outputs.

The double oracle method proposes pi and q by solving a floating-point LP on a working set of
outputs and completions. They are rounded to rationals with Fraction.limit_denominator at
several denominators, and both bounds are verified exactly. If the bounds do not meet, the
completion that is worst for pi and the output that is the best response to q are added to
the working set and the loop continues. If the loop stops before the bounds meet, the script
reports the certified interval between them. The additive game, whose payoff is post minus
pre, is treated in the same way.

The faces are cached in the file trap3_faces.npz. When that file is missing, they are first
computed by a pool of 12 worker processes. Every line the script prints is also written to
the file task1b_trap3_certify.out.
"""
import sys, time, itertools, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
# Add the parent folder to the import path, so that verify_dichotomy.py can be imported.
sys.path.insert(0, os.path.join(HERE, ".."))
from fractions import Fraction
import numpy as np
from multiprocessing import Pool
from common import View, solve_mixed_lp
import verify_dichotomy as vd

NA, NB, N = 3, 4, 7
P = (0, 0, 0, 9, 0, 0, 9)
VIO = {(0, 4): 8, (1, 4): 7, (1, 5): 6, (2, 5): 5}
VIEW = View(NA, NB, P, VIO)
HID = VIEW.hidden
RANGES = [range(P[i] + P[j] + 1) for i, j in HID]
CACHE = os.path.join(HERE, "trap3_faces.npz")


def comp_of_index(c):
    """Return the completion with index c, in the same order as View.completions.

    The index counts from zero.
    """
    h = []
    for r in reversed(RANGES):
        h.append(c % len(r)); c //= len(r)
    return tuple(reversed(h))


def _faces_chunk(args):
    """Compute the faces and the distances before the repair for one chunk of completions.

    args is the pair (lo, hi). The chunk is the completions with index from lo to hi minus 1,
    and it is the share of work that a worker does in one call. Three arrays are returned:
    the integral face points of these completions stacked one after another, the number of
    face points of each completion, and the distance of the prediction to the face of each
    completion.
    """
    lo, hi = args
    pts_all, counts, pres = [], [], []
    pf = np.array(P, float)
    for c in range(lo, hi):
        h = comp_of_index(c)
        W = VIEW.W_of(h); OPT = vd.matching_value(W)
        pts = vd.face_points_bip(W, OPT)
        pts_all.append(pts.astype(np.int8)); counts.append(len(pts))
        pres.append(vd.face_l0(pf, pts))
    return np.vstack(pts_all), np.array(counts, np.int32), np.array(pres, np.int16)


def all_feasible_z():
    """Return the 831,881 feasible outputs as an int8 array with one output per row.

    The outputs are the same as those of VIEW.feasible_z(), but each constraint is tested on
    all ten million candidate vectors at once instead of on one vector at a time.
    """
    top = dict(VIO)
    for i, j in HID: top[(i, j)] = P[i] + P[j]
    G = np.array(list(itertools.product(range(10), repeat=N)), dtype=np.int8)
    ok = np.ones(len(G), bool)
    for (i, j), w in top.items(): ok &= (G[:, i].astype(int) + G[:, j] >= w)
    return G[ok]


class Game:
    """The payoff table: the distance post for every output z and every completion c.

    The entry for z and c is the smallest number of coordinates in which z differs from a
    point of the face of c. The table is filled in lazily, one row or one column at a time.
    The method post_of_z gives the row of one output and post_of_c gives the column of one
    completion, and each result is kept once it is computed. allpts holds every face point of
    every completion, one after another. The face of completion c starts at the row of allpts
    whose number is entry c of starts. pres holds the distance before the repair for every
    completion, and Z holds the feasible outputs.
    """
    def __init__(self, allpts, starts, pres, Z):
        self.allpts, self.starts, self.pres, self.Z = allpts, starts, pres, Z
        self.nc = len(pres); self.nz = len(Z)
        self.cache_z = {}; self.cache_c = {}

    def post_of_z(self, zi):
        if zi not in self.cache_z:
            d = np.count_nonzero(self.allpts != self.Z[zi], axis=1)
            self.cache_z[zi] = np.minimum.reduceat(d, self.starts).astype(np.int16)
        return self.cache_z[zi]

    def post_of_c(self, ci):
        if ci not in self.cache_c:
            end = self.starts[ci + 1] if ci + 1 < self.nc else len(self.allpts)
            best = np.full(self.nz, 99, np.int16)
            for pt in self.allpts[self.starts[ci]:end]:
                np.minimum(best, np.count_nonzero(self.Z != pt, axis=1).astype(np.int16), out=best)
            self.cache_c[ci] = best
        return self.cache_c[ci]


def exact_upper(g, mode, pi):
    """Return the exact largest expected payoff of the mixed output pi over all completions.

    g is the Game, and mode is ratio or add. pi is a dictionary from the index of an output to
    its probability, a Fraction. The result is the largest expected payoff, as a Fraction,
    together with the index of a completion that attains it. That payoff is a rigorous upper
    bound on the game value.
    """
    den = 1
    for x in pi.values(): den = den * x.denominator // math.gcd(den, x.denominator)
    num = np.zeros(g.nc, np.int64)
    for zi, pr in pi.items():
        if pr > 0: num += int(pr * den) * g.post_of_z(zi).astype(np.int64)
    pres = g.pres.astype(np.int64)
    if mode == "ratio":
        # The value against a completion is its entry of num divided by the product of den and
        # its distance before the repair. Find the largest value exactly: take the 50 largest
        # in floating point, then compare those as exact fractions.
        f = num / (den * pres.astype(float))
        cand = np.argsort(-f)[:50]
        best = max(cand, key=lambda c: Fraction(int(num[c]), int(den * pres[c])))
        v = Fraction(int(num[best]), int(den * pres[best]))
        # Check exactly that no completion exceeds v. In integers, for every completion, its
        # entry of num times the denominator of v is at most the numerator of v times den
        # times its distance before the repair.
        assert (num * v.denominator <= v.numerator * den * pres).all()
        return v, int(best)
    else:
        val = num - den * pres
        best = int(val.argmax()); v = Fraction(int(val[best]), den)
        assert (val <= val[best]).all()
        return v, best


def exact_lower(g, mode, q):
    """Return the exact smallest expected payoff of any output against the mixed completion q.

    g is the Game, and mode is ratio or add. q is a dictionary from the index of a completion
    to its probability, a Fraction. The result is the smallest expected payoff, as a Fraction,
    together with the index of an output that attains it. That payoff is a rigorous lower
    bound on the game value.
    """
    pres = g.pres.astype(np.int64)
    if mode == "ratio":
        coef = {ci: pr / int(pres[ci]) for ci, pr in q.items() if pr > 0}
    else:
        coef = {ci: pr for ci, pr in q.items() if pr > 0}
    D = 1
    for x in coef.values(): D = D * x.denominator // math.gcd(D, x.denominator)
    num = np.zeros(g.nz, np.int64); shift = 0
    for ci, x in coef.items():
        num += int(x * D) * g.post_of_c(ci).astype(np.int64)
        if mode == "add": shift += int(x * D) * int(pres[ci])
    best = int(num.argmin())
    v = Fraction(int(num[best]) - shift, D)
    assert (num >= num[best]).all()
    return v, best


def rationalise(vec, dens=(1697, 5000, 10**4, 10**5, 10**6)):
    """Round a floating-point distribution to rationals, once per denominator bound in dens.

    vec is a dictionary from an index to a float probability. For each bound, every
    probability above a small threshold is rounded to the nearest fraction whose denominator
    is at most the bound, and the fractions are then divided by their sum so that they add up
    to 1. The list of these rational distributions is returned. 1697 is the denominator of
    the ratio value. It is tried first only to save time, since both bounds are checked
    exactly whatever the denominators.
    """
    outs = []
    for d in dens:
        fr = {k: Fraction(float(v)).limit_denominator(d) for k, v in vec.items() if v > 1e-9}
        s = sum(fr.values())
        if s <= 0: continue
        fr = {k: v / s for k, v in fr.items()}
        outs.append(fr)
    return outs


def solve_certified(g, mode, Zw, C, log, max_iter=60):
    """Run the double oracle method on a working set of outputs Zw and completions C.

    Each round solves the game restricted to the working set by a floating-point LP, rounds
    the two mixes to rationals, and computes the exact bounds they give over all completions
    and all outputs. If the best upper bound equals the best lower bound, that is the exact
    value, and it is returned with both optimal mixes. Otherwise violating outputs and
    completions are added to the working set and the next round starts. If the bounds never
    meet, that is, after max_iter rounds or when a round finds nothing new to add, None is
    returned with the best certified upper bound and the best certified lower bound, each as
    a pair of the bound and the rational mix that gives it. In both cases the final working
    set is returned as well. g is the Game, mode is ratio or add, and log is a function that
    prints one line of progress.
    """
    pres = g.pres.astype(float)
    best_up, best_lo = None, None
    for it in range(1, max_iter + 1):
        Zw = sorted(set(Zw)); C = sorted(set(C))
        R = np.array([g.post_of_z(zi)[C].astype(float) for zi in Zw])
        R = R / pres[C][None, :] if mode == "ratio" else R - pres[C][None, :]
        v, pi, q = solve_mixed_lp(R)
        pid = {Zw[k]: pi[k] for k in range(len(Zw))}; qd = {C[k]: q[k] for k in range(len(C))}
        # Rigorous upper bounds from the rationalised versions of pi, and rigorous lower bounds
        # from the rationalised versions of q.
        ups = [exact_upper(g, mode, pf) for pf in rationalise(pid)]
        los = [exact_lower(g, mode, qf) for qf in rationalise(qd)]
        (vu, cu), piu = min(zip(ups, rationalise(pid)), key=lambda t: t[0][0])
        (vl, zl), ql = max(zip(los, rationalise(qd)), key=lambda t: t[0][0])
        if best_up is None or vu < best_up[0]: best_up = (vu, piu)
        if best_lo is None or vl > best_lo[0]: best_lo = (vl, ql)
        log(f"    [{mode}] iter {it}: |Zw|={len(Zw)} |C|={len(C)} LP {v:.7f}; certified upper {float(best_up[0]):.7f} = {best_up[0]}, certified lower {float(best_lo[0]):.7f} = {best_lo[0]}")
        if best_up[0] == best_lo[0]:
            return best_up[0], best_up[1], best_lo[1], Zw, C
        # Add the violators: the completion that is worst for the mix pi over the outputs, and
        # the output that is the best response to the mix q over the completions.
        added = False
        if cu not in C: C.append(cu); added = True
        if zl not in Zw: Zw.append(zl); added = True
        # Also add a few more outputs and completions, picked by their floating-point scores,
        # to speed up the search.
        supp_c = [ci for ci, pr in qd.items() if pr > 1e-9]
        score = np.zeros(g.nz)
        for ci in supp_c:
            pc = g.post_of_c(ci).astype(float)
            score += qd[ci] * (pc / pres[ci] if mode == "ratio" else pc - pres[ci])
        for zi in np.argsort(score)[:6]:
            if int(zi) not in Zw: Zw.append(int(zi)); added = True
        expv = np.zeros(g.nc)
        for zi, pr in pid.items():
            if pr > 1e-9:
                pz = g.post_of_z(zi).astype(float)
                expv += pr * (pz / pres if mode == "ratio" else pz - pres)
        for ci in np.argsort(-expv)[:6]:
            if int(ci) not in C: C.append(int(ci)); added = True
        if not added:
            log("    no new violators but bounds differ: rationalisation gap; enlarging denominators")
            return None, best_up, best_lo, Zw, C
    return None, best_up, best_lo, Zw, C


def main():
    t0 = time.time()
    out = open(os.path.join(HERE, "task1b_trap3_certify.out"), "w")
    def log(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    nc = VIEW.n_completions()
    log(f"k=3 trap view: p={P}, violated={VIO}, hidden pairs {HID} in {[len(r) for r in RANGES]}: {nc} completions")
    Z = all_feasible_z(); log(f"feasible integral outputs: {len(Z)}  [{time.time()-t0:.0f}s]")
    if os.path.exists(CACHE):
        d = np.load(CACHE); allpts, counts, pres = d["allpts"], d["counts"], d["pres"]
        log(f"faces loaded from cache  [{time.time()-t0:.0f}s]")
    else:
        chunks = [(lo, min(lo + 20000, nc)) for lo in range(0, nc, 20000)]
        with Pool(12) as pool:
            res = pool.map(_faces_chunk, chunks)
        allpts = np.vstack([r[0] for r in res]); counts = np.concatenate([r[1] for r in res]); pres = np.concatenate([r[2] for r in res])
        np.savez_compressed(CACHE, allpts=allpts, counts=counts, pres=pres)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    log(f"faces: {len(allpts)} face points total, max face size {counts.max()}, pre range {pres.min()}..{pres.max()}  [{time.time()-t0:.0f}s]")
    g = Game(allpts, starts, pres, Z)

    # The deterministic value, exact, by the pool argument. It is the smallest, over all
    # outputs, of the largest ratio over all completions. A pool of completions gives every
    # output a lower bound on its largest ratio. The output with the smallest bound is then
    # checked against all completions. If none of them beats its bound, that output is the
    # best one. Otherwise its worst completions join the pool and the search repeats.
    presf = pres.astype(float)
    C = [0, nc - 1]
    best = np.zeros(g.nz)
    for ci in C: best = np.maximum(best, g.post_of_c(ci) / presf[ci])
    while True:
        zi = int(best.argmin()); full = g.post_of_z(zi) / presf; ci = int(full.argmax())
        if full[ci] <= best[zi] + 1e-9:
            vdet = max(Fraction(int(g.post_of_z(zi)[c]), int(pres[c])) for c in np.argsort(-full)[:20])
            log(f"  DETERMINISTIC value (exact, all {g.nz} outputs x all {nc} completions via pool of {len(C)}): {vdet} = {float(vdet):.6f} at z={tuple(int(x) for x in Z[zi])}")
            break
        for c2 in np.argsort(-full)[:3]:
            if int(c2) not in C: C.append(int(c2)); best = np.maximum(best, g.post_of_c(int(c2)) / presf[int(c2)])
    log(f"  [{time.time()-t0:.0f}s]")

    # Start the working set of outputs from the deterministic optimum and a few outputs that
    # earlier runs found in the support of the optimal mix. They only speed things up.
    zi0 = [zi]
    for z in [(0, 4, 3, 7, 6, 3, 7), (9, 7, 5, 0, 1, 0, 0), (0, 0, 0, 9, 8, 6, 9), (8, 6, 4, 1, 1, 1, 1), (9, 9, 9, 0, 0, 0, 0), (1, 6, 5, 8, 7, 5, 8)]:
        w = np.where((Z == np.array(z, np.int8)).all(axis=1))[0]
        if len(w): zi0.append(int(w[0]))
    rng = np.random.default_rng(0)
    C0 = [0, nc - 1] + [int(x) for x in rng.integers(0, nc, size=40)]
    for mode in ("ratio", "add"):
        v, up, lo, Zw, C = solve_certified(g, mode, list(zi0), list(C0), log)
        if v is not None:
            log(f"  EXACT {mode} value = {v} = {float(v):.7f}  (upper bound certified on all {nc} completions, lower bound certified on all {g.nz} outputs)")
            log("    optimal mixed output: " + "; ".join(f"z={tuple(int(x) for x in Z[zi])} p={pr}" for zi, pr in sorted(up.items(), key=lambda kv: -kv[1]) if pr > 0))
            log("    adversary mix (hidden w_u0v0,w_u0c,w_u1v0,w_u1c,w_u2v0,w_u2c ; pre): " +
                "; ".join(f"{comp_of_index(ci)} pre={int(pres[ci])} q={pr}" for ci, pr in sorted(lo.items(), key=lambda kv: -kv[1]) if pr > 0))
        else:
            log(f"  {mode}: certified interval [{lo[0]} = {float(lo[0]):.7f}, {up[0]} = {float(up[0]):.7f}] (not closed)")
            log("    upper-bound mix: " + "; ".join(f"z={tuple(int(x) for x in Z[zi])} p={pr}" for zi, pr in up[1].items() if pr > 0))
            log("    lower-bound adversary: " + "; ".join(f"{comp_of_index(ci)} pre={int(pres[ci])} q={pr}" for ci, pr in lo[1].items() if pr > 0))
        log(f"  [{time.time()-t0:.0f}s]")
    out.close()


if __name__ == "__main__":
    main()
