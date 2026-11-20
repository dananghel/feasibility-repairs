"""Run the exhaustive and face-exact checks of "Not All Feasibility Repairs Are Equal".

The program needs only the standard library. Run it with python3 verify_all.py. It takes
about five minutes.

The optimal face is the set of all optimal duals. Every distance to the optimal face is an
exact minimum over the whole face, which is what face-exact means. Two independent routines
compute these distances. The first, _l0_face, minimises over the whole face, fractional
points included. The second, _l0_face_integral, enumerates the integral points of the face,
those with whole-number coordinates. The function face_cross_validate checks that the two
routines agree. The labels in the printout follow the numbering of the paper.

The comments describe distances in words. The paper calls the number of coordinates that
differ the l0 distance and the sum of the absolute differences the l1 distance. The printed
labels l_inf and l_p stand for the largest absolute difference (the l-infinity distance) and
for the norm with a general exponent p. Unless another measure is named, the face distance
of a point is the number of coordinates in which it differs from the nearest point of the
optimal face.
"""
import itertools, math, random, time
from fractions import Fraction

# The repair operators.
def greedy_cover(cons, yhat, order):
    """Run the two-sided greedy repair on a covering system and return the repaired dual.

    A covering system asks that the two dual values of every constraint add up to at least
    its weight and that no dual value is negative. The list cons holds one triple per
    constraint: its two vertices and its weight. The prediction yhat is first clamped, with
    negative values replaced by zero. The constraints are then scanned in the given order,
    which lists positions in cons. At each violated constraint both endpoints are raised by
    the full residual, which is the weight minus the sum of the two dual values.
    """
    y = [max(0, t) for t in yhat]
    for idx in order:
        i, j, w = cons[idx]
        r = w - y[i] - y[j]
        if r > 0: y[i] += r; y[j] += r
    return y

def _sorted_repair(W, uh, vh, order=None, fired=None):
    """Run the same scan on a bipartite instance given by its weight matrix W.

    Every pair of a row and a column is a constraint, and its weight is the matrix entry. The
    lists uh and vh hold the row values and the column values of the prediction. If no order
    is given, the pairs are scanned by weight, largest first, with ties in row-major order
    (by row, then by column). A pair fires when the scan finds it violated and raises both
    of its endpoints. If fired is a list, every pair that fires is appended to it. The result
    is the list of repaired row values and the list of repaired column values.
    """
    na, nb = len(W), len(W[0])
    u = [max(0, t) for t in uh]; v = [max(0, t) for t in vh]
    if order is None:
        order = sorted(((a, b) for a in range(na) for b in range(nb)),
                       key=lambda ab: (-W[ab[0]][ab[1]], ab))
    for a, b in order:
        r = W[a][b] - u[a] - v[b]
        if r > 0:
            u[a] += r; v[b] += r
            if fired is not None: fired.append((a, b))
    return u, v

def _repair_1A(w, u, v, order):
    """Run the scan on an instance with a single row.

    The list w holds one weight per column, the weight of the pair formed by the row and
    that column. The number u is the row value of the prediction and the list v holds its
    column values. The columns are visited in the given order. The result is the repaired
    row value and the list of repaired column values.
    """
    (u2,), v2 = _sorted_repair([w], [u], v, [(0, k) for k in order])
    return u2, v2

def greedy_pack(C, u, v, order):
    """Run the greedy scan on a packing system and return the repaired dual.

    A packing system asks that the two dual values of every pair of a row and a column add
    up to at most the cost of the pair, held in the matrix C. The pairs are scanned in the
    given order. At each violated pair both endpoints are lowered by the full excess, which
    is the sum of the two dual values minus the cost.
    """
    u, v = list(u), list(v)
    for (a, b) in order:
        e = u[a] + v[b] - C[a][b]
        if e > 0: u[a] -= e; v[b] -= e
    return u, v

def greedy_pack_split(C, u, v, order):
    """Run the 'fair' variant of the packing scan, which splits the excess.

    At each violated pair the excess is split between the two endpoints instead of being
    taken off both. The column value is lowered by half the excess, rounded down, and the
    row value by the rest.
    """
    u, v = list(u), list(v)
    for (a, b) in order:
        e = u[a] + v[b] - C[a][b]
        if e > 0:
            lo = e // 2; u[a] -= e - lo; v[b] -= lo
    return u, v

def maximal_pack(C, u, v):
    """Run the alternating projection of Proposition 7(a) on a packing system.

    Each round has a row step and a column step. The row step lowers every row value to the
    smallest value, over all columns, of the cost of the pair minus the column value, unless
    the row value is already that low. The column step then does the same for every column
    value, against the new row values. The rounds repeat until nothing moves. After the
    first row step every constraint holds, so the column step never moves anything and the
    loop stops in the second round. The result is a maximal feasible point below the input
    pair u and v.
    """
    u, v = list(u), list(v)
    for _ in range(len(u) + len(v) + 2):
        nu = [min(u[a], min(C[a][b] - v[b] for b in range(len(v)))) for a in range(len(u))]
        nv = [min(v[b], min(C[a][b] - nu[a] for a in range(len(u)))) for b in range(len(v))]
        if nu == u and nv == v: break
        u, v = nu, nv
    return u, v

def feas_cover(cons, y): return all(y[i]+y[j] >= w for i,j,w in cons) and all(t >= 0 for t in y)
def feas_pack(C,u,v):    return all(u[a]+v[b] <= C[a][b] for a in range(len(u)) for b in range(len(v)))
def l1(a,b): return sum(abs(x-y) for x,y in zip(a,b))
def lp(a,b,p):
    # On vectors of whole numbers the value is an exact whole number when p is 1 or the
    # string inf. For any other p the result is a float that can carry a rounding error, so
    # callers allow a slack of one billionth.
    if p == 'inf': return max(abs(x-y) for x,y in zip(a,b))
    return sum(abs(x-y)**p for x,y in zip(a,b))**(1.0/p)

# Checks for Theorem 2 (covering) and Theorem 4 (packing).
def _random_cover_system(N, bipartite, wmax):
    """Draw a random covering system on N vertices.

    Some of the allowed pairs of vertices, a random selection of at least one, become the
    constraints, and each gets a random whole-number weight from 0 to wmax. If bipartite is
    true, the first half of the vertices (N divided by 2, rounded down) forms one side, the
    rest the other, and only pairs that join the two sides are allowed. Otherwise every pair
    of distinct vertices is allowed. The result is a list of triples: the two vertices and
    the weight.
    """
    if bipartite:
        h = N // 2; pool = [(i,j) for i in range(h) for j in range(h,N)]
    else:
        pool = [(i,j) for i in range(N) for j in range(i+1,N)]
    return [(i,j,random.randint(0,wmax)) for (i,j) in random.sample(pool, random.randint(1,len(pool)))]

def check_cover(bipartite, trials=1500, seed=21):
    """Check Theorem 2 for the sum of the absolute differences, on random covering systems.

    Theorem 2 says that the scan never increases the distance to any feasible point. Each
    trial draws a system on 2 to 4 vertices with weights from 0 to 5, a prediction with
    entries from minus 2 to 5 and a random scan order, and asserts that the scan output is
    feasible. The output is then compared with every feasible point whose coordinates are
    whole numbers from 0 to 6: the distance from the output to the point is set against the
    distance from the clamped prediction to the same point. The result is the number of
    comparisons, the number of them in which the distance grew, and the largest ratio of the
    distance after the repair to the distance before it.
    """
    random.seed(seed); viol = tot = 0; worst = 0.0
    for _ in range(trials):
        N = random.randint(2,4)
        cons = _random_cover_system(N, bipartite, 5)
        yh = [random.randint(-2,5) for _ in range(N)]
        order = list(range(len(cons))); random.shuffle(order)
        y2 = greedy_cover(cons, yh, order)
        assert feas_cover(cons, y2)
        for y in itertools.product(range(0,7), repeat=N):
            if not feas_cover(cons, list(y)): continue
            before = l1([max(0,t) for t in yh], y); after = l1(y2, y); tot += 1
            if after > before: viol += 1
            if before > 0: worst = max(worst, after/before)
    return tot, viol, worst

def check_all_norms(trials=1200, seed=5, PS=(1, 1.5, 2, 3, 'inf')):
    """Check Theorem 2 in the norm with exponent p, for every p in PS.

    Theorem 2 is stated for every such norm, and this function tries the exponents in PS.
    The norm with exponent p raises every absolute difference to the power p, adds them up
    and takes the root of order p of the sum. The string inf stands for the largest absolute
    difference. The comparisons are made as in check_cover, on systems drawn without the
    bipartite restriction, and every p is tried on the same comparisons. The result maps
    each p to a list of three numbers: the number of comparisons, the number of them in
    which the distance grew by more than the rounding slack, and the largest ratio of the
    distance after the repair to the distance before it.
    """
    random.seed(seed); res = {p: [0,0,0.0] for p in PS}
    for _ in range(trials):
        N = random.randint(2,4)
        cons = _random_cover_system(N, False, 5)
        yh = [random.randint(-2,5) for _ in range(N)]
        order = list(range(len(cons))); random.shuffle(order)
        y2 = greedy_cover(cons, yh, order)
        assert feas_cover(cons, y2)
        yc = [max(0,t) for t in yh]
        for y in itertools.product(range(0,7), repeat=N):
            if not feas_cover(cons, list(y)): continue
            for p in PS:
                b = lp(yc,y,p); a = lp(y2,y,p); res[p][0] += 1
                if a > b + 1e-9: res[p][1] += 1
                if b > 1e-12: res[p][2] = max(res[p][2], a/b)
    return res

def check_pack(op, trials=500, seed=7):
    """Run the same comparison for a packing operator (Theorem 4 and Proposition 7).

    Each trial draws a balanced instance, with as many rows as columns (1 or 2 of each) and
    costs from 0 to 5. It also draws a prediction with entries from minus 2 to 6 and a random
    scan order. It applies the operator op, or
    the alternating projection maximal_pack if op is None, and asserts that the output is
    feasible. The output is compared, in the sum of the absolute differences, with every
    feasible point whose coordinates are whole numbers from minus 4 to 6, against the
    prediction itself. The packing feasible set is unbounded below, so these points are a
    sample of it. The result is the number of comparisons, the number of them in which the
    distance grew, and the largest ratio of the distance after the repair to the distance
    before it.
    """
    random.seed(seed); viol = tot = 0; worst = 0.0
    for _ in range(trials):
        n = random.randint(1,2)
        C = [[random.randint(0,5) for _ in range(n)] for _ in range(n)]
        yh = ([random.randint(-2,6) for _ in range(n)], [random.randint(-2,6) for _ in range(n)])
        pairs = [(a,b) for a in range(n) for b in range(n)]
        o = pairs[:]; random.shuffle(o)
        u2, v2 = (maximal_pack(C, *yh) if op is None else op(C, yh[0], yh[1], o))
        assert feas_pack(C, u2, v2)
        for u in itertools.product(range(-4,7), repeat=n):
            for v in itertools.product(range(-4,7), repeat=n):
                if not feas_pack(C, list(u), list(v)): continue
                before = l1(yh[0],u)+l1(yh[1],v); after = l1(u2,u)+l1(v2,v); tot += 1
                if after > before: viol += 1
                if before > 0: worst = max(worst, after/before)
    return tot, viol, worst

# Random sweeps of the sorted scan.
def cold_start_check(seed=11):
    """Run a random sweep on the cold start, the setting of Proposition 15.

    For each of six shapes, 700 random weight matrices with entries from 0 to 4 are drawn.
    The sorted scan, with its default tie-break, is started from the all-zero prediction and
    the face distance of its output is computed. The result is a list with one entry per
    shape and the total number of instances. Each entry holds the number of rows, the number
    of columns, the number of trials, the largest face distance found, and twice the smaller
    side minus 1. The part of the printout numbered 4 in square brackets covers the same
    shapes exhaustively.
    """
    random.seed(seed)
    out = []; total = 0
    for (na, nb, T) in [(1,3,700),(1,4,700),(2,2,700),(2,3,700),(2,4,700),(3,3,700)]:
        mx = 0
        for _ in range(T):
            W = [[random.randint(0, 4) for _ in range(nb)] for _ in range(na)]
            u, v = _sorted_repair(W, [0]*na, [0]*nb)
            mx = max(mx, _l0_face_integral(W, u, v))
        out.append((na, nb, T, mx, 2*min(na, nb) - 1)); total += T
    return out, total

def sorted_repair_check(nb=10, trials=3000, seed=606):
    """Run the search for Proposition 10 on random instances with 1 row and 10 columns.

    Each trial draws weights from 1 to 6, picks a random integral point of the optimal face
    and moves k of its coordinates up or down by 1 to 4 to obtain the prediction. A moved
    value is clamped at zero, so the face distance of the prediction is at most k. The
    prediction is then repaired by the scan in order of decreasing weight (key DESC) and by
    the scan in index order (key FWD). For each k from 1 to 5 the result records, for both
    scans, the largest value found of the face distance after the repair minus the face
    distance before it.
    """
    random.seed(seed)
    def face(w):
        # Return the integral points of the optimal face of an instance with one row. A
        # candidate point is fixed by its row value, a whole number from 0 to the largest
        # weight: each of its column values is the weight of the column minus the row value,
        # or zero if that is negative. The points returned are the candidates whose values
        # add up to the smallest total. Integral points are enough here (Appendix A).
        best = min(u + sum(max(0,x-u) for x in w) for u in range(max(w)+1))
        return [(u,[max(0,x-u) for x in w]) for u in range(max(w)+1)
                if u + sum(max(0,x-u) for x in w) == best]
    def l0(u,v,F): return min((u!=fu)+sum(1 for a,b in zip(v,fv) if a!=b) for fu,fv in F)
    out = {}
    for k in (1,2,3,4,5):
        best = {'DESC': -nb - 1, 'FWD': -nb - 1}
        for _ in range(trials):
            w = [random.randint(1,6) for _ in range(nb)]
            F = face(w); fu,fv = random.choice(F); u0,v0 = fu,list(fv)
            for i in random.sample(range(nb+1), k):
                d = random.choice([-1,1]) * random.randint(1,4)
                if i == 0: u0 = max(0, fu + d)
                else:      v0[i-1] = max(0, v0[i-1] + d)
            P = list(range(nb))
            pre = l0(u0,v0,F)
            for name,o in (('DESC',sorted(P,key=lambda t:-w[t])), ('FWD',P)):
                u,v = _repair_1A(w,u0,v0,o)
                best[name] = max(best[name], l0(u,v,F) - pre)
        out[k] = best
    return out

# Matching and the optimal face.
def _mwm_brute(W):
    """Return the largest total weight of a matching in W, found by brute force.

    This is the reference implementation.
    """
    na, nb = len(W), len(W[0]); best = 0
    for r in range(min(na, nb) + 1):
        for A in itertools.combinations(range(na), r):
            for B in itertools.permutations(range(nb), r):
                best = max(best, sum(W[a][b] for a, b in zip(A, B)))
    return best

_INF = float('inf')

def _hungarian_min(cost, n, m):
    """Return the smallest total cost of assigning all n rows to distinct columns.

    The number of rows n must be at most the number of columns m. The method is the Hungarian
    method with potentials, in its e-maxx form. The costs may be integers or floats, and the
    result is the optimal value.
    """
    u = [0]*(n+1); v = [0]*(m+1); p = [0]*(m+1); way = [0]*(m+1)
    for i in range(1, n+1):
        p[0] = i; j0 = 0
        minv = [_INF]*(m+1); used = [False]*(m+1)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = _INF; j1 = -1
            row = cost[i0-1]; ui = u[i0]
            for j in range(1, m+1):
                if not used[j]:
                    cur = row[j-1] - ui - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(m+1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else:       minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    return -v[0]

def _mwm(W):
    """Return the largest total weight of a matching in a matrix W with no negative entry.

    Vertices may stay unmatched. By LP duality this value equals the optimal value of the
    covering LP, which minimises the sum of the dual values subject to two conditions: the
    two dual values of every pair of a row and a column add up to at least the weight of the
    pair, and no dual value is negative. This is the theorem of Egervary and Konig. It relies
    on the incidence matrix of a bipartite graph being totally unimodular.
    """
    na = len(W)
    if na == 0: return 0
    nb = len(W[0])
    if nb == 0: return 0
    if na <= nb:
        return -_hungarian_min([[-W[a][b] for b in range(nb)] for a in range(na)], na, nb)
    return -_hungarian_min([[-W[a][b] for a in range(na)] for b in range(nb)], nb, na)

# This comment explains how the exact distance to the whole optimal face is computed. The
# distance is the smallest number of coordinates in which a point differs from a point of
# the face, and the face is taken in full, fractional points included.
#
# A dual is feasible when none of its values is negative and the two dual values of every
# pair of a row and a column add up to at least the weight of the pair. The optimal face is
# the set of feasible duals whose values add up to the optimal value, called OPT in the code.
#
# Take a target point with whole-number coordinates and a set of coordinates. Among the
# feasible duals that take the values of the target on that set, look at the smallest sum of
# values, and call it the constrained minimum of the set. Some point of the face agrees with
# the target on the set exactly when the constrained minimum equals OPT. The function
# _face_min_sum computes the constrained minimum in closed form. Substituting the fixed
# values leaves two kinds of conditions on the free coordinates. Each free coordinate gets a
# lower bound: the largest value, over its fixed neighbours, of the weight of the pair minus
# the fixed value of the neighbour, or zero if none of these is positive. The constraints
# remain on the pairs whose two endpoints are both free. Writing each free value as its lower
# bound plus an amount that is not negative gives the constrained minimum as three terms
# added together: the sum of the fixed values, the sum of the lower bounds of the free
# coordinates, and the largest total weight of a matching between the free rows and the free
# columns. In that matching the weight of a pair is its original weight minus its two lower
# bounds, replaced by zero when negative. The third term again comes from LP duality, on the
# bipartite graph of the pairs whose two endpoints are both free. When the fixed values are
# infeasible by themselves (one of them is negative, or two of them add up to less than the
# weight of their pair), _face_min_sum returns infinity.
#
# The constrained minimum never decreases when the set grows. So if some face point agrees
# with the target on a set, the same holds for every subset of that set. A depth-first search
# that decides for one coordinate after another whether to agree with the target or not, and
# that follows the agreeing branch only when the constrained minimum is still at most OPT,
# is therefore exact. The result is a minimum over the continuous face, and no cap on the
# coordinate values is involved.
#
# Enumerating only the integral points of the face is exact here too, for the following
# reason. The constraint matrix of the set of feasible duals is the incidence matrix, edges
# against vertices, of a bipartite graph, which is totally unimodular. The weights are whole
# numbers. Hence the set of feasible duals is an integral polyhedron. Fixing some of the
# coordinates to whole-number values deletes their columns from the matrix and moves whole
# numbers to the right-hand side. The system that remains is again totally unimodular with
# whole-number data, so the polyhedron of the feasible duals that take the fixed values is
# integral too. Therefore, whenever this polyhedron contains a point whose values add up to
# OPT, it contains such a point with whole-number coordinates, and that point lies on the
# face and agrees with the target on the fixed coordinates. So, for a target with
# whole-number coordinates, the smallest number of differing coordinates over the continuous
# face equals the smallest number over the integral points of the face, and enumerating the
# integral points, as _l0_face_integral does, is exact. The function face_cross_validate
# below checks this again numerically. The argument uses that the system is bipartite: for a
# covering system that is not bipartite the incidence matrix is not totally unimodular and
# the argument fails.

def _face_min_sum(W, na, nb, fixA, fixB):
    tot = 0
    for a in range(na):
        za = fixA[a]
        if za is not None:
            if za < 0: return _INF
            tot += za
    for b in range(nb):
        zb = fixB[b]
        if zb is not None:
            if zb < 0: return _INF
            tot += zb
    LA = [0]*na; LB = [0]*nb
    for a in range(na):
        fa = fixA[a]; row = W[a]
        for b in range(nb):
            w = row[b]
            if w == 0: continue
            fb = fixB[b]
            if fa is not None:
                if fb is not None:
                    if fa + fb < w: return _INF
                elif w - fa > LB[b]: LB[b] = w - fa
            elif fb is not None:
                if w - fb > LA[a]: LA[a] = w - fb
    freeA = [a for a in range(na) if fixA[a] is None]
    freeB = [b for b in range(nb) if fixB[b] is None]
    for a in freeA: tot += LA[a]
    for b in freeB: tot += LB[b]
    if freeA and freeB:
        R = [[max(0, W[a][b] - LA[a] - LB[b]) for b in freeB] for a in freeA]
        if any(any(r) for r in R): tot += _mwm(R)
    return tot

def _l0_face(W, u, v, OPT=None):
    """Return the exact face distance of the point with row values u and column values v.

    This is the smallest number of coordinates in which the point differs from a point of
    the whole optimal face, fractional points included. The search finds the largest set of
    coordinates on which some face point agrees with the given point, and the distance is
    the number of coordinates outside that set. If OPT is not given, the optimal value is
    computed by _mwm.
    """
    na, nb = len(W), len(W[0]); n = na + nb
    if OPT is None: OPT = _mwm(W)
    z = list(u) + list(v)
    fixA = [None]*na; fixB = [None]*nb
    best = [0]
    def rec(i, cnt):
        if cnt + (n - i) <= best[0]: return
        if i == n:
            if cnt > best[0]: best[0] = cnt
            return
        if i < na: fixA[i] = z[i]
        else:      fixB[i-na] = z[i]
        if _face_min_sum(W, na, nb, fixA, fixB) <= OPT:
            rec(i+1, cnt+1)
        if i < na: fixA[i] = None
        else:      fixB[i-na] = None
        rec(i+1, cnt)
    rec(0, 0)
    return n - best[0]

def _l0_face_integral(W, u, v):
    """Return the exact face distance measured to the integral points of the optimal face.

    This is the smallest number of coordinates in which the point with row values u and
    column values v differs from an integral point of the face, found by a pruned
    enumeration of those points. No coordinate of an optimal dual exceeds the largest entry
    of W, so capping the values at the largest entry plus 1 loses nothing. By the argument
    in the long comment above _face_min_sum, the result equals that of _l0_face, and
    face_cross_validate checks it.
    """
    na, nb = len(W), len(W[0]); opt = _mwm(W)
    cap = max(max(r) for r in W) + 1
    best = [na + nb]
    def vrec(uu, b, ssofar, mism):
        if mism >= best[0]: return
        if b == nb:
            if ssofar == opt: best[0] = mism
            return
        lb = max([0] + [W[a][b] - uu[a] for a in range(na)])
        rem = sum(max([0] + [W[a][bb] - uu[a] for a in range(na)]) for bb in range(b + 1, nb))
        for vb in range(lb, cap + 1):
            if ssofar + vb + rem > opt: break
            vrec(uu, b + 1, ssofar + vb, mism + (vb != v[b]))
    def urec(a, uu, ssofar, mism):
        if mism >= best[0] or ssofar > opt: return
        if a == na: vrec(tuple(uu), 0, ssofar, mism); return
        for ua in range(cap + 1):
            urec(a + 1, uu + [ua], ssofar + ua, mism + (ua != u[a]))
    urec(0, [], 0, 0)
    return best[0]

# The one-sided walk.
def _walk_repair(W, uh, vh):
    """Run the one-sided walk repair, our covering transplant of Algorithm 1 of Dinitz et al.

    The walk visits one vertex at a time. It raises that vertex alone by the largest residual
    among its remaining violated constraints, removes those constraints and steps to the
    neighbour that had the largest residual. This version needs only the standard library
    and follows the same conventions as experiments.py. The residuals are fixed at the
    values they have at the clamped prediction. A walk starts at the lowest-index vertex
    that still has a violated constraint, with the rows numbered before the columns. Ties
    for the largest residual go to the lowest index. The result is one list of repaired dual
    values, rows first.
    """
    na, nb = len(W), len(W[0])
    u = [max(0, t) for t in uh]; v = [max(0, t) for t in vh]
    y = u + v
    adj = {}
    for a in range(na):
        for b in range(nb):
            res = W[a][b] - y[a] - y[na + b]
            if res > 0:
                adj.setdefault(a, {})[na + b] = res
                adj.setdefault(na + b, {})[a] = res
    delta = [0] * (na + nb)
    while any(adj.get(i) for i in range(na + nb)):
        i = min(k for k in range(na + nb) if adj.get(k))
        while adj.get(i):
            j = max(adj[i], key=lambda t: (adj[i][t], -t))
            delta[i] = max(delta[i], max(adj[i].values()))
            for k in list(adj[i]):
                adj[k].pop(i, None)
            adj[i] = {}
            i = j
    return [y[t] + delta[t] for t in range(na + nb)]


def walk_vs_sorted(trials=400, seed=5):
    """Compare the walk with the sorted scan on a witness and on a random sweep.

    The walk is the lowest of the repairs in every cell of Table 5, which counts the
    coordinates in which the repaired dual differs from the reference optimum, and of
    Table 6, which counts the dual adjustments of the solver. Proposition 21 bounds its
    number of coordinates that differ by twice that of the prediction. Yet under a fixed
    tie-break it can end strictly worse than the sorted scan. The first part of the result
    comes from a witness with 3 rows and 4 columns: the face distance of the prediction, of
    the output of the sorted scan and of the output of the walk, which are asserted to be 5,
    4 and 6. The second part says how often the walk loses on a random sweep: the number of
    instances on which its output is strictly farther from the face than that of the sorted
    scan, and the number of instances tried.
    """
    W0 = [[2, 0, 3, 0], [4, 0, 0, 5], [0, 5, 4, 2]]
    uh0, vh0 = [2, 1, 0], [4, 6, 0, 1]
    pre0 = _l0_face_integral(W0, uh0, vh0)
    su, sv = _sorted_repair(W0, uh0, vh0)
    yw = _walk_repair(W0, uh0, vh0)
    s0 = _l0_face_integral(W0, su, sv)
    w0 = _l0_face_integral(W0, yw[:3], yw[3:])
    assert (pre0, s0, w0) == (5, 4, 6), (pre0, s0, w0)
    rng = random.Random(seed); worse = tot = 0
    for _ in range(trials):
        na, nb = rng.choice([(2, 3), (3, 3), (3, 4), (2, 4)])
        W = [[rng.randint(0, 6) for _ in range(nb)] for _ in range(na)]
        uh = [rng.randint(0, 6) for _ in range(na)]
        vh = [rng.randint(0, 6) for _ in range(nb)]
        su, sv = _sorted_repair(W, uh, vh)
        yw = _walk_repair(W, uh, vh)
        assert all(yw[a] + yw[na + b] >= W[a][b] for a in range(na) for b in range(nb))
        tot += 1
        worse += _l0_face_integral(W, yw[:na], yw[na:]) > _l0_face_integral(W, su, sv)
    return (pre0, s0, w0), (worse, tot)

# Weakly-descending orders. Such an order scans the pairs by decreasing weight, and pairs of
# equal weight may come in any order.
def _grouped_orders(groups):
    """Yield every order that takes the groups by decreasing key, in any order inside a group.

    The dictionary groups maps each key to the list of its items.
    """
    keys = sorted(groups, reverse=True)
    for parts in itertools.product(*[itertools.permutations(groups[k]) for k in keys]):
        yield [x for part in parts for x in part]

def _weight_groups(W, skip_zero=False):
    """Group the pairs of a row and a column of W by weight, each group in row-major order.

    If skip_zero is true, the zero-weight pairs are left out.
    """
    groups = {}
    for a in range(len(W)):
        for b in range(len(W[0])):
            if W[a][b] or not skip_zero:
                groups.setdefault(W[a][b], []).append((a, b))
    return groups

def _all_desc_orders(W):
    """Yield every weakly-descending order of the pairs of W."""
    return _grouped_orders(_weight_groups(W))

def _pos_desc_orders(W):
    """Yield every weakly-descending order, up to the order of the zero-weight pairs.

    The zero-weight pairs come last, in row-major order. A zero-weight pair never fires:
    after the clamp its residual is minus the sum of its two dual values, which is at most
    zero. So this list is exhaustive.
    """
    zeros = [(a, b) for a in range(len(W)) for b in range(len(W[0])) if W[a][b] == 0]
    for o in _grouped_orders(_weight_groups(W, skip_zero=True)):
        yield o + zeros

def _desc_orders_1A(w):
    """Yield every weakly-descending order of the columns of an instance with a single row.

    The list w holds the weight of each column.
    """
    return _grouped_orders({x: [k for k in range(len(w)) if w[k] == x] for x in set(w)})

def _desc_order_counts(W):
    """Count the weakly-descending orders of the pairs of W.

    The result holds three numbers: the number of weakly-descending orders, the number of
    different ways in which they arrange the positive-weight pairs, and the number of
    zero-weight pairs.
    """
    cnt = {}
    for a in range(len(W)):
        for b in range(len(W[0])):
            cnt[W[a][b]] = cnt.get(W[a][b], 0) + 1
    tot = 1
    for m in cnt.values(): tot *= math.factorial(m)
    pos = 1
    for w, m in cnt.items():
        if w > 0: pos *= math.factorial(m)
    return tot, pos, cnt.get(0, 0)

def _big(n):
    """Render an exact count that may be astronomically large in a readable form.

    A count below one billion is written out in full. A count of one billion or more is cut
    down to its first five digits, followed by the power of ten.
    """
    if n < 10**9: return str(n)
    s = str(n)
    return "%s.%se+%d" % (s[0], s[1:5], len(s) - 1)

# The witnesses of Appendix D. The first is a weight matrix. Each of the other four is a
# triple: the weight matrix, the row values of the prediction and its column values.
COLD_3X4 = [[4,6,5,3],[5,4,3,0],[4,4,0,1]]
WARM_2X3 = ([[1,2,5],[4,5,6]], [0,0], [0,0,3])
WARM_2X6 = ([[6,5,1,4,3,6],[6,3,2,2,1,1]], [0,1], [3,3,0,0,0,3])
SHARP_K2 = ([[10,9,5,0,0,10],[0,8,0,2,3,0]], [0,1], [10,5,0,0,5,10])
SHARP_K3 = ([[13,12,11,7,0,0,0,0,13],[0,11,0,0,2,0,3,0,0],[0,0,10,0,0,4,0,5,0]],
            [0,1,3], [13,8,5,0,0,0,5,7,13])

def refutation_check():
    """Check the witnesses of Appendix D and return a label and a message for each.

    Part (a) is the cold start. On an instance with 3 rows and 4 columns the sorted scan
    started from the all-zero prediction ends at face distance 6, which is twice the smaller
    side, under every weakly-descending order. So twice the smaller side minus 1 is not an
    envelope, that is, not an upper bound on the face distance after a cold start.

    Part (b) is the warm start. The bound of Proposition 10, an increase of at most 1 in the
    face distance, fails once the smaller side is at least 2. The increase, which the
    printout calls the excess, is 2 on an instance with 2 rows and 3 columns, and it is 3,
    which is twice the smaller side minus 1, on an instance with 2 rows and 6 columns. The
    last two entries reach the bound of Theorem 11.
    """
    out = []
    # Part (a), the cold start. All 192 weakly-descending orders give one and the same output,
    # at face distance 6.
    orders = list(_all_desc_orders(COLD_3X4))
    assert len(orders) == 192
    ds = {tuple(sum(_sorted_repair(COLD_3X4, [0]*3, [0]*4, o), [])) for o in orders}
    assert len(ds) == 1, ds
    d1 = {_l0_face_integral(COLD_3X4, y[:3], y[3:]) for y in ds}
    assert d1 == {6}, d1
    out.append(("cold 3x4 W1 (all 192 orders)", "l0-to-face = 6 > 5 = 2min-1"))
    # A variant of part (a). Weights of at most 3 suffice under the canonical tie-break, which
    # takes ties in row-major (lexicographic) order.
    Wc = [[3,3,2,2],[3,3,2,2],[3,3,1,1]]
    u, v = _sorted_repair(Wc, [0]*3, [0]*4)
    assert _l0_face_integral(Wc, u, v) == 6
    out.append(("cold 3x4 Wc (weights <= 3, lex)", "l0-to-face = 6 > 5"))
    # Part (b), the warm start. At the minimal shape, 2 rows and 3 columns, the face distance
    # increases by 2 under both weakly-descending orders.
    W2, uh, vh = WARM_2X3
    pre = _l0_face_integral(W2, uh, vh)
    exs = set()
    for o in _all_desc_orders(W2):
        u, v = _sorted_repair(W2, uh, vh, o)
        exs.add(_l0_face_integral(W2, u, v) - pre)
    assert pre == 2 and exs == {2}, (pre, exs)
    out.append(("warm 2x3 (both orders)", "pre 2 -> post 4, excess 2 > 1"))
    # A variant of part (b). With 2 rows and 6 columns the increase reaches 3, which is twice
    # the smaller side minus 1, under the canonical order.
    W3, uh3, vh3 = WARM_2X6
    pre3 = _l0_face_integral(W3, uh3, vh3)
    u, v = _sorted_repair(W3, uh3, vh3)
    post3 = _l0_face_integral(W3, u, v)
    assert (pre3, post3) == (4, 7), (pre3, post3)
    out.append(("warm 2x6 (canonical)", "pre 4 -> post 7, excess 3 = 2min-1"))
    # The member with k equal to 2 of the witnesses for the sharp bound. The increase is 3
    # under all 240 weakly-descending orders.
    Wf2, uf2, vf2 = SHARP_K2
    orders = list(_all_desc_orders(Wf2))
    assert len(orders) == 240
    pref2 = _l0_face_integral(Wf2, uf2, vf2)
    outs = {tuple(sum(_sorted_repair(Wf2, uf2, vf2, o), [])) for o in orders}
    exf2 = {_l0_face_integral(Wf2, list(y[:2]), list(y[2:])) - pref2 for y in outs}
    assert pref2 == 5 and exf2 == {3}, (pref2, exf2)
    out.append(("sharp family k=2 (all 240 orders)", "pre 5 -> post 8, excess 3 = 2min-1"))
    # The member with k equal to 3. The increase is 5 under the canonical order.
    Wf3, uf3, vf3 = SHARP_K3
    pref3 = _l0_face_integral(Wf3, uf3, vf3)
    u, v = _sorted_repair(Wf3, uf3, vf3)
    assert (pref3, _l0_face_integral(Wf3, u, v)) == (7, 12), pref3
    out.append(("sharp family k=3 (canonical)", "pre 7 -> post 12, excess 5 = 2min-1"))
    return out

def envelope_sweep(seed=29):
    """Run a random sweep of two envelopes for the sorted scan.

    The first envelope sets the face distance after a cold start against twice the smaller
    side. The second sets the increase of the face distance after a warm start against twice
    the smaller side minus 1 (Theorem 11). The first is not a theorem: Proposition 26 breaks
    it on an instance with 4 rows and 5 columns. On the shapes of this sweep both hold for
    every instance drawn, which the function asserts. The result holds, for the cold start
    and then for the warm start, the number of instances and the largest amount by which the
    measured value exceeds its envelope. That amount is zero or negative when the envelope
    holds.
    """
    random.seed(seed)
    worst_cold = worst_warm = -10**9
    n_cold = n_warm = 0
    for (na, nb, T) in [(3,4,150),(2,5,150),(3,5,100),(4,4,100)]:
        m = min(na, nb)
        for _ in range(T):
            W = [[random.randint(0, 4) for _ in range(nb)] for _ in range(na)]
            u, v = _sorted_repair(W, [0]*na, [0]*nb)
            d = _l0_face_integral(W, u, v)
            worst_cold = max(worst_cold, d - 2*m); n_cold += 1
            assert d <= 2*m, (W, d)
            cap = max(max(r) for r in W) + 1
            uh = [random.randint(0, cap) for _ in range(na)]
            vh = [random.randint(0, cap) for _ in range(nb)]
            pre = _l0_face_integral(W, uh, vh)
            uq, vq = _sorted_repair(W, uh, vh)
            post = _l0_face_integral(W, uq, vq)
            worst_warm = max(worst_warm, post - pre - (2*m - 1)); n_warm += 1
            assert post - pre <= 2*m - 1, (W, uh, vh, pre, post)
    return n_cold, worst_cold, n_warm, worst_warm

# Face-exact checks, under every tie-break order.

def face_cross_validate(trials=3000, seed=1):
    """Cross-check the two face routines against each other, and _mwm against brute force.

    On random instances with 1 to 3 rows, 1 to 4 columns and weights from 0 to 4, the routine
    for the continuous face, _l0_face, is compared with the enumeration of the integral
    face, _l0_face_integral. Three points are tried on each instance: a random prediction,
    its sorted repair and the sorted repair of the all-zero prediction. On each instance
    _mwm is also compared with the brute-force routine _mwm_brute. This confirms numerically
    the total-unimodularity argument in the long comment above _face_min_sum. The function
    asserts that there is no mismatch. The result is the number of face comparisons and the
    number of mismatches among them, then the same two numbers for the matching routine.
    """
    random.seed(seed)
    n_face = mism_face = 0
    n_mwm = mism_mwm = 0
    for _ in range(trials):
        na = random.randint(1, 3); nb = random.randint(1, 4)
        W = [[random.randint(0, 4) for _ in range(nb)] for _ in range(na)]
        n_mwm += 1
        if _mwm(W) != _mwm_brute(W): mism_mwm += 1
        cap = max(max(r) for r in W) + 1
        uh = [random.randint(0, cap) for _ in range(na)]
        vh = [random.randint(0, cap) for _ in range(nb)]
        pts = [(uh, vh), _sorted_repair(W, uh, vh), _sorted_repair(W, [0]*na, [0]*nb)]
        for (u, v) in pts:
            n_face += 1
            if _l0_face(W, list(u), list(v)) != _l0_face_integral(W, list(u), list(v)):
                mism_face += 1
    assert mism_face == 0 and mism_mwm == 0, (mism_face, mism_mwm)
    return n_face, mism_face, n_mwm, mism_mwm

# The sharpness family, which the printout calls F_k.
def family_Fk(k, t=2):
    """Build the member with k rows of the family that attains the bound (Proposition 35).

    The instance has k rows and 3 times k plus t columns. The last t columns are inert
    padding: all their weights are zero. The paper uses t equal to 2. This puts the family
    strictly inside the informative regime, where the face distance of the prediction is at
    most the difference in size between the two sides. It also makes the additive branch of
    the minimum in Theorem 11 the binding one, that is, the smaller of the two branches. The
    result is the weight matrix, the row values and the column values of the prediction, and
    a closed-form value that sharp_family_check asserts to be the optimal value.
    """
    assert k >= 1 and t >= 0
    M = 4*k + 2; s = 2*k + 1
    na, nb = k, 3*k + t
    W = [[0]*nb for _ in range(na)]
    X1, X2, Z0 = 0, 1, k + 1
    L = {i: 1 + i        for i in range(1, k)}
    H = {i: k + 2*i      for i in range(1, k)}
    Z = {i: k + 2*i + 1  for i in range(1, k)}
    p = {i: M - 2*i - 1  for i in range(1, k)}
    q = {i: M - 2*i - 2  for i in range(1, k)}
    h = {i: 2*i + 1      for i in range(1, k)}
    l = {i: 2*i          for i in range(1, k)}
    W[0][X1] = M; W[0][X2] = M - 1; W[0][Z0] = s
    for i in range(1, k):
        W[0][L[i]] = p[i]; W[i][L[i]] = q[i]
        W[i][H[i]] = h[i]; W[i][Z[i]] = l[i]
    uh = [0]*na; vh = [0]*nb
    vh[X1] = M; vh[X2] = M - 1
    for i in range(1, k):
        uh[i] = l[i] - 1; vh[L[i]] = q[i] - h[i]; vh[H[i]] = h[i] - l[i] + 1
    return W, uh, vh, M + sum(q.values())

def sharp_family_check(kmax=12, kmax_int=5):
    """Check the sharpness family for every k from 1 to kmax and return one row per k.

    The face distances before and after the sorted repair are computed by exact minimisation
    over the fractional face and, for k up to kmax_int, also by enumeration of the integral
    face. Every weakly-descending order is run, through _pos_desc_orders: every arrangement
    of the positive-weight pairs, with the zero-weight pairs placed last. The function
    asserts that no zero-weight pair fires and that all these orders give one single output.
    It also asserts the optimal value, the two face distances (2 times k plus 1 before the
    repair and 4 times k after it), an increase of twice the smaller side minus 1, and the
    agreement of the two face routines where both run. Each row is a dictionary of the
    values measured for one k.
    """
    rows = []
    for k in range(1, kmax + 1):
        W, uh, vh, OPTclosed = family_Fk(k)
        na, nb = len(W), len(W[0])
        OPT = _mwm(W)
        assert OPT == OPTclosed == 3*k*k + k + 2, (k, OPT, OPTclosed)
        pre = _l0_face(W, uh, vh, OPT)
        u, v = _sorted_repair(W, uh, vh)
        post = _l0_face(W, u, v, OPT)
        assert pre == 2*k + 1 and post == 4*k, (k, pre, post)
        assert post - pre == 2*min(na, nb) - 1
        # Record whether the instance lies in the informative regime, and which branch of the
        # minimum in Theorem 11 binds. The regime asks that the face distance of the prediction
        # is at most the difference in size between the two sides. The additive branch binds
        # when the face distance of the prediction plus twice the smaller side minus 1 is less
        # than the trivial branch, the number of rows plus the number of columns.
        informative = pre <= abs(nb - na)
        additive_binding = pre + 2*min(na, nb) - 1 < na + nb
        # Tie-breaks: every order must give the same single output.
        tot, pos, nz = _desc_order_counts(W)
        outs = set()
        for o in _pos_desc_orders(W):
            fired = []
            outs.add(tuple(sum(_sorted_repair(W, uh, vh, o, fired), [])))
            assert all(W[a][b] > 0 for a, b in fired)
        assert len(outs) == 1, (k, len(outs))
        posts = {_l0_face(W, list(y[:na]), list(y[na:]), OPT) for y in outs}
        assert posts == {post}, (k, posts)
        integral = None
        if k <= kmax_int:
            ip = _l0_face_integral(W, uh, vh)
            iq = _l0_face_integral(W, u, v)
            assert (ip, iq) == (pre, post), (k, ip, iq)
            integral = (ip, iq)
        rows.append(dict(k=k, na=na, nb=nb, n=na+nb, OPT=OPT, pre=pre, post=post,
                         excess=post - pre, informative=informative,
                         additive=additive_binding, orders_total=tot,
                         orders_pos=pos, zero_pairs=nz, n_outputs=len(outs),
                         integral=integral))
    return rows

def hardcoded_member_regime():
    """Compare the two hard-coded witnesses with the sharpness family on two regime tests.

    The two hard-coded witnesses reach the constant (an increase of twice the smaller side
    minus 1) only because the trivial branch of the minimum in Theorem 11, the number of
    rows plus the number of columns, is tight there. They also sit outside the informative
    regime of the paper, in which the face distance of the prediction is at most the
    difference in size between the two sides. The sharpness family, with its two padding
    columns, has neither defect.

    The result has one row for each of the two witnesses and for the family members with k
    equal to 2 and 3. A row holds a name, the number of rows, the number of columns, the
    face distances before and after the sorted repair, the difference and the sum of the two
    sides, whether the instance is in the informative regime, and whether the additive
    branch binds.
    """
    jobs = [("Wf2 (k=2)",) + SHARP_K2, ("Wf3 (k=3)",) + SHARP_K3]
    jobs += [("F_%d" % k,) + family_Fk(k)[:3] for k in (2, 3)]
    out = []
    for name, W, uh, vh in jobs:
        na, nb = len(W), len(W[0])
        OPT = _mwm(W)
        pre = _l0_face(W, uh, vh, OPT)
        u, v = _sorted_repair(W, uh, vh)
        post = _l0_face(W, u, v, OPT)
        out.append((name, na, nb, pre, post, abs(nb - na), na + nb,
                    pre <= abs(nb - na), pre + 2*min(na, nb) - 1 < na + nb))
    return out

def tiebreak_check():
    """Run every weakly-descending order on each witness whose tie-breaks the paper discusses.

    Zero-weight pairs come last and never fire, which is checked on every run of the scan,
    so enumerating the permutations of the positive-weight pairs is exhaustive. The function
    counts the distinct outputs and does not assert that there is only one. The warm witness
    with 2 rows and 6 columns has more than one, and Appendix G of the paper names it as the
    exception to order-robustness.

    The result has one row per witness: its name, its shape, the number of weakly-descending
    orders, the number of different ways in which they arrange the positive-weight pairs,
    the number of zero-weight pairs, the number of distinct outputs, the face distance of
    the prediction, and a list that says how many of the enumerated orders end at each face
    distance.
    """
    jobs = [("cold 3x4 W1", COLD_3X4, [0]*3, [0]*4),
            ("warm 2x3",) + WARM_2X3,
            ("warm 2x6",) + WARM_2X6,
            ("sharp Wf2 k=2",) + SHARP_K2,
            ("sharp Wf3 k=3",) + SHARP_K3]
    jobs += [("sharp F_%d" % k,) + family_Fk(k)[:3] for k in (2, 3, 4)]
    out = []
    for name, W, uh, vh in jobs:
        na, nb = len(W), len(W[0]); OPT = _mwm(W)
        tot, pos, nz = _desc_order_counts(W)
        pre = _l0_face(W, uh, vh, OPT)
        outs = {}
        for o in _pos_desc_orders(W):
            fired = []
            key = tuple(sum(_sorted_repair(W, uh, vh, o, fired), []))
            assert all(W[a][b] > 0 for a, b in fired)
            outs[key] = outs.get(key, 0) + 1
        # Count how many of the orders of the positive-weight pairs end at each face distance
        # after the repair.
        hist = {}
        for y, c in outs.items():
            d = _l0_face(W, list(y[:na]), list(y[na:]), OPT)
            hist[d] = hist.get(d, 0) + c
        out.append((name, "%dx%d" % (na, nb), tot, pos, nz, len(outs),
                    pre, sorted(hist.items())))
    return out

# Proposition 10, on instances with a single row.
def _face_1A(w, u, v):
    """Return the exact face distance of a point when the instance has a single row.

    This is the smallest number of coordinates in which the point, with row value u and
    column values v, differs from a point of the continuous optimal face. The list w holds
    the column weights. In the code W is the largest weight and W2 is the second largest
    weight, counted with multiplicity, so W2 equals W when the largest weight occurs more
    than once. W2 is zero when there is only one column. The face is a segment of points:
    the row value runs from W2 to W, a column of largest weight holds W minus the row value,
    and every other column holds zero. The function sortedwarm_all_orders cross-checks this
    closed form against _l0_face.
    """
    d = len(w); W = max(w)
    ks = max(range(d), key=lambda k: w[k])
    W2 = 0 if d == 1 else sorted(w, reverse=True)[1]
    N = sum(1 for k in range(d) if k != ks and v[k] != 0)
    a_ok = (W2 <= u <= W)
    b_ok = (0 <= v[ks] <= W - W2)
    if a_ok and v[ks] == W - u: delta = 0
    elif a_ok or b_ok:          delta = 1
    else:                       delta = 2
    return N + delta

def _l1_face_1A(w, u, v):
    """Return the exact smallest sum of absolute differences to the same continuous face.

    As a function of the row value of the face point, this distance is convex and piecewise
    linear. It is therefore evaluated only at the two ends of the segment and at its two
    breakpoints, each moved to the nearer end if it falls outside the segment.
    """
    d = len(w); W = max(w)
    ks = max(range(d), key=lambda k: w[k])
    W2 = 0 if d == 1 else sorted(w, reverse=True)[1]
    base = sum(abs(v[k]) for k in range(d) if k != ks)
    cands = {W2, W, min(W, max(W2, u)), min(W, max(W2, W - v[ks]))}
    return base + min(abs(u - t) + abs(v[ks] - (W - t)) for t in cands)

def _linf_faces_1A(w, u, v):
    """Return two exact distances to the continuous face of an instance with a single row.

    The first is the smallest value of the largest absolute difference. The second is the
    smallest value of the largest increase plus the largest decrease, which the paper calls
    the two-sided l-infinity distance. The face is the segment described in _face_1A: the
    row value runs from W2 to W, a column of largest weight holds W minus the row value, and
    every other column holds zero. Both distances are convex and piecewise linear in the row
    value of the face point. They are built from lines of slope plus 1 or minus 1 with
    whole-number intercepts and from constants, so every breakpoint is a multiple of one
    half. Evaluating at every multiple of one half from W2 to W, in exact rational
    arithmetic, therefore finds the true minimum. Enumerating the integral points of the
    face is not valid for these two distances, because the total-unimodularity argument of
    Appendix A covers only the number of coordinates that differ and the sum of the absolute
    differences. This is why the routine minimises over the full face, fractional points
    included.
    """
    d = len(w); W = max(w)
    ks = max(range(d), key=lambda k: w[k])
    W2 = 0 if d == 1 else sorted(w, reverse=True)[1]
    rest = [v[k] for k in range(d) if k != ks]
    best_inf = best_pm = None
    t = Fraction(W2)
    while t <= W:
        diffs = [Fraction(u) - t, Fraction(v[ks]) - (W - t)] + [Fraction(x) for x in rest]
        li = max(abs(x) for x in diffs)
        pm = max([Fraction(0)] + diffs) + max([Fraction(0)] + [-x for x in diffs])
        best_inf = li if best_inf is None else min(best_inf, li)
        best_pm = pm if best_pm is None else min(best_pm, pm)
        t += Fraction(1, 2)
    return best_inf, best_pm

def sortedwarm_all_orders(dmax=5, Wmax=4, Pmax=5, xval=4000, seed=17):
    """Check Proposition 10, on instances with one row, under every weakly-descending order.

    The sweep is exhaustive over the number of columns from 1 to dmax, all weights from 0 to
    Wmax, all predictions with entries from 0 to Pmax, and every weakly-descending tie-break
    order. A prediction with negative entries is clamped to one that is already in the
    sweep. A permutation of the columns moves the weight and the predicted value of each
    column together. Applied to the scan order as well, it rearranges the output of the
    repair in the same way and leaves the face distance unchanged. The sweep
    therefore enumerates one representative per orbit, a multiset of pairs of a column
    weight and a column value, and reports the exact numbers of instances and of pairs of an
    instance and an order that the representatives stand for. Before the sweep, the closed
    form _face_1A is cross-checked against _l0_face on xval random cases. The function
    asserts that the cross-check finds no mismatch, that the face distance never increases
    by more than 1, and that the largest increase is exactly 1. The result is a dictionary
    with the parameters of the sweep and its counts. Violations and attainments of the
    increase of 1 are counted over pairs of an instance and an order.
    """
    random.seed(seed)
    xn = xbad = 0
    for _ in range(xval):
        d = random.randint(1, dmax)
        w = [random.randint(0, Wmax) for _ in range(d)]
        u = random.randint(0, Pmax); v = [random.randint(0, Pmax) for _ in range(d)]
        xn += 1
        if _face_1A(w, u, v) != _l0_face([w], [u], v): xbad += 1
    n_inst = n_pairs = 0
    viol = attained = 0; worst = -10**9
    for d in range(1, dmax + 1):
        opts = [(ww, vv) for ww in range(Wmax + 1) for vv in range(Pmax + 1)]
        fd = math.factorial(d)
        for u in range(Pmax + 1):
            for combo in itertools.combinations_with_replacement(opts, d):
                w = [c[0] for c in combo]; v = [c[1] for c in combo]
                mult = {}
                for c in combo: mult[c] = mult.get(c, 0) + 1
                osize = fd
                for m in mult.values(): osize //= math.factorial(m)
                pre = _face_1A(w, u, v)
                for o in _desc_orders_1A(w):
                    u2, v2 = _repair_1A(w, u, v, o)
                    ex = _face_1A(w, u2, v2) - pre
                    if ex > worst: worst = ex
                    if ex > 1: viol += osize
                    if ex == 1: attained += osize
                    n_pairs += osize
                n_inst += osize
    assert xbad == 0 and viol == 0 and worst == 1, (xbad, viol, worst)
    return dict(xval_n=xn, xval_bad=xbad, dmax=dmax, Wmax=Wmax, Pmax=Pmax,
                instances=n_inst, pairs=n_pairs, violations=viol,
                worst=worst, attained=attained)

# Proposition 15: the exhaustive check of the cold-start envelope.
def _cold_outputs(W):
    """Return every distinct output of the sorted repair started from the all-zero prediction.

    The outputs are those of every weakly-descending order. In a cold start a pair can fire
    only while both of its endpoints are still at zero, and the firing sets both of them to
    the weight of the pair (Proposition 15). So, within a group of pairs of equal weight,
    the pairs that fire are exactly a maximal matching on the vertices that are still free.
    Zero-weight pairs never fire. The result is a set of outputs. Each output is written as
    the tuple of its row values followed by the tuple of its column values.
    """
    na, nb = len(W), len(W[0])
    groups = {}
    for a in range(na):
        for b in range(nb):
            if W[a][b] > 0: groups.setdefault(W[a][b], []).append((a, b))
    keys = sorted(groups, reverse=True)
    out = set()
    def maximal_matchings(edges, ra, rb):
        """Return every maximal matching of edges that avoids the vertices already used.

        The sets ra and rb hold the rows and the columns that are already used. Each path of
        decisions gives a different set of edges, so there are no repeats.
        """
        res = []
        def mrec(i, ua, ub, cur):
            if i == len(edges):
                if all(e[0] in ua or e[1] in ub for e in edges):
                    res.append(cur)
                return
            a, b = edges[i]
            if a in ua or b in ub: mrec(i+1, ua, ub, cur); return
            mrec(i+1, ua | {a}, ub | {b}, cur + [(a, b)])
            mrec(i+1, ua, ub, cur)
        mrec(0, set(ra), set(rb), [])
        return res
    def rec(gi, ra, rb, u, v):
        if gi == len(keys):
            out.add((tuple(u), tuple(v))); return
        w = keys[gi]
        for m in maximal_matchings(groups[w], ra, rb):
            u2 = list(u); v2 = list(v)
            for (a, b) in m: u2[a] = w; v2[b] = w
            rec(gi + 1, ra | {e[0] for e in m}, rb | {e[1] for e in m}, u2, v2)
    rec(0, frozenset(), frozenset(), [0]*na, [0]*nb)
    return out

def _orbit_reps(na, nb, V):
    """Return one representative of each orbit of the weight matrices under permutations.

    The weight matrices are all those of the given shape with entries from 0 to V. Their
    number is V plus 1, raised to the power of the number of entries. Two matrices lie in
    the same orbit when permuting the rows and the columns of one gives the other. Such a
    permutation rearranges the outputs of the repair in the same way and leaves their face
    distances unchanged, so the largest value over the representatives equals the largest
    value over all matrices.
    """
    rowsp = list(itertools.product(range(V + 1), repeat=nb))
    colperms = list(itertools.permutations(range(nb)))
    reps = []
    for rows in itertools.combinations_with_replacement(rowsp, na):
        best = None
        for cp in colperms:
            cand = tuple(sorted(tuple(r[c] for c in cp) for r in rows))
            if best is None or cand < best: best = cand
        if best == rows: reps.append([list(r) for r in rows])
    return reps

def cold_start_exhaustive(V=4, shapes=((1,3),(1,4),(2,2),(2,3),(2,4),(3,3)),
                          audit_shape=(2,3)):
    """Check the cold-start envelope, twice the smaller side minus 1, exhaustively at six shapes.

    At each shape the check runs over every weight matrix with entries from 0 to V, through
    one representative per orbit, and over every weakly-descending tie-break order, through
    _cold_outputs. The function asserts that the largest face distance equals the envelope
    at every shape, so the bound is reached at every shape. One shape, audit_shape, is then
    redone without the orbit reduction, and the function asserts that the largest face
    distance is the same. The result holds one row per shape (the number of rows, the number
    of columns, the number of matrices, the number of representatives, the number of outputs
    enumerated, the largest face distance and the envelope), the total number of matrices,
    the total number of representatives, and the figures of the audit.
    """
    rows = []; total = 0; total_reps = 0; mx_by_shape = {}
    for (na, nb) in shapes:
        reps = _orbit_reps(na, nb, V)
        m = min(na, nb); mx = 0; nout = 0
        for W in reps:
            OPT = _mwm(W)
            outs = _cold_outputs(W); nout += len(outs)
            for (u, v) in outs:
                d = _l0_face(W, list(u), list(v), OPT)
                if d > mx: mx = d
        n_all = (V + 1) ** (na * nb)
        assert mx == 2*m - 1, (na, nb, mx)
        rows.append((na, nb, n_all, len(reps), nout, mx, 2*m - 1))
        mx_by_shape[(na, nb)] = mx
        total += n_all; total_reps += len(reps)
    # Audit: redo one shape without the orbit reduction.
    an, ab = audit_shape
    amx = 0; acount = 0
    for flat in itertools.product(range(V + 1), repeat=an*ab):
        W = [list(flat[i*ab:(i+1)*ab]) for i in range(an)]
        OPT = _mwm(W); acount += 1
        for (u, v) in _cold_outputs(W):
            d = _l0_face(W, list(u), list(v), OPT)
            if d > amx: amx = d
    assert amx == mx_by_shape[audit_shape], (amx, mx_by_shape[audit_shape])
    return rows, total, total_reps, (an, ab, acount, amx)

# Proposition 7: no finite constant.
def fail_growth(kmax=12):
    """Show, for k from 1 to kmax, how the two rejected operators of Proposition 7 expand.

    Neither operator has any finite expansion constant for the sum of the absolute
    differences: no fixed factor bounds the distance after the repair by the distance before
    it. In both instances all costs (the paper calls them capacities) are zero, the feasible
    point is the all-zero dual, and the prediction has one column value equal to 1 and zero
    elsewhere. Part (a) runs the alternating projection on the instance with k rows and 2
    columns. Part (b) runs the integral split on the instance with k rows and 1 column. Both
    outputs lie at distance k plus 1 from the feasible point, against a distance of 1 before
    the repair, and the function asserts it. Each row of the result holds k and then, for
    each part, the number of coordinates, the distance after the repair and its ratio to the
    distance before.
    """
    rows = []
    for k in range(1, kmax + 1):
        C = [[0, 0] for _ in range(k)]
        u0, v0 = [0]*k, [1, 0]
        u1, v1 = maximal_pack(C, u0, v0)
        assert feas_pack(C, u1, v1)
        pre_a = l1(u0, [0]*k) + l1(v0, [0, 0])
        post_a = l1(u1, [0]*k) + l1(v1, [0, 0])
        Cs = [[0] for _ in range(k)]
        us, vs = [0]*k, [1]
        us, vs = greedy_pack_split(Cs, us, vs, [(a, 0) for a in range(k)])
        assert feas_pack(Cs, us, vs)
        pre_b = l1([0]*k, [0]*k) + l1([1], [0])
        post_b = l1(us, [0]*k) + l1(vs, [0])
        assert post_a == k + 1 and post_b == k + 1 and pre_a == pre_b == 1
        rows.append((k, k + 2, post_a, post_a // pre_a, k + 1, post_b, post_b // pre_b))
    return rows

# Corollary 3: the two-input Lipschitz property.
def lipschitz_two_input(trials=6000, seed=101, PS=(1, 1.5, 2, 3, 7, 'inf')):
    """Check the two-input Lipschitz property of the scan (Corollary 3) on random systems.

    Theorem 2 bounds the distance to feasible points only. The two-input statement also
    holds: when two arbitrary predictions are repaired with the same scan order, the
    distance between the two outputs is at most the distance between the two clamped
    predictions, in the norm with exponent p. The argument is the same convexity argument.
    Consider the two differences between the two runs at the endpoints of a constraint. A
    step at that constraint maps this pair of differences either to itself, or to the
    swapped pair with both signs reversed, or to a point on the segment between those two.
    Along that segment the same amount is added to both differences, and the sum of their
    absolute values, each raised to the power p, is convex and equal at both ends.

    Each trial draws a system on 2 to 6 vertices, bipartite with probability one half, with
    weights from 0 to 6. It also draws two predictions with entries from minus 3 to 7 and
    one random scan order for both. The result has two parts. The first maps each p in PS to
    the number of comparisons, the number of violations and the largest ratio of the
    distance between the outputs to the distance between the clamped predictions. The second
    counts the bipartite systems and the others.
    """
    random.seed(seed)
    res = {p: [0, 0, 0.0] for p in PS}
    nbip = [0, 0]
    for _ in range(trials):
        N = random.randint(2, 6)
        bip = random.random() < 0.5
        nbip[0 if bip else 1] += 1
        cons = _random_cover_system(N, bip, 6)
        x = [random.randint(-3, 7) for _ in range(N)]
        y = [random.randint(-3, 7) for _ in range(N)]
        order = list(range(len(cons))); random.shuffle(order)
        rx = greedy_cover(cons, x, order); ry = greedy_cover(cons, y, order)
        cx = [max(0, t) for t in x]; cy = [max(0, t) for t in y]
        for p in PS:
            b = lp(cx, cy, p); a = lp(rx, ry, p)
            res[p][0] += 1
            if a > b + 1e-9: res[p][1] += 1
            if b > 1e-12: res[p][2] = max(res[p][2], a/b)
    return res, nbip

# Theorem 8 and Table 4: the order-dependence family.
def l0_table_ext():
    """Compute one row of Table 4 for each member of the family of Theorem 8, d from 2 to 6.

    The member has one row and d columns with weights 1 to d, and its prediction is all
    zero. It is repaired by the forward scan and by the reverse scan. A row of the result
    starts with d. Then come five measures, each for the forward and the reverse output: the
    number of coordinates that differ from the face, the dual gap (the sum of the dual
    values minus the optimal value), the sum of the absolute differences to the face, the
    realised displacement (the sum of the absolute differences between the output and the
    prediction) and the largest absolute difference to the face. The first and the third
    measure are preceded by their value for the prediction.

    The sum of the absolute differences to the face is the same for both orders and equal to
    that of the prediction itself. For d at least 3 the largest absolute difference to the
    face ranks the two orders backwards, that is, the opposite way to the number of
    coordinates that differ. For d equal to 2 it is the same for both orders. The function
    asserts the values of the sum of the absolute differences, of the realised displacement
    and of the largest absolute difference. It only cross-checks the number of coordinates
    that differ against _l0_face.
    """
    rows = []
    for d in range(2, 7):
        w = [k + 1 for k in range(d)]
        opt = max(w)
        fwd = _repair_1A(w, 0, [0]*d, list(range(d)))
        rev = _repair_1A(w, 0, [0]*d, list(reversed(range(d))))
        l0pre = _face_1A(w, 0, [0]*d)
        l0f = _face_1A(w, *fwd); l0r = _face_1A(w, *rev)
        # Cross-check against the general routine for the continuous face.
        assert l0pre == _l0_face([w], [0], [0]*d)
        assert l0f == _l0_face([w], [fwd[0]], fwd[1])
        assert l0r == _l0_face([w], [rev[0]], rev[1])
        gf = fwd[0] + sum(fwd[1]) - opt; gr = rev[0] + sum(rev[1]) - opt
        l1pre = _l1_face_1A(w, 0, [0]*d)
        l1f = _l1_face_1A(w, *fwd); l1r = _l1_face_1A(w, *rev)
        dispf = abs(fwd[0]) + sum(fwd[1]); dispr = abs(rev[0]) + sum(rev[1])
        assert l1f == l1r == l1pre == d and dispf == dispr == 2*d
        # The columns of Table 4 for the largest absolute difference. For d at least 3 they
        # rank the two orders the opposite way to the number of coordinates that differ.
        lif, pmf = _linf_faces_1A(w, *fwd)
        lir, pmr = _linf_faces_1A(w, *rev)
        assert lif == 1, f"fwd l_inf to face = {lif} != 1 at d={d}"
        assert lir == d - 1, f"rev l_inf to face = {lir} != d-1 at d={d}"
        assert pmf == lif and pmr == lir, "l_inf^pm differs from l_inf"
        assert (lif < lir) == (d >= 3), "the l_inf ranking is strict exactly for d >= 3"
        rows.append((d, l0pre, l0f, l0r, gf, gr, l1pre, l1f, l1r, dispf, dispr,
                     int(lif), int(lir)))
    return rows

def l0_table():
    """Return the columns of Table 4 for the number of coordinates that differ and the dual gap.

    These are the first six entries of each row returned by l0_table_ext: d, then the
    number of coordinates that differ from the face for the prediction, for the forward
    output and for the reverse output, then the dual gap of the forward output and of the
    reverse output.
    """
    return [r[:6] for r in l0_table_ext()]

def _no_violations(t, v, w):
    """Assert that a non-expansion check found no violation, then pass its numbers on.

    The numbers are the comparisons made, the violations found and the largest ratio. The
    ratio may exceed 1 only by the rounding slack of one billionth.
    """
    assert v == 0 and w <= 1 + 1e-9, (t, v, w)
    return t, v, w

if __name__ == "__main__":
    _T0 = time.time()
    for p, (t,v,w) in check_all_norms().items():
        _no_violations(t, v, w)
        print("Thm 2  every l_p:  p=%-4s      : %9d comps, %d violations, max ratio %.6f" % (str(p), t, v, w))
    print()
    print("Thm 2  covering, bipartite      : %9d comps, %d violations, max ratio %.6f" % _no_violations(*check_cover(True)))
    print("Thm 2  covering, NON-bipartite  : %9d comps, %d violations, max ratio %.6f" % _no_violations(*check_cover(False)))
    print("Thm 4  packing, full excess     : %9d comps, %d violations, max ratio %.6f" % _no_violations(*check_pack(greedy_pack)))
    print("Prop 7 packing, SPLIT excess    : %9d comps, %d violations, max ratio %.6f" % check_pack(greedy_pack_split))
    print("Prop 7 residuated (A-side first): %9d comps, %d violations, max ratio %.6f" % check_pack(None))
    print()
    rows, total = cold_start_check()
    print("Prop 15 cold-start l0 displacement to the enumerated face (sorted scan), %d instances" % total)
    print("          shape  trials  max l0  2min-1")
    for na, nb, T, mx, bound in rows:
        print("          %dx%-3d %7d %7d %7d" % (na, nb, T, mx, bound))
    print()
    print("Prop 10  sorted scan: max (post-repair l0 MINUS pre-repair l0), 1 x 10")
    print("          k   DESC   FWD")
    for k,b in sorted_repair_check().items():
        print("          %d %6d %5d" % (k, b['DESC'], b['FWD']))
    print()
    print("Thm 8  l0 order-dependence   d  l0pre  l0fwd  l0rev  gapfwd  gaprev")
    for r in l0_table(): print("                             %2d %6d %6d %6d %7d %7d" % r)
    print()
    print("Table 4  the l1 distance to the face is identical for both orders too,")
    print("         and the l_inf distance to the face ranks them BACKWARDS")
    print("            d  l0pre l0fwd l0rev | gapf gapr | l1pre l1fwd l1rev | disp_f disp_r"
          " | linf_f linf_r")
    for r in l0_table_ext():
        print("           %2d %5d %5d %5d | %4d %4d | %5d %5d %5d | %6d %6d | %6d %6d" % r)
    print()
    print("Appendix D witnesses: the cold start can exceed 2min-1, and the warm-start +1 fails once min >= 2")
    for name, msg in refutation_check():
        print("          %-34s %s" % (name, msg))
    nc, wc, nw, ww = envelope_sweep()
    print("Envelopes on a random sweep over 3x4, 2x5, 3x5 and 4x4 (the cold one fails at 4x5, Prop 26):")
    print("          cold  d <= 2min      : %4d instances, max(d - 2min)        = %d" % (nc, wc))
    print("          warm  excess <= 2min-1: %4d instances, max(excess-(2min-1)) = %d" % (nw, ww))
    print()
    nf, mf, nm, mm = face_cross_validate()
    print()
    print("[0] Face routines agree (continuous-face minimisation vs INTEGRAL face")
    print("    enumeration, which total unimodularity justifies, Appendix A):")
    print("          %6d (instance, point) pairs, %d mismatches" % (nf, mf))
    print("          %6d max-weight-matching values vs brute force, %d mismatches" % (nm, mm))

    print()
    print("[1] Sharpness family F_k of Prop 35 / Thm 11 -- EXACT face minimisation, general k")
    print("      k  shape    n   OPT   pre  post  excess  2k-1  informative  additive")
    print("         (informative: pre <= ||B|-|A||;  additive: pre+2k-1 < |A|+|B|,")
    print("          i.e. the additive branch of the theorem's min is the binding one)")
    fam = sharp_family_check()
    for r in fam:
        print("     %3d  %2dx%-3d %4d %5d %5d %5d %7d %5d      %-5s      %-5s"
              % (r['k'], r['na'], r['nb'], r['n'], r['OPT'], r['pre'], r['post'],
                 r['excess'], 2*r['k']-1, r['informative'], r['additive']))
    kint = [r['k'] for r in fam if r['integral'] is not None]
    print("    verified through min(|A|,|B|) = %d by exact minimisation over the full"
          % max(r['k'] for r in fam))
    print("    (fractional) optimal face; independently through min(|A|,|B|) = %d by"
          % max(kint))
    print("    brute-force enumeration of the INTEGRAL optimal face, same values.")
    print("    tie-breaks: every weakly-descending order gives ONE output at every k")
    print("      k  total desc orders  positive-part orders  zero pairs  distinct outputs")
    for r in fam:
        print("     %3d  %-18s %-21d %-11d %d"
              % (r['k'], _big(r['orders_total']), r['orders_pos'],
                 r['zero_pairs'], r['n_outputs']))

    print()
    print("[1b] ADVERSE: the two hard-coded witnesses vs F_k")
    print("      member      shape   pre  post  ||B|-|A||  |A|+|B|  informative  additive-branch-binding")
    for (name, na, nb, pre, post, diff, tot, inf, add) in hardcoded_member_regime():
        print("      %-10s %2dx%-3d %5d %5d %9d %8d  %-11s  %s"
              % (name, na, nb, pre, post, diff, tot, inf, add))

    print()
    print("[2] Tie-breaks: EVERY weakly-descending order.  Zero-weight pairs are")
    print("    last in any descending order and never fire (checked at run time), so")
    print("    enumerating the positive-weight permutations is exhaustive.")
    print("      witness         shape   total orders  pos-part orders  zero pairs  outputs  pre  post: #orders")
    for (name, shp, tot, pos, nz, nout, pre, posts) in tiebreak_check():
        print("      %-15s %-6s %13s %16d %11d %8d %4d  %s"
              % (name, shp, _big(tot), pos, nz, nout, pre,
                 ", ".join("%d:%d" % t for t in posts)))

    print()
    print("[3] Prop 10 (|A|=1, the +1 bound) holds for EVERY weakly-descending order")
    sw = sortedwarm_all_orders()
    print("    closed-form |A|=1 face cross-checked against the general routine:")
    print("          %d random cases, %d mismatches" % (sw['xval_n'], sw['xval_bad']))
    print("    EXHAUSTIVE: |B| = 1..%d, weights 0..%d, predictions 0..%d, all orders"
          % (sw['dmax'], sw['Wmax'], sw['Pmax']))
    print("          instances                : %d" % sw['instances'])
    print("          (instance, order) pairs  : %d" % sw['pairs'])
    print("          excess > 1 violations    : %d" % sw['violations'])
    print("          max excess               : %d   (+1 attained on %d pairs)"
          % (sw['worst'], sw['attained']))
    print("          nothing capped or sampled in this sweep; violations and")
    print("          attainments are counted over (instance, order) pairs.")

    print()
    print("[4] Prop 15 cold-start envelope: EXHAUSTIVE, not a random sample")
    rows, tot, treps, audit = cold_start_exhaustive()
    print("      shape   matrices   orbit reps  outputs enumerated  max l0  2min-1")
    for (na, nb, n_all, nrep, nout, mx, bd) in rows:
        print("      %dx%-3d %10d %12d %20d %7d %7d" % (na, nb, n_all, nrep, nout, mx, bd))
    print("      TOTAL  %10d %12d" % (tot, treps))
    print("    every weight matrix with entries in 0..4 at those six shapes, and every")
    print("    weakly-descending tie-break order (via the maximal-matching")
    print("    characterisation of the cold-start firings); nothing capped or sampled.")
    print("    audit -- shape %dx%d redone WITHOUT the orbit reduction: %d matrices, max l0 = %d"
          % audit)

    print()
    print("[5] Prop 7: neither rejected operator has ANY finite l1 expansion constant")
    print("      k   (a) n=k+2  post l1  ratio |  (b) n=k+1  post l1  ratio")
    for (k, na_, pa, ra, nb_, pb, rb) in fail_growth():
        print("     %2d %11d %8d %6d | %10d %8d %6d" % (k, na_, pa, ra, nb_, pb, rb))
    print("    pre-repair l1 distance is 1 in every row, so the ratio IS the post")
    print("    distance: it grows as n-1 in (a) and as n in (b).")

    print()
    print("[6] Cor 3, the two-input Lipschitz property (both inputs arbitrary):")
    print("      ||R_s(x) - R_s(y)||_p <= ||max(x,0) - max(y,0)||_p")
    _lz, _nbip = lipschitz_two_input()
    for p, (t, v, w) in _lz.items():
        _no_violations(t, v, w)
        print("          p=%-4s : %7d comparisons, %d violations, max ratio %.6f"
              % (str(p), t, v, w))
    print("          instance pool: %d bipartite + %d non-bipartite covering systems,"
          % (_nbip[0], _nbip[1]))
    print("          both predictions drawn with negative entries allowed.")
    print()
    print("[7] Section 7: the one-sided walk is the most l0-frugal operator measured and is within")
    print("    a factor two in the worst case (Prop 21), yet it can end strictly worse than the sorted scan:")
    (_p, _s, _w), (_worse, _tot) = walk_vs_sorted()
    print("      3x4 witness w=((2,0,3,0),(4,0,0,5),(0,5,4,2)), yhat=(2,1,0|4,6,0,1):")
    print("          pre %d -> sorted %d, walk %d   (face-exact)" % (_p, _s, _w))
    print("      random sweep: the walk is strictly worse than sorted on %d of %d instances"
          % (_worse, _tot))
    print()
    print("total wall-clock time: %.1f s" % (time.time() - _T0))
