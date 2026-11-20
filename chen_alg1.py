#!/usr/bin/env python3
"""
This module implements Algorithm 1 of Chen, Silwal, Vakilian and Zhang, line by line.

The algorithm comes from their paper "Faster Fundamental Graph Algorithms via Learned
Predictions", ICML 2022, arXiv:2204.12055v1. Its name there is MWPM-PD followed by two plus
signs. The code is transcribed line by line from the arXiv LaTeX source (the file matching.tex
in the folder src, in the algorithm environment with the label alg:fastmatching) and from the
appendix proof (the file omit3.tex in the folder icml22, Lemmas B.4 to B.10).

The two papers work in different orientations. Chen et al. work in the packing orientation:
minimum-cost perfect matching, where every edge has a cost and the two dual values of every
edge add up to at most the edge cost. The repairs paper, the one this code accompanies, works
in the covering orientation: maximum-weight matching, where the two dual values of every edge
add up to at least the edge weight. In this file W holds the weights, u holds the dual values
of the rows and v holds the dual values of the columns.

The two orientations correspond one to one through a change of sign. The cost of an edge is
minus its weight. The left vertices are the rows and the right vertices are the columns, and
the packing dual value of every vertex is minus its covering dual value. The two packing dual
values of an edge then add up to minus the sum of its two covering dual values. When the
covering dual is feasible, that is at most minus the weight, which is the cost, so the result
is feasible for the packing dual.

Lines 8 to 10 of Algorithm 1 then give the following. The potential of a left vertex (the
algorithm writes the potentials as z) is minus its packing dual value, which is the u value of
the row. The potential of a right vertex is its packing dual value, which is minus the v value
of the column. The reduced cost of an edge is its cost, plus the potential of its left
endpoint, minus the potential of its right endpoint. That equals the sum of the two covering
dual values of the edge minus its weight, which is the covering slack.

The change of sign preserves the tight edges (the edges whose constraint holds with equality),
the optimal matching and the number of coordinates in which two duals differ (known as the l0
distance). So the code carries integers throughout, and the reduced costs are the covering
slacks.
"""
import numpy as np
from collections import deque
import heapq

INF = float("inf")


# Line 4 of the algorithm. The function below finds the maximum matching that this line needs.
def hopcroft_karp(adj, na, nb):
    """Find a maximum cardinality bipartite matching.

    The entry of adj for a row vertex lists the column vertices it is joined to. The other two
    arguments are the number of row vertices and the number of column vertices. The result is
    a pair of lists: the column matched to each row and the row matched to each column, with
    minus 1 for a vertex that is left unmatched.
    """
    matchL = [-1] * na
    matchR = [-1] * nb
    while True:
        dist = [INF] * na
        q = deque()
        for a in range(na):
            if matchL[a] == -1:
                dist[a] = 0
                q.append(a)
        found = False
        while q:
            a = q.popleft()
            for b in adj[a]:
                a2 = matchR[b]
                if a2 == -1:
                    found = True
                elif dist[a2] == INF:
                    dist[a2] = dist[a] + 1
                    q.append(a2)
        if not found:
            return matchL, matchR
        # Search depth first along the layers. The search is iterative, not recursive, so
        # there is no risk of exceeding the recursion depth.
        for root in range(na):
            if matchL[root] != -1:
                continue
            stack = [(root, 0)]
            path = []
            it = {}
            while stack:
                a, _ = stack[-1]
                if a not in it:
                    it[a] = 0
                advanced = False
                while it[a] < len(adj[a]):
                    b = adj[a][it[a]]
                    it[a] += 1
                    a2 = matchR[b]
                    if a2 == -1:
                        # Augment the matching along the path found.
                        path.append((a, b))
                        for (aa, bb) in reversed(path):
                            matchL[aa] = bb
                            matchR[bb] = aa
                        stack = []
                        path = []
                        advanced = True
                        break
                    if dist[a2] == dist[a] + 1:
                        path.append((a, b))
                        stack.append((a2, 0))
                        advanced = True
                        break
                if not stack:
                    break
                if not advanced:
                    dist[a] = INF
                    stack.pop()
                    if path:
                        path.pop()


# Line 16 of the algorithm. The function below finds the maximum flow that this line needs.
def dinic_maxflow(nnodes, arcs, s, t):
    """Find a maximum flow from the node s to the node t.

    The network has nnodes nodes. Each entry of arcs is a triple: the node the arc leaves, the
    node it enters and its capacity. The result is a pair: the value of the flow, and a list
    with the flow on each arc, in the same order as arcs.
    """
    head = [[] for _ in range(nnodes)]
    to, cap = [], []
    for (u, v, c) in arcs:
        head[u].append(len(to)); to.append(v); cap.append(c)
        head[v].append(len(to)); to.append(u); cap.append(0)
    orig = list(cap)
    value = 0
    while True:
        level = [-1] * nnodes
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for e in head[u]:
                if cap[e] > 0 and level[to[e]] < 0:
                    level[to[e]] = level[u] + 1
                    q.append(to[e])
        if level[t] < 0:
            break
        it = [0] * nnodes

        def dfs(u, f):
            if u == t:
                return f
            while it[u] < len(head[u]):
                e = head[u][it[u]]
                w = to[e]
                if cap[e] > 0 and level[w] == level[u] + 1:
                    d = dfs(w, min(f, cap[e]))
                    if d > 0:
                        cap[e] -= d
                        cap[e ^ 1] += d
                        return d
                it[u] += 1
            return 0

        while True:
            f = dfs(s, 10**9)
            if f == 0:
                break
            value += f
    flows = [orig[2 * i] - cap[2 * i] for i in range(len(arcs))]
    return value, flows


# The main function. It runs Algorithm 1 itself.
def chen_alg1(W, u, v, st_residual=True, check=True):
    """Run Algorithm 1 of Chen et al. 2022 on the covering data W, u and v.

    W is an n by n matrix of integer weights, for maximum-weight perfect matching.

    The arguments u and v form a covering dual that is both integer and feasible: the two dual
    values of every edge add up to at least the edge weight. This is the repaired dual of their
    paper, up to the change of sign described at the top of this file.

    The flag st_residual says whether the Dijkstra run of line 12 includes the residual arcs
    into the source S and out of the sink T. These arcs exist in the residual graph whenever
    the corresponding unit-capacity arc carries flow. Chen et al. ask for the "shortest path
    distance" in the residual graph, which taken literally includes them. Every standard
    implementation of successive shortest paths (SSP) omits them. The same flag also decides
    whether these arcs are added to the maximum-flow subproblems of lines 14 to 16. The
    program measure_chen.py runs the algorithm both ways.

    The flag check says whether to check Lemma B.5 again on all edges in every iteration, once
    the reduced costs have been recomputed: no edge has a negative reduced cost, and every
    matched edge has reduced cost zero. The part about matched edges is always checked inside
    the Dijkstra run. This flag adds the full sweep.

    The result is a dictionary with the following entries.
    The entry iters is the number of iterations of the while loop, which is the quantity of
    Lemma B.9.
    The entry deficiency is n minus the size of the matching after line 4, that is, the number
    of rows that this matching leaves unmatched. It is the quantity of Lemma B.8.
    The entry ff_units is the total flow pushed by the Ford-Fulkerson calls of line 16, which
    this code makes through dinic_maxflow.
    The entry weight is the weight of the perfect matching that is returned.
    The entry match is the matching, given as the column matched to each row.
    The entry init_matching_size is the size of the matching after line 4.
    The entry dij_relax counts the row to column arcs scanned by all the Dijkstra runs. Each time
    a run processes a row vertex, the count goes up by the number of columns, so the saturated
    arc to the matched column is counted too. Arcs that leave the source, the sink or a column
    vertex are not counted.
    The entry dinic_arcs is the number of arcs in all the maximum-flow subproblems.
    The entry work is dij_relax plus dinic_arcs, an operation count for the whole run.
    """
    W = np.asarray(W)
    assert np.all(W == np.rint(W)), "weights must be integral"
    W = np.rint(W).astype(np.int64)
    na, nb = W.shape
    assert na == nb, "Algorithm 1 is stated for perfect matching on n+n"
    n = na
    u = np.rint(np.asarray(u, dtype=float)).astype(np.int64)
    v = np.rint(np.asarray(v, dtype=float)).astype(np.int64)

    # Line 10 of the algorithm, computed directly from u, v and W. The result is the same as
    # running lines 8 and 9 and then line 10. The reduced cost of an edge is the sum of its two
    # dual values minus its weight, which is the covering slack.
    red = u[:, None] + v[None, :] - W
    assert red.min() >= 0, "input dual is not feasible (line 10 / Lemma B.5 base case)"

    # Lines 2 to 4 of the algorithm. Find a maximum matching on the tight edges, which are the
    # edges of reduced cost zero.
    adj = [np.flatnonzero(red[a] == 0).tolist() for a in range(na)]
    matchL, matchR = hopcroft_karp(adj, na, nb)
    msize = sum(1 for a in range(na) if matchL[a] != -1)
    deficiency = n - msize

    # Lines 5 to 7 of the algorithm. The flow is exactly the matching. For every matched edge,
    # one unit of flow runs along the arc from the source to its row, along the edge itself,
    # and along the arc from its column to the sink.
    S, T = na + nb, na + nb + 1
    NN = na + nb + 2
    flow_val = msize

    iters = 0
    ff_units = 0
    # Counts the row to column arcs scanned by Dijkstra. Each time a run processes a row vertex,
    # the count goes up by the number of columns.
    dij_relax = 0
    # Counts the arcs present in the maximum-flow subproblems, which are built from the residual
    # arcs of reduced cost zero.
    dinic_arcs = 0
    while flow_val < n:                                    # Line 11 of the algorithm.
        # Line 12 of the algorithm. Run Dijkstra from the source in the residual graph, with the
        # slacks as arc lengths.
        dist = np.full(NN, INF)
        dist[S] = 0.0
        done = np.zeros(NN, dtype=bool)
        pq = [(0.0, S)]
        while pq:
            d, x = heapq.heappop(pq)
            if done[x] or d > dist[x]:
                continue
            done[x] = True
            if x == S:
                for a in range(na):
                    # Residual arc from the source to a row vertex.
                    if matchL[a] == -1 and d < dist[a]:
                        dist[a] = d
                        heapq.heappush(pq, (d, a))
            elif x == T:
                if st_residual:
                    for b in range(nb):
                        # Residual arc from the sink to a column vertex.
                        if matchR[b] != -1 and d < dist[na + b]:
                            dist[na + b] = d
                            heapq.heappush(pq, (d, na + b))
            elif x < na:
                a = x
                # Residual arcs from this row to every column whose edge carries no flow. The
                # length of each arc is the reduced cost of the edge, which is at least zero.
                nd = d + red[a]
                mb = matchL[a]
                better = nd < dist[na:na + nb]
                if mb != -1:
                    # The arc from this row to its matched column is saturated.
                    better[mb] = False
                dij_relax += int(better.size)
                for b in np.flatnonzero(better):
                    dist[na + b] = nd[b]
                    heapq.heappush(pq, (nd[b], na + b))
                if st_residual and mb != -1 and d < dist[S]:
                    # Residual arc from a row vertex to the source.
                    dist[S] = d
                    heapq.heappush(pq, (d, S))
            else:
                b = x - na
                a2 = matchR[b]
                if a2 != -1:
                    # Residual arc from this column back to its matched row. Its length is minus
                    # the reduced cost of the matched edge, which equals zero by Lemma B.5.
                    w = -int(red[a2, b])
                    assert w == 0, "matched edge lost zero reduced cost (Lemma B.5)"
                    if d + w < dist[a2]:
                        dist[a2] = d + w
                        heapq.heappush(pq, (d + w, a2))
                else:
                    # Residual arc from a column vertex to the sink.
                    if d < dist[T]:
                        dist[T] = d
                        heapq.heappush(pq, (d, T))
        assert np.isfinite(dist[:na + nb]).all(), "unreachable vertex in G_f"
        assert np.isfinite(dist[T]), "t unreachable: no perfect matching"

        dL = np.rint(dist[:na]).astype(np.int64)
        dR = np.rint(dist[na:na + nb]).astype(np.int64)

        # Lines 12 and 13 of the algorithm. Add to the potential of every vertex its shortest
        # path distance from the source, then recompute the reduced costs. The potential of a
        # left vertex is the u value of the row, so dL is added to u. The potential of a right
        # vertex is minus the v value of the column, so dR is subtracted from v.
        u = u + dL
        v = v - dR
        red = u[:, None] + v[None, :] - W
        if check:
            assert red.min() >= 0, "Lemma B.5 violated: negative reduced cost"
            for a in range(na):
                if matchL[a] != -1:
                    assert red[a, matchL[a]] == 0

        # Lines 14 to 16 of the algorithm. Find a maximum flow in the part of the residual graph
        # whose arcs have reduced cost zero.
        arcs = []
        for a in range(na):
            if matchL[a] == -1:
                arcs.append((S, a, 1))
            elif st_residual:
                arcs.append((a, S, 1))
            mb = matchL[a]
            for b in np.flatnonzero(red[a] == 0):
                b = int(b)
                if b != mb:
                    arcs.append((a, na + b, 1))
        for b in range(nb):
            if matchR[b] == -1:
                arcs.append((na + b, T, 1))
            else:
                arcs.append((na + b, matchR[b], 1))   # Backward arc, with reduced cost zero.
                if st_residual:
                    arcs.append((T, na + b, 1))
        dinic_arcs += len(arcs)
        val, flows = dinic_maxflow(NN, arcs, S, T)
        assert val >= 1, "Lemma B.6 violated: no progress"

        # Line 17 of the algorithm. Augment the current flow, in the residual graph, along the
        # maximum flow just found.
        f = np.zeros((na, nb), dtype=bool)
        for a in range(na):
            if matchL[a] != -1:
                f[a, matchL[a]] = True
        for (arc, fl) in zip(arcs, flows):
            if fl <= 0:
                continue
            x, y, _ = arc
            if x < na and na <= y < na + nb:              # Forward arc from a row to a column.
                assert not f[x, y - na]
                f[x, y - na] = True
            elif na <= x < na + nb and y < na:            # Backward arc from a column to a row.
                assert f[y, x - na]
                f[y, x - na] = False
        assert (f.sum(axis=1) <= 1).all() and (f.sum(axis=0) <= 1).all(), \
            "augmented flow is not a matching"
        matchL = [-1] * na
        matchR = [-1] * nb
        for a, b in zip(*np.nonzero(f)):
            matchL[int(a)] = int(b)
            matchR[int(b)] = int(a)
        flow_val = int(f.sum())
        ff_units += val
        iters += 1
        assert iters <= 4 * n + 10, "runaway loop"

    weight = int(sum(W[a, matchL[a]] for a in range(na)))
    return {"iters": iters, "deficiency": deficiency, "ff_units": ff_units,
            "weight": weight, "match": list(matchL),
            "init_matching_size": msize,
            "dij_relax": dij_relax, "dinic_arcs": dinic_arcs,
            "work": dij_relax + dinic_arcs}
