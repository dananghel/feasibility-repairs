"""Compute the mixed game values of the view V0 (M equals 5) and of its scaled copy (M equals 3).

The view has two rows, u1 and u2, and three columns, v1, v2 and c. The clamped prediction is
zero on both rows, M on v1, zero on v2 and M on c. The edges (u1, v2) and (u2, v2) are
violated, with residuals M minus 1 and M minus 2. Their weights are the same two numbers,
because the predicted values at their endpoints are zero. The other four pairs have hidden
weights, each a whole number from 0 to M. In the order the code uses, they are (u1, v1),
(u1, c), (u2, v1) and (u2, c).

The repair side returns an integral output, a vector of five whole numbers from 0 to M that
is feasible for every completion, and the adversary picks the completion. Here pre is the
distance of the prediction to the optimal face of the completion, and post is the distance of
the output to the same face. Both count the coordinates that differ from the nearest point of
the face. In the mixed game the repair side picks a probability distribution over the outputs.
The mixed ratio value is the smallest value, over all such distributions, of the largest
expected ratio of post to pre over the completions. The mixed additive value
is the same with post minus pre in place of the ratio.

Both games are solved by LP (with the HiGHS solver). The script then tries to certify each
value in exact rational arithmetic on the supports of the optimal strategies. When that
attempt fails, the script says so and reports the LP value only.
"""
import sys, os, time
# Add the parent folder to the import path, so that verify_dichotomy.py can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fractions import Fraction
import numpy as np
from common import View, face_and_pre, solve_mixed_lp, rational_certify
import verify_dichotomy as vd

def run(M):
    t0 = time.time()
    view = View(2, 3, (0, 0, M, 0, M), {(0, 3): M - 1, (1, 3): M - 2})
    assert view.hidden == [(0, 2), (0, 4), (1, 2), (1, 4)], view.hidden
    comps = list(view.completions()); assert len(comps) == (M + 1) ** 4
    faces = []; pres = []
    for h in comps:
        pts, pre = face_and_pre(view, h); faces.append(pts); pres.append(pre)
    Z = view.feasible_z()
    print(f"M={M}: {len(comps)} completions, {len(Z)} feasible integral z, pre range {min(pres)}..{max(pres)}")
    post = np.zeros((len(Z), len(comps)), dtype=np.int64)
    for zi, z in enumerate(Z):
        za = np.array(z, float)
        for ci, pts in enumerate(faces):
            post[zi, ci] = vd.face_l0(za, pts)
    pres = np.array(pres)
    # The deterministic value: the best that a single output can do against its worst completion.
    det_ratio = (post / pres[None, :]).max(axis=1); zbest = int(det_ratio.argmin())
    det_add = (post - pres[None, :]).max(axis=1)
    print(f"  deterministic: min_z max_c ratio = {det_ratio.min():.6f} at z={Z[zbest]}; min_z max_c (post-pre) = {det_add.min()}")
    out = {}
    for name, R in (("ratio", post / pres[None, :]), ("additive", (post - pres[None, :]).astype(float))):
        t, pi, q = solve_mixed_lp(R)
        # Exact payoffs for every output and every completion, so that the certificate covers
        # all of them.
        Rf = [[Fraction(int(post[z, c]), int(pres[c])) if name == "ratio" else Fraction(int(post[z, c] - pres[c]))
               for c in range(len(comps))] for z in range(len(Z))]
        cert = rational_certify(Rf, pi, q)
        supp_z = [(Z[z], round(float(pi[z]), 6)) for z in range(len(Z)) if pi[z] > 1e-7]
        supp_c = [(comps[c], int(pres[c]), round(float(q[c]), 6)) for c in range(len(comps)) if q[c] > 1e-7]
        print(f"  mixed {name}: LP value {t:.9f}; primal support {len(supp_z)} outputs, dual support {len(supp_c)} completions")
        if cert is None:
            print("    exact certification FAILED (degenerate support); LP value only")
        else:
            v, pie, qe = cert
            print(f"    EXACT value = {v} = {float(v):.9f}")
            print("    optimal mixed strategy (exact):")
            for z, pr in sorted(pie.items(), key=lambda kv: -kv[1]):
                if pr > 0: print(f"      z={Z[z]}  prob {pr}")
            print("    adversary's optimal mixed completion (exact), hidden (w_u1v1, w_u1c, w_u2v1, w_u2c):")
            for c, pr in sorted(qe.items(), key=lambda kv: -kv[1]):
                if pr > 0: print(f"      hidden={comps[c]} pre={pres[c]}  prob {pr}")
        out[name] = (t, cert)
    print(f"  [{time.time()-t0:.1f}s]")
    return out

if __name__ == "__main__":
    for M in (5, 3):
        run(M)
