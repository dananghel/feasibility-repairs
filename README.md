# Not All Feasibility Repairs Are Equal: code

Code for the paper **"Not All Feasibility Repairs Are Equal: Non-Expansive Retractions for
Warm-Started Solvers"** by Dan Anghel (New In ML, NeurIPS 2026). Seven Python programs regenerate
every theorem check, table and constant in the paper. One companion folder, `mixed_values/`, holds
scripts for a result that the paper does not include.

Table, theorem and appendix numbers, in this README and in the programs' output, follow the
NewInML camera-ready, with two exceptions: `verify_dichotomy.py` calls Table 5 of the paper
"Table 2" and Table 6 "Table 3". `experiments_itcs.py` labels its experiments E1-E4 and prints
the rows of Tables 6, 7 and 8 under their LaTeX labels (`tab:solver`, `tab:rect`, `tab:drift`).
Three more LaTeX labels appear in the output: `thm:order` (Theorem 8) and `prop:coldrefuted`
(Proposition 26) in `verify_dichotomy.py`, and `prop:tightrows` (Proposition 43) in
`experiments_itcs.py`.

## Requirements

Python 3.10 or later. `verify_all.py` needs only the standard library; everything else needs NumPy
and SciPy (tested with NumPy 1.26 and SciPy 1.15). All runs are CPU only; the timings below are on
an Apple M3 Pro laptop.

## Programs

| file | needs | regenerates | runtime |
|---|---|---|---|
| `verify_all.py` | standard library only | the exhaustive checks and the random searches: non-expansion in each norm (on random systems), the two failing operators, the order-dependence table (Table 4) including its l_inf columns (the l_inf^pm values are asserted equal to them), the cold-start envelope, the sorted-scan warm-start sweep, the sharpness family `F_k`, and the walk-vs-sorted witness and sweep | ~5 min |
| `experiments.py` | numpy, scipy | Table 5 (post-repair l0 error by repair) | ~20 s |
| `solver.py` | numpy, scipy | the warm-startable successive-shortest-path solver, plus `verify_reference()`, its correctness suite | ~3 s |
| `experiments_itcs.py` | numpy, scipy (imports `experiments.py` and `solver.py`) | Tables 6-8, the correlation figures and their significance tests, the paired sorted-vs-forward comparison of scan orders with its exact sign test, the per-draw Table 7 analyses with the alternate-seed disclosures, and the cold-start comparison | ~2 min |
| `verify_balanced.py` | numpy, scipy | the balanced lower bound (sorting is not l0-additive on `B_n`), the l_inf / l_inf^pm reversal on the same family, and the l1 distance to the optimal face (equal for both orders) | ~1 min |
| `chen_alg1.py` + `measure_chen.py` | numpy, scipy (`measure_chen.py` imports `chen_alg1.py`, `experiments.py`, `experiments_itcs.py` and `solver.py`) | a line-by-line implementation of Chen et al.'s Algorithm 1 (weight-checked against SciPy on 1,440 solves), and the deficiency and iteration-count measurements on the Table 6 draws | ~1 min |
| `verify_dichotomy.py` | numpy, scipy (imports `solver.py`, `experiments.py` and `experiments_itcs.py`) | Section 6 and Appendix E: the no-additive-constant gadget, the cover-directed scan over every minimum cover, order and tie-break and its maximal-matching variant over every maximal matching, order and tie-break, the walk over every tie-break with its 3x4 witness and the ladder family, the chain family, the class constant 3/2, the packing mirror of the cover-directed scan, and the face-only admissibility tests. Also the balanced face lemma, the cold-start witnesses and the exact cold envelope floor(5 min/2) - 1 for min >= 3 (Appendix D), the tight-row lower bound (Appendix F), and the cover-directed cells of Table 2. Some sections check further results that the NewInML paper does not include | ~12 min |
| `mixed_values/` | numpy, scipy (imports `verify_dichotomy.py`) | companion scripts certifying game values for randomised repair rules, a result the NewInML paper does not include; see `mixed_values/README.md`. **`task1b_trap3_certify.py` needs several GB of memory and, when its cache file `trap3_faces.npz` is missing (as on a first run), opens a pool of 12 workers.** | see its README |

Run each with `python3 <file>`; for the pair, run `measure_chen.py`, which imports `chen_alg1.py`. All configurations are pinned; no arguments are needed. The seven
programs together take about twenty minutes. Do not run them in parallel on a machine with little
memory.

## How the checks work

Distances to the optimal face are exact. Both `verify_all.py` and `verify_balanced.py` compute
the minimum Hamming (or l1) distance from a point to the full optimal face, fractional points
included. They do not measure to one selected optimum.
`verify_all.py` does this in the standard library: Hamming by a pruned agree/disagree search
whose feasibility test is an LP-duality identity; and, for the single-row family of Table 4 only,
l1 by evaluating a convex piecewise-linear function at its breakpoints and l_inf / l_inf^pm by
exact rational minimisation over the continuous face. `verify_all.py` also has a second Hamming
routine, `_l0_face_integral`, which enumerates the integral face points with coordinates up to the
largest weight plus 1; total unimodularity makes it exact, and some checks (the walk-vs-sorted
witness and sweep, the random sweeps) use it alone. Integral enumeration is not valid for those two norms; the
paper's total-unimodularity argument (Section 2, footnote 2, and Appendix A) covers l0 and l1
only. `verify_balanced.py` does all four as linear or mixed-integer programs. In `verify_all.py`
the Hamming search over the full face is cross-checked against brute-force enumeration of the
integral face (9,000 instance/point pairs, no mismatches).

Tie-breaks are enumerated wherever a claim of order-robustness rests on computation. Zero-weight
pairs can never fire (their residual is `-y_a - y_b <= 0` for `y >= 0`), so enumerating the
permutations of the positive weights is exhaustive even when the total number of
weakly-descending orders is astronomical. For the balanced family, `verify_balanced.py` enumerates
the orders for n <= 4 and samples them for larger n. There the paper proves that the output does
not depend on the order, so the sampling only spot-checks the proof.

Adverse results are reported too. `experiments_itcs.py` includes the endpoint shift and prints
its comparison with the uniform shift cell by cell. It has
exactly the same realised l_inf displacement as the uniform shift, between 0.34 and 0.76 of its l0
and between 0.30 and 0.72 of its dual gap, and yet costs the solver more work in 8 of 12 cells. That measurement contradicts a naive
reading of the paper's own correlation claim, and the paper reports it for that reason.

## Reproducing the main numbers

```
python3 verify_all.py        # Zero violations and a largest ratio of exactly 1.000000 in every non-expansion check.
python3 verify_balanced.py   # Face distance 1 before the repair and n after it, for n from 2 to 8, under
                             # every weakly descending order up to n = 4 and 300 sampled orders above.
                             # For n from 2 to 7, the largest coordinate difference to the face is 1
                             # for the descending scan and n minus 1 for the increasing one.
python3 verify_dichotomy.py  # Every section prints its numbers and ends with 'all assertions in [k] hold'.
python3 measure_chen.py      # Runs Algorithm 1 of Chen et al. on the Table 6 draws, in about one minute.
python3 solver.py            # All 1954 instances match the exact optimum computed by SciPy.
python3 experiments.py       # Table 5.
python3 experiments_itcs.py  # Tables 6 to 8, the correlations and the significance tests.
```

## Citation

```bibtex
@inproceedings{anghel2026repairs,
  title     = {Not All Feasibility Repairs Are Equal: Non-Expansive Retractions for Warm-Started Solvers},
  author    = {Anghel, Dan},
  booktitle = {New In ML Workshop at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  note      = {Non-archival}
}
```

## Licence

MIT; see `LICENSE`.
