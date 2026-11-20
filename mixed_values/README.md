# Mixed (randomised) oblivious game values

Companion scripts certifying game values for a result on randomised (oblivious) repair rules that
the NewInML paper does not include. Nothing in the paper depends on them.
They import `verify_dichotomy.py` from the parent directory.

- `task1_v0_mixed.py`: the view V0 (M = 5) and its scaled copy (M = 3): mixed ratio value 15/16,
  mixed additive value -1/5, by LP plus exact rational certification. About one minute. The
  exact certificate is found for the M = 5 additive game; on the other three games the LP's
  optimal support is degenerate and the script reports "LP value only". The exact 15/16
  certificate (both players) for V0 (M = 5) is checked directly in `verify_dichotomy.py` section [21].
- `task1b_trap3_certify.py`: the view of the trap instance T_3 (10^6 completions, 831,881
  feasible outputs): deterministic value exactly 6/5, mixed ratio value exactly 1766/1697,
  mixed additive value exactly 3/16, each certified on every completion and every output.
  WARNING: this script needs several GB of memory and, when its cache file `trap3_faces.npz`
  is missing (as on a first run), opens a multiprocessing pool of 12 workers; it takes about ten minutes on a 12-core machine. Do not run it alongside other
  heavy jobs.
- `common.py`: the view model, face and pre computation, the mixed LP and the certifier.
