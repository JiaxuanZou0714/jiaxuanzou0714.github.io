"""
Recover FSL exponents (s, beta) from the 3 constant-batch loss curves, then
compute the theory-optimal clipped power-law batch schedule and its power-of-2
discretization (switch points). Outputs ramp_fracs to feed back into bsched_gpt.py.

FSL model for a constant batch B, at optimizer step T (tokens = B*T):
    L(T,B) = Lstar + A*T^{-s} + (C/B) * Phi_beta(T)
    Phi_beta(T) = integral_0^T (tau+1)^{1/beta-2} dtau
                = (1 - (T+1)^{1/beta-1}) / (1 - 1/beta)

Caveat: FSL is derived for vanilla SGD; fitting it to AdamW curves yields an
*effective* beta. The schedule is "exact" only within this fitted FSL model.
"""
import json, os
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "gpu_logs")

CONST = {"const_big": 524288, "const_med": 131072, "const_small": 65536}
data = {k: json.load(open(os.path.join(LOG, k + ".json"))) for k in CONST}

# gather (T=steps, B, L=val) points; drop the very first noisy eval
pts = []
for k, B in CONST.items():
    for r in data[k]["records"]:
        if r["step"] >= 2:
            pts.append((r["step"], B, r["val"]))
T = np.array([p[0] for p in pts], float)
B = np.array([p[1] for p in pts], float)
L = np.array([p[2] for p in pts], float)


def phi(Tv, beta):
    e = 1.0 / beta - 1.0          # < 0 for beta>1
    return (1.0 - (Tv + 1.0) ** e) / (1.0 - 1.0 / beta)


def model(theta, Tv, Bv):
    Lstar, A, s, C, beta = theta
    return Lstar + A * Tv ** (-s) + (C / Bv) * phi(Tv, beta)


def resid(theta):
    return model(theta, T, B) - L


# fit
x0 = [3.0, 20.0, 0.4, 1e4, 2.0]
lb = [2.0, 1e-3, 0.05, 0.0, 1.05]
ub = [4.5, 1e3, 2.0, 1e9, 6.0]
sol = least_squares(resid, x0, bounds=(lb, ub), max_nfev=20000)
Lstar, A, s, C, beta = sol.x
rms = np.sqrt(np.mean(sol.fun ** 2))
print("=== FSL joint fit on 3 constant curves ===")
print(f"Lstar={Lstar:.3f}  A={A:.3f}  s={s:.3f}  C={C:.4g}  beta={beta:.3f}")
print(f"residual RMS = {rms:.4f}  (n={len(L)} points)")
# per-run residual
for k, Bv in CONST.items():
    rr = [r for r in data[k]["records"] if r["step"] >= 2]
    tt = np.array([r["step"] for r in rr], float)
    ll = np.array([r["val"] for r in rr], float)
    pr = model(sol.x, tt, Bv) - ll
    print(f"  {k:12s} B={int(Bv):7d} rms={np.sqrt(np.mean(pr**2)):.4f} "
          f"floor(C*Phi_inf/B)={C/Bv/(1-1/beta):.3f}")

# crude beta identifiability: refit with beta fixed at +/-30% and see RMS change
for bf in [beta * 0.7, beta * 1.3]:
    def r2(th4):
        return model([th4[0], th4[1], th4[2], th4[3], bf], T, B) - L
    s2 = least_squares(r2, [Lstar, A, s, C], bounds=([2, 1e-3, 0.05, 0], [4.5, 1e3, 2, 1e9]))
    print(f"  beta fixed {bf:.2f}: RMS={np.sqrt(np.mean(s2.fun**2)):.4f}")

# --------------------------------------------------------------------------- #
# theory-optimal schedule from fitted (s, beta), over the SAME hardware range
# --------------------------------------------------------------------------- #
D = 6.0e8
B_MIN, B_MAX = 65536.0, 524288.0
levels = np.array([B_MIN, 2 * B_MIN, 4 * B_MIN, B_MAX])


def compute_schedule(beta_use, A_use, s_use, C_use):
    p = 1.0 / (2.0 * beta_use) - 1.0

    def clipped_opt(Th):
        j = np.arange(Th)[::-1]
        raw = (j + 1.0) ** p
        def budget(mu):
            return np.sum(np.clip(mu * raw, B_MIN, B_MAX))
        lo, hi = 0.0, 1.0
        while budget(hi) < D and hi < 1e12:
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if budget(mid) < D else (lo, mid)
        return np.clip(0.5 * (lo + hi) * raw, B_MIN, B_MAX)

    def fsl_loss(Th, b):
        j = np.arange(Th - 1, -1, -1)
        return A_use * Th ** (-s_use) + C_use * np.sum((j + 1.0) ** (1.0 / beta_use - 2.0) / b)

    best = None
    for Th in np.unique(np.round(np.geomspace(D / B_MAX, D / B_MIN, 120)).astype(int)):
        b = clipped_opt(int(Th))
        Lo = fsl_loss(int(Th), b)
        if best is None or Lo < best[0]:
            best = (Lo, int(Th), b)
    _, Th, b_opt = best
    qb = levels[np.argmin(np.abs(np.log2(b_opt[:, None]) - np.log2(levels)[None, :]), axis=1)]
    fracs = [float(qb[qb == lv].sum()) / float(qb.sum()) for lv in levels]
    zt = np.cumsum(qb)
    switches = [zt[np.where(qb == lv)[0][0]] - lv for lv in levels[1:]]
    return Th, fracs, switches


print(f"\npower-of-2 levels: {[int(x) for x in levels]}  (heuristic fracs were [0.5,0.25,0.15,0.1])")
for bu in [2.0, 1.5, 3.0]:
    Th, fracs, sw = compute_schedule(bu, A, s, C)
    print(f"assumed beta={bu}: T*={Th:5d}  ramp_fracs={[round(f,3) for f in fracs]}  "
          f"switch@{[f'{z/1e6:.0f}M' for z in sw]}")
