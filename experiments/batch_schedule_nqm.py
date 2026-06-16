"""
Noisy Quadratic Model (NQM) validation of the optimal batch-size schedule
derived in the blog post "为什么 LLM pretrain 过程中途要把 batch size 翻倍".

The NQM is the standard surrogate for the large-batch / critical-batch-size
theory (McCandlish et al.; FSL). Its expected loss obeys an *exact* recursion,
so we get clean, Monte-Carlo-free loss curves.

Per-coordinate i with curvature h_i, per-sample gradient noise variance sig2_i,
constant lr eta, batch B_k at step k:

    v_{i,k+1} = (1 - eta h_i)^2 v_{i,k} + eta^2 sig2_i / B_k ,   v_{i,k}=E[theta_i^2]
    L_k       = 0.5 * sum_i h_i v_{i,k}

Final loss decomposes exactly as

    L_T = S(T) + sum_k kappa(T-1-k) / B_k ,
    S(T)     = 0.5 sum_i h_i v_{i,0} a_i^T          (signal/bias term, indep. of B)
    kappa(j) = 0.5 eta^2 sum_i h_i sig2_i a_i^j      (noise kernel),  a_i=(1-eta h_i)^2

which is exactly the FSL form  A T^{-s} + C * integral K(T-t)/b(t) dt.
Fixed budget D = sum_k B_k, Cauchy-Schwarz gives  B_k* ∝ sqrt(kappa(T-1-k)),
clipped to [Bmin, Bmax] -> the blog's clipped power-law (late-switch in hard regime).
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "img", "post-06-16")
OUT_DIR = os.path.abspath(OUT_DIR)

# --------------------------------------------------------------------------- #
# 1. NQM spectrum
# --------------------------------------------------------------------------- #
N = 1500
eta = 0.5
H_MAX, H_MIN = 1.0, 1e-5
S_SRC = 0.3        # source-exponent knob (signal energy ~ h^S_SRC); <1-1/beta -> hard regime
C_NOISE = 0.5      # noise-shape knob (sig2 ~ h^C_NOISE) -> capacity exponent beta

# budget / batch bounds (defined early so the noise scale + kernel can use them)
D = 3.0e6
B_MIN = 8.0
B_MAX = 8192.0
TARGET_FLOOR = 0.5  # calibrate noise magnitude so the smallest batch's loss floor < 1
                    # (keeps loss curves from spiking above the initial loss)

h = np.geomspace(H_MAX, H_MIN, N)          # curvatures (descending)
a = (1.0 - eta * h) ** 2                    # per-step contraction
assert np.all(eta * h < 1.0), "lr too large -> unstable"

v0 = h ** (S_SRC - 1.0)                      # initial E[theta_i^2]
sig2 = h ** C_NOISE                          # per-sample gradient-noise variance (shape)

# normalise initial loss to 1
L0 = 0.5 * np.sum(h * v0)
v0 = v0 / L0

# calibrate global noise magnitude so that L_inf(B_MIN) = TARGET_FLOOR
base_floor_Bmin = 0.5 * np.sum(h * (eta ** 2) * sig2 / (B_MIN * (1.0 - a)))
sig2 *= TARGET_FLOOR / base_floor_Bmin

c_sig = 0.5 * (h * v0)                        # signal coefficients  (S(T)=sum c_sig a^T)
c_noi = 0.5 * eta ** 2 * (h * sig2)           # kernel coefficients  (kappa(j)=sum c_noi a^j)


def signal_term(T):
    return float(np.sum(c_sig * a ** T))


# pre-compute kernel kappa(j) for j = 0 .. JMAX (covers the longest schedule, T=D/Bmin)
JMAX = int(np.ceil(D / B_MIN)) + 5
kappa = np.empty(JMAX + 1)
p = np.ones_like(a)
for j in range(JMAX + 1):
    kappa[j] = np.sum(c_noi * p)
    p *= a
Kcum = np.concatenate([[0.0], np.cumsum(kappa)])   # Kcum[m] = sum_{j<m} kappa[j]


def final_loss(T, B):
    """Exact final loss for schedule B[0..T-1] via the decomposition."""
    j = np.arange(T - 1, -1, -1)              # remaining steps for k=0..T-1
    return signal_term(T) + float(np.sum(kappa[j] / B))


def run_curve(B):
    """Run the exact recursion, return (cumulative_tokens, loss) per step."""
    v = v0.copy()
    losses = np.empty(len(B) + 1)
    toks = np.empty(len(B) + 1)
    cum = 0.0
    for k, Bk in enumerate(B):
        losses[k] = 0.5 * np.sum(h * v)
        toks[k] = cum
        v = a * v + (eta ** 2) * sig2 / Bk
        cum += Bk
    losses[-1] = 0.5 * np.sum(h * v)
    toks[-1] = cum
    return toks, losses


# --------------------------------------------------------------------------- #
# 2. measure the effective exponents s, beta
# --------------------------------------------------------------------------- #
def fit_power(x, y):
    lx, ly = np.log(x), np.log(y)
    sl, _ = np.polyfit(lx, ly, 1)
    return sl


Tgrid_fit = np.unique(np.round(np.geomspace(50, 50_000, 60)).astype(int))
S_of_T = np.array([signal_term(int(T)) for T in Tgrid_fit])
s_exp = -fit_power(Tgrid_fit, S_of_T)        # S(T) ~ T^{-s}

jfit = np.unique(np.round(np.geomspace(2, 50_000, 60)).astype(int))
q = -fit_power(jfit, kappa[jfit])            # kappa(j) ~ j^{-(2-1/beta)} -> slope = -(2-1/beta)
beta_eff = 1.0 / (2.0 - q)
p_exp = 1.0 / (2.0 * beta_eff) - 1.0          # blog exponent of b*(t) ~ (T-t+1)^p

# --------------------------------------------------------------------------- #
# 3. schedules at a fixed token budget D  (D, B_MIN, B_MAX defined above)
# --------------------------------------------------------------------------- #
def clip_optimal(T, D, Bmin, Bmax):
    """B_k* = clip(mu * sqrt(kappa(T-1-k)), Bmin, Bmax) with mu set by budget D."""
    j = np.arange(T - 1, -1, -1)
    sw = np.sqrt(kappa[j])

    def budget(mu):
        return np.sum(np.clip(mu * sw, Bmin, Bmax))

    lo, hi = 0.0, 1.0
    while budget(hi) < D:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if budget(mid) < D:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    return np.clip(mu * sw, Bmin, Bmax)


# 3a. optimal continuous schedule, optimised over T as well
T_candidates = np.unique(np.round(np.geomspace(200, 350_000, 110)).astype(int))
best = None
for T in T_candidates:
    if D / T < B_MIN or D / T > B_MAX:
        # still allow; clip handles it, but skip extremes where all steps hit a bound
        pass
    B = clip_optimal(int(T), D, B_MIN, B_MAX)
    L = final_loss(int(T), B)
    if best is None or L < best[0]:
        best = (L, int(T), B)
L_opt, T_opt, B_opt = best

# 3b. best constant batch (the fair "default" baseline)
best_c = None
for B0 in np.geomspace(B_MIN, B_MAX, 120):
    T = int(round(D / B0))
    if T < 2:
        continue
    L = final_loss(T, np.full(T, B0))
    if best_c is None or L < best_c[0]:
        best_c = (L, T, B0)
L_const, T_const, B_const = best_c

# 3c. two-stage (one switch): search B1, B2>B1 and switch token D1 (generalises constant,
#     so it is provably >= best constant). Closed-form loss via prefix sums Kcum.
def two_stage_loss(B1, B2, T1, T2):
    T = T1 + T2
    if T > JMAX:
        return np.inf
    noise = (Kcum[T] - Kcum[T2]) / B1 + Kcum[T2] / B2
    return signal_term(T) + noise


best_2 = None
B1_grid = np.geomspace(B_MIN, B_MAX / 2, 26)
for B1 in B1_grid:
    for B2 in np.geomspace(B1 * 1.3, B_MAX, 26):
        for frac in np.linspace(0.05, 0.97, 50):
            D1 = frac * D
            T1 = int(round(D1 / B1))
            T2 = int(round((D - D1) / B2))
            if T1 < 1 or T2 < 1:
                continue
            L = two_stage_loss(B1, B2, T1, T2)
            if best_2 is None or L < best_2[0]:
                best_2 = (L, B1, B2, T1, T2)
L_two, B1_two, B2_two, T1_two, T2_two = best_2
B_two = np.concatenate([np.full(T1_two, B1_two), np.full(T2_two, B2_two)])

# 3d. multi-stage doubling: snap the continuous optimum to powers of two
log2B = np.log2(np.clip(B_opt, B_MIN, B_MAX))
B_dbl = 2.0 ** np.round(log2B)
B_dbl = np.clip(B_dbl, B_MIN, B_MAX)
B_dbl *= D / np.sum(B_dbl)                     # rescale to hit the budget
L_dbl = final_loss(len(B_dbl), B_dbl)

# --------------------------------------------------------------------------- #
# 4. report
# --------------------------------------------------------------------------- #
print(f"measured source exponent      s    = {s_exp:.3f}")
print(f"measured capacity exponent    beta = {beta_eff:.3f}")
print(f"blog schedule exponent  1/(2b)-1   = {p_exp:.3f}   (b*(t) ~ (T-t+1)^p)")
print(f"regime: s {'<=' if s_exp <= 1 - 1/beta_eff else '>'} 1-1/beta "
      f"({1 - 1/beta_eff:.3f}) -> {'HARD (late switch)' if s_exp <= 1-1/beta_eff else 'EASY (grow throughout)'}")
print(f"budget D = {D:.2e} tokens,  Bmin={B_MIN:.0f}, Bmax={B_MAX:.0f}")
print("-" * 60)
print(f"{'schedule':<26}{'T (steps)':>12}{'final loss':>14}{'vs const':>10}")
for name, L, T in [
    ("constant (default/best)", L_const, T_const),
    ("two-stage (one switch)", L_two, len(B_two)),
    ("multi-stage doubling", L_dbl, len(B_dbl)),
    ("optimal clipped power-law", L_opt, T_opt),
]:
    print(f"{name:<26}{T:>12d}{L:>14.4e}{L_const / L:>9.2f}x")
print("-" * 60)

# exact noise floor of a constant batch B (fixed point v_inf = eta^2 sig2 / (B(1-a)))
def noise_floor(B):
    return 0.5 * float(np.sum(h * (eta ** 2) * sig2 / (B * (1.0 - a))))


# token budget that constant needs to reach the optimal's final loss
def tokens_to_reach(B_single_value, target):
    if target < noise_floor(B_single_value):
        return np.inf                          # constant B can never get below its floor
    v = v0.copy(); cum = 0.0
    for _ in range(5_000_000):
        L = 0.5 * np.sum(h * v)
        if L <= target:
            return cum
        v = a * v + (eta ** 2) * sig2 / B_single_value
        cum += B_single_value
    return np.inf

floor_const = noise_floor(B_const)
print(f"best-constant batch B={B_const:.0f}, its noise floor L_inf = {floor_const:.3e}")
print(f"optimal final loss = {L_opt:.3e}")
if L_opt < floor_const:
    print("-> optimal beats the constant-batch noise floor: constant can NEVER reach it, "
          "at any token budget.")
else:
    tok_const = tokens_to_reach(B_const, L_opt)
    print(f"-> best-constant needs {tok_const:.2e} tokens = {tok_const / D:.2f}x budget "
          f"to match optimal's final loss")

# --------------------------------------------------------------------------- #
# 5. headline figure: loss curves (left) + schedules (right)
# --------------------------------------------------------------------------- #
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5,
                     "axes.labelsize": 12, "legend.fontsize": 9.5})
M = 1e6  # plot tokens in millions

scheds = [
    ("Constant batch (default)", np.full(T_const, B_const), dict(color="#7f7f7f", lw=2.2, ls="-")),
    ("Two-stage (one switch)",   B_two,                     dict(color="#2ca02c", lw=2.4, ls="-")),
    ("Multi-stage doubling",     B_dbl,                     dict(color="#1f77b4", lw=1.8, ls="--")),
    ("Optimal (variational)",    B_opt,                     dict(color="#d62728", lw=3.4, ls="-", alpha=0.9)),
]
runs = [(name, B, *run_curve(B), st) for name, B, st in scheds]  # name, B, tok, loss, style
xmax = max(r[2][-1] for r in runs) / M

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5.2))

# panel 0: loss, log-log overview (shows the power-law decay + the end crash)
for name, B, tok, loss, st in runs:
    ax0.loglog(np.maximum(tok, 1), loss, label=name, **st)
ax0.set_xlabel("consumed tokens")
ax0.set_ylabel("excess loss")
ax0.set_title("Loss vs tokens (log-log overview)")
ax0.grid(True, which="both", alpha=0.25)
ax0.legend(frameon=False, loc="lower left")

# panel 1: loss, LINEAR token axis -> the schedules separate clearly here
for name, B, tok, loss, st in runs:
    ax1.semilogy(tok / M, loss, label=name, **st)
ax1.set_xlabel("consumed tokens (millions)")
ax1.set_ylabel("excess loss")
ax1.set_title("Loss vs tokens (linear axis)")
ax1.set_xlim(0, xmax)
ax1.grid(True, which="both", alpha=0.25)
ax1.legend(frameon=False, loc="upper right")

# panel 2: batch schedules, LINEAR token axis -> plateau + ramp is visible
ax2.semilogy(runs[3][2][:-1] / M, B_opt, color="#d62728", lw=3.2, alpha=0.9,
             label="Optimal $b^*(t)$")
k = np.arange(T_opt)                                       # analytic blog form
ana = (T_opt - k) ** p_exp
ana *= D / np.sum(ana)
ana = np.clip(ana, B_MIN, B_MAX)
tok_ana = np.concatenate([[0.0], np.cumsum(ana)[:-1]])
ax2.semilogy(tok_ana / M, ana, "--", color="black", lw=1.7,
             label=r"analytic $(T-t{+}1)^{1/2\beta-1}$")
ax2.step(runs[2][2][:-1] / M, B_dbl, where="post", color="#1f77b4", lw=1.8,
         label="Doubling staircase")
ax2.axhline(B_const, color="#7f7f7f", lw=2.2, label="Constant (default)")
ax2.set_xlabel("consumed tokens (millions)")
ax2.set_ylabel("batch size $b(t)$")
ax2.set_title(fr"Batch schedules ($s$={s_exp:.2f}, $\beta$={beta_eff:.2f})")
ax2.set_xlim(0, xmax)
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(frameon=False, loc="upper left")

fig.tight_layout()
out_main = os.path.join(OUT_DIR, "batch_schedule_experiment.png")
fig.savefig(out_main, dpi=150, bbox_inches="tight")
print("saved", out_main)

# --------------------------------------------------------------------------- #
# 6. diagnostics figure: noise floor ~ 1/B  and  kernel power law
# --------------------------------------------------------------------------- #
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

# noise floor vs B (Section 1): exact fixed point v_inf = eta^2 sig2 / (B(1-a))
Bs = np.geomspace(8, 8192, 18)
floors = np.array([noise_floor(B0) for B0 in Bs])
ax1.loglog(Bs, floors, "o", color="#d62728", ms=6, label="exact floor $L_\\infty(B)$")
ax1.loglog(Bs, floors[0] * Bs[0] / Bs, "--", color="black", label=r"$\propto 1/B$")
ax1.set_xlabel("constant batch size $B$")
ax1.set_ylabel("noise loss floor $L_\\infty(B)$")
ax1.set_title("Section 1: floor halves when $B$ doubles")
ax1.grid(True, which="both", alpha=0.25)
ax1.legend(frameon=False)

ax2.loglog(jfit, kappa[jfit], "o", color="#1f77b4", ms=4, label="kernel $\\kappa(j)$")
fit_line = kappa[jfit][0] * (jfit / jfit[0]) ** (-q)
ax2.loglog(jfit, fit_line, "--", color="black",
           label=fr"$\propto j^{{-(2-1/\beta)}},\ \beta$={beta_eff:.2f}")
ax2.set_xlabel("steps remaining $j$")
ax2.set_ylabel(r"noise kernel $\kappa(j)$")
ax2.set_title("Section 3: kernel is a power law")
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(frameon=False)

fig2.tight_layout()
out_diag = os.path.join(OUT_DIR, "batch_schedule_diagnostics.png")
fig2.savefig(out_diag, dpi=150, bbox_inches="tight")
print("saved", out_diag)
