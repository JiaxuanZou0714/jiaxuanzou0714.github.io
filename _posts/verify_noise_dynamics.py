import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def fit_loglog_slope(x: np.ndarray, y: np.ndarray, tail: int = 4) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    if x.size > tail:
        x = x[-tail:]
        y = y[-tail:]
    slope, _ = np.polyfit(np.log(x), np.log(y), 1)
    return float(slope)


def experiment_direction_invariance(rng: np.random.Generator, out_dir: Path) -> dict:
    d = 64
    n = 50000
    sigma = 5.0

    s = np.zeros(d)
    s[0] = 1.0

    z = rng.standard_normal((n, d))
    g = s + sigma * z
    u = normalize_rows(g)

    cos_g = g @ s / np.maximum(np.linalg.norm(g, axis=1), EPS)
    cos_u = u @ s / np.maximum(np.linalg.norm(u, axis=1), EPS)
    abs_diff = np.abs(cos_g - cos_u)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    idx = np.linspace(0, n - 1, 3000, dtype=int)
    axes[0].scatter(cos_g[idx], cos_u[idx], s=6, alpha=0.35, color="#1f77b4")
    axes[0].plot([-1, 1], [-1, 1], "--", color="black", linewidth=1.5)
    axes[0].set_xlabel("cos(g, s)")
    axes[0].set_ylabel("cos(g/||g||, s)")
    axes[0].set_title("Direction Cosine Is Preserved")
    axes[0].grid(alpha=0.25)

    axes[1].hist(abs_diff, bins=60, color="#ff7f0e", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("|cos(g, s) - cos(g/||g||, s)|")
    axes[1].set_ylabel("Count (log scale)")
    axes[1].set_title("Numerical Difference Distribution")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "01_direction_invariance.png", dpi=160)
    plt.close(fig)

    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
    }


def experiment_noise_and_eta_scaling(rng: np.random.Generator, out_dir: Path) -> dict:
    d = 128
    n = 70000
    sigmas = np.array([0.3, 0.5, 0.8, 1.2, 1.8, 2.7, 4.0, 6.0], dtype=float)

    s = np.zeros(d)
    s[0] = 1.0
    x = s.copy()  # For F(x)=0.5||x||^2, gradient equals x.

    drift_sgd = []
    drift_norm = []
    var_trace_sgd = []
    var_trace_norm = []
    eta_max_sgd = []
    eta_max_norm = []

    for sigma in sigmas:
        z = rng.standard_normal((n, d))
        g = s + sigma * z
        u = normalize_rows(g)

        # Drift along true gradient direction.
        drift_sgd.append(float(np.mean(g @ s)))
        drift_norm.append(float(np.mean(u @ s)))

        # Trace of covariance: sum_i Var(update_i).
        var_trace_sgd.append(float(np.var(g, axis=0).sum()))
        var_trace_norm.append(float(np.var(u, axis=0).sum()))

        # One-step expected descent condition for F(x)=0.5||x||^2.
        ex_dot_u_sgd = float(np.mean(g @ x))
        e_u2_sgd = float(np.mean(np.sum(g * g, axis=1)))
        eta_max_sgd.append(2.0 * ex_dot_u_sgd / max(e_u2_sgd, EPS))

        ex_dot_u_norm = float(np.mean(u @ x))
        e_u2_norm = float(np.mean(np.sum(u * u, axis=1)))
        eta_max_norm.append(2.0 * ex_dot_u_norm / max(e_u2_norm, EPS))

    drift_sgd = np.array(drift_sgd)
    drift_norm = np.array(drift_norm)
    var_trace_sgd = np.array(var_trace_sgd)
    var_trace_norm = np.array(var_trace_norm)
    eta_max_sgd = np.array(eta_max_sgd)
    eta_max_norm = np.array(eta_max_norm)

    # Plot A: drift and covariance scaling.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].loglog(sigmas, np.abs(drift_sgd), "o-", linewidth=2, label="SGD drift |E[<u,s>]|")
    axes[0].loglog(sigmas, np.abs(drift_norm), "o-", linewidth=2, label="Normalized drift |E[<u,s>]|")
    ref_inv_sigma = np.abs(drift_norm[-1]) * (sigmas / sigmas[-1]) ** (-1)
    axes[0].loglog(sigmas, ref_inv_sigma, "--", linewidth=1.8, label="Reference ~ sigma^-1")
    axes[0].set_xlabel("sigma")
    axes[0].set_ylabel("Drift magnitude")
    axes[0].set_title("Drift Scaling in Noise-Dominated Regime")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].loglog(sigmas, var_trace_sgd, "o-", linewidth=2, label="SGD trace(Cov)")
    axes[1].loglog(sigmas, var_trace_norm, "o-", linewidth=2, label="Normalized trace(Cov)")
    ref_sigma2 = var_trace_sgd[-1] * (sigmas / sigmas[-1]) ** 2
    axes[1].loglog(sigmas, ref_sigma2, "--", linewidth=1.8, label="Reference ~ sigma^2")
    axes[1].set_xlabel("sigma")
    axes[1].set_ylabel("Trace covariance")
    axes[1].set_title("Noise Amplification vs Noise Compression")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "02_noise_compression.png", dpi=160)
    plt.close(fig)

    # Plot B: max stable learning-rate scaling.
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.loglog(sigmas, eta_max_sgd, "o-", linewidth=2.2, label="SGD empirical eta_max")
    ax.loglog(sigmas, eta_max_norm, "o-", linewidth=2.2, label="Normalized empirical eta_max")

    eta_sgd_theory = 2.0 / (1.0 + d * sigmas**2)
    ax.loglog(sigmas, eta_sgd_theory, "--", linewidth=1.8, label="SGD theory 2/(1+d sigma^2)")

    ref_eta_sgd = eta_max_sgd[-1] * (sigmas / sigmas[-1]) ** (-2)
    ref_eta_norm = eta_max_norm[-1] * (sigmas / sigmas[-1]) ** (-1)
    ax.loglog(sigmas, ref_eta_sgd, ":", linewidth=2, label="Reference ~ sigma^-2")
    ax.loglog(sigmas, ref_eta_norm, ":", linewidth=2, label="Reference ~ sigma^-1")

    ax.set_xlabel("sigma")
    ax.set_ylabel("Estimated max stable eta")
    ax.set_title("Max Stable Learning Rate Scaling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "03_eta_scaling.png", dpi=160)
    plt.close(fig)

    return {
        "sigmas": sigmas.tolist(),
        "drift_sgd": drift_sgd.tolist(),
        "drift_norm": drift_norm.tolist(),
        "var_trace_sgd": var_trace_sgd.tolist(),
        "var_trace_norm": var_trace_norm.tolist(),
        "eta_max_sgd": eta_max_sgd.tolist(),
        "eta_max_norm": eta_max_norm.tolist(),
        "slope_high_sigma_drift_norm": fit_loglog_slope(sigmas, np.abs(drift_norm)),
        "slope_high_sigma_var_sgd": fit_loglog_slope(sigmas, var_trace_sgd),
        "slope_high_sigma_var_norm": fit_loglog_slope(sigmas, var_trace_norm),
        "slope_high_sigma_eta_sgd": fit_loglog_slope(sigmas, eta_max_sgd),
        "slope_high_sigma_eta_norm": fit_loglog_slope(sigmas, eta_max_norm),
    }


def simulate_steady_state_error(
    rng: np.random.Generator,
    d: int,
    sigma: float,
    eta: float,
    steps: int,
    burnin: int,
    runs: int,
    normalized: bool,
) -> float:
    x = rng.standard_normal((runs, d)) * 2.0
    acc = 0.0
    count = 0

    for t in range(steps):
        g = x + sigma * rng.standard_normal((runs, d))
        if normalized:
            u = normalize_rows(g)
        else:
            u = g
        x = x - eta * u

        if t >= burnin:
            acc += float(np.mean(np.sum(x * x, axis=1)))
            count += 1

    return acc / max(count, 1)


def experiment_steady_state_scaling(rng: np.random.Generator, out_dir: Path) -> dict:
    sigmas = np.array([0.4, 0.7, 1.2, 2.0, 3.3], dtype=float)
    d = 32
    eta = 0.05
    steps = 3500
    burnin = 1800
    runs = 256

    err_sgd = []
    err_norm = []
    for sigma in sigmas:
        err_sgd.append(
            simulate_steady_state_error(
                rng,
                d=d,
                sigma=sigma,
                eta=eta,
                steps=steps,
                burnin=burnin,
                runs=runs,
                normalized=False,
            )
        )
        err_norm.append(
            simulate_steady_state_error(
                rng,
                d=d,
                sigma=sigma,
                eta=eta,
                steps=steps,
                burnin=burnin,
                runs=runs,
                normalized=True,
            )
        )

    err_sgd = np.array(err_sgd)
    err_norm = np.array(err_norm)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.loglog(sigmas, err_sgd, "o-", linewidth=2.2, label="SGD steady-state E||x||^2")
    ax.loglog(sigmas, err_norm, "o-", linewidth=2.2, label="Normalized steady-state E||x||^2")

    ref_sgd = err_sgd[0] * (sigmas / sigmas[0]) ** 2
    ref_norm = err_norm[0] * (sigmas / sigmas[0])
    ax.loglog(sigmas, ref_sgd, "--", linewidth=1.8, label="Reference ~ sigma^2")
    ax.loglog(sigmas, ref_norm, "--", linewidth=1.8, label="Reference ~ sigma")

    ax.set_xlabel("sigma")
    ax.set_ylabel("Steady-state E||x||^2")
    ax.set_title("Steady-State Error Floor Scaling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "04_steady_state_scaling.png", dpi=160)
    plt.close(fig)

    return {
        "sigmas": sigmas.tolist(),
        "steady_state_error_sgd": err_sgd.tolist(),
        "steady_state_error_norm": err_norm.tolist(),
        "slope_high_sigma_error_sgd": fit_loglog_slope(sigmas, err_sgd),
        "slope_high_sigma_error_norm": fit_loglog_slope(sigmas, err_norm),
    }


def experiment_heteroscedastic_blocks(rng: np.random.Generator, out_dir: Path) -> dict:
    d_block = 32
    runs = 512
    steps = 2500
    sigma_low = 0.5
    sigma_high = 4.0

    # A global eta constrained by the noisiest block for SGD-like updates.
    eta = 0.003

    x0_low = rng.standard_normal((runs, d_block)) * 2.0
    x0_high = rng.standard_normal((runs, d_block)) * 2.0

    x_sgd_low = x0_low.copy()
    x_sgd_high = x0_high.copy()
    x_norm_low = x0_low.copy()
    x_norm_high = x0_high.copy()

    curve_sgd_low = np.zeros(steps)
    curve_sgd_high = np.zeros(steps)
    curve_norm_low = np.zeros(steps)
    curve_norm_high = np.zeros(steps)

    for t in range(steps):
        g_low = x_sgd_low + sigma_low * rng.standard_normal((runs, d_block))
        g_high = x_sgd_high + sigma_high * rng.standard_normal((runs, d_block))
        x_sgd_low = x_sgd_low - eta * g_low
        x_sgd_high = x_sgd_high - eta * g_high

        gn_low = x_norm_low + sigma_low * rng.standard_normal((runs, d_block))
        gn_high = x_norm_high + sigma_high * rng.standard_normal((runs, d_block))
        x_norm_low = x_norm_low - eta * normalize_rows(gn_low)
        x_norm_high = x_norm_high - eta * normalize_rows(gn_high)

        curve_sgd_low[t] = float(np.mean(np.sum(x_sgd_low * x_sgd_low, axis=1)))
        curve_sgd_high[t] = float(np.mean(np.sum(x_sgd_high * x_sgd_high, axis=1)))
        curve_norm_low[t] = float(np.mean(np.sum(x_norm_low * x_norm_low, axis=1)))
        curve_norm_high[t] = float(np.mean(np.sum(x_norm_high * x_norm_high, axis=1)))

    # Directly estimate effective contraction coefficients kappa_r = E[<u,s>]/||s||^2.
    n_kappa = 120000
    s = np.ones(d_block)
    s = s / np.linalg.norm(s)

    z_low = rng.standard_normal((n_kappa, d_block))
    z_high = rng.standard_normal((n_kappa, d_block))
    u_low = normalize_rows(s + sigma_low * z_low)
    u_high = normalize_rows(s + sigma_high * z_high)

    kappa_norm_low = float(np.mean(u_low @ s) / np.dot(s, s))
    kappa_norm_high = float(np.mean(u_high @ s) / np.dot(s, s))

    kappa_sgd_low = 1.0
    kappa_sgd_high = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    t = np.arange(1, steps + 1)
    axes[0].plot(t, curve_sgd_low, label="SGD low-noise block", linewidth=2)
    axes[0].plot(t, curve_sgd_high, label="SGD high-noise block", linewidth=2)
    axes[0].plot(t, curve_norm_low, label="Norm low-noise block", linewidth=2)
    axes[0].plot(t, curve_norm_high, label="Norm high-noise block", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Block E||x_r||^2")
    axes[0].set_title("Global eta Under Heteroscedastic Noise")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = ["Low-noise block", "High-noise block"]
    x_pos = np.arange(2)
    width = 0.35
    axes[1].bar(x_pos - width / 2, [kappa_sgd_low, kappa_sgd_high], width=width, label="SGD kappa")
    axes[1].bar(x_pos + width / 2, [kappa_norm_low, kappa_norm_high], width=width, label="Norm kappa")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Effective contraction coefficient kappa")
    axes[1].set_title("Implicit Inverse-Variance Weighting")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "05_heteroscedastic_blocks.png", dpi=160)
    plt.close(fig)

    return {
        "eta": eta,
        "sigma_low": sigma_low,
        "sigma_high": sigma_high,
        "kappa_sgd_low": kappa_sgd_low,
        "kappa_sgd_high": kappa_sgd_high,
        "kappa_norm_low": kappa_norm_low,
        "kappa_norm_high": kappa_norm_high,
        "kappa_norm_ratio_low_over_high": kappa_norm_low / max(kappa_norm_high, EPS),
    }


def main() -> None:
    out_dir = Path("figures_noise_dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260414)

    summary = {
        "experiment_1_direction_invariance": experiment_direction_invariance(rng, out_dir),
        "experiment_2_and_3_noise_eta_scaling": experiment_noise_and_eta_scaling(rng, out_dir),
        "experiment_4_steady_state_scaling": experiment_steady_state_scaling(rng, out_dir),
        "experiment_5_heteroscedastic_blocks": experiment_heteroscedastic_blocks(rng, out_dir),
    }

    summary_path = out_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved figures and metrics to:", out_dir.resolve())
    print("Summary metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
