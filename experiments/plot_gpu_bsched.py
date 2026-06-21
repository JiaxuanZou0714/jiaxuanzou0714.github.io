"""Plot the real-transformer batch-schedule experiment (FineWeb, 45M GPT)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "gpu_logs")
OUT = os.path.abspath(os.path.join(HERE, "..", "assets", "img", "post-06-16"))

runs = [
    ("const_big",   "Constant 512k (large)", "#7f7f7f", "-"),
    ("const_med",   "Constant 128k",         "#9467bd", "-"),
    ("const_small", "Constant 64k (small)",  "#1f77b4", "-"),
    ("ramp",        "Doubling ramp 64k→512k", "#d62728", "-"),
]
data = {k: json.load(open(os.path.join(LOG, k + ".json"))) for k, *_ in runs}
def arr(k, f): return np.array([r[f] for r in data[k]["records"]])

plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 9.5})
M = 1e6
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5.2))

# panel 1: val loss vs tokens (zoom on the informative region)
for k, lab, col, ls in runs:
    a1.plot(arr(k, "tokens") / M, arr(k, "val"), ls, color=col,
            lw=3.0 if k == "ramp" else 1.9, label=lab)
a1.set_xlabel("consumed tokens (millions)"); a1.set_ylabel("val loss")
a1.set_title("Val loss vs tokens (45M GPT, FineWeb, constant lr)")
a1.set_ylim(4.0, 6.0); a1.grid(True, alpha=0.25); a1.legend(frameon=False)

# panel 2: val loss vs optimizer steps (efficiency)
for k, lab, col, ls in runs:
    a2.plot(arr(k, "step"), arr(k, "val"), ls, color=col,
            lw=3.0 if k == "ramp" else 1.9, label=lab)
a2.set_xlabel("optimizer steps"); a2.set_ylabel("val loss")
a2.set_title("Val loss vs optimizer steps"); a2.set_xscale("log")
a2.set_ylim(4.0, 6.0); a2.grid(True, which="both", alpha=0.25); a2.legend(frameon=False)

# panel 3: batch schedules
for k, lab, col, ls in runs:
    a3.plot(arr(k, "tokens") / M, arr(k, "batch"), color=col,
            lw=3.0 if k == "ramp" else 1.9, label=lab, drawstyle="steps-post")
a3.set_xlabel("consumed tokens (millions)"); a3.set_ylabel("batch size (tokens / step)")
a3.set_title("Batch schedules"); a3.set_yscale("log")
a3.grid(True, which="both", alpha=0.25); a3.legend(frameon=False)

fig.tight_layout()
out = os.path.join(OUT, "gpu_batch_schedule.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# ---- report ----
print("\n=== final ===")
for k, lab, *_ in runs:
    r = data[k]["records"][-1]
    print(f"{lab:26s} val={r['val']:.4f}  steps={r['step']:5d}  tokens={r['tokens']/1e6:.0f}M")
best_const = min((data[k]["records"][-1]["val"], k) for k in ["const_big", "const_med", "const_small"])
print(f"best constant: {best_const[1]} val={best_const[0]:.4f}")
# how early did ramp reach the best constant's final loss?
tgt = best_const[0]
hit = next((r for r in data["ramp"]["records"] if r["val"] <= tgt), None)
if hit:
    cs = data[best_const[1]]["records"][-1]
    print(f"ramp reached best-constant final val ({tgt:.3f}) at step {hit['step']} "
          f"({hit['tokens']/1e6:.0f}M tokens) vs {best_const[1]} {cs['step']} steps "
          f"({cs['tokens']/1e6:.0f}M) -> {100*(1-hit['step']/cs['step']):.0f}% fewer steps")
