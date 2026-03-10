#!/usr/bin/env python3

import os, glob, re, ast, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "Dataset", "Attention Task Validation", "data_brsm")
FIG   = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
PALETTE = {"Single": "#3498db", "Multiple": "#e74c3c"}
MOD_PAL = {"Lab": "#2c3e50", "Game": "#27ae60"}


def _parse_lab_rt(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [float(x) for x in parsed]
        return [float(parsed)]
    except Exception:
        pass
    nums = re.findall(r"[\d.]+", s)
    return [float(n) for n in nums]


def load_lab_data(group_dir, group_label):
    rows = []
    for fp in sorted(glob.glob(os.path.join(group_dir, "*.csv"))):
        fname = os.path.basename(fp)
        pid = int(fname.split("_")[0])
        df = pd.read_csv(fp)
        rt_col = None
        for c in df.columns:
            if "mouse.time" in c.lower() and "start" not in c.lower():
                rt_col = c
                break
        if rt_col is None:
            continue
        color_col = "target_col" if "target_col" in df.columns else None
        trial_idx = 0
        for _, r in df.iterrows():
            rts = _parse_lab_rt(r.get(rt_col))
            color = r.get(color_col, None) if color_col else None
            if not rts or (isinstance(color, float) and np.isnan(color)):
                continue
            if pd.isna(color) or str(color).strip() == "":
                continue
            if group_label == "Single":
                for rt in rts:
                    rows.append({"participant": pid, "group": group_label,
                                 "modality": "Lab", "trial": trial_idx,
                                 "rt_s": rt, "correct": 1})
                    trial_idx += 1
            else:
                first_rt = rts[0] if rts else np.nan
                rows.append({"participant": pid, "group": group_label,
                             "modality": "Lab", "trial": trial_idx,
                             "rt_s": first_rt, "correct": 1})
                trial_idx += 1
    return pd.DataFrame(rows)


def load_phone_data(group_dir, group_label):
    rows = []
    for fp in sorted(glob.glob(os.path.join(group_dir, "*.csv"))):
        fname = os.path.basename(fp)
        pid = int(fname.split("_")[0])
        df = pd.read_csv(fp)
        for _, r in df.iterrows():
            rows.append({"participant": pid, "group": group_label,
                         "modality": "Game", "level": int(r["Level"]),
                         "rt_ms": float(r["InitialResponseTime(ms)"]),
                         "hit_rate": float(r["HitRate(%)"]),
                         "false_alarms": int(r["FalseAlarms"])})
    return pd.DataFrame(rows)


lab_single   = load_lab_data(os.path.join(DATA, "single", "lab"),   "Single")
lab_multi    = load_lab_data(os.path.join(DATA, "multiple", "lab"), "Multiple")
phone_single = load_phone_data(os.path.join(DATA, "single", "phone"),   "Single")
phone_multi  = load_phone_data(os.path.join(DATA, "multiple", "phone"), "Multiple")

lab_all   = pd.concat([lab_single, lab_multi], ignore_index=True)
phone_all = pd.concat([phone_single, phone_multi], ignore_index=True)

phone_all = phone_all[phone_all["level"] <= 10].copy()
print(f"[INFO] Filtered phone data to levels 1-10: {len(phone_all)} rows retained.")

lab_all["rt_ms"] = lab_all["rt_s"] * 1000.0

lab_trials = lab_all[["participant", "group", "modality", "rt_ms"]].copy()
phone_trials = phone_all[phone_all["rt_ms"] > 0][["participant", "group", "modality", "rt_ms"]].copy()
all_trials = pd.concat([lab_trials, phone_trials], ignore_index=True)

lab_summary = lab_all.groupby(["participant", "group"]).agg(
    mean_rt_ms=("rt_ms", "mean")).reset_index()
lab_summary["modality"] = "Lab"

phone_summary = phone_all[phone_all["rt_ms"] > 0].groupby(["participant", "group"]).agg(
    mean_rt_ms=("rt_ms", "mean")).reset_index()
phone_summary["modality"] = "Game"

summary = pd.concat([lab_summary, phone_summary], ignore_index=True)

all_trials["condition"] = all_trials["group"] + " – " + all_trials["modality"]
summary["condition"] = summary["group"] + " – " + summary["modality"]

print("Generating RT histograms (4-panel)...")

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
conditions = [
    ("Single", "Lab", axes[0, 0], "#3498db"),
    ("Single", "Game", axes[0, 1], "#2ecc71"),
    ("Multiple", "Lab", axes[1, 0], "#e74c3c"),
    ("Multiple", "Game", axes[1, 1], "#e67e22"),
]

for grp, mod, ax, color in conditions:
    subset = all_trials[(all_trials["group"] == grp) & (all_trials["modality"] == mod)]["rt_ms"]
    ax.hist(subset, bins=25, color=color, alpha=0.75, edgecolor="white", linewidth=0.8)
    ax.axvline(subset.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean = {subset.mean():.0f} ms")
    ax.axvline(subset.median(), color="gray", linestyle=":", linewidth=1.5, label=f"Median = {subset.median():.0f} ms")
    ax.set_title(f"{grp} Target – {mod}", fontweight="bold", fontsize=13)
    ax.set_xlabel("Reaction Time (ms)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9, loc="upper right")

    skew = subset.skew()
    ax.annotate(f"Skew = {skew:.2f}\nN = {len(subset)}",
                xy=(0.97, 0.65), xycoords="axes fraction",
                ha="right", fontsize=9, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle("Distribution of Trial-Level Reaction Times", fontweight="bold", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rt_histograms.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {os.path.join(FIG, 'rt_histograms.png')}")

print("Generating overlaid Lab vs Game histograms...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, grp in enumerate(["Single", "Multiple"]):
    ax = axes[i]
    lab_data = all_trials[(all_trials["group"] == grp) & (all_trials["modality"] == "Lab")]["rt_ms"]
    game_data = all_trials[(all_trials["group"] == grp) & (all_trials["modality"] == "Game")]["rt_ms"]

    ax.hist(lab_data, bins=20, alpha=0.6, color="#2c3e50", edgecolor="white",
            linewidth=0.6, label=f"Lab (M={lab_data.mean():.0f})")
    ax.hist(game_data, bins=20, alpha=0.6, color="#27ae60", edgecolor="white",
            linewidth=0.6, label=f"Game (M={game_data.mean():.0f})")
    ax.set_title(f"{grp} Target Group", fontweight="bold", fontsize=13)
    ax.set_xlabel("Reaction Time (ms)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)

plt.suptitle("Lab vs Game RT Distributions", fontweight="bold", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rt_histograms_overlaid.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {os.path.join(FIG, 'rt_histograms_overlaid.png')}")

print("Generating per-participant strip plot...")

fig, ax = plt.subplots(figsize=(12, 6))
sns.stripplot(data=summary, x="condition", y="mean_rt_ms",
              hue="group", palette=PALETTE, dodge=False,
              size=8, alpha=0.7, jitter=0.2, edgecolor="white", linewidth=0.5, ax=ax)

for cond in summary["condition"].unique():
    m = summary[summary["condition"] == cond]["mean_rt_ms"].mean()
    idx = list(summary["condition"].unique()).index(cond)
    ax.hlines(m, idx - 0.25, idx + 0.25, color="black", linewidth=2.5, zorder=10)

ax.set_xlabel("Condition")
ax.set_ylabel("Mean RT per Participant (ms)")
ax.set_title("Individual Participant RTs Across Conditions", fontweight="bold", fontsize=14)
ax.legend(title="Target Load", loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "participant_rt_strip.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {os.path.join(FIG, 'participant_rt_strip.png')}")

print("Generating log(RT) histograms...")

fig_log, axes_log = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
log_conditions = [
    ("Single", "Lab", axes_log[0, 0], "#3498db"),
    ("Single", "Game", axes_log[0, 1], "#2ecc71"),
    ("Multiple", "Lab", axes_log[1, 0], "#e74c3c"),
    ("Multiple", "Game", axes_log[1, 1], "#e67e22"),
]

for grp, mod, ax, color in log_conditions:
    subset = all_trials[(all_trials["group"] == grp) & (all_trials["modality"] == mod)]["rt_ms"]
    log_rt = np.log(subset)
    ax.hist(log_rt, bins=25, color=color, alpha=0.75, edgecolor="white", linewidth=0.8)
    ax.axvline(log_rt.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean = {log_rt.mean():.2f}")
    ax.axvline(log_rt.median(), color="gray", linestyle=":", linewidth=1.5, label=f"Median = {log_rt.median():.2f}")
    ax.set_title(f"{grp} Target – {mod}", fontweight="bold", fontsize=13)
    ax.set_xlabel("log(RT) [log ms]")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9, loc="upper right")

    skew = log_rt.skew()
    ax.annotate(f"Skew = {skew:.2f}\nN = {len(subset)}",
                xy=(0.97, 0.65), xycoords="axes fraction",
                ha="right", fontsize=9, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

fig_log.suptitle("Distribution of log(RT) — Assessing Normality After Transformation",
                 fontweight="bold", fontsize=14, y=1.01)
fig_log.tight_layout()
fig_log.savefig(os.path.join(FIG, "rt_log_histograms.png"), dpi=200, bbox_inches="tight")
plt.close(fig_log)
print(f"  Saved: {os.path.join(FIG, 'rt_log_histograms.png')}")

print("Generating paired difference plot...")

fig, ax = plt.subplots(figsize=(10, 6))
for grp, color in PALETTE.items():
    sub = summary[summary["group"] == grp]
    lab_rt = sub[sub["modality"] == "Lab"].set_index("participant")["mean_rt_ms"]
    game_rt = sub[sub["modality"] == "Game"].set_index("participant")["mean_rt_ms"]
    common = lab_rt.index.intersection(game_rt.index)
    diff = game_rt[common] - lab_rt[common]

    ax.bar(range(len(diff)), diff.values, color=color, alpha=0.7, edgecolor="white",
           label=f"{grp} (Mean Δ = {diff.mean():.0f} ms)")

ax.axhline(0, color="black", linewidth=1, linestyle="-")
ax.set_xlabel("Participant (index)")
ax.set_ylabel("Game RT − Lab RT (ms)")
ax.set_title("Per-Participant Modality Effect (Game − Lab)", fontweight="bold", fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "paired_difference_bar.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {os.path.join(FIG, 'paired_difference_bar.png')}")

print("\n✓ All additional visualizations generated.")
