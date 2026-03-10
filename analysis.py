#!/usr/bin/env python3

import os, glob, re, ast, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "Dataset", "Attention Task Validation", "data_brsm")
FIG   = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
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
                    rows.append({
                        "participant": pid,
                        "group": group_label,
                        "modality": "Lab",
                        "trial": trial_idx,
                        "rt_s": rt,
                        "target_color": str(color).strip(),
                        "correct": 1
                    })
                    trial_idx += 1
            else:
                first_rt = rts[0] if rts else np.nan
                rows.append({
                    "participant": pid,
                    "group": group_label,
                    "modality": "Lab",
                    "trial": trial_idx,
                    "rt_s": first_rt,
                    "target_color": str(color).strip(),
                    "correct": 1
                })
                trial_idx += 1
    return pd.DataFrame(rows)


def load_phone_data(group_dir, group_label):
    rows = []
    for fp in sorted(glob.glob(os.path.join(group_dir, "*.csv"))):
        fname = os.path.basename(fp)
        pid = int(fname.split("_")[0])
        df = pd.read_csv(fp)
        for _, r in df.iterrows():
            rows.append({
                "participant": pid,
                "group": group_label,
                "modality": "Game",
                "level": int(r["Level"]),
                "rt_ms": float(r["InitialResponseTime(ms)"]),
                "success_rate": float(r["SuccessRate(%)"]),
                "hit_rate": float(r["HitRate(%)"]),
                "false_alarms": int(r["FalseAlarms"]),
                "completed": str(r["Completed"]).strip().lower() == "true",
            })
    return pd.DataFrame(rows)


lab_single   = load_lab_data(os.path.join(DATA, "single", "lab"),   "Single")
lab_multi    = load_lab_data(os.path.join(DATA, "multiple", "lab"), "Multiple")
phone_single = load_phone_data(os.path.join(DATA, "single", "phone"),   "Single")
phone_multi  = load_phone_data(os.path.join(DATA, "multiple", "phone"), "Multiple")

lab_all   = pd.concat([lab_single, lab_multi], ignore_index=True)
phone_all = pd.concat([phone_single, phone_multi], ignore_index=True)

phone_all = phone_all[phone_all["level"] <= 10].copy()
print(f"\n[INFO] Filtered phone data to levels 1-10: {len(phone_all)} rows retained.")

lab_all["rt_ms"] = lab_all["rt_s"] * 1000.0


lab_summary = (
    lab_all
    .groupby(["participant", "group"])
    .agg(mean_rt_ms=("rt_ms", "mean"),
         accuracy=("correct", "mean"),
         n_trials=("correct", "count"))
    .reset_index()
)
lab_summary["modality"] = "Lab"

phone_completed = phone_all[phone_all["rt_ms"] > 0].copy()
phone_summary = (
    phone_completed
    .groupby(["participant", "group"])
    .agg(mean_rt_ms=("rt_ms", "mean"),
         accuracy=("hit_rate", "mean"),
         n_trials=("hit_rate", "count"))
    .reset_index()
)
phone_summary["modality"] = "Game"

summary = pd.concat([lab_summary, phone_summary], ignore_index=True)
summary["accuracy_pct"] = summary["accuracy"]
mask_lab = summary["modality"] == "Lab"
summary.loc[mask_lab, "accuracy_pct"] = summary.loc[mask_lab, "accuracy"] * 100.0

print("\n" + "=" * 65)
print("PARTICIPANT-LEVEL SUMMARY (first 10 rows)")
print("=" * 65)
print(summary.head(10).to_string(index=False))


desc = (
    summary
    .groupby(["group", "modality"])
    .agg(
        N=("mean_rt_ms", "count"),
        RT_Mean=("mean_rt_ms", "mean"),
        RT_SD=("mean_rt_ms", "std"),
        Acc_Mean=("accuracy_pct", "mean"),
        Acc_SD=("accuracy_pct", "std"),
    )
    .round(2)
    .reset_index()
)

print("\n" + "=" * 65)
print("DESCRIPTIVE STATISTICS  (per cell)")
print("=" * 65)
print(desc.to_string(index=False))
desc.to_csv(os.path.join(FIG, "descriptive_stats.csv"), index=False)


print("\n" + "=" * 65)
print("RQ1 – CONCURRENT VALIDITY  (Game ↔ Lab correlations)")
print("=" * 65)

piv = summary.pivot_table(index=["participant", "group"],
                           columns="modality",
                           values=["mean_rt_ms", "accuracy_pct"]).reset_index()
piv.columns = ["_".join(c).strip("_") for c in piv.columns]

corr_results = []
for grp in ["Single", "Multiple"]:
    sub = piv[piv["group"] == grp].dropna()
    if len(sub) < 3:
        continue
    r_rt, p_rt = stats.pearsonr(sub["mean_rt_ms_Lab"], sub["mean_rt_ms_Game"])
    rho_rt, prho_rt = stats.spearmanr(sub["mean_rt_ms_Lab"], sub["mean_rt_ms_Game"])
    r_acc, p_acc = stats.pearsonr(sub["accuracy_pct_Lab"], sub["accuracy_pct_Game"])
    rho_acc, prho_acc = stats.spearmanr(sub["accuracy_pct_Lab"], sub["accuracy_pct_Game"])

    print(f"\n--- {grp} target group (n = {len(sub)}) ---")
    print(f"  RT:       Pearson r = {r_rt:.3f}  (p = {p_rt:.4f}),  Spearman ρ = {rho_rt:.3f}  (p = {prho_rt:.4f})")
    print(f"  Accuracy: Pearson r = {r_acc:.3f}  (p = {p_acc:.4f}),  Spearman ρ = {rho_acc:.3f}  (p = {prho_acc:.4f})")

    corr_results.append({"Group": grp, "Metric": "RT", "Pearson_r": round(r_rt, 3),
                          "p_pearson": round(p_rt, 4), "Spearman_rho": round(rho_rt, 3),
                          "p_spearman": round(prho_rt, 4), "n": len(sub)})
    corr_results.append({"Group": grp, "Metric": "Accuracy", "Pearson_r": round(r_acc, 3),
                          "p_pearson": round(p_acc, 4), "Spearman_rho": round(rho_acc, 3),
                          "p_spearman": round(prho_acc, 4), "n": len(sub)})

pd.DataFrame(corr_results).to_csv(os.path.join(FIG, "rq1_correlations.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, grp in enumerate(["Single", "Multiple"]):
    sub = piv[piv["group"] == grp].dropna()
    ax = axes[i]
    ax.scatter(sub["mean_rt_ms_Lab"], sub["mean_rt_ms_Game"],
               color=PALETTE[grp], s=60, edgecolor="white", linewidth=0.5)
    if len(sub) > 2:
        z = np.polyfit(sub["mean_rt_ms_Lab"], sub["mean_rt_ms_Game"], 1)
        xs = np.linspace(sub["mean_rt_ms_Lab"].min(), sub["mean_rt_ms_Lab"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "--", color=PALETTE[grp], alpha=0.7)
        r_val = stats.pearsonr(sub["mean_rt_ms_Lab"], sub["mean_rt_ms_Game"])[0]
        ax.set_title(f"{grp} Target  (r = {r_val:.2f})", fontweight="bold")
    ax.set_xlabel("Lab Mean RT (ms)")
    ax.set_ylabel("Game Mean RT (ms)")
plt.suptitle("RQ1 – Concurrent Validity: Lab vs Game RT", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq1_correlation_scatter.png"), dpi=200, bbox_inches="tight")
plt.close()


print("\n" + "=" * 65)
print("RQ2 – TARGET LOAD EFFECT  (Single vs Multiple)")
print("=" * 65)

rq2_results = []
for mod in ["Lab", "Game"]:
    sub = summary[summary["modality"] == mod]
    single = sub[sub["group"] == "Single"]["mean_rt_ms"]
    multi  = sub[sub["group"] == "Multiple"]["mean_rt_ms"]
    t_val, p_val = stats.ttest_ind(single, multi, equal_var=False)
    d = pg.compute_effsize(single, multi, eftype="cohen")
    print(f"\n  {mod}:  t = {t_val:.3f},  p = {p_val:.4f},  Cohen's d = {d:.3f}")
    print(f"         Single M = {single.mean():.1f}  SD = {single.std():.1f}")
    print(f"         Multiple M = {multi.mean():.1f}  SD = {multi.std():.1f}")
    rq2_results.append({"Modality": mod, "t": round(t_val, 3), "p": round(p_val, 4),
                          "Cohen_d": round(d, 3),
                          "Single_M": round(single.mean(), 1), "Single_SD": round(single.std(), 1),
                          "Multiple_M": round(multi.mean(), 1), "Multiple_SD": round(multi.std(), 1)})

pd.DataFrame(rq2_results).to_csv(os.path.join(FIG, "rq2_target_load.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=summary, x="modality", y="mean_rt_ms", hue="group",
            palette=PALETTE, ax=ax, ci=95, capsize=0.1, edgecolor="black", linewidth=0.8)
ax.set_xlabel("Modality")
ax.set_ylabel("Mean RT (ms)")
ax.set_title("RQ2 – Target Load Effect on Reaction Time", fontweight="bold")
ax.legend(title="Target Load")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq2_target_load_bar.png"), dpi=200, bbox_inches="tight")
plt.close()


print("\n" + "=" * 65)
print("RQ3 – MODALITY EFFECT  (Game vs Lab, paired)")
print("=" * 65)

rq3_results = []
for grp in ["Single", "Multiple"]:
    sub = summary[summary["group"] == grp]
    lab_rt  = sub[sub["modality"] == "Lab"].set_index("participant")["mean_rt_ms"]
    game_rt = sub[sub["modality"] == "Game"].set_index("participant")["mean_rt_ms"]
    common = lab_rt.index.intersection(game_rt.index)
    if len(common) < 3:
        continue
    t_val, p_val = stats.ttest_rel(lab_rt[common], game_rt[common])
    d = pg.compute_effsize(lab_rt[common], game_rt[common], paired=True, eftype="cohen")
    print(f"\n  {grp} group (n = {len(common)}):")
    print(f"    Lab  M = {lab_rt[common].mean():.1f}  SD = {lab_rt[common].std():.1f}")
    print(f"    Game M = {game_rt[common].mean():.1f}  SD = {game_rt[common].std():.1f}")
    print(f"    t = {t_val:.3f},  p = {p_val:.4f},  Cohen's d = {d:.3f}")
    rq3_results.append({"Group": grp, "t": round(t_val, 3), "p": round(p_val, 4),
                          "Cohen_d": round(d, 3), "n": len(common),
                          "Lab_M": round(lab_rt[common].mean(), 1),
                          "Game_M": round(game_rt[common].mean(), 1)})

pd.DataFrame(rq3_results).to_csv(os.path.join(FIG, "rq3_modality.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=summary, x="group", y="mean_rt_ms", hue="modality",
               split=True, palette=MOD_PAL, inner="quartile", ax=ax)
ax.set_xlabel("Target Load Group")
ax.set_ylabel("Mean RT (ms)")
ax.set_title("RQ3 – Modality Effect (Lab vs Game)", fontweight="bold")
ax.legend(title="Modality")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq3_modality_violin.png"), dpi=200, bbox_inches="tight")
plt.close()


print("\n" + "=" * 65)
print("RQ4 – PROGRESSION EFFECT  (Game level → RT)")
print("=" * 65)

prog = phone_completed.copy()
prog_agg = prog.groupby(["group", "level"]).agg(
    mean_rt=("rt_ms", "mean"),
    sd_rt=("rt_ms", "std"),
    n=("rt_ms", "count")
).reset_index()

for grp in ["Single", "Multiple"]:
    sub = prog[prog["group"] == grp]
    r, p = stats.pearsonr(sub["level"], sub["rt_ms"])
    print(f"  {grp}: Pearson r(level, RT) = {r:.3f},  p = {p:.4f},  N_trials = {len(sub)}")

fig, ax = plt.subplots(figsize=(9, 5))
for grp in ["Single", "Multiple"]:
    sub = prog_agg[prog_agg["group"] == grp]
    ax.errorbar(sub["level"], sub["mean_rt"], yerr=sub["sd_rt"] / np.sqrt(sub["n"]),
                marker="o", label=grp, color=PALETTE[grp], capsize=3, linewidth=1.5)
ax.set_xlabel("Game Level")
ax.set_ylabel("Mean Initial RT (ms)")
ax.set_title("RQ4 – Progression Effect: RT Across Game Levels", fontweight="bold")
ax.legend(title="Target Load")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rq4_progression.png"), dpi=200, bbox_inches="tight")
plt.close()


print("\n" + "=" * 65)
print("MIXED ANOVA  (Target Load × Modality on RT)")
print("=" * 65)

anova_df = summary.copy()
counts = anova_df.groupby("participant")["modality"].nunique()
valid_pids = counts[counts == 2].index
anova_df = anova_df[anova_df["participant"].isin(valid_pids)].copy()

aov = pg.mixed_anova(
    data=anova_df,
    dv="mean_rt_ms",
    within="modality",
    between="group",
    subject="participant",
)
print(aov.to_string(index=False))
aov.to_csv(os.path.join(FIG, "mixed_anova.csv"), index=False)

print("\n--- Accuracy ---")
aov_acc = pg.mixed_anova(
    data=anova_df,
    dv="accuracy_pct",
    within="modality",
    between="group",
    subject="participant",
)
print(aov_acc.to_string(index=False))
aov_acc.to_csv(os.path.join(FIG, "mixed_anova_accuracy.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cell_means = anova_df.groupby(["group", "modality"])["mean_rt_ms"].agg(["mean", "sem"]).reset_index()
for grp in ["Single", "Multiple"]:
    sub = cell_means[cell_means["group"] == grp]
    axes[0].errorbar(sub["modality"], sub["mean"], yerr=sub["sem"],
                     marker="o", label=grp, color=PALETTE[grp], linewidth=2, capsize=4)
axes[0].set_title("Interaction Plot – RT", fontweight="bold")
axes[0].set_ylabel("Mean RT (ms)")
axes[0].set_xlabel("Modality")
axes[0].legend(title="Target Load")

cell_acc = anova_df.groupby(["group", "modality"])["accuracy_pct"].agg(["mean", "sem"]).reset_index()
for grp in ["Single", "Multiple"]:
    sub = cell_acc[cell_acc["group"] == grp]
    axes[1].errorbar(sub["modality"], sub["mean"], yerr=sub["sem"],
                     marker="o", label=grp, color=PALETTE[grp], linewidth=2, capsize=4)
axes[1].set_title("Interaction Plot – Accuracy", fontweight="bold")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_xlabel("Modality")
axes[1].legend(title="Target Load")

plt.suptitle("2 × 2 Mixed ANOVA Interaction Plots", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "interaction_plots.png"), dpi=200, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=summary, x="group", y="mean_rt_ms", hue="modality",
            palette=MOD_PAL, ax=axes[0], linewidth=1.2)
axes[0].set_title("RT Distribution by Condition", fontweight="bold")
axes[0].set_xlabel("Target Load")
axes[0].set_ylabel("Mean RT (ms)")
axes[0].legend(title="Modality")

sns.boxplot(data=summary, x="group", y="accuracy_pct", hue="modality",
            palette=MOD_PAL, ax=axes[1], linewidth=1.2)
axes[1].set_title("Accuracy Distribution by Condition", fontweight="bold")
axes[1].set_xlabel("Target Load")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(title="Modality")

plt.suptitle("Overview – RT and Accuracy Across Conditions", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "overview_boxplots.png"), dpi=200, bbox_inches="tight")
plt.close()

print("\n" + "=" * 65)
print("✓  All analyses complete.  Figures saved to:", FIG)
print("=" * 65)
