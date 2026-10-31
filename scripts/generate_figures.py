import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Publication-quality CVPR styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

NAVY = "#1B365D"
TEAL = "#008080"
CORAL = "#D9534F"
AMBER = "#D9822B"
SLATE = "#5C6B73"
PURPLE = "#6A4C93"

os.makedirs("figures", exist_ok=True)

with open("logs/experiment_results.json", "r") as f:
    res = json.load(f)

d = np.load("data/olivetti_faces.npz")
images_all = d["images"]
y_all = d["target"]
sp = np.load("data/split_P26CS0001.npz")
train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]
images_test = images_all[test_idx]
y_test = y_all[test_idx]

def save_both(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved figures/{name}.pdf and .png")

# ---------------------------------------------------------
# Figure 1: Sample Clean vs. Masked Olivetti Faces (rows/cols 24..39 zeroed)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 6, figsize=(8.2, 3.0))
sample_test_positions = [0, 2, 11, 20, 29, 44]
for col, tpos in enumerate(sample_test_positions):
    img_c = images_test[tpos]
    img_m = img_c.copy()
    img_m[24:40, 24:40] = 0.0
    cid = y_test[tpos]
    gidx = test_idx[tpos]

    ax_top = axes[0, col]
    ax_top.imshow(img_c, cmap="gray", vmin=0.0, vmax=1.0)
    rect = patches.Rectangle((23.5, 23.5), 16, 16, linewidth=1.0, edgecolor=CORAL, facecolor="none", linestyle="--")
    ax_top.add_patch(rect)
    ax_top.set_title(f"ID {cid} (#{gidx})", fontsize=8.5, pad=3)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    if col == 0:
        ax_top.set_ylabel("Clean Test", fontweight="bold", fontsize=9)

    ax_bot = axes[1, col]
    ax_bot.imshow(img_m, cmap="gray", vmin=0.0, vmax=1.0)
    rect2 = patches.Rectangle((23.5, 23.5), 16, 16, linewidth=1.0, edgecolor=CORAL, facecolor="none")
    ax_bot.add_patch(rect2)
    ax_bot.set_xticks([])
    ax_bot.set_yticks([])
    if col == 0:
        ax_bot.set_ylabel("Masked (24:39)", fontweight="bold", fontsize=9)

plt.subplots_adjust(wspace=0.08, hspace=0.12)
save_both(fig, "fig1_dataset_and_mask")

# ---------------------------------------------------------
# Figure 2: Q1(a) Decision Tree Baseline Depth & Leaf Analysis
# ---------------------------------------------------------
q1a = res["q1"]["q1a_grid"]
depths_labels = ["2", "4", "8", "16", "None"]
x_pos = np.arange(len(depths_labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.1))

# Left: Accuracy curves for min_samples_leaf=1 (solid) and min_samples_leaf=4 (dashed)
msl1 = [r for r in q1a if r["min_samples_leaf"] == 1]
msl2 = [r for r in q1a if r["min_samples_leaf"] == 2]
msl4 = [r for r in q1a if r["min_samples_leaf"] == 4]

ax1.plot(x_pos, [r["train_acc"]*100 for r in msl1], "-o", color=NAVY, lw=1.8, label="Train (msl=1)")
ax1.plot(x_pos, [r["val_acc"]*100 for r in msl1], "-s", color=TEAL, lw=2.0, label="Validation (msl=1)")
ax1.plot(x_pos, [r["test_acc"]*100 for r in msl1], "-^", color=CORAL, lw=1.8, label="Clean Test (msl=1)")
ax1.plot(x_pos, [r["val_acc"]*100 for r in msl4], "--s", color=SLATE, lw=1.3, alpha=0.85, label="Validation (msl=4)")

# Highlight frozen setting (max_depth=16, msl=1)
ax1.scatter([3], [msl1[3]["val_acc"]*100], s=110, facecolors="none", edgecolors=AMBER, lw=2.2, zorder=6, label="Frozen (depth=16, msl=1)")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(depths_labels)
ax1.set_xlabel("Maximum Tree Depth (max_depth)")
ax1.set_ylabel("Classification Accuracy (%)")
ax1.set_title("(a) Single Tree Accuracy vs. Depth")
ax1.set_ylim(0, 106)
ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

# Right: Number of leaves & singleton leaf fraction
ax2_twin = ax2.twinx()
ax2_twin.spines["top"].set_visible(False)
ax2_twin.grid(False)

w_bar = 0.25
ax2.bar(x_pos - w_bar, [r["n_leaves"] for r in msl1], width=w_bar, color=NAVY, alpha=0.88, label="Leaves (msl=1)")
ax2.bar(x_pos,         [r["n_leaves"] for r in msl2], width=w_bar, color=TEAL, alpha=0.82, label="Leaves (msl=2)")
ax2.bar(x_pos + w_bar, [r["n_leaves"] for r in msl4], width=w_bar, color=SLATE, alpha=0.78, label="Leaves (msl=4)")

ax2_twin.plot(x_pos, [r["frac_leaves_one_sample"]*100 for r in msl1], "-D", color=CORAL, lw=2.0, markersize=5.5, label="Singleton Leaf % (msl=1)")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(depths_labels)
ax2.set_xlabel("Maximum Tree Depth (max_depth)")
ax2.set_ylabel("Number of Terminal Leaves", color=NAVY)
ax2_twin.set_ylabel("Leaves with 1 Sample (%)", color=CORAL)
ax2.set_ylim(0, 56)
ax2_twin.set_ylim(0, 22)
ax2.set_title("(b) Tree Capacity & Singleton Leaf Fraction")

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, framealpha=0.9, fontsize=7.8)

plt.tight_layout()
save_both(fig, "fig2_q1a_tree_depth_analysis")

# ---------------------------------------------------------
# Figure 3: Q1(b)-(c) Bagging Accuracy, OOB, Disagreement & Vote Margin vs. B
# ---------------------------------------------------------
q1bc = res["q1"]["q1bc_bagging"]
B_vals = [r["B"] for r in q1bc]
x_b = np.arange(len(B_vals))
b_labels = [str(b) for b in B_vals]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.15))

ax1.plot(x_b, [r["train_acc"]*100 for r in q1bc], "-o", color=NAVY, lw=1.7, label="Train Accuracy")
ax1.plot(x_b, [r["val_acc"]*100 for r in q1bc], "-s", color=TEAL, lw=2.0, label="Validation Accuracy")
ax1.plot(x_b, [r["test_acc"]*100 for r in q1bc], "-^", color=CORAL, lw=2.0, label="Clean Test Accuracy")
ax1.plot(x_b, [r["oob_acc"]*100 for r in q1bc], "--d", color=PURPLE, lw=1.7, label="Out-of-Bag (OOB) Accuracy")
ax1.axvline(x=5, color=AMBER, linestyle=":", lw=1.6, label="Selected B* = 100")

ax1.set_xticks(x_b)
ax1.set_xticklabels(b_labels)
ax1.set_xlabel("Ensemble Size B (Number of Bootstrap Trees)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("(a) Bagging Accuracy Scaling vs. B")
ax1.set_ylim(15, 105)
ax1.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7.8)

# Right: Required Q1(c) Triple Plot (OOB Accuracy, Pairwise Disagreement, Mean Vote Margin vs B)
ax2.plot(x_b, [r["oob_acc"] for r in q1bc], "-o", color=PURPLE, lw=2.0, label="Out-of-Bag (OOB) Accuracy")
ax2.plot(x_b, [r["pairwise_disagreement_val"] for r in q1bc], "-s", color=CORAL, lw=1.9, label="Pairwise Tree Disagreement (Val)")
ax2.plot(x_b, [r[" pairwise_disagreement_test"] if " pairwise_disagreement_test" in r else r["pairwise_disagreement_test"] for r in q1bc], "--s", color=CORAL, lw=1.3, alpha=0.75, label="Pairwise Tree Disagreement (Test)")
ax2.plot(x_b, [r["mean_vote_margin_val"] for r in q1bc], "-^", color=TEAL, lw=1.9, label="Mean Vote Margin (Val)")
ax2.plot(x_b, [r["mean_vote_margin_test"] for r in q1bc], "--^", color=TEAL, lw=1.3, alpha=0.75, label="Mean Vote Margin (Test)")
ax2.axvline(x=5, color=AMBER, linestyle=":", lw=1.6)

ax2.set_xticks(x_b)
ax2.set_xticklabels(b_labels)
ax2.set_xlabel("Ensemble Size B (Number of Bootstrap Trees)")
ax2.set_ylabel("Metric Value (Fraction in [0, 1])")
ax2.set_title("(b) OOB Accuracy, Disagreement & Vote Margin")
ax2.set_ylim(-0.03, 1.05)
ax2.legend(loc="center right", frameon=True, framealpha=0.9, fontsize=7.4)

plt.tight_layout()
save_both(fig, "fig3_q1bc_bagging_curves")

# ---------------------------------------------------------
# Figure 4: Q1(d) Four Diagnostic Face Cases (2 Corrected, 2 Both Wrong)
# ---------------------------------------------------------
cases_corr = res["q1"]["q1d_cases"]["bagging_corrects_tree"][:2]
cases_wrong = res["q1"]["q1d_cases"]["both_wrong"][:2]
all_4_cases = [
    ("Bagging Corrects #1", cases_corr[0], TEAL),
    ("Bagging Corrects #2", cases_corr[1], TEAL),
    ("Both Wrong #1", cases_wrong[0], CORAL),
    ("Both Wrong #2", cases_wrong[1], CORAL),
]

fig, axes = plt.subplots(2, 4, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [1.0, 1.45, 1.0, 1.45]})
for idx_case, (tag, cinfo, col_accent) in enumerate(all_4_cases):
    row = idx_case // 2
    col_base = (idx_case % 2) * 2
    ax_img = axes[row, col_base]
    ax_bar = axes[row, col_base + 1]

    tpos = cinfo["test_pos"]
    ax_img.imshow(images_test[tpos], cmap="gray", vmin=0.0, vmax=1.0)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"{tag}\nTrue ID={cinfo['true_identity']} | Tree->{cinfo['single_tree_pred']}", fontsize=8.2, color=NAVY)

    top5 = cinfo["bagging_top5_votes"][:5]
    cls_labels = [f"ID {v['class']}" + ("*" if v["class"] == cinfo["true_identity"] else "") for v in top5][::-1]
    vote_vals = [v["votes"] for v in top5][::-1]
    bar_colors = [TEAL if v["class"] == cinfo["true_identity"] else (CORAL if v["class"] == cinfo["single_tree_pred"] else SLATE) for v in top5][::-1]

    y_p = np.arange(len(cls_labels))
    bars = ax_bar.barh(y_p, vote_vals, color=bar_colors, height=0.62, alpha=0.9)
    for b_rect, v_cnt in zip(bars, vote_vals):
        ax_bar.text(v_cnt + 1.2, b_rect.get_y() + b_rect.get_height()/2, f"{v_cnt}", va="center", fontsize=7.5)
    ax_bar.set_yticks(y_p)
    ax_bar.set_yticklabels(cls_labels, fontsize=7.8)
    ax_bar.set_xlim(0, 56)
    ax_bar.set_xlabel("Votes (out of B=100)", fontsize=7.8)
    ax_bar.set_title(f"Bagging Pred: ID {cinfo['bagging_pred']} ({cinfo['bagging_true_class_votes']} true votes)", fontsize=8.0)

plt.tight_layout()
save_both(fig, "fig4_q1d_case_studies")

# ---------------------------------------------------------
# Figure 5: Q2(b) SAMME Trajectories (Selected Model: depth=4, T=50)
# ---------------------------------------------------------
traj = res["q2"]["selected_boosting_trajectories"]
rounds_x = np.arange(1, len(traj["eps_history"]) + 1)

fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.2))

ax = axes[0, 0]
ax.plot(rounds_x, traj["eps_history"], "-o", color=CORAL, markersize=3.0, lw=1.5, label=r"Weighted Error $\epsilon_t$")
ax.axhline(y=39.0/40.0, color=SLATE, linestyle="--", lw=1.2, label="Random Guess (39/40 = 0.975)")
ax.set_xlabel("Boosting Round t")
ax.set_ylabel(r"Weighted Training Error $\epsilon_t$")
ax.set_title(r"(a) Weak Learner Weighted Error $\epsilon_t$")
ax.legend(loc="lower right", fontsize=7.5)

ax = axes[0, 1]
ax.plot(rounds_x, traj["alpha_history"], "-s", color=NAVY, markersize=3.0, lw=1.5, label=r"Learner Weight $\alpha_t$")
ax.set_xlabel("Boosting Round t")
ax.set_ylabel(r"SAMME Weight $\alpha_t$")
ax.set_title(r"(b) Weak Learner Weight $\alpha_t$ over Rounds")
ax.legend(loc="upper right", fontsize=7.5)

ax = axes[1, 0]
ax.plot(rounds_x, traj["max_w_history"], "-^", color=AMBER, markersize=3.2, lw=1.6, label=r"$\max_i w_i^{(t)}$")
ax.axhline(y=1.0/240.0, color=SLATE, linestyle=":", lw=1.2, label="Uniform Weight (1/240 = 0.0042)")
ax.set_xlabel("Boosting Round t")
ax.set_ylabel("Maximum Sample Weight")
ax.set_title("(c) Maximum Single Sample Weight vs. Round")
ax.legend(loc="upper right", fontsize=7.5)

ax = axes[1, 1]
ax.plot(rounds_x, traj["top10_w_history"], "-D", color=PURPLE, markersize=3.0, lw=1.6, label="Top-10 Heaviest Samples")
ax.axhline(y=10.0/240.0, color=SLATE, linestyle=":", lw=1.2, label="Uniform Top-10 (10/240 = 0.0417)")
ax.set_xlabel("Boosting Round t")
ax.set_ylabel("Cumulative Weight of Top-10")
ax.set_title("(d) Total Weight of 10 Heaviest Training Images")
ax.legend(loc="lower right", fontsize=7.5)

plt.tight_layout()
save_both(fig, "fig5_q2b_samme_dynamics")

# ---------------------------------------------------------
# Figure 6: Q2(c) Hardest Training Face & Q2(d) Differential Mask Response
# ---------------------------------------------------------
hard_info = res["q2"]["q2c_hard_sample"]
diff_case = res["q2"]["q2d_clean_vs_masked"]["differential_mask_cases"][0]

fig = plt.figure(figsize=(8.4, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.55, 1.35], wspace=0.30)

# Panel (a): Hardest training face & normal peer
ax0 = fig.add_subplot(gs[0, 0])
hard_img = images_all[hard_info["global_dataset_idx"]]
ax0.imshow(hard_img, cmap="gray", vmin=0.0, vmax=1.0)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title(f"(a) Hardest Training Face\nID {hard_info['true_identity']} (Train #{hard_info['train_pos']}, Global #{hard_info['global_dataset_idx']})\nMisclassified {hard_info['times_misclassified']}/{hard_info['total_rounds']} rounds", fontsize=8.3)

# Panel (b): Sample weight trajectory & error bars for hardest training face
ax1 = fig.add_subplot(gs[0, 1])
r_idx = np.arange(1, hard_info["total_rounds"] + 1)
w_traj = np.array(hard_info["weight_by_round"])
corr_flags = np.array(hard_info["correct_by_round"])
ax1.plot(r_idx, w_traj, "-", color=NAVY, lw=1.5, label=r"Sample Weight $w_{149}^{(t)}$")
ax1.scatter(r_idx[corr_flags == 0], w_traj[corr_flags == 0], color=CORAL, s=18, zorder=5, label="Misclassified Round")
ax1.scatter(r_idx[corr_flags == 1], w_traj[corr_flags == 1], color=TEAL, marker="s", s=26, zorder=6, label="Correctly Classified")
ax1.set_xlabel("SAMME Round t")
ax1.set_ylabel(r"Normalised Sample Weight $w_i^{(t)}$")
ax1.set_title("(b) Sawtooth Sample-Weight Trajectory", fontsize=9.0)
ax1.legend(loc="upper right", fontsize=7.3)

# Panel (c): Q2(d) Clean vs Masked Accuracy & Differential Case
ax2 = fig.add_subplot(gs[0, 2])
models_lbl = ["Single Tree", "Bagging\n(B=100)", "SAMME\n(d=4, T=50)"]
clean_vals = [
    res["q2"]["q2d_clean_vs_masked"]["single_tree"]["clean_acc"]*100,
    res["q2"]["q2d_clean_vs_masked"]["bagging"]["clean_acc"]*100,
    res["q2"]["q2d_clean_vs_masked"]["boosting"]["clean_acc"]*100,
]
mask_vals = [
    res["q2"]["q2d_clean_vs_masked"]["single_tree"]["masked_acc"]*100,
    res["q2"]["q2d_clean_vs_masked"]["bagging"]["masked_acc"]*100,
    res["q2"]["q2d_clean_vs_masked"]["boosting"]["masked_acc"]*100,
]
xm = np.arange(3)
bw = 0.34
b1 = ax2.bar(xm - bw/2, clean_vals, width=bw, color=NAVY, label="Clean Test")
b2 = ax2.bar(xm + bw/2, mask_vals, width=bw, color=CORAL, label="16x16 Masked")
for rect in b1:
    ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.2, f"{rect.get_height():.1f}%", ha="center", fontsize=7.2)
for rect in b2:
    ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.2, f"{rect.get_height():.1f}%", ha="center", fontsize=7.2)
ax2.set_xticks(xm)
ax2.set_xticklabels(models_lbl, fontsize=8.0)
ax2.set_ylabel("Test Accuracy (%)")
ax2.set_ylim(0, 108)
ax2.set_title("(c) Clean vs. 16x16 Masked Accuracy", fontsize=9.0)
ax2.legend(loc="upper right", fontsize=7.5)

plt.tight_layout()
save_both(fig, "fig6_q2cd_hard_face_and_mask")

# ---------------------------------------------------------
# Figure 7: Q3 Bias-Variance Decomposition Across Models & Depths
# ---------------------------------------------------------
q3s = res["q3"]["q3b_single_by_depth"]
q3b = res["q3"]["q3b_bag100_by_depth"]
q3_bst = res["q3"]["q3a_selected_models"]["boosting"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2))

labels_bv = ["Single\n(d=4)", "Bag100\n(d=4)", "Single\n(d=16)", "Bag100\n(d=16)", "Single\n(None)", "Bag100\n(None)", "SAMME\n(d=4,T=50)"]
bias_vals = [
    q3s["4"]["bias_sq"], q3b["4"]["bias_sq"],
    q3s["16"]["bias_sq"], q3b["16"]["bias_sq"],
    q3s["None"]["bias_sq"], q3b["None"]["bias_sq"],
    q3_bst["bias_sq"]
]
var_vals = [
    q3s["4"]["variance"], q3b["4"]["variance"],
    q3s["16"]["variance"], q3b["16"]["variance"],
    q3s["None"]["variance"], q3b["None"]["variance"],
    q3_bst["variance"]
]

xbv = np.arange(len(labels_bv))
ax1.bar(xbv, bias_vals, width=0.55, color=NAVY, label=r"Measured $\mathrm{Bias}^2$")
ax1.bar(xbv, var_vals, width=0.55, bottom=bias_vals, color=CORAL, label=r"Measured $\mathrm{Variance}$")
ax1.set_xticks(xbv)
ax1.set_xticklabels(labels_bv, fontsize=7.6)
ax1.set_ylabel(r"Squared Error ($\mathrm{Bias}^2 + \mathrm{Variance}$)")
ax1.set_title(r"(a) $\mathrm{Bias}^2$ & $\mathrm{Variance}$ (30 Refits)")
ax1.set_ylim(0, 1.08)
ax1.legend(loc="upper right", fontsize=7.8)

# Right: Test accuracy across 30 refits (mean +/- std) & variance removed %
d_keys = ["4", "16", "None"]
d_names = ["Shallow\n(depth=4)", "Selected\n(depth=16)", "Much Deeper\n(depth=None)"]
xd = np.arange(3)
s_acc = [q3s[k]["mean_test_acc"]*100 for k in d_keys]
s_std = [q3s[k]["std_test_acc"]*100 for k in d_keys]
b_acc = [q3b[k]["mean_test_acc"]*100 for k in d_keys]
b_std = [q3b[k]["std_test_acc"]*100 for k in d_keys]

ax2.errorbar(xd - 0.12, s_acc, yerr=s_std, fmt="-o", color=CORAL, lw=1.8, capsize=4, label="Single Tree (30 refits)")
ax2.errorbar(xd + 0.12, b_acc, yerr=b_std, fmt="-s", color=TEAL, lw=2.0, capsize=4, label="100-Tree Bagging (30 refits)")
for i, k in enumerate(d_keys):
    gain = q3b[k]["acc_gain_over_single"] * 100
    vrem = q3b[k]["var_removed_pct"]
    ax2.annotate(f"+{gain:.1f}% acc\n(-{vrem:.1f}% var)", xy=(xd[i]+0.12, b_acc[i]), xytext=(xd[i]-0.18, b_acc[i]+7.0),
                 fontsize=7.3, fontweight="bold", color=NAVY)

ax2.set_xticks(xd)
ax2.set_xticklabels(d_names, fontsize=8.2)
ax2.set_ylabel("Test Accuracy Across 30 Refits (%)")
ax2.set_ylim(10, 102)
ax2.set_title("(b) Accuracy Gain & Variance Reduction vs. Depth")
ax2.legend(loc="lower right", fontsize=7.8)

plt.tight_layout()
save_both(fig, "fig7_q3_bias_variance")
