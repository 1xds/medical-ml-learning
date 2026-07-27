# =========================
# 11b. Two-Step Feature Selection Distribution by Family
# =========================

from collections import Counter

# Step 2: after correlation filtering
deredundant_counter = Counter()
family_total = {name: len(cols) for name, cols in feature_categories.items()}

for feat in deredundant_cols:
    for prefix, name in [("original_shape2D","shape2D"), ("original_firstorder","firstorder"),
                          ("original_glcm","glcm"), ("original_glrlm","glrlm"),
                          ("original_glszm","glszm"), ("original_gldm","gldm"), ("original_ngtdm","ngtdm")]:
        if feat.startswith(prefix):
            deredundant_counter[name] += 1
            break

# Step 3: after SelectKBest
selected_counter = Counter()
for feat in selected_features:
    for prefix, name in [("original_shape2D","shape2D"), ("original_firstorder","firstorder"),
                          ("original_glcm","glcm"), ("original_glrlm","glrlm"),
                          ("original_glszm","glszm"), ("original_gldm","gldm"), ("original_ngtdm","ngtdm")]:
        if feat.startswith(prefix):
            selected_counter[name] += 1
            break

# Print summary
print("Feature selection distribution per family:")
print(f"{'Family':12s} {'Original':>8s} {'After corr':>10s} {'Selected':>10s}")
for name in sorted(family_total, key=lambda x: -family_total[x]):
    orig = family_total[name]
    dered = deredundant_counter.get(name, 0)
    sel = selected_counter.get(name, 0)
    pct = sel / orig * 100 if orig > 0 else 0
    print(f"  {name:12s} {orig:>8d} {dered:>10d} {sel:>10d} ({pct:.0f}%)")

# Bar chart: 3-stage comparison
fig, ax = plt.subplots(figsize=(10, 6))
family_order = sorted(family_total.keys(), key=lambda x: -family_total[x])
orig_counts = [family_total[name] for name in family_order]
dered_counts = [deredundant_counter.get(name, 0) for name in family_order]
sel_counts = [selected_counter.get(name, 0) for name in family_order]

x = np.arange(len(family_order))
width = 0.25

bars_orig = ax.bar(x - width, orig_counts, width, label=f"Original ({len(feature_cols)})",
                    color="#EEEDFE", edgecolor="#534AB7", linewidth=1)
bars_dered = ax.bar(x, dered_counts, width, label=f"After |r|>{CORR_THRESHOLD} ({len(deredundant_cols)})",
                     color="#E1F5EE", edgecolor="#0F6E56", linewidth=1)
bars_sel = ax.bar(x + width, sel_counts, width, label=f"SelectKBest k={K_BEST_FINAL} ({len(selected_features)})",
                   color="#185FA5", edgecolor="white", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(family_order, fontsize=11)
ax.set_ylabel("Number of features")
ax.set_title("Two-step feature selection distribution by family",
            fontsize=14, fontweight=500)
ax.legend(fontsize=10)

for bar, count in zip(bars_sel, sel_counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha="center", fontsize=10, fontweight=500)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_feature_selection_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.show()
