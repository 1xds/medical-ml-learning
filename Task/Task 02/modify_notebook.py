"""Modify task.ipynb: replace 102x102 with 7x7, add family distribution chart, fix bugs."""
import json

nb_path = r"D:\Research\medical-ml-learning\Task\Task 02\task.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
cells = nb["cells"]

# ============================================================
# 1. Fix Cell 0 — matplotlib style compatibility
# ============================================================
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    if src.startswith("# =========================\n# 0. Imports"):
        new_lines = []
        for line in cell["source"]:
            if line.strip() == 'plt.style.use("seaborn-v0_8-whitegrid")':
                new_lines.append("try:\n")
                new_lines.append("    plt.style.use(\"seaborn-v0_8-whitegrid\")\n")
                new_lines.append("except OSError:\n")
                new_lines.append("    plt.style.use(\"seaborn-whitegrid\")\n")
            else:
                new_lines.append(line)
        cells[i]["source"] = new_lines
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None
        print(f"[Fix] Cell {i}: matplotlib style compatibility")
        break

# ============================================================
# 2. Fix Cell 14 — greycoprops API change
# ============================================================
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    if "greycoprops" in src and cell["cell_type"] == "code":
        new_lines = []
        skip_greycoprops_import = False
        skip_props_block = False
        props_block_lines = []
        in_props_block = False

        # First pass: find and replace import
        for line in cell["source"]:
            if "from skimage.feature import graycomatrix, greycoprops" in line:
                # Replace with manual computation approach
                new_lines.append("from skimage.feature import graycomatrix\n")
                new_lines.append("\n")
                new_lines.append("# Compute GLCM properties manually\n")
                new_lines.append("# (avoids greycoprops API changes across scikit-image versions)\n")
                new_lines.append("def compute_glcm_props(glcm_matrix):\n")
                new_lines.append("    \"\"\"Compute 6 common GLCM texture properties from normalized matrix.\"\"\"\n")
                new_lines.append("    Ng = glcm_matrix.shape[0]\n")
                new_lines.append("    i_idx = np.arange(Ng)\n")
                new_lines.append("    j_idx = np.arange(Ng)\n")
                new_lines.append("    diff = i_idx[:, None] - j_idx[None, :]\n")
                new_lines.append("    contrast = np.sum(diff ** 2 * glcm_matrix)\n")
                new_lines.append("    dissimilarity = np.sum(np.abs(diff) * glcm_matrix)\n")
                new_lines.append("    homogeneity = np.sum(glcm_matrix / (1.0 + diff ** 2))\n")
                new_lines.append("    energy = np.sum(glcm_matrix ** 2)\n")
                new_lines.append("    asm = energy\n")
                new_lines.append("    mu_i = np.sum(i_idx[:, None] * glcm_matrix)\n")
                new_lines.append("    mu_j = np.sum(j_idx[None, :] * glcm_matrix)\n")
                new_lines.append("    sigma_i = np.sqrt(np.sum((i_idx[:, None] - mu_i) ** 2 * glcm_matrix))\n")
                new_lines.append("    sigma_j = np.sqrt(np.sum((j_idx[None, :] - mu_j) ** 2 * glcm_matrix))\n")
                new_lines.append("    if sigma_i > 0 and sigma_j > 0:\n")
                new_lines.append("        correlation = np.sum((i_idx[:, None] - mu_i) * (j_idx[None, :] - mu_j) * glcm_matrix) / (sigma_i * sigma_j)\n")
                new_lines.append("    else:\n")
                new_lines.append("        correlation = 0.0\n")
                new_lines.append("    return {\n")
                new_lines.append("        \"contrast\": contrast, \"dissimilarity\": dissimilarity,\n")
                new_lines.append("        \"homogeneity\": homogeneity, \"energy\": energy,\n")
                new_lines.append("        \"correlation\": correlation, \"asm\": asm,\n")
                new_lines.append("    }\n")
                new_lines.append("\n")
            else:
                new_lines.append(line)

        # Second pass: replace greycoprops usage
        joined = "".join(new_lines)

        old_props = """# Step 6: Texture features
props_demo = {}
for prop in ["contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "asm"]:
    props_demo[prop] = greycoprops(glcm, prop)[0, 0]"""

        new_props = """# Step 6: Texture features (computed from GLCM matrix directly)
props_demo = compute_glcm_props(glcm_matrix)"""

        joined = joined.replace(old_props, new_props)

        # Also update print section: props_demo is now a dict, iteration still works
        # No change needed — for k, v in props_demo.items() still works

        # Convert back to line array
        final_lines = []
        split_lines = joined.split("\n")
        for idx, line in enumerate(split_lines):
            if idx < len(split_lines) - 1:
                final_lines.append(line + "\n")
            else:
                if line:  # last non-empty line
                    final_lines.append(line)

        cells[i]["source"] = final_lines
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None
        print(f"[Fix] Cell {i}: greycoprops → manual computation")
        break

# ============================================================
# 3. Replace Cell 15 (markdown) — 7×7 description
# ============================================================
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    if "Feature Correlation Analysis" in src and "inter-feature correlations" in src:
        cells[i]["source"] = [
            "## Feature Correlation Analysis (Family-Level)\n",
            "\n",
            "102 individual features are organized into **7 mathematical families**:\n",
            "- **shape2D** (9), **firstorder** (18), **glcm** (24), **glrlm** (16), **glszm** (16), **gldm** (14), **ngtdm** (5)\n",
            "\n",
            "Instead of the unreadable 102×102 full matrix, we use a **7×7 family-level average correlation matrix**:\n",
            "- **Diagonal**: intra-family redundancy (features within the same family repeating each other)\n",
            "- **Off-diagonal**: inter-family information overlap (two families describing similar patterns)\n",
            "\n",
            "Higher values → more redundancy/overlap; lower values → more complementary information.\n",
            "\n",
            "This is **EDA only** — visualization for understanding, not for deleting features before train/test split.\n",
        ]
        print(f"[Replace] Cell {i}: markdown → 7×7 family-level description")
        break

# ============================================================
# 4. Replace Cell 16 (code) — 7×7 family-level matrix + top pairs
# ============================================================
cell16_source = [
    "# =========================\n",
    "# 10. Feature Correlation Analysis (7×7 Family-Level + Top Redundant Pairs)\n",
    "# =========================\n",
    "\n",
    "families = {\n",
    "    \"shape2D\": [c for c in feature_cols if c.startswith(\"original_shape2D\")],\n",
    "    \"firstorder\": [c for c in feature_cols if c.startswith(\"original_firstorder\")],\n",
    "    \"glcm\": [c for c in feature_cols if c.startswith(\"original_glcm\")],\n",
    "    \"glrlm\": [c for c in feature_cols if c.startswith(\"original_glrlm\")],\n",
    "    \"glszm\": [c for c in feature_cols if c.startswith(\"original_glszm\")],\n",
    "    \"gldm\": [c for c in feature_cols if c.startswith(\"original_gldm\")],\n",
    "    \"ngtdm\": [c for c in feature_cols if c.startswith(\"original_ngtdm\")],\n",
    "}\n",
    "\n",
    "corr = df_features[feature_cols].corr()\n",
    "family_names = list(families.keys())\n",
    "family_corr = pd.DataFrame(index=family_names, columns=family_names, dtype=float)\n",
    "\n",
    "for fa in family_names:\n",
    "    for fb in family_names:\n",
    "        cols_a, cols_b = families[fa], families[fb]\n",
    "        if fa == fb:\n",
    "            vals = np.abs(corr.loc[cols_a, cols_b].values)\n",
    "            np.fill_diagonal(vals, np.nan)\n",
    "            family_corr.loc[fa, fb] = np.nanmean(vals)\n",
    "        else:\n",
    "            family_corr.loc[fa, fb] = np.abs(corr.loc[cols_a, cols_b]).mean().mean()\n",
    "\n",
    "# 7×7 heatmap\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "sns.heatmap(family_corr, annot=True, fmt=\".2f\", cmap=\"RdBu_r\",\n",
    "            center=0, vmin=0, vmax=1, square=True, ax=ax,\n",
    "            linewidths=1, linecolor=\"white\",\n",
    "            cbar_kws={\"shrink\": 0.8, \"label\": \"Mean |Pearson r|\"})\n",
    "ax.set_title(\"Feature family-level correlation matrix (7×7)\",\n",
    "             fontsize=14, fontweight=500)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIG_DIR, \"04_family_corr_heatmap.png\"),\n",
    "            dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "\n",
    "# Print interpretation\n",
    "print(\"\\n7×7 Family-level correlation interpretation:\")\n",
    "for fa in family_names:\n",
    "    diag = family_corr.loc[fa, fa]\n",
    "    level = \"high\" if diag > 0.5 else \"moderate\" if diag > 0.3 else \"low\"\n",
    "    print(f\"  {fa:12s}  intra-family redundancy = {diag:.2f}  ({level})\")\n",
    "\n",
    "# Top |r| > 0.95 redundant pairs\n",
    "high_pairs = []\n",
    "for i_idx in range(len(feature_cols)):\n",
    "    for j_idx in range(i_idx + 1, len(feature_cols)):\n",
    "        r = corr.iloc[i_idx, j_idx]\n",
    "        if abs(r) > 0.95:\n",
    "            high_pairs.append((feature_cols[i_idx].replace(\"original_\", \"\"),\n",
    "                               feature_cols[j_idx].replace(\"original_\", \"\"), r))\n",
    "high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)\n",
    "\n",
    "print(f\"\\n|r| > 0.95 redundant pairs ({len(high_pairs)} total, showing top 15):\")\n",
    "for f1, f2, r in high_pairs[:15]:\n",
    "    print(f\"  {f1:40s} vs {f2:40s}  r = {r:.3f}\")\n",
    "\n",
    "# Top redundant pairs bar chart\n",
    "if len(high_pairs) > 0:\n",
    "    top_n = min(20, len(high_pairs))\n",
    "    labels = [f\"{a[:30]}\\nvs {b[:30]}\" for a, b, _ in high_pairs[:top_n]]\n",
    "    values = [abs(r) for _, _, r in high_pairs[:top_n]]\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))\n",
    "    ax.barh(range(top_n), values, color=\"#D85A30\",\n",
    "            edgecolor=\"white\", linewidth=0.5)\n",
    "    ax.set_yticks(range(top_n))\n",
    "    ax.set_yticklabels(labels, fontsize=7)\n",
    "    ax.invert_yaxis()\n",
    "    ax.set_xlabel(\"|Pearson r|\")\n",
    "    ax.set_title(f\"Top {top_n} highly redundant feature pairs (|r| > 0.95)\",\n",
    "                fontsize=13, fontweight=500)\n",
    "    ax.set_xlim(0.95, 1.01)\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(os.path.join(FIG_DIR, \"04b_redundant_pairs.png\"),\n",
    "                dpi=150, bbox_inches=\"tight\")\n",
    "    plt.show()\n",
]

for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    if "# 10. Feature Correlation Analysis" in src and "corr_matrix" in src:
        cells[i]["source"] = cell16_source
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None
        print(f"[Replace] Cell {i}: 102×102 → 7×7 family-level matrix")
        break

# ============================================================
# 5. Insert 2 new cells after Cell 18 (SelectKBest family distribution)
# ============================================================
# Find Cell 18 — the preprocessing & feature selection code cell
insert_idx = None
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    if "# 11. Preprocessing & Feature Selection" in src:
        insert_idx = i + 1  # insert after this cell
        break

if insert_idx is not None:
    # New markdown cell
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## SelectKBest Feature Distribution by Family\n",
            "\n",
            "Which feature families are represented in the selected 20 features?\n",
            "This directly reflects the **information balance** across families and validates whether SelectKBest selects complementary features from multiple families.\n",
        ]
    }

    # New code cell
    code_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# =========================\n",
            "# 11b. Selected Feature Family Distribution\n",
            "# =========================\n",
            "\n",
            "from collections import Counter\n",
            "\n",
            "family_counter = Counter()\n",
            "family_total = {name: len(cols) for name, cols in feature_categories.items()}\n",
            "\n",
            "for feat in selected_features:\n",
            "    for prefix, name in [(\"original_shape2D\",\"shape2D\"), (\"original_firstorder\",\"firstorder\"),\n",
            "                          (\"original_glcm\",\"glcm\"), (\"original_glrlm\",\"glrlm\"),\n",
            "                          (\"original_glszm\",\"glszm\"), (\"original_gldm\",\"gldm\"), (\"original_ngtdm\",\"ngtdm\")]:\n",
            "        if feat.startswith(prefix):\n",
            "            family_counter[name] += 1\n",
            "            break\n",
            "\n",
            "print(\"SelectKBest selected features per family:\")\n",
            "for name in sorted(family_counter, key=lambda x: -family_counter[x]):\n",
            "    count = family_counter[name]\n",
            "    total = family_total[name]\n",
            "    print(f\"  {name:12s}: {count}/{total} selected ({count/total*100:.0f}%)\")\n",
            "\n",
            "# Bar chart: total vs selected per family\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "family_order = sorted(family_total.keys(), key=lambda x: -family_total[x])\n",
            "selected_counts = [family_counter.get(name, 0) for name in family_order]\n",
            "total_counts = [family_total[name] for name in family_order]\n",
            "\n",
            "sel_colors = [\"#185FA5\", \"#0F6E56\", \"#BA7517\", \"#534AB7\", \"#D85A30\", \"#AFA9EC\", \"#E8B8B8\"]\n",
            "x = np.arange(len(family_order))\n",
            "width = 0.35\n",
            "\n",
            "bars_total = ax.bar(x - width/2, total_counts, width, label=\"Total features\",\n",
            "                    color=\"#EEEDFE\", edgecolor=\"#534AB7\", linewidth=1)\n",
            "bars_sel = ax.bar(x + width/2, selected_counts, width, label=\"Selected features\",\n",
            "                   color=sel_colors[:len(family_order)], edgecolor=\"white\", linewidth=1)\n",
            "\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(family_order, fontsize=11)\n",
            "ax.set_ylabel(\"Number of features\")\n",
            "ax.set_title(f\"Feature selection distribution (SelectKBest k={K_BEST})\",\n",
            "            fontsize=14, fontweight=500)\n",
            "ax.legend(fontsize=10)\n",
            "\n",
            "for bar, count in zip(bars_sel, selected_counts):\n",
            "    if count > 0:\n",
            "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,\n",
            "                str(count), ha=\"center\", fontsize=10, fontweight=500)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIG_DIR, \"05_feature_selection_distribution.png\"),\n",
            "            dpi=150, bbox_inches=\"tight\")\n",
            "plt.show()\n",
        ],
        "outputs": [],
        "execution_count": None,
    }

    # Insert at position
    cells.insert(insert_idx, md_cell)
    cells.insert(insert_idx + 1, code_cell)
    print(f"[Insert] 2 new cells after preprocessing cell (position {insert_idx})")

# ============================================================
# 6. Update figure numbering in subsequent cells
# ============================================================
# The inserted cell uses "05_feature_selection_distribution.png"
# This conflicts with the existing Cell 22 (CV boxplot) which also uses "05_cv_boxplot.png"
# Need to renumber figures from the CV cell onwards

# Find cells that save figures and update numbering
fig_num = 5  # Start from 06 since we added 05 for distribution
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    if cell["cell_type"] == "code" and "05_cv_boxplot.png" in src:
        # CV boxplot → 06
        new_lines = [line.replace("05_cv_boxplot.png", "06_cv_boxplot.png") for line in cell["source"]]
        cells[i]["source"] = new_lines
        print(f"[Fix] Cell {i}: figure 05 → 06 (cv_boxplot)")
        fig_num = 6
    elif cell["cell_type"] == "code" and "06_roc_pr_curves.png" in src:
        # ROC/PR → 07
        new_lines = [line.replace("06_roc_pr_curves.png", "07_roc_pr_curves.png") for line in cell["source"]]
        cells[i]["source"] = new_lines
        print(f"[Fix] Cell {i}: figure 06 → 07 (roc_pr)")
    elif cell["cell_type"] == "code" and "07_confusion_matrix.png" in src:
        # Confusion → 08
        new_lines = [line.replace("07_confusion_matrix.png", "08_confusion_matrix.png") for line in cell["source"]]
        cells[i]["source"] = new_lines
        print(f"[Fix] Cell {i}: figure 07 → 08 (confusion)")
    elif cell["cell_type"] == "code" and "08_feature_importance.png" in src:
        # Feature importance → 09
        new_lines = [line.replace("08_feature_importance.png", "09_feature_importance.png") for line in cell["source"]]
        cells[i]["source"] = new_lines
        print(f"[Fix] Cell {i}: figure 08 → 09 (importance)")
    elif cell["cell_type"] == "code" and "09_learning_curves.png" in src:
        # Learning curves → 10
        new_lines = [line.replace("09_learning_curves.png", "10_learning_curves.png") for line in cell["source"]]
        cells[i]["source"] = new_lines
        print(f"[Fix] Cell {i}: figure 09 → 10 (learning)")
    elif cell["cell_type"] == "code" and "10_shap_summary.png" in src:
        # SHAP summary → 11
        new_lines = [line.replace("10_shap_summary.png", "11_shap_summary.png") for line in cell["source"]]
        cells[i]["source"] = new_lines
        print(f"[Fix] Cell {i}: figure 10 → 11 (shap_summary)")
    elif cell["cell_type"] == "code" and "11_shap_force.png" in src:
        # SHAP force → 12
        new_lines = [line.replace("11_shap_force.png", "12_shap_force.png") for line in cell["source"]]
        cells[i]["source"] = new_lines
        print(f"[Fix] Cell {i}: figure 11 → 12 (shap_force)")

# ============================================================
# Save
# ============================================================
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nSaved: {nb_path}")
print(f"Total cells: {len(cells)}")
print("Done!")
