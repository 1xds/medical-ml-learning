# =========================
# 11. Preprocessing & Two-Step Feature Selection
# =========================

X = df_features[feature_cols].values
y = df_features["label"].values
print(f"Feature matrix: {X.shape}, Labels: benign={np.sum(y==0)}, malignant={np.sum(y==1)}")

# ---------- Step 0: Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} (benign={np.sum(y_train==0)}, malignant={np.sum(y_train==1)})")
print(f"Test:  {X_test.shape[0]}  (benign={np.sum(y_test==0)}, malignant={np.sum(y_test==1)})")

# ---------- Step 1: Standardization (fit on train only) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Step 2: Remove High-Correlation Features (|r| > CORR_THRESHOLD) ----------
# CRITICAL: correlation computed on TRAINING data only to avoid data leakage
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
corr_train = X_train_df.corr().abs()

# Upper triangle: find features where any |r| > threshold
upper_mask = np.triu(np.ones(corr_train.shape), k=1).astype(bool)
upper = corr_train.where(upper_mask)

to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]

# Retained features after correlation filtering
deredundant_cols = [c for c in feature_cols if c not in to_drop]
deredundant_idx = [feature_cols.index(c) for c in deredundant_cols]

X_train_dered = X_train_scaled[:, deredundant_idx]
X_test_dered = X_test_scaled[:, deredundant_idx]

print()
print(f"--- Step 2: Correlation Filtering (|r| > {CORR_THRESHOLD}) ---")
print(f"  Original features: {len(feature_cols)}")
print(f"  Removed features:  {len(to_drop)}")
print(f"  Retained features: {len(deredundant_cols)}")

# Show removed features and their correlation partners
for col in to_drop:
    corr_vals = upper[col]
    partner = corr_vals[corr_vals > CORR_THRESHOLD].idxmax()
    r_val = corr_vals[partner]
    col_short = col.replace("original_", "")
    partner_short = partner.replace("original_", "")
    print(f"  Drop: {col_short:40s}  |r|={r_val:.3f} with {partner_short}")

# ---------- Step 3: SelectKBest (supervised, ANOVA F-test) ----------
K_BEST_FINAL = min(K_BEST, len(deredundant_cols))

selector = SelectKBest(score_func=f_classif, k=K_BEST_FINAL)
X_train_sel = selector.fit_transform(X_train_dered, y_train)
X_test_sel = selector.transform(X_test_dered)

selected_mask = selector.get_support()
selected_features = [deredundant_cols[i] for i in range(len(deredundant_cols)) if selected_mask[i]]

print()
print(f"--- Step 3: SelectKBest (k={K_BEST_FINAL}) ---")
print(f"  Input:   {len(deredundant_cols)} features (after correlation filtering)")
print(f"  Selected: {len(selected_features)} features")

for i, feat in enumerate(selected_features, 1):
    score_idx = np.where(selected_mask)[0][i - 1]
    score = selector.scores_[score_idx]
    cat = "other"
    for prefix, name in [("original_shape2D","shape2D"), ("original_firstorder","firstorder"),
                          ("original_glcm","glcm"), ("original_glrlm","glrlm"),
                          ("original_glszm","glszm"), ("original_gldm","gldm"), ("original_ngtdm","ngtdm")]:
        if feat.startswith(prefix):
            cat = name
            break
    feat_short = feat.replace("original_", "")
    print(f"  {i:2d}. [{cat:10s}] {feat_short:40s}  F-score = {score:.2f}")

# Summary
print()
print("=" * 60)
print("Two-Step Feature Selection Summary:")
print(f"  {len(feature_cols)} features -> {len(deredundant_cols)} (after |r|>{CORR_THRESHOLD}) -> {len(selected_features)} (SelectKBest k={K_BEST_FINAL})")
print("=" * 60)
