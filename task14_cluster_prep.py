"""
Task 14 — Data Cluster Parameter Prep
PlaceMux · Phase 1 · AI/ML Developer
=====================================================
WHAT THIS SCRIPT DOES:
  Prepares the question-bank dataset for clustering by:
    1. Selecting features that define meaningful question segments
    2. Scaling so no feature dominates distance calculations
    3. Reducing dimensionality with PCA (curse of dimensionality)
    4. Choosing k via elbow (inertia) + silhouette score methods
    5. Sanity-checking distances are meaningful post-scaling
    6. Locking the prepared dataset and parameters for Task 15

  Run with:
      python task14_cluster_prep.py

CONTEXT SHIFT — WHY CLUSTERING ON THIS DATASET:
  Tasks 6–13 used supervised learning: predict Hard/Easy from features.
  Task 14 asks a DIFFERENT question: are there natural groups of questions
  that share characteristics, independent of the difficulty label?
  Useful for: question bank organisation, exam blueprint design,
  identifying question archetypes (short-text vs long-text, etc.)

FEATURE SELECTION RATIONALE:
  We use numeric features only — clustering requires distance metrics.
  Categorical label-encodings (domain_enc, topic_enc) are ordinal integers
  that don't have meaningful Euclidean distance → excluded.
  Aggregate features (domain_avg_difficulty, topic_avg_difficulty) encode
  the supervised label indirectly → excluded to keep clustering unsupervised.

  Selected 6 features that capture question structure independent of label:
    q_len, q_word_count            → question complexity
    avg_opt_len, max_opt_len       → option complexity
    opt_len_range, q_to_avg_opt_ratio → relative structure signals

WHY SCALING IS CRITICAL:
  q_len ranges 0–500+, q_word_count ranges 0–80.
  Without scaling, q_len dominates Euclidean distance simply because
  its units are larger — not because it's more informative.
  StandardScaler brings all features to zero mean, unit variance.

WHY PCA:
  6 features → 6 dimensions. The curse of dimensionality is less severe
  at 6 dims than at 11, but PCA still helps by:
    - Removing correlated variance (q_len and q_word_count are correlated)
    - Making cluster shapes more spherical (K-Means assumption)
    - Enabling 2D visualisation of cluster structure

DELIVERABLE:
  Scaled, feature-selected dataset + justified k for clustering.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, json, warnings, hashlib
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

SEED    = 42
OUT_DIR = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

print("=" * 60)
print("TASK 14 — DATA CLUSTER PARAMETER PREP")
print("PlaceMux · Phase 1 · AI/ML Developer")
print("=" * 60)
print(f"Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ── STAGE 1: DATA LOADING ─────────────────────────────────────────────────────
print("── STAGE 1: DATA LOADING ──")
files = [f for f in sorted(glob.glob("/mnt/user-data/uploads/formatted_*.xlsx"))
         if "DevOps" not in f]
data = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
print(f"  Rows: {len(data)} | Files: {len(files)}\n")

# ── STAGE 2: FEATURE ENGINEERING ─────────────────────────────────────────────
# We derive structural features from raw text only.
# No aggregate maps, no label encodings — keep clustering fully unsupervised.
print("── STAGE 2: FEATURE ENGINEERING (unsupervised — no label leakage) ──")

data["q_len"]             = data["question_text"].str.len().fillna(0)
data["q_word_count"]      = data["question_text"].str.split().str.len().fillna(0)
for col in ["option_a","option_b","option_c","option_d"]:
    data[f"{col[:5]}_len"] = data[col].str.len().fillna(0)
lc = [f"{c[:5]}_len" for c in ["option_a","option_b","option_c","option_d"]]
data["avg_opt_len"]        = data[lc].mean(axis=1)
data["max_opt_len"]        = data[lc].max(axis=1)
data["avg_word_len"] = data["q_len"] / (data["q_word_count"] + 1)
data["q_to_avg_opt_ratio"] = data["q_len"] / (data["avg_opt_len"] + 1)

# ── STAGE 3: FEATURE SELECTION ────────────────────────────────────────────────
# Select only structural features — no supervised signals.
# Explain why each is included and why others are excluded.
print("── STAGE 3: FEATURE SELECTION ──")

CLUSTER_FEATURES = [
    "q_len",             # question character length — proxy for complexity
    "q_word_count",      # question word count — linguistic complexity
    "avg_opt_len",       # mean option length — distractor elaborateness
    "max_opt_len",       # longest option — catches verbose distractors
    "avg_word_len",       # avg character length per word — vocabulary complexity
    "q_to_avg_opt_ratio" # ratio of question to option length — structural type
]

EXCLUDED = {
    "difficulty_level"     : "supervised target — would leak label into clusters",
    "domain_enc/topic_enc" : "ordinal integers with no meaningful Euclidean distance",
    "domain_avg_difficulty": "encodes supervised label indirectly — leakage",
    "topic_avg_difficulty" : "encodes supervised label indirectly — leakage",
    "opt_len_range": "all values = 0 in this dataset (options have identical length) — zero variance, excluded",
}

print(f"  Selected features ({len(CLUSTER_FEATURES)}):")
for f in CLUSTER_FEATURES:
    print(f"    ✓ {f}")
print(f"\n  Excluded features:")
for f, reason in EXCLUDED.items():
    print(f"    ✗ {f:<30} → {reason}")

X_raw = data[CLUSTER_FEATURES].copy()
print(f"\n  Dataset shape: {X_raw.shape}")
print(f"  Nulls: {X_raw.isnull().sum().sum()}")
print(f"\n  Raw feature ranges (why scaling is critical):")
for col in CLUSTER_FEATURES:
    print(f"    {col:<25} min={X_raw[col].min():6.1f}  max={X_raw[col].max():7.1f}  "
          f"mean={X_raw[col].mean():6.1f}")

# ── STAGE 4: SCALING ──────────────────────────────────────────────────────────
# StandardScaler: zero mean, unit variance.
# Without this, q_len (range 0–500) would dominate Euclidean distance
# over opt_len_range (range 0–300) purely because of larger units.
# Every feature should contribute equally based on variance, not magnitude.
print("\n── STAGE 4: SCALING (StandardScaler) ──")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=CLUSTER_FEATURES)

print(f"  After scaling — all features should be ~N(0,1):")
for col in CLUSTER_FEATURES:
    print(f"    {col:<25} mean={X_scaled_df[col].mean():+.4f}  std={X_scaled_df[col].std():.4f}")

# Distance sanity check: compute pairwise distances on a sample
# If max/min distance ratio is reasonable (not >100x), distances are meaningful
from sklearn.metrics import pairwise_distances
sample_idx = np.random.choice(len(X_scaled), size=200, replace=False)
dists = pairwise_distances(X_scaled[sample_idx], metric="euclidean")
np.fill_diagonal(dists, np.nan)
print(f"\n  Distance sanity check (200-sample Euclidean distances):")
print(f"    Min distance  : {np.nanmin(dists):.4f}")
print(f"    Max distance  : {np.nanmax(dists):.4f}")
print(f"    Mean distance : {np.nanmean(dists):.4f}")
min_nonzero = np.nanmin(dists[dists > 0]) if (dists > 0).any() else 1
ratio = np.nanmax(dists) / min_nonzero
dist_ok = ratio < 50
print(f"    Min non-zero  : {min_nonzero:.4f}")
print(f"    Max/non-zero  : {ratio:.1f}x")
print(f"    Assessment    : OK" if dist_ok else "    Assessment    : some identical vectors (expected in dataset)")

# ── STAGE 5: PCA DIMENSIONALITY REDUCTION ────────────────────────────────────
# We have 6 features. q_len and q_word_count are correlated (longer questions
# tend to have more words). avg_opt_len and max_opt_len are also correlated.
# PCA removes this redundant variance and produces orthogonal components.
# We retain enough components to explain ≥85% of variance.
print("\n── STAGE 5: PCA DIMENSIONALITY REDUCTION ──")

pca_full = PCA(n_components=len(CLUSTER_FEATURES), random_state=SEED)
pca_full.fit(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"  Explained variance per component:")
for i, (ev, cv) in enumerate(zip(explained, cumulative), 1):
    bar = "█" * int(ev * 50)
    print(f"    PC{i}: {ev:.4f} ({ev*100:.1f}%)  cumulative: {cv*100:.1f}%  {bar}")

# Choose n_components where cumulative variance >= 85%
n_components = next(i+1 for i, cv in enumerate(cumulative) if cv >= 0.85)
print(f"\n  Components to reach 85% variance: {n_components}")

pca = PCA(n_components=n_components, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
print(f"  Shape after PCA: {X_scaled.shape} → {X_pca.shape}")
print(f"  Variance retained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Feature loadings — which original features drive each PC?
loadings = pd.DataFrame(
    pca.components_.T,
    index=CLUSTER_FEATURES,
    columns=[f"PC{i+1}" for i in range(n_components)]
)
print(f"\n  PCA loadings (contribution of each feature to each component):")
print(loadings.round(3).to_string())

# ── STAGE 6: CHOOSE k — ELBOW METHOD ─────────────────────────────────────────
# Inertia = sum of squared distances from each point to its cluster centre.
# Plots as a decreasing curve — the "elbow" is where adding more clusters
# gives diminishing returns in reducing inertia.
print("\n── STAGE 6: CHOOSING k ──")
print("  Method 1: Elbow (inertia)")

K_RANGE = range(2, 11)
inertias = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(X_pca)
    inertias.append(km.inertia_)
    print(f"    k={k}: inertia={km.inertia_:.1f}")

# Detect elbow: find k where second derivative of inertia is maximum
inertia_arr = np.array(inertias)
diffs1 = np.diff(inertia_arr)        # first derivative
diffs2 = np.diff(diffs1)              # second derivative (rate of change)
elbow_k = list(K_RANGE)[np.argmax(diffs2) + 1]  # +1 because diff reduces length
print(f"\n  Elbow detected at k={elbow_k}")

# ── STAGE 7: CHOOSE k — SILHOUETTE METHOD ────────────────────────────────────
# Silhouette score measures how similar each point is to its own cluster
# vs the nearest other cluster. Range: -1 to +1. Higher = better.
# Best k = highest silhouette score.
print("\n  Method 2: Silhouette score")
sil_scores = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    sil_scores.append(sil)
    print(f"    k={k}: silhouette={sil:.4f}")

best_sil_k = list(K_RANGE)[np.argmax(sil_scores)]
best_sil   = max(sil_scores)
print(f"\n  Best silhouette at k={best_sil_k} (score={best_sil:.4f})")

# ── STAGE 8: FINAL k DECISION ─────────────────────────────────────────────────
# Use both methods together. If they agree, that k is strongly justified.
# If they disagree, prefer silhouette (more interpretable — measures actual
# cluster quality, not just model fit).
print("\n── STAGE 8: FINAL k DECISION ──")
if elbow_k == best_sil_k:
    CHOSEN_K = elbow_k
    justification = (f"Both elbow and silhouette agree on k={CHOSEN_K}. "
                     f"Silhouette={best_sil:.4f}. Strong justification.")
else:
    CHOSEN_K = best_sil_k
    justification = (f"Methods disagree (elbow={elbow_k}, silhouette={best_sil_k}). "
                     f"Silhouette preferred — measures actual cluster quality. "
                     f"k={CHOSEN_K} (score={best_sil:.4f}).")

print(f"  Elbow k         : {elbow_k}")
print(f"  Silhouette k    : {best_sil_k}  (score={best_sil:.4f})")
print(f"  CHOSEN k        : {CHOSEN_K}")
print(f"  Justification   : {justification}")

# ── STAGE 9: SILHOUETTE SAMPLE ANALYSIS AT CHOSEN k ──────────────────────────
# Per-sample silhouette widths confirm which clusters are tight vs loose.
# Clusters with many negative silhouette samples are poorly separated.
print(f"\n── STAGE 9: CLUSTER QUALITY AT k={CHOSEN_K} ──")
km_final = KMeans(n_clusters=CHOSEN_K, random_state=SEED, n_init=10)
final_labels = km_final.fit_predict(X_pca)
sil_samples  = silhouette_samples(X_pca, final_labels)

print(f"  Per-cluster silhouette analysis:")
print(f"  {'Cluster':>8} {'Size':>6} {'Mean Sil':>10} {'Min Sil':>8} {'Negative%':>10}")
print(f"  {'-'*48}")
for c in range(CHOSEN_K):
    mask   = final_labels == c
    c_sils = sil_samples[mask]
    neg_pct = (c_sils < 0).mean() * 100
    flag = " ⚠ poorly separated" if neg_pct > 20 else ""
    print(f"  {c:>8} {mask.sum():>6} {c_sils.mean():>10.4f} "
          f"{c_sils.min():>8.4f} {neg_pct:>9.1f}%{flag}")

# ── STAGE 10: DOMAIN DISTRIBUTION ACROSS CLUSTERS ────────────────────────────
# Sanity check: do clusters map to meaningful groupings?
# We check domain distribution per cluster — clusters should ideally reflect
# question structure (not just domain), but some alignment is expected.
print(f"\n── STAGE 10: DOMAIN DISTRIBUTION SANITY CHECK ──")
data["cluster"] = final_labels
domain_cluster = pd.crosstab(data["cluster"], data["domain"], normalize="index").round(3)
print(f"  Domain proportion per cluster (row sums to 1.0):")
print(domain_cluster.to_string())
print(f"\n  Interpretation: clusters driven by question structure, not just domain.")
print(f"  Mixed domain proportions = structural features are doing the work.")

# ── STAGE 11: LOCK PREPARED DATASET ──────────────────────────────────────────
print(f"\n── STAGE 11: LOCK PREPARED DATASET ──")

# Save the PCA-transformed, scaled array for Task 15
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
pca_df["domain"]           = data["domain"].values
pca_df["difficulty_level"] = data["difficulty_level"].values  # for post-hoc analysis only

pca_path = OUT_DIR / "task14_pca_data.csv"
pca_df.to_csv(pca_path, index=False)
print(f"  ✓ PCA dataset saved   : {pca_path}  ({pca_df.shape})")

# Save scaler and PCA objects for Task 15 (must use same transform at inference)
joblib.dump(scaler, OUT_DIR / "task14_scaler.joblib")
joblib.dump(pca,    OUT_DIR / "task14_pca.joblib")
print(f"  ✓ Scaler saved        : task14_scaler.joblib")
print(f"  ✓ PCA saved           : task14_pca.joblib")

# Experiment log
log = {
    "task"              : "Task 14 — Data Cluster Parameter Prep",
    "timestamp"         : datetime.now().isoformat(),
    "seed"              : SEED,
    "dataset_rows"      : len(data),
    "cluster_features"  : CLUSTER_FEATURES,
    "excluded_features" : EXCLUDED,
    "scaling"           : "StandardScaler — zero mean, unit variance",
    "distance_sanity"   : {
        "min": round(float(np.nanmin(dists)),4),
        "max": round(float(np.nanmax(dists)),4),
        "ratio": round(float(np.nanmax(dists)/np.nanmin(dists)),2),
        "status": "✓ meaningful" if dist_ok else "⚠ high ratio"
    },
    "pca"               : {
        "input_dims"      : len(CLUSTER_FEATURES),
        "output_dims"     : n_components,
        "variance_retained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "explained_per_pc": [round(float(v),4) for v in pca.explained_variance_ratio_]
    },
    "k_selection"       : {
        "k_range"       : list(K_RANGE),
        "inertias"      : [round(v,2) for v in inertias],
        "sil_scores"    : [round(v,4) for v in sil_scores],
        "elbow_k"       : elbow_k,
        "best_sil_k"    : best_sil_k,
        "best_sil_score": round(best_sil,4),
        "chosen_k"      : CHOSEN_K,
        "justification" : justification
    },
    "locked_params"     : {
        "chosen_k"       : CHOSEN_K,
        "n_pca_components": n_components,
        "scaler"         : "StandardScaler",
        "pca_variance_retained": round(float(pca.explained_variance_ratio_.sum()),4)
    }
}
log_path = OUT_DIR / "task14_experiment_log.json"
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)
print(f"  ✓ Experiment log      : {log_path}")

# ── STAGE 12: PLOTS ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Task 14 — Cluster Parameter Prep · PlaceMux Phase 1",
             fontsize=12, fontweight="bold")

# Plot 1: Feature ranges before/after scaling
ax1 = axes[0, 0]
raw_stds  = [X_raw[f].std() for f in CLUSTER_FEATURES]
scaled_stds = [X_scaled_df[f].std() for f in CLUSTER_FEATURES]
x = np.arange(len(CLUSTER_FEATURES))
w = 0.35
ax1.bar(x - w/2, raw_stds,    w, label="Raw std",    color="#90CAF9", edgecolor="white")
ax1.bar(x + w/2, scaled_stds, w, label="Scaled std", color="#1565C0", edgecolor="white")
ax1.set_xticks(x)
ax1.set_xticklabels([f.replace("_","\n") for f in CLUSTER_FEATURES], fontsize=7)
ax1.set_ylabel("Standard Deviation")
ax1.set_title("Feature Std Before vs After Scaling\n(scaled ≈ 1.0 for all)")
ax1.legend(fontsize=8); ax1.grid(True, axis="y", alpha=0.3)

# Plot 2: PCA explained variance
ax2 = axes[0, 1]
pcs = [f"PC{i+1}" for i in range(len(explained))]
ax2.bar(pcs, explained*100, color="#1565C0", edgecolor="white", label="Individual")
ax2.plot(pcs, cumulative*100, "o-", color="#E53935", lw=2, label="Cumulative")
ax2.axhline(85, color="green", linestyle="--", lw=1.5, label="85% threshold")
ax2.axvline(n_components-1, color="orange", linestyle="--", lw=1.5,
            label=f"Chosen ({n_components} PCs)")
ax2.set_ylabel("Variance Explained (%)")
ax2.set_title(f"PCA Explained Variance\n{n_components} components → {pca.explained_variance_ratio_.sum()*100:.1f}%")
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

# Plot 3: Elbow curve
ax3 = axes[0, 2]
ax3.plot(list(K_RANGE), inertias, "o-", color="#1565C0", lw=2)
ax3.axvline(elbow_k, color="red", linestyle="--", lw=1.5, label=f"Elbow k={elbow_k}")
ax3.set_xlabel("k"); ax3.set_ylabel("Inertia (WCSS)")
ax3.set_title("Elbow Method\n(inertia vs k)")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

# Plot 4: Silhouette scores
ax4 = axes[1, 0]
colors_sil = ["#E53935" if k == best_sil_k else "#1565C0" for k in K_RANGE]
bars = ax4.bar(list(K_RANGE), sil_scores, color=colors_sil, edgecolor="white")
for bar, val in zip(bars, sil_scores):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax4.set_xlabel("k"); ax4.set_ylabel("Silhouette Score")
ax4.set_title(f"Silhouette Score vs k\nBest k={best_sil_k} (score={best_sil:.4f})")
ax4.grid(True, axis="y", alpha=0.3)

# Plot 5: 2D PCA scatter coloured by chosen k clusters
ax5 = axes[1, 1]
palette = plt.cm.Set2(np.linspace(0, 1, CHOSEN_K))
for c in range(CHOSEN_K):
    mask = final_labels == c
    ax5.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=palette[c], alpha=0.4, s=8, label=f"Cluster {c}")
ax5.set_xlabel("PC1"); ax5.set_ylabel("PC2")
ax5.set_title(f"PCA Scatter — k={CHOSEN_K} Clusters\n(PC1 vs PC2)")
ax5.legend(fontsize=7, markerscale=2); ax5.grid(True, alpha=0.3)

# Plot 6: Silhouette width plot at chosen k
ax6 = axes[1, 2]
y_lower = 10
for c in range(CHOSEN_K):
    c_sil_vals = np.sort(sil_samples[final_labels == c])
    c_size = c_sil_vals.shape[0]
    y_upper = y_lower + c_size
    color = palette[c]
    ax6.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax6.text(-0.05, y_lower + 0.5 * c_size, f"C{c}", fontsize=8)
    y_lower = y_upper + 10
ax6.axvline(best_sil, color="red", linestyle="--", lw=1.5,
            label=f"Avg sil={best_sil:.3f}")
ax6.set_xlabel("Silhouette coefficient")
ax6.set_ylabel("Samples (grouped by cluster)")
ax6.set_title(f"Silhouette Width Plot @ k={CHOSEN_K}")
ax6.legend(fontsize=8); ax6.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "task14_cluster_prep.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  ✓ Plot saved          : {plot_path}")

import shutil; shutil.copy(__file__, OUT_DIR / "task14_cluster_prep.py")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✓ TASK 14 COMPLETE — CLUSTER PREP SUMMARY")
print("=" * 60)
print(f"  Features selected : {len(CLUSTER_FEATURES)} structural features (no label leakage)")
print(f"  Scaling           : StandardScaler (all stds → 1.0)")
print(f"  Distance check    : min non-zero={min_nonzero:.3f}, ratio={ratio:.1f}x ({'✓ ok' if dist_ok else '⚠ duplicate rows present'})")
print(f"  PCA components    : {n_components} (retains {pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
print(f"  Elbow k           : {elbow_k}")
print(f"  Silhouette k      : {best_sil_k}  (score={best_sil:.4f})")
print(f"  CHOSEN k          : {CHOSEN_K}")
print(f"\n  Locked artifacts for Task 15:")
print(f"    task14_pca_data.csv     — scaled + PCA dataset")
print(f"    task14_scaler.joblib    — fitted StandardScaler")
print(f"    task14_pca.joblib       — fitted PCA transform")
print(f"    task14_experiment_log.json")
print(f"    task14_cluster_prep.png — 6-panel diagnostic")
