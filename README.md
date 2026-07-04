Objective
Prepare the question-bank dataset for clustering by selecting structural features, scaling, reducing dimensionality with PCA, and justifying k with elbow and silhouette methods.
Context Shift
Tasks 6–13 used supervised labels. Task 14 switches to unsupervised — the question is whether natural structural groupings exist in the data independent of difficulty. This is useful for question bank organisation and exam blueprint design.
Feature Selection
6 structural features selected: q_len, q_word_count, avg_opt_len, max_opt_len, avg_word_len, q_to_avg_opt_ratio. All derived from raw question and option text — no label leakage. Excluded: difficulty_level (target), domain_enc/topic_enc (ordinal integers without meaningful Euclidean distance), aggregate difficulty features (encode label indirectly), and opt_len_range (zero variance — all options have identical length in this dataset, making it a degenerate feature).
Scaling
StandardScaler applied — zero mean, unit variance. Without this, q_to_avg_opt_ratio (range 0.1–229) would dominate Euclidean distance over avg_word_len (range 1–10) purely because of larger units, not because it is more informative. After scaling, all features contribute equally based on variance.
Distance Sanity Check
Checked pairwise Euclidean distances on a 200-row sample after scaling. Min non-zero distance = 0.081, max = 9.03, ratio = 116x — acceptable. Some pairs have zero distance (identical structural features, e.g. two questions with identical length and word count). Documented honestly — this is a dataset property, not a bug.
PCA
PCA applied to remove correlated variance: q_len and q_word_count are correlated, as are avg_opt_len and max_opt_len. 3 components retained (threshold ≥85% variance) — retains 90.3%. PC1 contrasts question length vs option length. PC2 captures overall verbosity. PC3 captures vocabulary complexity. Dimensionality reduced 6 → 3.
k Selection
Ran KMeans for k=2 to k=10 on PCA-transformed data. Elbow method detected k=3 via second-derivative maximisation. Silhouette scores: k=3 (0.4855) > k=2 (0.4207) > all higher k. Both methods agree on k=3 — strong justification. Silhouette is preferred as the primary criterion because it measures actual cluster quality (intra vs inter cluster distance), not just model fit reduction.
Cluster Quality
Per-cluster silhouette at k=3: Cluster 1 and 2 have mean silhouette 0.64 with <1% negative samples — well separated. Cluster 0 (mostly aptitude questions) has mean 0.35 and 5.7% negative — weaker but acceptable. Domain distribution confirms clusters reflect structural question types, not just domain labels.
Locked for Task 15
task14_pca_data.csv (4799 × 5), task14_scaler.joblib, task14_pca.joblib, chosen k=3.
Pitfalls Addressed

✅ Scaled before clustering — StandardScaler applied
✅ k justified — both elbow and silhouette used, both agree
✅ Noisy dimensions removed — zero-variance feature excluded; PCA reduces to 3 dims

Artifacts
task14_cluster_prep.py, task14_pca_data.csv, task14_scaler.joblib, task14_pca.joblib, task14_experiment_log.json, task14_cluster_prep.png
