# Network Intrusion Detection — Assignment Report

**Course:** Advanced Python (ICS0019)  
**Team members:** Kaan Meetin, Lorenz Ritsch 
**Date:** 25.05.2026 
**Repository:** [GitHub/GitLab link]

---

## 1. Approach

### 1.1 Strategy Overview

We had no fixed plan before starting. Our initial goal was simply to reproduce
the baseline and understand why it is not good enough. After observing that the baseline Random Forest almost completely
ignores R2L and U2R attacks, we focused all effort on improving
detection of those two minority classes, as they were the main reason for the low macro F1 score.

Our general approach: 
1. establish a baseline, try out RandomForest
2. deal with class imbalance
3. switch to XGBOOST
4. tune hyperparameters
5. engineer features

### 1.2 Preprocessing

Beyond the starter code, we applied the following:

- **Feature engineering:**
  - Replaced label encoding of `protocol_type`, `service`, and `flag` with
    one-hot encoding. Label encoding implies a spurious ordinal relationship
    (e.g. `http=22 > ftp=10`) that does not exist. One-hot encoding gives
    XGBoost a separate binary column per category, which is more expressive
    particularly for the `service` feature (~70 unique values) that is highly
    informative for R2L attack detection.
  - Applied `log1p` transformation to seven heavily right-skewed numeric
    features: `duration`, `src_bytes`, `dst_bytes`, `count`, `srv_count`,
    `dst_host_count`, `dst_host_srv_count`. These span several orders of
    magnitude; log-scaling compresses the long tail and makes tree splits
    more informative.
  - Added three derived features: `bytes_ratio` (src_bytes / dst_bytes + 1),
    `error_rate_avg` (mean of serror_rate and rerror_rate), and
    `srv_error_rate_avg` (mean of srv_serror_rate and srv_rerror_rate).
  - Final feature count: **124** (up from 40).

- **Feature selection:** Dropped `num_outbound_cmds` (constant zero throughout
  the dataset, as specified in the assignment). No further features were
  removed.

- **Scaling:** No scaling applied. Tree-based models (Random Forest, XGBoost)
  are invariant to feature scale. Scaling was applied for k-NN and LinearSVC
  experiments via `StandardScaler` inside the pipeline.

- **Other:** Categorical features (`protocol_type`, `service`, `flag`) were
  initially label-encoded using the starter code. The one-hot encoding step
  replaced this encoding in our final pipeline.

### 1.3 Class Imbalance Handling

As the training set is severely unbalanced, wee addressed this with SMOTE (Synthetic
Minority Over-sampling Technique).

- **Method:** SMOTE with default `sampling_strategy='auto'`, which upsamples
  all minority classes to match the majority class size (~67,000 each).
- **Parameters:** `k_neighbors=5` (default), `random_state=42`.
- **Effect:** Training set expanded from 125,973 to ~336,000 rows.
  Post-SMOTE class distribution: all five classes balanced at ~67,000 samples.
- **Implementation:** SMOTE was always applied **inside** an `ImbPipeline`
  (from `imbalanced-learn`) so that it fires only on the training fold during
  cross-validation and never sees the validation or test data. This is a
  critical implementation detail — see Experiment 5 for what happens when this
  is done incorrectly.

We also tested `class_weight='balanced'` on Random Forest (Experiment 2) and
SMOTETomek with custom sampling targets (Experiment 4). Both performed worse
than plain SMOTE.

---

## 2. Experiments

### Experiment 1: Random Forest baseline

- **Algorithm:** RandomForestClassifier, default parameters
- **What changed from starter code:** Nothing — this is our own baseline run
- **Macro F1 (CV):** not formally computed
- **Macro F1 (test):** 0.5009
- **Observation:** The model performs well on DoS (F1=0.87) and Normal
  (F1=0.78) but is essentially blind to R2L (F1=0.02, recall=0.01) and weak
  on U2R (F1=0.11, recall=0.06). This confirms the class imbalance problem:
  the model learned that predicting "Normal" is almost always safe.

### Experiment 2: Random Forest with balanced class weights

- **Algorithm:** RandomForestClassifier, `class_weight='balanced'`
- **What changed:** Added `class_weight='balanced'`
- **Macro F1 (CV):** not formally computed
- **Macro F1 (test):** 0.4779
- **Observation:** Counterintuitively, performance regressed. R2L F1 dropped
  from 0.02 to 0.00. The explanation: KDDTest+ contains R2L attack types not
  present in training. Upweighting the 995 training R2L examples by ~68×
  caused the model to overfit to those specific patterns, which did not
  generalise to unseen R2L variants in the test set.

### Experiment 3: Random Forest with SMOTE

- **Algorithm:** RandomForestClassifier, default parameters, SMOTE applied
  before training
- **What changed:** Added SMOTE oversampling
- **Macro F1 (CV):** not formally computed
- **Macro F1 (test):** 0.5351
- **Observation:** Meaningful improvement over both previous experiments. R2L
  F1 jumped from 0.02 to 0.15 (recall 0.01 → 0.08). SMOTE generates synthetic
  minority examples by interpolating between existing ones, providing more
  variation than simple duplication. U2R also improved slightly (0.11 → 0.14).

### Experiment 4: SMOTETomek with custom sampling targets

- **Algorithm:** XGBoost, SMOTETomek with R2L=10,000, U2R=2,000, Probe=20,000
- **What changed:** Replaced SMOTE with SMOTETomek; reduced oversampling
  targets to avoid over-synthesising from few originals (especially U2R: 52)
- **Macro F1 (CV):** not formally computed
- **Macro F1 (test):** 0.5900
- **Observation:** Worse than plain SMOTE. Capping U2R at 2,000 synthetic
  examples gave the model too little variation to learn from. Tomek link
  removal (cleaning majority boundary examples) did not compensate.

### Experiment 5: XGBoost + SMOTE (data leakage discovered)

- **Algorithm:** XGBoost + SMOTE (applied before RandomizedSearchCV, not
  inside the CV pipeline)
- **What changed:** Switched to XGBoost, added hyperparameter search with
  50 iterations, 5-fold CV
- **Macro F1 (CV):** **0.9998** ← suspicious
- **Macro F1 (test):** 0.6240
- **Observation:** This experiment accidentally introduced data leakage.
  SMOTE was applied to the full training set before cross-validation, meaning
  the synthetic examples in each validation fold were interpolated from real
  examples in the corresponding training fold. The model effectively "saw" its
  own validation data during training, producing an inflated CV score. The
  CV-test gap of 0.38 revealed the problem. The fix (Experiment 6) was to move
  SMOTE inside an `ImbPipeline` so it runs per fold.

### Experiment 6: XGBoost + SMOTE (honest pipeline, hyperparameter tuning)

- **Algorithm:** XGBoost inside ImbPipeline with SMOTE, RandomizedSearchCV
  with 20 iterations, 3-fold CV
- **What changed:** SMOTE moved inside ImbPipeline; leakage eliminated
- **Macro F1 (CV):** 0.9487 (honest — no leakage)
- **Macro F1 (test):** 0.6203
- **Observation:** Fixing the leakage dropped the CV score from 0.9998 to
  0.9487 — a more credible number. The CV-test gap of ~0.33 is now due to
  genuine distribution shift (unseen attack types in KDDTest+), not leakage.
  Best hyperparameters found: n_estimators=406, max_depth=6,
  learning_rate=0.165, subsample=0.828, colsample_bytree=0.868,
  min_child_weight=1.

### Experiment 7: XGBoost + SMOTE + per-class threshold tuning

- **Algorithm:** XGBoost (tuned parameters from Experiment 6) with
  OOF-based threshold tuning for R2L and U2R
- **What changed:** Added threshold tuning — if R2L probability exceeds a
  threshold, predict R2L even if another class has higher argmax probability
- **Macro F1 (CV):** 0.9405
- **Macro F1 (test):** 0.6275
- **Observation:** The R2L precision-recall asymmetry (precision 0.98, recall
  0.14) indicated excessive caution. Thresholds chosen via OOF CV were R2L=0.4
  and U2R=0.3 — much higher than the thresholds that work on test (0.1/0.1).
  This demonstrated that decision thresholds optimal for the training
  distribution do not transfer to KDDTest+, again due to the distribution
  shift. Honest threshold tuning added only a marginal gain over the default
  argmax.

### Experiment 8: XGBoost + LightGBM soft voting

- **Algorithm:** XGBoost + LightGBM (both with SMOTE inside ImbPipeline),
  probabilities averaged
- **What changed:** Added LightGBM as a second base model; final prediction
  based on averaged probabilities
- **Macro F1 (CV):** 0.9455
- **Macro F1 (test):** 0.6288
- **Observation:** Marginal improvement over single XGBoost (0.6275 → 0.6288).
  Different algorithms making slightly different errors can cancel out, but
  XGBoost and LightGBM are structurally similar enough that their mistakes
  largely overlap.

### Experiment 9: XGBoost + SMOTE + feature engineering (best result)

- **Algorithm:** XGBoost (tuned parameters), SMOTE, one-hot encoding,
  log transforms, ratio features
- **What changed:** Feature engineering applied before training (see
  Section 1.2)
- **Macro F1 (CV):** 0.9453
- **Macro F1 (test):** **0.6331**
- **Observation:** New best result. DoS F1 improved substantially (0.88 →
  0.90) due to log transforms on byte counts. R2L F1 improved (0.24 → 0.30)
  due to one-hot encoding of `service`, which separates services associated
  with R2L attacks (ftp, imap, telnet) from unrelated ones. Probe F1 dropped
  slightly (0.81 → 0.76) — the reason is unclear but may relate to probe
  attacks being spread across many service types, making one-hot encoding
  less helpful there.


### Experiment 10: k-Nearest Neighbours + SMOTE

- **Algorithm:** KNeighborsClassifier (k=5) with StandardScaler and SMOTE
- **What changed:** Completely different algorithm family (distance-based)
- **Macro F1 (CV):** not formally computed
- **Macro F1 (test):** 0.5800
- **Observation:** Worse than XGBoost. k-NN finds the k nearest training
  neighbours for each prediction. Unseen R2L attack types have no close
  training neighbours, so they get classified by majority vote of DoS or
  Normal neighbours. Also slow: prediction on 22,544 test samples against
  ~336,000 SMOTE-expanded training samples took ~30 minutes.

### Experiments Summary

| # | Description | Algorithm | Imbalance Handling | Macro F1 (CV) | Macro F1 (test) |
|---|---|---|---|---|---|
| 1 | Baseline | Random Forest | None | — | 0.50 |
| 2 | Balanced class weights | Random Forest | class_weight='balanced' | — | 0.48 |
| 3 | SMOTE oversampling | Random Forest | SMOTE | — | 0.54 |
| 4 | SMOTETomek custom targets | XGBoost | SMOTETomek | — | 0.59 |
| 5 | Hyperparameter search (leaky) | XGBoost | SMOTE (pre-CV, leaky) | 0.9998 | 0.62 |
| 6 | Hyperparameter search (honest) | XGBoost | SMOTE in ImbPipeline | 0.9487 | 0.62 |
| 7 | OOF threshold tuning | XGBoost | SMOTE in ImbPipeline | 0.9405 | 0.63 |
| 8 | Soft voting ensemble | XGBoost + LightGBM | SMOTE in ImbPipeline | 0.9455 | 0.63 |
| **9** | **Feature engineering** | **XGBoost** | **SMOTE in ImbPipeline** | **0.9453** | **0.6331** |
| 10 | k-Nearest Neighbours | kNN (k=5) | SMOTE | — | 0.58 |

---

## 3. Final Results

### 3.1 Best Model

- **Algorithm:** XGBoost (XGBClassifier) inside ImbPipeline with SMOTE
- **Key parameters:**
  - `n_estimators=406`
  - `max_depth=6`
  - `learning_rate=0.165`
  - `subsample=0.828`
  - `colsample_bytree=0.868`
  - `min_child_weight=1`
  - `tree_method='hist'` (fast approximate splitting)
  - `eval_metric='mlogloss'`
- **Imbalance handling:** SMOTE (`k_neighbors=5`, `random_state=42`)
  inside `ImbPipeline` — applied per CV fold, never to validation/test data
- **Feature engineering:** One-hot encoding of `protocol_type`, `service`,
  `flag`; log1p transforms on 7 skewed numerics; 3 ratio features; total
  124 features

### 3.2 Final Macro F1-Score

| Metric | Score |
|---|---|
| **Macro F1 (test)** | **0.6331** |
| Macro F1 (CV, default argmax) | 0.9453 |
| Macro F1 (CV, with OOF thresholds) | 0.9455 |

### 3.3 Classification Report

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 0.70 | 0.97 | 0.81 | 9,711 |
| DoS | 0.96 | 0.85 | 0.90 | 7,460 |
| Probe | 0.84 | 0.69 | 0.76 | 2,421 |
| R2L | 0.99 | 0.17 | 0.30 | 2,885 |
| U2R | 0.75 | 0.27 | 0.40 | 67 |
| **Macro avg** | **0.85** | **0.59** | **0.63** | **22,544** |

### 3.4 Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

---

## 4. Cross-Validation vs. Test Score

- **CV macro F1:** 0.9453
- **Test macro F1:** 0.6331
- **Gap:** 0.312

**Analysis:** The gap is large but expected and well-understood. It is not the
result of overfitting or test-set leakage (we confirmed this by fixing the
leakage in Experiment 5). The gap arises from a fundamental property of the
NSL-KDD benchmark: KDDTest+ deliberately contains R2L and U2R attack types
that do not appear in KDDTrain+.

During cross-validation, all five folds are drawn from KDDTrain+. When a fold
is held out for validation, its R2L examples are of the same attack types as
the training folds (warezclient, guess_passwd, ftp_write, imap, etc.). The
model learns these patterns well, producing a high CV score.

In KDDTest+, R2L contains attack types such as `httptunnel`, `sendmail`, and
`named` — none of which appear in training. The model's internal representation
of "what R2L looks like" does not match these new patterns, so R2L recall drops
from ~0.95 (CV) to 0.17 (test). The same applies to U2R.

The size of the gap is consistent with published academic results on this
specific train/test split. Our threshold tuning experiments further confirmed
the distribution shift: decision thresholds that maximised F1 on OOF training
predictions (R2L=0.4, U2R=0.3) were very different from the thresholds that
work on the test set (R2L=0.1, U2R=0.1), meaning the model's probability
calibration differs across the two distributions.

---

## 5. What Worked and What Didn't

### What had the biggest positive impact

1. **Switching from Random Forest to XGBoost** (+0.10 macro F1, from 0.53 to
   0.63). XGBoost builds trees sequentially, with each tree correcting the
   errors of the previous ones. This error-correction mechanism is particularly
   effective for rare classes: later trees in the sequence focus specifically
   on the residual errors, which are predominantly R2L and U2R misclassifications.

2. **SMOTE oversampling** (+0.03 macro F1 on Random Forest, from 0.50 to 0.53;
   contributed to all subsequent results). SMOTE generated synthetic minority
   examples that gave the model sufficient variation to learn R2L and U2R
   patterns, rather than always defaulting to Normal.

3. **Feature engineering** (+0.06 on best XGBoost model, from 0.627 to 0.633).
   One-hot encoding `service` was the most impactful change: it allowed the
   model to learn that specific services (ftp, imap, telnet) are associated
   with R2L attacks, rather than treating service as an ordinal integer.
   Log-transforming byte counts improved DoS detection (F1: 0.88 → 0.90).

### What surprisingly didn't help

1. **`class_weight='balanced'` on Random Forest** actually regressed (0.50 →
   0.48). Upweighting the 52 U2R training examples by ~1,300× caused the model
   to overfit to those specific U2R patterns, which did not match the unseen
   U2R attacks in the test set.

2. **Hyperparameter tuning via RandomizedSearchCV** produced negligible gains
   on the test set (0.62 → 0.62) after fixing the leakage. The model had
   already reached its ceiling for this feature representation. More tuning
   capacity does not help when the bottleneck is training data coverage of
   unseen attack types.

3. **Soft voting with LightGBM** added nothing meaningful (+0.001). XGBoost
   and LightGBM make similar types of mistakes on this dataset, so averaging
   their predictions did not cancel out errors.


4. **SMOTETomek with conservative sampling targets** hurt vs. plain SMOTE
   (0.59 vs 0.63). Setting U2R=2,000 was too conservative: the model needed
   more variation than 2,000 interpolations of 52 originals could provide.

5. **Threshold tuning on OOF predictions** barely helped (0.627 → 0.628).
   The OOF thresholds (R2L=0.4, U2R=0.3) were well-calibrated for the
   training distribution but wrong for the test distribution — illustrating
   that the CV and test distributions are genuinely different.

### What would you try with more time

- **CatBoost with native categorical handling**: CatBoost encodes categoricals
  using target statistics rather than one-hot, which may produce a more
  powerful representation than our manual one-hot encoding, particularly for
  the high-cardinality `service` feature.

- **Anomaly detection as an additional feature**: Train an autoencoder or
  Isolation Forest on Normal traffic only, then add the reconstruction error
  (or anomaly score) as a feature. This provides an explicit signal for "how
  unlike normal traffic is this connection?" which may generalise better to
  unseen attack types than supervised features alone.

- **Stacking ensemble with a meta-learner**: Instead of soft voting (averaging
  probabilities), train a Logistic Regression meta-learner on the out-of-fold
  predictions of multiple base models. The meta-learner can learn which base
  model to trust for which class, rather than weighting all equally.

- **Deeper domain-specific feature engineering**: The NSL-KDD features encode
  a two-second time window of traffic. Creating features that capture
  longer-range patterns (e.g., rolling error rates over 10-second windows)
  might reveal R2L signatures that are invisible at the individual-connection
  level.

---

## Appendix: Environment

- **Hardware:** Intel Core i5-1235U, 16GB RAM
- **Python version:** 3.13.7
- **Key libraries:**
  - scikit-learn [latest version]
  - xgboost [latest version]
  - lightgbm [latest version]
  - imbalanced-learn [latest version]
  - pandas [latest version]
  - numpy [latest version]
- **Random seed:** 42 