import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ==============================================================
# 1. LOAD DATA
# ==============================================================
train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test  = pd.read_csv(test_url,  names=columns)

df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'],  inplace=True)

# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================
df_full = pd.concat([df_train, df_test])

cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================
category_map = {
    'normal': 'Normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed  = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']
X_test  = df_test_processed.drop(columns=['category'])
y_test  = df_test_processed['category']

print(f"Training set: {X_train.shape[0]} records, {X_train.shape[1]} features")
print(f"Test set:     {X_test.shape[0]} records,  {X_test.shape[1]} features")
print(f"\nTraining class distribution:\n{y_train.value_counts()}")
print(f"\nTest class distribution:\n{y_test.value_counts()}")

# ==============================================================
# 5. FEATURE ENGINEERING
# ==============================================================
skewed_numeric = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
                  'dst_host_count', 'dst_host_srv_count']

# One-hot encode categoricals (fit on train only, transform both)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(X_train[cat_cols])

train_ohe = pd.DataFrame(ohe.transform(X_train[cat_cols]),
                         columns=ohe.get_feature_names_out(cat_cols),
                         index=X_train.index)
test_ohe  = pd.DataFrame(ohe.transform(X_test[cat_cols]),
                         columns=ohe.get_feature_names_out(cat_cols),
                         index=X_test.index)

def engineer(df):
    df = df.copy()
    for col in skewed_numeric:
        df[col] = np.log1p(df[col])
    df['bytes_ratio']        = df['src_bytes'] / (df['dst_bytes'] + 1)
    df['error_rate_avg']     = (df['serror_rate']     + df['rerror_rate'])     / 2
    df['srv_error_rate_avg'] = (df['srv_serror_rate'] + df['srv_rerror_rate']) / 2
    return df

X_train = pd.concat([engineer(X_train).drop(columns=cat_cols), train_ohe], axis=1)
X_test  = pd.concat([engineer(X_test).drop(columns=cat_cols),  test_ohe],  axis=1)

print(f"\nFeature count after engineering: {X_train.shape[1]}")

# ==============================================================
# 6. ENCODE LABELS FOR XGBOOST
# ==============================================================
le_y = LabelEncoder()
y_train_enc = le_y.fit_transform(y_train)

# ==============================================================
# 7. BUILD PIPELINE
# ==============================================================
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(
        n_estimators=406,
        max_depth=6,
        learning_rate=0.16564507699318415,
        subsample=0.8282623055075649,
        colsample_bytree=0.8683831592708489,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        tree_method='hist',
    ))
])

# ==============================================================
# 8. CROSS-VALIDATION (mandatory for report)
# ==============================================================
print("\nRunning 3-fold cross-validation...")
cv_scores = cross_val_score(
    pipeline, X_train, y_train_enc,
    cv=3, scoring='f1_macro', n_jobs=-1
)
print(f"CV macro F1: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# ==============================================================
# 9. TRAIN FINAL MODEL AND EVALUATE ON TEST SET
# ==============================================================
print("\nFitting final model on full training set...")
pipeline.fit(X_train, y_train_enc)

y_pred_enc = pipeline.predict(X_test)
y_pred     = le_y.inverse_transform(y_pred_enc)

test_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n=== Final Results on KDDTest+ ===")
print(f"CV macro F1:   {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
print(f"Test macro F1: {test_f1:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# ==============================================================
# 9b. THRESHOLD TUNING (OOF-based, no test leakage)
# ==============================================================
from sklearn.model_selection import StratifiedKFold

r2l_idx = list(le_y.classes_).index('R2L')
u2r_idx = list(le_y.classes_).index('U2R')

def threshold_predict(proba, r2l_thresh, u2r_thresh):
    pred = np.argmax(proba, axis=1)
    pred = np.where(proba[:, r2l_idx] > r2l_thresh, r2l_idx, pred)
    pred = np.where(proba[:, u2r_idx] > u2r_thresh, u2r_idx, pred)
    return pred

print("\nRunning OOF predictions for threshold selection...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_proba = np.zeros((len(X_train), len(le_y.classes_)))

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc)):
    print(f"  Fold {fold_idx + 1}/3...")
    fold_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=406, max_depth=6,
            learning_rate=0.16564507699318415,
            subsample=0.8282623055075649,
            colsample_bytree=0.8683831592708489,
            min_child_weight=1,
            random_state=42, n_jobs=-1,
            eval_metric='mlogloss', tree_method='hist',
        ))
    ])
    fold_pipe.fit(X_train.iloc[tr_idx], y_train_enc[tr_idx])
    oof_proba[val_idx] = fold_pipe.predict_proba(X_train.iloc[val_idx])

# Find best thresholds on OOF predictions only
print("Sweeping thresholds...")
best = (0.0, None, None)
for rt in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    for ut in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        pred = threshold_predict(oof_proba, rt, ut)
        score = f1_score(y_train_enc, pred, average='macro')
        if score > best[0]:
            best = (score, rt, ut)

_, best_r2l, best_u2r = best
print(f"Chosen thresholds: R2L={best_r2l}, U2R={best_u2r}")

# Apply frozen thresholds to test predictions
y_proba_test = pipeline.predict_proba(X_test)
pred_enc = threshold_predict(y_proba_test, best_r2l, best_u2r)
y_pred = le_y.inverse_transform(pred_enc)

test_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n=== Final Results on KDDTest+ ===")
print(f"CV macro F1:   {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
print(f"Test macro F1: {test_f1:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# ==============================================================
# 10. CONFUSION MATRIX
# ==============================================================
import matplotlib.pyplot as plt  # installed automatically with seaborn
import seaborn as sns
from sklearn.metrics import confusion_matrix

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()