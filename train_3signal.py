"""
Retrains a classifier that actually uses Track + grades + Skills/Interests
together (unlike final_model_pipeline.pkl, whose drop_skill_interest step
throws Skills/Interests away before the classifier ever sees them).

Reuses model_utils' existing transformer classes in the exact same order
already proven to work in profile_pipeline.pkl, just with a classifier
bolted on the end instead of stopping after the multi-hot encoding.

Output: models/track_grades_skills_model.pkl
    - Standard sklearn Pipeline, decodable with the SAME label_enc.pkl
      already bundled (classes were merged 33->29 to match it exactly).
    - app.py auto-discovers anything in models/ as a selectable classifier,
      so this shows up as a second option next to "Final Model Pipeline"
      without needing to touch that one.
"""
import sys, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/isa")

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, classification_report

import model_utils
from model_utils import (FeatureEngineer, TrackOneHotEncoder, TextToList,
                          TopKMultiHotEncoder, DropRawTextCols, DropNearZeroVariance,
                          SUBJECT_COLS)

t0 = time.time()
DATA_PATH = "/home/claude/isa/thanaweya_final_with_profiles_modified.csv"
LABEL_ENC_PATH = "/home/claude/isa/label_enc.pkl"
OUT_PATH = "/home/claude/isa/models/track_grades_skills_model.pkl"

print("Loading data...")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

# ---- align target classes with the existing label_enc.pkl (29 classes) ----
label_enc = joblib.load(LABEL_ENC_PATH)
KNOWN = set(label_enc.classes_.tolist())
CATEGORY_MERGE = {
    "Science Petroleum and Mining": "Science",
    "Information Systems": "Computers and Information",
    "Artificial Intelligence": "Computers and Information",
    "Archaeology and Languages": "Archaeology",
}
df["Faculty_Category"] = df["Faculty_Category"].replace(CATEGORY_MERGE)
unknown = set(df["Faculty_Category"].unique()) - KNOWN
assert not unknown, f"Still have unmapped categories: {unknown}"
print("Faculty_Category classes now match label_enc.pkl's 29 classes.")

# ---- Track: long labels -> short codes the pipelines expect ----
TRACK_MAP = {
    "Literary Track": "Literacy",
    "Scientific - Science Track": "Science",
    "Scientific - Math Track": "Math",
}
df["Track"] = df["Track"].map(TRACK_MAP)
assert df["Track"].isna().sum() == 0, "Unmapped Track values found"

# ---- build feature frame ----
keep_cols = SUBJECT_COLS + ["Track", "Skills", "Interests", "Faculty_Category"]
data = df[keep_cols].copy()
data["Skills"] = data["Skills"].fillna("")
data["Interests"] = data["Interests"].fillna("")

y_full = label_enc.transform(data["Faculty_Category"])
X_full = data.drop(columns=["Faculty_Category"])

# ---- stratified subsample (single-CPU sandbox; full 666k would be too slow) ----
SAMPLE_N = 180_000
if len(X_full) > SAMPLE_N:
    X_sample, _, y_sample, _ = train_test_split(
        X_full, y_full, train_size=SAMPLE_N, stratify=y_full, random_state=42)
else:
    X_sample, y_sample = X_full, y_full
print(f"Working sample: {len(X_sample):,} rows (stratified from {len(X_full):,})")

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.15, stratify=y_sample, random_state=42)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# ---- pipeline: same feature-engineering order as profile_pipeline.pkl ----
pipeline = Pipeline(steps=[
    ("feature_engineer", FeatureEngineer()),
    ("track_ohe", TrackOneHotEncoder()),
    ("text_to_list", TextToList(cols=["Skills", "Interests"])),
    ("skills_mh", TopKMultiHotEncoder(list_col="Skills", prefix="skill", k=30)),
    ("interests_mh", TopKMultiHotEncoder(list_col="Interests", prefix="interest", k=30)),
    ("drop_raw_text", DropRawTextCols(cols=["Skills", "Interests"])),
    ("drop_low_var", DropNearZeroVariance(threshold=0.001)),
    ("classifier", HistGradientBoostingClassifier(
        max_iter=150, max_depth=7, learning_rate=0.1, l2_regularization=0.5,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, random_state=42)),
])

print("Fitting pipeline (feature engineering + classifier)... this can take a few minutes on 1 CPU.")
t1 = time.time()
pipeline.fit(X_train, y_train)
print(f"  fit done in {time.time()-t1:.1f}s")

# ---- evaluate ----
*pre, (_, clf) = pipeline.steps
Xt_test = X_test
for _, step in pre:
    Xt_test = step.transform(Xt_test)
proba_test = clf.predict_proba(Xt_test)
pred_test = clf.classes_[np.argmax(proba_test, axis=1)]

acc = accuracy_score(y_test, pred_test)
f1m = f1_score(y_test, pred_test, average="macro")
top3 = top_k_accuracy_score(y_test, proba_test, k=3, labels=clf.classes_)
print(f"Test accuracy: {acc:.4f}")
print(f"Test macro-F1: {f1m:.4f}")
print(f"Test top-3 accuracy: {top3:.4f}")

# also report train accuracy for reference (mild overfit check)
Xt_train = X_train
for _, step in pre:
    Xt_train = step.transform(Xt_train)
pred_train = clf.classes_[np.argmax(clf.predict_proba(Xt_train), axis=1)]
print(f"Train accuracy: {accuracy_score(y_train, pred_train):.4f}")

report = classification_report(y_test, pred_test, target_names=label_enc.classes_,
                                labels=np.arange(len(label_enc.classes_)), zero_division=0)
with open("/home/claude/isa/train_3signal_report.txt", "w") as f:
    f.write(f"Test accuracy: {acc:.4f}\nTest macro-F1: {f1m:.4f}\nTest top-3 accuracy: {top3:.4f}\n\n")
    f.write(report)
print("Wrote classification report to train_3signal_report.txt")

# ---- save ----
import os
os.makedirs("/home/claude/isa/models", exist_ok=True)
joblib.dump(pipeline, OUT_PATH)
print(f"Saved pipeline to {OUT_PATH}")
print(f"Total time: {time.time()-t0:.1f}s")
