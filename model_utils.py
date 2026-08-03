"""
model_utils.py
================
Custom scikit-learn transformer classes used by the Masar AI capstone
pipeline (ML_Final_Capstone_Project_Masar_AI.ipynb). Matches the classes
needed to unpickle: profile_pipeline.pkl, and any exported classifier
pipeline (best_rf.pkl / cat_pipeline.pkl / final_model_pipeline.pkl).

Why this file exists
---------------------
joblib/pickle store a *reference* to a class (module + name), not the class
code itself. A pipeline built in a notebook has its classes attached to
`__main__`. To load that same pipeline inside the Streamlit app (a
different `__main__`), Python needs classes with the exact same names
available again. app.py imports this file and calls
`register_classes_on_main()` before loading any uploaded/bundled model, so
unpickling works regardless of where the pipeline was originally trained.

Keep this file in sync with the notebook if the preprocessing classes ever
change.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

SCIENCE_COLS = ['chemistry', 'biology', 'geology', 'physics', 'pure_mathematics', 'applied_math']
HUMANITIES_COLS = ['history', 'geography', 'philosophy', 'psychology']
LANGUAGE_COLS = ['arabic', 'first_foreign_lang', 'second_foreign_lang']
SUBJECT_COLS = LANGUAGE_COLS + ['pure_mathematics', 'history', 'geography', 'philosophy',
                                 'psychology', 'chemistry', 'biology', 'geology',
                                 'applied_math', 'physics']


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Structural imputation (subject not taken -> 0) + engineered aggregate features.
    Learns only the per-Track max total (needed for total_pct) from the training data."""

    def fit(self, X, y=None):
        X = X.copy()
        X[SUBJECT_COLS] = X[SUBJECT_COLS].fillna(0)
        total = X[SUBJECT_COLS].sum(axis=1)
        self.track_max_total_ = total.groupby(X['Track']).max().to_dict()
        self.global_max_total_ = total.max()
        return self

    def transform(self, X):
        X = X.copy()
        X[SUBJECT_COLS] = X[SUBJECT_COLS].fillna(0)
        X['total'] = X[SUBJECT_COLS].sum(axis=1)
        X['avg_subject_score'] = X[SUBJECT_COLS].replace(0, np.nan).mean(axis=1).fillna(0)
        track_max = X['Track'].map(self.track_max_total_).fillna(self.global_max_total_)
        X['total_pct'] = (X['total'] / track_max).clip(upper=1.0)
        X['science_score_avg'] = X[SCIENCE_COLS].replace(0, np.nan).mean(axis=1).fillna(0)
        X['humanities_score_avg'] = X[HUMANITIES_COLS].replace(0, np.nan).mean(axis=1).fillna(0)
        X['language_score_avg'] = X[LANGUAGE_COLS].replace(0, np.nan).mean(axis=1).fillna(0)
        return X


class TopKMultiHotEncoder(BaseEstimator, TransformerMixin):
    """Multi-hot encodes a free-text comma separated column using the top-K tokens
    learned from the training set only, with an 'Other' catch-all bucket."""

    def __init__(self, list_col, prefix, k=30):
        self.list_col = list_col
        self.prefix = prefix
        self.k = k

    def fit(self, X, y=None):
        counter = Counter()
        for lst in X[self.list_col]:
            counter.update(lst)
        self.vocab_ = [tok for tok, _ in counter.most_common(self.k)]
        return self

    def transform(self, X):
        X = X.copy()
        for tok in self.vocab_:
            col = f'{self.prefix}_{tok.replace(" ", "_")}'
            X[col] = X[self.list_col].apply(lambda lst: int(tok in lst))
        X[f'{self.prefix}_Other'] = X[self.list_col].apply(lambda lst: int(any(t not in self.vocab_ for t in lst)))
        return X


class DropRawTextCols(BaseEstimator, TransformerMixin):
    """Drops the intermediate free-text / list columns once they've been encoded."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])


class TrackOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encodes the 3-category Track column."""

    def fit(self, X, y=None):
        self.categories_ = sorted(X['Track'].dropna().unique().tolist())
        return self

    def transform(self, X):
        X = X.copy()
        for cat in self.categories_:
            X[f'Track_{cat}'] = (X['Track'] == cat).astype(int)
        return X.drop(columns=['Track'])


class DropPrefixedCols(BaseEstimator, TransformerMixin):
    """Drops any column whose name starts with one of the given prefixes.
    Used to keep skill_/interest_ multi-hot columns OUT of the classifier's
    feature selection + model training, while the separate profile pipeline
    (used only for the recommender's content-based similarity) keeps them."""

    def __init__(self, prefixes):
        self.prefixes = prefixes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_to_drop = [c for c in X.columns if any(c.startswith(p) for p in self.prefixes)]
        return X.drop(columns=cols_to_drop)


class TextToList(BaseEstimator, TransformerMixin):
    """Converts a comma-separated string column into a list of strings."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.cols:
            X_copy[col] = X_copy[col].apply(
                lambda x: [item.strip() for item in str(x).split(',') if item.strip()] if pd.notna(x) else []
            )
        return X_copy


class DropSpecificCols(BaseEstimator, TransformerMixin):
    """Drops a list of specified columns."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])


class SelectiveYeoJohnson(BaseEstimator, TransformerMixin):
    """Applies Yeo-Johnson only to numeric columns whose |skew| > threshold on the
    training data."""

    def __init__(self, cols, skew_threshold=0.75):
        self.cols = cols
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        self.skewed_cols_ = [c for c in self.cols if abs(X[c].skew()) > self.skew_threshold]
        if self.skewed_cols_:
            self.pt_ = PowerTransformer(method='yeo-johnson')
            self.pt_.fit(X[self.skewed_cols_])
        else:
            self.pt_ = None
        return self

    def transform(self, X):
        X = X.copy()
        if self.skewed_cols_:
            X[self.skewed_cols_] = self.pt_.transform(X[self.skewed_cols_])
        return X


class SelectiveScaler(BaseEstimator, TransformerMixin):
    """StandardScaler applied only to the numeric feature columns."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = self.scaler_.transform(X[self.cols])
        return X


class DropNearZeroVariance(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.001):
        self.threshold = threshold

    def fit(self, X, y=None):
        vt = VarianceThreshold(threshold=self.threshold)
        vt.fit(X)
        self.kept_cols_ = X.columns[vt.get_support()].tolist()
        self.dropped_cols_ = X.columns[~vt.get_support()].tolist()
        return self

    def transform(self, X):
        return X[self.kept_cols_]


class DropCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_)


class SelectTopKImportance(BaseEstimator, TransformerMixin):
    """Ranks features by RandomForest importance (training fold only) and keeps the top-k."""

    def __init__(self, k=40, n_estimators=150, max_depth=20, min_samples_leaf=10, random_state=42):
        self.k = k
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        rf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf, max_features='sqrt',
                                     random_state=self.random_state, n_jobs=-1, class_weight='balanced')
        rf.fit(X, y, sample_weight=sample_weight)
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        self.selected_features_ = importances.head(min(self.k, len(importances))).index.tolist()
        self.importances_ = importances
        return self

    def transform(self, X):
        return X[self.selected_features_]


def register_classes_on_main():
    """Injects every transformer class defined here into sys.modules['__main__'],
    so joblib/pickle can resolve classes that were originally pickled from a
    notebook's __main__ namespace. Call this once, before loading a model file."""
    import sys
    main_mod = sys.modules.get('__main__')
    if main_mod is None:
        return
    for name, obj in list(globals().items()):
        if isinstance(obj, type):
            setattr(main_mod, name, obj)
