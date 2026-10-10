"""Final predictors trained on the selected genes."""

from __future__ import annotations

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .survival import RidgeCox


def make_ensemble(seed: int = 42) -> VotingClassifier:
    """Five-member soft-voting ensemble used in the published HFW-NPO."""
    return VotingClassifier(
        estimators=[
            ("et", ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                        min_samples_leaf=2, random_state=seed, n_jobs=1)),
            ("svm", make_pipeline(StandardScaler(),
                                  SVC(kernel="rbf", C=1.0, probability=True,
                                      class_weight="balanced", random_state=seed))),
            ("lr", make_pipeline(StandardScaler(),
                                 LogisticRegression(C=1.0, class_weight="balanced",
                                                    max_iter=5000, random_state=seed))),
            ("rf", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                          min_samples_leaf=2, random_state=seed, n_jobs=1)),
            ("gb", GradientBoostingClassifier(random_state=seed)),
        ],
        voting="soft",
    )


def smote_balance(X, y, seed: int = 42):
    """Adaptive SMOTE as in HFW-NPO (skipped when the minority class is tiny)."""
    counts = Counter(np.asarray(y).tolist())
    if len(counts) < 2 or min(counts.values()) < 2:
        return X, y
    k = min(5, min(counts.values()) - 1)
    return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)


def make_cox() -> RidgeCox:
    return RidgeCox(l2=1.0)
