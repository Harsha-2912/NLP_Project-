"""
Model Architectures for Toxic Comment Classification.
Implements:
1. Multilabel Binary Relevance (One-vs-Rest) with Calibrated Logistic Regression.
2. Multilabel Classifier Chains (capturing label correlation dependencies).
3. Binary Toxic/Non-toxic Classifier with threshold calibration.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
import joblib

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_base_estimator(model_type: str = 'logistic_regression', class_weight: str = 'balanced'):
    """Returns the base classification algorithm."""
    if model_type == 'logistic_regression':
        return LogisticRegression(
            C=2.5,
            max_iter=1000,
            class_weight=class_weight,
            solver='liblinear',
            random_state=42
        )
    elif model_type == 'calibrated_svc':
        base_svc = LinearSVC(
            C=1.0,
            class_weight=class_weight,
            random_state=42,
            dual=False
        )
        return CalibratedClassifierCV(estimator=base_svc, cv=3)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class MultilabelToxicClassifier:
    """
    Multilabel Classifier for the 6 Jigsaw toxic labels:
    ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'].
    
    Supports:
    - Binary Relevance (independent One-vs-Rest)
    - Classifier Chains (modeling dependencies across labels)
    - Per-label threshold calibration
    """
    def __init__(
        self,
        strategy: str = 'binary_relevance',  # 'binary_relevance' or 'classifier_chain'
        model_type: str = 'logistic_regression',
        labels: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        self.strategy = strategy
        self.model_type = model_type
        self.labels = labels or LABELS
        self.models: Dict[str, any] = {}
        self.chain_model = None
        # Default decision threshold (0.5 or custom per-label tuned)
        self.thresholds = thresholds or {l: 0.5 for l in self.labels}
        self.is_fitted = False

    def fit(self, X, Y):
        """
        Fits the multilabel classifier.
        Y: 2D array or DataFrame of shape (n_samples, n_labels)
        """
        if isinstance(Y, np.ndarray):
            y_arr = Y
        else:
            y_arr = Y[self.labels].values

        if self.strategy == 'binary_relevance':
            for idx, label in enumerate(self.labels):
                y_label = y_arr[:, idx]
                clf = get_base_estimator(self.model_type, class_weight='balanced')
                clf.fit(X, y_label)
                self.models[label] = clf
        elif self.strategy == 'classifier_chain':
            base_clf = get_base_estimator(self.model_type, class_weight='balanced')
            self.chain_model = ClassifierChain(base_clf, order='random', random_state=42)
            self.chain_model.fit(X, y_arr)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Returns probability estimates for each label as a 2D array: (n_samples, n_labels).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")

        if self.strategy == 'binary_relevance':
            prob_list = []
            for label in self.labels:
                clf = self.models[label]
                # P(class=1)
                p1 = clf.predict_proba(X)[:, 1]
                prob_list.append(p1)
            return np.column_stack(prob_list)
        elif self.strategy == 'classifier_chain':
            return self.chain_model.predict_proba(X)

    def predict(self, X, custom_thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Generates binary predictions based on decision thresholds.
        """
        probs = self.predict_proba(X)
        thresholds = custom_thresholds or self.thresholds
        thresh_arr = np.array([thresholds.get(lbl, 0.5) for lbl in self.labels])
        return (probs >= thresh_arr).astype(int)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)


class BinaryToxicClassifier:
    """
    Binary classifier for overall toxic vs. non-toxic classification.
    """
    def __init__(self, model_type: str = 'logistic_regression', threshold: float = 0.5):
        self.model_type = model_type
        self.threshold = threshold
        self.model = get_base_estimator(model_type, class_weight='balanced')
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
