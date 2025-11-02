import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def train_hazard_logit(X, y, class_weight="balanced", calibrate=True, splits=None):
    # Simple global fit; CPCV used only for OOF predictions in evaluation for now.
    model = LogisticRegression(max_iter=200, class_weight=class_weight, n_jobs=None)
    model.fit(X, y)
    cal_model = None
    if calibrate:
        cal_model = CalibratedClassifierCV(model, method="isotonic", cv=5)
        cal_model.fit(X, y)
    return model, cal_model
