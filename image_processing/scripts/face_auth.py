from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage import data as skdata
from skimage.feature import Cascade, hog, local_binary_pattern

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_model.pkl"

FACE_SIZE = 128
LBP_POINTS, LBP_RADIUS, LBP_GRID = 8, 1, 4
HOG_SETTINGS = dict(orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), block_norm="L2-Hys")
face_detector = Cascade(skdata.lbp_frontal_face_cascade_filename())
_bundles = {}


def limit_size(img, max_side=1200):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def crop_face(image, size=FACE_SIZE, margin=0.15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smallest = max(int(0.04 * min(gray.shape)), 24)
    faces = face_detector.detect_multi_scale(img=gray, scale_factor=1.1, step_ratio=1,
                                             min_size=(smallest, smallest),
                                             max_size=(min(gray.shape), min(gray.shape)))
    if len(faces) == 0:
        return cv2.resize(image, (size, size)), False
    f = max(faces, key=lambda d: d["width"] * d["height"])
    x, y, w, h = f["c"], f["r"], f["width"], f["height"]
    m = int(margin * w)
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, image.shape[1]), min(y + h + m, image.shape[0])
    return cv2.resize(image[y0:y1, x0:x1], (size, size)), True


def extract_features(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    feats = {}
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / hist.sum()
    for i, v in enumerate(hist):
        feats[f"hist_{i}"] = float(v)
    equalised = cv2.equalizeHist(gray)
    lbp = local_binary_pattern(equalised, LBP_POINTS, LBP_RADIUS, method="uniform")
    cell = FACE_SIZE // LBP_GRID
    for r in range(LBP_GRID):
        for c in range(LBP_GRID):
            patch = lbp[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            h, _ = np.histogram(patch, bins=np.arange(0, LBP_POINTS + 3))
            h = h / h.sum()
            for b, v in enumerate(h):
                feats[f"lbp_{r}{c}_{b}"] = float(v)
    for i, v in enumerate(hog(equalised, **HOG_SETTINGS)):
        feats[f"hog_{i}"] = float(v)
    return feats


def predict_face(image_path, model_path=DEFAULT_MODEL_PATH, details=False):
    key = str(model_path)
    if key not in _bundles:
        _bundles[key] = joblib.load(model_path)
    bundle = _bundles[key]

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    face, found = crop_face(limit_size(image))
    if not found:
        label, confidence = "unknown", 0.0
    else:
        x = pd.DataFrame([extract_features(face)])[bundle["feature_cols"]]
        proba = bundle["model"].predict_proba(x)[0]
        confidence = float(proba.max())
        label = str(bundle["model"].classes_[proba.argmax()]) if confidence >= bundle["threshold"] else "unknown"
        if "max_distance" in bundle:
            v = bundle["gate_scaler"].transform(x)
            if float(np.sqrt(((bundle["gate_matrix"] - v) ** 2).sum(axis=1)).min()) > bundle["max_distance"]:
                label = "unknown"
    return (label, confidence) if details else label
