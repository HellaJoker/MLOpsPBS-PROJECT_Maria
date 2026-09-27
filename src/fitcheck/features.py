"""Feature engineering shared between training (Notebooks/MLOps.ipynb) and serving (api/app.py).

Why this lives in a module and not in the notebook:
the optimized model is a scikit-learn Pipeline that *contains* `add_fit_features`.
Pickles store functions by import path, so a function defined in a notebook cell
(`__main__.add_fit_features`) cannot be loaded by the API process. Keeping it here
makes the same code importable as `fitcheck.features` everywhere.
"""

import pandas as pd

# Class ids produced by the labelling rule in the notebook.
CLASS_LABELS = {0: "Too Small", 1: "Good Fit", 2: "Too Large"}

# Raw inputs a client sends. The optimized pipeline consumes exactly these columns.
RAW_FEATURES = [
    "height_cm",
    "weight_kg",
    "garment_chest_cm",
    "fabric_stretch_pct",
    "product_type_id",  # 0: Tops, 1: Bottoms, 2: Jackets
]

# Physical bounds used when the training data was built (np.clip in the notebook).
# The API validates against these so it never scores inputs outside the training range.
BOUNDS = {
    "height_cm": (140.0, 220.0),
    "weight_kg": (40.0, 150.0),
    "garment_chest_cm": (60.0, 140.0),
    "fabric_stretch_pct": (0.0, 20.0),
}


def bmi(height_cm, weight_kg):
    """Body Mass Index (the notebook calls it `imc_index`)."""
    return weight_kg / ((height_cm / 100.0) ** 2)


# ---------------------------------------------------------------------------
# Baseline model (fitcheck_gb_model.joblib)
# ---------------------------------------------------------------------------
def build_baseline_features(raw: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """Reproduce the baseline's training transform: BMI + one-hot product type.

    `feature_columns` is the list saved in model_features.joblib; reindexing to it
    guarantees the same column order and fills absent dummy columns with 0.
    """
    df = raw.copy()
    df["imc_index"] = bmi(df["height_cm"], df["weight_kg"])
    for type_id in (0, 1, 2):
        df[f"product_type_id_{type_id}"] = (df["product_type_id"] == type_id).astype(int)
    return df.reindex(columns=list(feature_columns), fill_value=0)


# ---------------------------------------------------------------------------
# Optimized model (fitcheck_gb_model_optimized.joblib)
# ---------------------------------------------------------------------------
def add_fit_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add body-vs-garment features for the optimized pipeline.

    Gradient-boosted trees split on one feature at a time, so a boundary like
    "garment is big enough *for this body*" (a diagonal in height/weight/garment space)
    is approximated by a staircase of many splits. Giving the model explicit
    garment-relative-to-body features turns that diagonal into something a few
    splits can capture.

    Note: these are generic, physically motivated features. They deliberately do NOT
    copy the coefficients of the synthetic labelling rule (0.45*weight + 0.35*height),
    which would make the model trivially reproduce the rule instead of learning it.
    """
    df = raw[RAW_FEATURES].copy()
    df["imc_index"] = bmi(df["height_cm"], df["weight_kg"])

    # Stretch lets the garment open up: a 100 cm chest with 5% stretch behaves like ~105 cm.
    df["effective_chest_cm"] = df["garment_chest_cm"] * (1.0 + df["fabric_stretch_pct"] / 100.0)

    # Garment size relative to each body dimension (ratios and differences).
    df["chest_per_kg"] = df["effective_chest_cm"] / df["weight_kg"]
    df["chest_per_height_cm"] = df["effective_chest_cm"] / df["height_cm"]
    df["chest_minus_weight"] = df["effective_chest_cm"] - df["weight_kg"]
    df["chest_minus_height"] = df["effective_chest_cm"] - df["height_cm"]
    return df
