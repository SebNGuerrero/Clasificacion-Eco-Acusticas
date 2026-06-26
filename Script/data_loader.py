import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_FILE, TEST_FILE, MODEL_DIR,
    FEATURE_COLS, TARGET_COL, META_COLS,
    SPECIES_IDS, SPECIES_META, N_FEATURES, N_CLASSES,
    RANDOM_SEED, VAL_SIZE,
)


# ─── 1. Carga y Validación de Schema ──────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    for path in [TRAIN_FILE, TEST_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Archivo no encontrado: {path}\n"
                f"Coloca los CSVs en la carpeta '{os.path.dirname(path)}/'"
            )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    _validate_schema(train_df, "train")
    _validate_schema(test_df,  "test")

    print(f"[data_loader] Train: {train_df.shape} | Test: {test_df.shape}")
    return train_df, test_df


def _validate_schema(df: pd.DataFrame, split_name: str) -> None:
    """Verifica columnas, tipos y ausencia de nulos."""
    required_cols = FEATURE_COLS + [TARGET_COL] + META_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{split_name}] Columnas faltantes: {missing}")

    null_counts = df[FEATURE_COLS].isnull().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"[{split_name}] Nulos en features: {bad}")

    invalid_targets = set(df[TARGET_COL].unique()) - set(SPECIES_IDS)
    if invalid_targets and split_name == "train":
        raise ValueError(f"[train] Clases no esperadas en {TARGET_COL}: {invalid_targets}")

    print(f"[data_loader] Schema '{split_name}' OK — {len(df)} observaciones, 0 nulos.")


# ─── 2. Preprocesamiento ──────────────────────────────────────────────────────

def preprocess(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    save_artifacts: bool = True,
) -> dict:

    # ── Extracción ────────────────────────────────────────────────────────────
    X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
    X_test_raw  = test_df[FEATURE_COLS].values.astype(np.float32)
    y_train_raw = train_df[TARGET_COL].values
    y_test_raw  = test_df[TARGET_COL].values

    # ── Label Encoding: {10,12,17,18,23} → {0,1,2,3,4} ──────────────────────
    le = LabelEncoder()
    le.fit(SPECIES_IDS)
    y_train = le.transform(y_train_raw).astype(np.int64)
    y_test  = le.transform(y_test_raw).astype(np.int64)

    species_names = [SPECIES_META[sid]["name"] for sid in le.classes_]

    # ── StandardScaler (fit solo en X_train) ─────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    # ── Split interno 80/20 para validación (stratified) ─────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )

    # ── Pesos de clase: w_c = N / (C * count_c) ──────────────────────────────
    # Equivalente a class_weight='balanced' de sklearn
    class_counts  = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    class_weights = len(y_train) / (N_CLASSES * class_counts)
    class_weights = class_weights.astype(np.float32)

    print("[data_loader] Preprocesamiento completado.")
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  Distribución de clases (train):")
    for c, sid in enumerate(le.classes_):
        print(f"    clase {c} (species_id={sid}, {SPECIES_META[sid]['name']}): "
              f"{int(class_counts[c])} obs ({100*class_counts[c]/len(y_train):.1f}%) "
              f"→ weight={class_weights[c]:.3f}")

    # ── Guardar artefactos ────────────────────────────────────────────────────
    if save_artifacts:
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))
        np.save(os.path.join(MODEL_DIR, "class_weights.npy"), class_weights)
        print(f"[data_loader] Artefactos guardados en '{MODEL_DIR}/'")

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "y_train_raw":   y_train_raw,
        "y_test_raw":    y_test_raw,
        "X_train_val":   X_tr,
        "X_val":         X_val,
        "y_train_val":   y_tr,
        "y_val":         y_val,
        "class_weights": class_weights,
        "scaler":        scaler,
        "label_encoder": le,
        "species_names": species_names,
    }


def load_preprocessed() -> dict:

    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le      = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    weights = np.load(os.path.join(MODEL_DIR,  "class_weights.npy"))
    return {"scaler": scaler, "label_encoder": le, "class_weights": weights}


# ─── 3. Utilidades de Evaluación ──────────────────────────────────────────────

def decode_labels(y_encoded: np.ndarray, le: LabelEncoder) -> np.ndarray:
    return le.inverse_transform(y_encoded)


if __name__ == "__main__":
    train_df, test_df = load_data()
    data = preprocess(train_df, test_df, save_artifacts=True)
    print("\n[OK] data_loader.py ejecutado correctamente.")
    print(f"     class_weights = {data['class_weights']}")