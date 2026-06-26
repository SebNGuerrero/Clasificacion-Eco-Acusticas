import os
import pandas as pd
 
# ─── Rutas ────────────────────────────────────────────────────────────────────
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# subir un nivel → MachineLearning/
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")

TRAIN_FILE = os.path.join(DATA_DIR, "eco_acoustic_train.csv")
TEST_FILE  = os.path.join(DATA_DIR, "eco_acoustic_test.csv")

for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)
 
# ─── Dataset ──────────────────────────────────────────────────────────────────
 
N_FEATURES   = 64
N_CLASSES    = 5
SPECIES_IDS  = [10, 12, 17, 18, 23]
 
# Mapeo especie_id → nombre científico y tipo de fauna
SPECIES_META = {
    10: {"name": "Leptodactylus discodactylus", "type": "Anfibio"},
    12: {"name": "Osteocephalus taurinus",       "type": "Anfibio"},
    17: {"name": "Chiroxiphia lineata",           "type": "Ave"},
    18: {"name": "Saltator grossus",              "type": "Ave"},
    23: {"name": "Pheucticus chrysopeplus",       "type": "Ave"},
}
 
# Columnas del CSV
FEATURE_COLS = [f"mel_{i}" for i in range(N_FEATURES)]
TARGET_COL   = "species_id"
META_COLS    = ["recording_id", "songtype_id", "is_tp"]  # excluir de X
 
# ─── Reproducibilidad ─────────────────────────────────────────────────────────
 
RANDOM_SEED = 42
 
# ─── Preprocesamiento ─────────────────────────────────────────────────────────
 
VAL_SIZE = 0.20  # 20% del train para validación interna (MLP y LightGBM)
 
# ─── Reducción de Dimensionalidad ─────────────────────────────────────────────
 
PCA_VAR_THRESHOLD = 0.95    # Varianza acumulada mínima a retener
PCA_CLUSTER_DIMS  = 20      # Componentes PCA como input de clustering
 
TSNE_PERPLEXITY   = 40
TSNE_N_ITER       = 1000
TSNE_INIT         = "pca"   # Inicialización PCA acelera convergencia
 
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.1
UMAP_METRIC       = "euclidean"
 
# ─── Clustering ───────────────────────────────────────────────────────────────
 
# DBSCAN: grid de búsqueda de hiperparámetros
DBSCAN_EPS_GRID         = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
DBSCAN_MIN_SAMPLES_GRID = [5, 10, 15, 20, 30]
 
# GMM: rango de componentes y tipos de covarianza a evaluar
GMM_N_COMPONENTS_GRID = list(range(2, 11))   # {2, 3, ..., 10}
GMM_COV_TYPES         = ["full", "tied", "diag", "spherical"]
GMM_MAX_ITER          = 200
 
# ─── MLP (PyTorch) ────────────────────────────────────────────────────────────
 
MLP_INPUT_DIM    = N_FEATURES         # 64
MLP_HIDDEN_DIMS  = [512, 256, 128]    # Capas ocultas
MLP_OUTPUT_DIM   = N_CLASSES          # 5 logits → softmax
MLP_DROPOUT_RATE = 0.3
 
MLP_LR           = 1e-3
MLP_WEIGHT_DECAY = 1e-4               # L2 regularización en Adam
MLP_BATCH_SIZE   = 64
MLP_MAX_EPOCHS   = 200
MLP_PATIENCE     = 20                 # Early stopping patience
 
# Configuraciones de regularización a comparar:
#   A: Linear → BN → ReLU → Dropout  (estándar)
#   B: Linear → ReLU → Dropout → BN
#   C: Linear → ReLU → BN → Dropout
MLP_CONFIGS = ["A", "B", "C"]
 
# ─── LightGBM ─────────────────────────────────────────────────────────────────
 
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         N_CLASSES,
    "metric":            "multi_logloss",
    "num_leaves":        31,
    "learning_rate":     0.05,
    "n_estimators":      1000,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,       # L1
    "reg_lambda":        0.1,       # L2
    "random_state":      RANDOM_SEED,
    "class_weight":      "balanced",
    "verbosity":         -1,
    "n_jobs":            -1,
}
 
LGBM_EARLY_STOPPING_ROUNDS = 50
LGBM_CV_FOLDS              = 5
 
# ─── Umbrales de Confianza ─────────────────────────
 
THRESHOLD_ACCEPT = 0.85   # P ≥ 0.85 → ACCEPT  (Verde: clasificación automática)
THRESHOLD_REVIEW = 0.40   # P ≥ 0.40 → REVIEW  (Amarillo: revisión humana)
                           # P <  0.40 → REJECT  (Rojo: descarte automático)
 
# ─── Figuras  ───────────────────────
 
FONT_SIZE   = 14
FIGURE_DPI  = 150
FIGURE_SIZE = (10, 6)