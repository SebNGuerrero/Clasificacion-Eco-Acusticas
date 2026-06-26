import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

from config import (
    FEATURE_COLS, TARGET_COL, SPECIES_META, FIG_DIR,
    FONT_SIZE, FIGURE_DPI, FIGURE_SIZE, RANDOM_SEED, N_FEATURES,
)
from data_loader import load_data, preprocess

matplotlib.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE + 2,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE,
    "ytick.labelsize":  FONT_SIZE,
    "legend.fontsize":  FONT_SIZE,
})

SHOW_PLOTS = True
SAVE_PLOTS = True

# ─── Análisis de Distribución de Clases ───────────────────────────────────────

def plot_class_distribution(y_train_raw: np.ndarray, species_meta: dict) -> None:
    """
    Gráfica de barras con la distribución de clases en el conjunto de entrenamiento.
    Evidencia el desbalance de clases y justifica el uso de pesos en la loss function.
    """
    species_ids = sorted(species_meta.keys())
    labels = [f"sp_{sid}\n{species_meta[sid]['name'].split()[0]}" for sid in species_ids]
    counts = [np.sum(y_train_raw == sid) for sid in species_ids]
    percentages = [c / len(y_train_raw) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(labels, counts, color="steelblue", edgecolor="white", linewidth=1.2)

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=FONT_SIZE)

    ax.set_title("Distribución de Clases — eco_acoustic_train.csv")
    ax.set_xlabel("Especie (species_id — nombre abreviado)")
    ax.set_ylabel("Número de observaciones")
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    path = f"{FIG_DIR}/eda_class_distribution.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Figura guardada: {path}")

    print("\n[EDA] Distribución de clases:")
    for sid, cnt, pct in zip(species_ids, counts, percentages):
        print(f"  species_id={sid} ({SPECIES_META[sid]['name']}): "
              f"{cnt} obs ({pct:.1f}%)")


# ─── Estadísticas Descriptivas de Features ────────────────────────────────────

def analyze_features(X_train: np.ndarray) -> pd.DataFrame:
    """
    Calcula media, desviación estándar, min, max y varianza por feature.
    Identifica features con varianza muy baja (potencialmente no informativas).
    """
    df_stats = pd.DataFrame(X_train, columns=FEATURE_COLS).describe().T
    df_stats["variance"] = df_stats["std"] ** 2
    df_stats["cv"] = df_stats["std"] / (df_stats["mean"].abs() + 1e-8)  # coef. de variación

    low_var = df_stats[df_stats["variance"] < 0.01]
    if len(low_var) > 0:
        print(f"[EDA] ADVERTENCIA: {len(low_var)} features con varianza < 0.01: "
              f"{low_var.index.tolist()}")
    else:
        print(f"[EDA] Todas las {N_FEATURES} features tienen varianza aceptable.")

    print(f"\n[EDA] Estadísticas de mel_0..mel_63 (antes de escalar):")
    print(df_stats[["mean", "std", "min", "max", "variance"]].round(4).to_string())

    return df_stats


def plot_feature_statistics(X_train_raw: np.ndarray) -> None:
    """
    Dos subplots: media y std por índice de feature mel_k.
    Ilustra la heterogeneidad de escala que justifica el StandardScaler.
    """
    means = X_train_raw.mean(axis=0)
    stds  = X_train_raw.std(axis=0)
    k_idx = np.arange(N_FEATURES)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(k_idx, means, color="navy", linewidth=1.5)
    axes[0].fill_between(k_idx, means - stds, means + stds,
                          alpha=0.3, color="steelblue", label="±1σ")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Media μ_k")
    axes[0].set_title("Estadísticas de Features mel_k por Índice (datos crudos)")
    axes[0].legend(fontsize=FONT_SIZE)

    axes[1].bar(k_idx, stds, color="darkorange", width=0.7, alpha=0.85)
    axes[1].set_xlabel("Índice de Feature k  (mel_k)")
    axes[1].set_ylabel("Desv. Estándar σ_k")

    plt.tight_layout()
    path = f"{FIG_DIR}/eda_feature_statistics.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Figura guardada: {path}")


# ─── Heatmap de Correlación ───────────────────────────────────────────────────

def plot_correlation_heatmap(X_train: np.ndarray) -> None:
    """
    Heatmap de la matriz de correlación de Pearson entre las 64 features.
    Identifica grupos de features altamente correlacionadas
    (relevante para interpretar comportamiento de PCA).
    """
    corr_matrix = np.corrcoef(X_train.T)  # shape (64, 64)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlación de Pearson")

    ax.set_title("Matriz de Correlación — Features mel_0..mel_63 (escaladas)")
    ax.set_xlabel("Índice de Feature k")
    ax.set_ylabel("Índice de Feature k")
    ax.set_xticks(range(0, N_FEATURES, 8))
    ax.set_yticks(range(0, N_FEATURES, 8))

    plt.tight_layout()
    path = f"{FIG_DIR}/eda_correlation_heatmap.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Figura guardada: {path}")

    # Reporte de pares altamente correlacionados
    high_corr_pairs = []
    for i in range(N_FEATURES):
        for j in range(i + 1, N_FEATURES):
            if abs(corr_matrix[i, j]) > 0.90:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    print(f"[EDA] Pares de features con |corr| > 0.90: {len(high_corr_pairs)}")
    if high_corr_pairs:
        for i, j, c in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:10]:
            print(f"  mel_{i} ↔ mel_{j}: {c:.3f}")


# ─── Análisis de Outliers ─────────────────────────────────────────────────────

def analyze_outliers(X_train: np.ndarray) -> dict:
    """
    Detecta outliers por el método IQR (Q1 - 1.5·IQR, Q3 + 1.5·IQR).
    Devuelve fracción de observaciones con al menos un outlier en alguna feature.
    """
    Q1 = np.percentile(X_train, 25, axis=0)
    Q3 = np.percentile(X_train, 75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Máscara: True si la observación tiene outlier en CUALQUIER feature
    outlier_mask = np.any((X_train < lower) | (X_train > upper), axis=1)
    n_outliers = outlier_mask.sum()
    pct = n_outliers / len(X_train) * 100

    print(f"\n[EDA] Análisis de Outliers (IQR method):")
    print(f"  Observaciones con ≥1 feature outlier: {n_outliers} ({pct:.1f}%)")
    print(f"  Nota: StandardScaler atenúa pero NO elimina outliers.")

    return {"outlier_mask": outlier_mask, "n_outliers": n_outliers, "pct": pct}


# ─── Análisis por Clase ───────────────────────────────────────────────────────

def plot_class_feature_means(X_train: np.ndarray, y_train: np.ndarray,
                              label_encoder) -> None:
    """
    Media de cada mel_k por clase. Revela el perfil espectral característico
    de cada especie y visualiza la separabilidad de clases en el espacio original.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    colors  = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
    k_idx   = np.arange(N_FEATURES)

    for c in range(len(label_encoder.classes_)):
        mask = (y_train == c)
        class_mean = X_train[mask].mean(axis=0)
        sid  = label_encoder.classes_[c]
        name = SPECIES_META[sid]["name"].split()[0] + " " + SPECIES_META[sid]["name"].split()[1]
        ax.plot(k_idx, class_mean, label=f"sp_{sid} {name}",
                linewidth=1.8, color=colors[c])

    ax.set_xlabel("Índice de Feature k  (mel_k, escaladas)")
    ax.set_ylabel("Media de la feature por clase")
    ax.set_title("Perfil Espectral Medio por Especie — Espacio Original R^64")
    ax.legend(fontsize=FONT_SIZE - 1, ncol=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    path = f"{FIG_DIR}/eda_class_feature_means.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Figura guardada: {path}")


# ─── Punto de Entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EDA — Exploración del Dataset Eco-Acústico")
    print("=" * 60)

    # Cargar y preprocesar datos
    train_df, test_df = load_data()
    data = preprocess(train_df, test_df, save_artifacts=True)

    X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
    X_train  = data["X_train"]
    y_train  = data["y_train"]
    le       = data["label_encoder"]

    # ── Distribución de clases ────────────────────────────────────────────────
    plot_class_distribution(data["y_train_raw"], SPECIES_META)

    # ── Estadísticas de features (datos crudos antes de escalar) ─────────────
    _ = analyze_features(X_train_raw)
    plot_feature_statistics(X_train_raw)

    # ── Correlación entre features (datos escalados) ──────────────────────────
    plot_correlation_heatmap(X_train)

    # ── Outliers ──────────────────────────────────────────────────────────────
    _ = analyze_outliers(X_train)

    # ── Perfil espectral medio por especie ────────────────────────────────────
    plot_class_feature_means(X_train, y_train, le)

    print("\n[OK] EDA completado. Figuras en:", FIG_DIR)