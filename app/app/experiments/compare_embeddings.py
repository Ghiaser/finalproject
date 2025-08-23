# app/app/experiments/compare_embeddings.py
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ---------- Paths ----------
BASE_DIR = Path("app/app/experiments")
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Base models / formats we expect
MODELS = {
    "ViT-B/32": {"tag": "vit_b32", "dim": 512},
    "ViT-L/14": {"tag": "vit_l14", "dim": 768},
}
FORMATS = ["jpg", "png", "webp"]


# ---------- IO helpers ----------
def load_embeddings(path: Path):
    """
    Load a .npy dict saved as {"embeddings": ndarray, "filenames": list[str], "meta": {...}}
    Backward-compatible with "image_names" key if present.
    """
    d = np.load(str(path), allow_pickle=True).item()
    X = np.asarray(d["embeddings"])
    names = list(d.get("filenames", d.get("image_names", [])))
    return X, names, d.get("meta", {})


def save_plot(fig_path: Path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved plot: {fig_path}")


# ---------- Similarity / stats ----------
def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """
    Return NxN cosine similarity matrix for rows of X.
    """
    Xn = l2_normalize(X)
    return Xn @ Xn.T


def spearman_agreement(simA: np.ndarray, simB: np.ndarray):
    """
    Compare two similarity matrices by taking the upper triangle (excluding diagonal)
    and computing Spearman correlation.
    """
    if simA.shape != simB.shape:
        raise ValueError("Similarity matrices must have the same shape for comparison.")
    iu = np.triu_indices(simA.shape[0], k=1)
    a = simA[iu].ravel()
    b = simB[iu].ravel()
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


# ---------- Name alignment ----------
def align_by_names(Xa, names_a, Xb, names_b):
    """
    Intersect filenames, return Xa', Xb', names in the same order.
    """
    idx_a = {n: i for i, n in enumerate(names_a)}
    common = [n for n in names_b if n in idx_a]
    if len(common) < 3:
        return None, None, []  # too few samples for a robust correlation
    A = Xa[[idx_a[n] for n in common]]
    B = Xb[[names_b.index(n) for n in common]]
    return A, B, common


# ---------- Heatmap ----------
def heatmap(sim: np.ndarray, title: str, out_path: Path):
    plt.figure()
    plt.imshow(sim, interpolation="nearest")  # default colormap; no explicit colors
    plt.title(title)
    plt.colorbar()
    save_plot(out_path)


# ---------- Main ----------
def main():
    t0 = time.time()
    report = {
        "per_format": {},       # heatmaps by model/format/dimension
        "cross_model": {},      # Spearman for ViT-B/32 vs ViT-L/14 per format
        "dim_correlations": {}, # Spearman base vs reduced dims per model/format
    }

    # -------- 1) Collect all available embedding files (base + reduced) --------
    # Base files (full dims)
    base_files = {}  # (model_name, fmt) -> path
    for model_name, meta in MODELS.items():
        tag = meta["tag"]
        for fmt in FORMATS:
            p = BASE_DIR / f"{tag}_{fmt}.npy"
            if p.exists():
                base_files[(model_name, fmt)] = p

    # Reduced files (any *_d*.npy)
    reduced_files = {}  # (model_name, fmt, d_str) -> path
    for model_name, meta in MODELS.items():
        tag = meta["tag"]
        for fmt in FORMATS:
            for p in BASE_DIR.glob(f"{tag}_{fmt}_d*.npy"):
                # extract dimension token from filename e.g., vit_b32_jpg_d128.npy -> "128"
                stem = p.stem  # vit_b32_jpg_d128
                if "_d" in stem:
                    d_str = stem.split("_d")[-1]
                    reduced_files[(model_name, fmt, d_str)] = p

    # Load everything once
    loaded = {}  # path -> (X, names, meta)
    for p in set(base_files.values()) | set(reduced_files.values()):
        X, names, meta = load_embeddings(p)
        loaded[str(p)] = (X, names, meta)

    # -------- 2) Heatmaps for every available file --------
    for (model_name, fmt), p in base_files.items():
        X, names, _ = loaded[str(p)]
        base_dim = MODELS[model_name]["dim"]
        sim = cosine_sim_matrix(X)
        out = PLOTS_DIR / f"{model_name.replace('/','_').replace(' ','_')}_{fmt}_d{base_dim}_heatmap.png"
        heatmap(sim, f"{model_name} {fmt.upper()} (d={base_dim})", out)
        report["per_format"].setdefault(model_name, {}).setdefault(fmt, {})[str(base_dim)] = {"heatmap": str(out)}

    for (model_name, fmt, d_str), p in reduced_files.items():
        X, names, _ = loaded[str(p)]
        sim = cosine_sim_matrix(X)
        out = PLOTS_DIR / f"{model_name.replace('/','_').replace(' ','_')}_{fmt}_d{d_str}_heatmap.png"
        heatmap(sim, f"{model_name} {fmt.upper()} (d={d_str})", out)
        report["per_format"].setdefault(model_name, {}).setdefault(fmt, {})[f"{d_str}"] = {"heatmap": str(out)}

    # -------- 3) Cross-model (base) comparisons: ViT-B/32 vs ViT-L/14 for each format --------
    for fmt in FORMATS:
        key_b = ("ViT-B/32", fmt)
        key_l = ("ViT-L/14", fmt)
        if key_b in base_files and key_l in base_files:
            Xb, nb, _ = loaded[str(base_files[key_b])]
            Xl, nl, _ = loaded[str(base_files[key_l])]
            Ab, Al, common = align_by_names(Xb, nb, Xl, nl)
            if len(common) >= 3:
                sim_b = cosine_sim_matrix(Ab)
                sim_l = cosine_sim_matrix(Al)
                rho, p = spearman_agreement(sim_b, sim_l)
                report["cross_model"][fmt] = {"spearman_rho": rho, "p_value": p}
                print(f"[OK] Cross-model (B/32 vs L/14) for {fmt}: Spearman ρ={rho:.3f} (p={p:.2e})")

    # -------- 4) Cross-dimension comparisons (base vs reduced) per model/format --------
    # Build and store barplots summarizing Spearman(base vs reduced) for each model/format.
    for model_name, meta in MODELS.items():
        base_dim = meta["dim"]
        report["dim_correlations"].setdefault(model_name, {})
        for fmt in FORMATS:
            if (model_name, fmt) not in base_files:
                continue

            X_base, names_base, _ = loaded[str(base_files[(model_name, fmt)])]
            sim_vals = []  # (label, rho)
            dim_results = {}

            # Iterate all reduced dims found for this model/format
            for (m2, f2, d_str), p_red in reduced_files.items():
                if m2 != model_name or f2 != fmt:
                    continue
                X_red, names_red, _ = loaded[str(p_red)]
                A, B, common = align_by_names(X_base, names_base, X_red, names_red)
                if len(common) < 3:
                    continue
                sim_base = cosine_sim_matrix(A)
                sim_red = cosine_sim_matrix(B)
                rho, p = spearman_agreement(sim_base, sim_red)
                dim_key = f"d{d_str}"
                dim_results[dim_key] = {"spearman_rho": rho, "p_value": p}
                sim_vals.append((dim_key, rho))
                print(f"[OK] Dim compare {model_name}::{fmt} base(d{base_dim}) vs {p_red.name}: ρ={rho:.3f} (p={p:.2e})")

            # Save to report
            report["dim_correlations"][model_name][fmt] = dim_results

            # Plot a small bar chart if we have any results
            if sim_vals:
                sim_vals.sort(key=lambda t: int(t[0].lstrip("d")))
                labels = [k for k, _ in sim_vals]
                values = [v for _, v in sim_vals]
                plt.figure()
                plt.bar(labels, values)
                plt.ylim(0.0, 1.0)
                plt.title(f"{model_name} {fmt.upper()} – Spearman(base d{base_dim} vs reduced)")
                plt.ylabel("Spearman ρ")
                out_png = PLOTS_DIR / f"{model_name.replace('/','_').replace(' ','_')}_{fmt}_dim_compare.png"
                save_plot(out_png)

    # -------- 5) Save summary JSON --------
    out_json = BASE_DIR / "results_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Saved report: {out_json}")
    print(f"⏱ Total time: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
