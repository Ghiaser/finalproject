import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ------------- Config -------------

BASE_EXP_DIR = os.path.join("app", "app", "experiments")
EMB_DIR = BASE_EXP_DIR  # where *.npy were saved by clip_embeddings.py
PLOTS_DIR = os.path.join(BASE_EXP_DIR, "plots")
REPORT_JSON = os.path.join(BASE_EXP_DIR, "results_summary.json")

# expected files produced earlier
FILEMAP = {
    "ViT-B/32": {
        "jpg":  os.path.join(EMB_DIR, "vit_b32_jpg.npy"),
        "png":  os.path.join(EMB_DIR, "vit_b32_png.npy"),
        "webp": os.path.join(EMB_DIR, "vit_b32_webp.npy"),
    },
    "ViT-L/14": {
        "jpg":  os.path.join(EMB_DIR, "vit_l14_jpg.npy"),
        "png":  os.path.join(EMB_DIR, "vit_l14_png.npy"),
        "webp": os.path.join(EMB_DIR, "vit_l14_webp.npy"),
    },
}

# ------------- Helpers -------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_embeddings_sorted(path: str):
    """
    Load embeddings .npy (dict with 'embeddings' and 'image_names'),
    and return (names, embs) sorted by file name to guarantee identical order.
    """
    d = np.load(path, allow_pickle=True).item()
    names = list(map(str, d["image_names"]))
    embs = np.array(d["embeddings"])
    order = np.argsort(names)
    names = [names[i] for i in order]
    embs = embs[order]
    return names, embs

def cosine_sim_matrix(embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for a batch of row-embeddings.
    """
    if embs.ndim != 2:
        raise ValueError("embs should be 2D: [N, D]")
    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    z = embs / norms
    return z @ z.T  # [N, N]

def upper_triangle_values(M: np.ndarray):
    """
    Return the upper-triangular (i<j) values as a 1D array,
    to compare matrices without the diagonal.
    """
    n = M.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    return M[triu_idx]

def plot_heatmap(sim: np.ndarray, names, title: str, out_path: str):
    """
    Save a clean heatmap (matplotlib only) to out_path.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(sim, vmin=-1.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)
    # keep ticks light for readability with many images
    plt.xticks(range(len(names)), [os.path.splitext(n)[0] for n in names], rotation=90, fontsize=8)
    plt.yticks(range(len(names)), [os.path.splitext(n)[0] for n in names], fontsize=8)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] Saved heatmap: {out_path}")

def pairwise_dict(sim: np.ndarray, names):
    """
    Build a JSON-safe dict of pairwise similarities: "name_i__vs__name_j": float(sim).
    Only i<j to avoid duplicates and diagonal.
    """
    out = {}
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{names[i]}__vs__{names[j]}"
            out[key] = float(sim[i, j])
    return out

def sanitize_for_filename(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")

# ------------- Main -------------

def main():
    t0 = time.time()
    ensure_dir(PLOTS_DIR)

    report = {
        "per_model": {},     # per model -> per format -> { image_names, avg_upper, pairwise }
        "cross_model": {},   # per format -> { spearman_rho, p_value }
        "notes": [
            "All images are sorted by filename before computing similarities.",
            "Cosine similarities are computed from L2-normalized embeddings.",
            "Spearman correlation is computed on the upper-triangle (i<j) of the similarity matrices."
        ],
    }

    # load, compute similarities, plot
    sims = {}  # sims[model][format] = (names, sim_matrix)
    for model, fmts in FILEMAP.items():
        sims[model] = {}
        report["per_model"][model] = {}
        for fmt, path in fmts.items():
            if not os.path.isfile(path):
                print(f"[WARN] Missing file: {path} (skipping {model}::{fmt})")
                continue
            names, embs = load_embeddings_sorted(path)
            sim = cosine_sim_matrix(embs)
            sims[model][fmt] = (names, sim)

            # plot
            title = f"{model} :: {fmt} cosine similarity"
            out_png = os.path.join(PLOTS_DIR, f"{sanitize_for_filename(model)}_{fmt}_heatmap.png")
            plot_heatmap(sim, names, title, out_png)

            # add to report
            upp = upper_triangle_values(sim)
            report["per_model"][model][fmt] = {
                "image_names": names,
                "pairwise_similarity": pairwise_dict(sim, names),
                "avg_upper_triangle": float(np.mean(upp)) if upp.size else None,
                "min_upper_triangle": float(np.min(upp)) if upp.size else None,
                "max_upper_triangle": float(np.max(upp)) if upp.size else None,
            }

    # cross-model comparison (Spearman) per format where both models exist
    for fmt in ("jpg", "png", "webp"):
        if fmt in sims.get("ViT-B/32", {}) and fmt in sims.get("ViT-L/14", {}):
            names_b, sim_b = sims["ViT-B/32"][fmt]
            names_l, sim_l = sims["ViT-L/14"][fmt]

            # names are sorted independently; ensure same order
            if names_b != names_l:
                # align by name (shouldn't happen with identical sets, but let's be safe)
                name_to_idx_b = {n: i for i, n in enumerate(names_b)}
                order_l = [name_to_idx_b[n] for n in names_l if n in name_to_idx_b]
                sim_b = sim_b[np.ix_(order_l, order_l)]
                names_b = [names_b[i] for i in order_l]

            v1 = upper_triangle_values(sim_b)
            v2 = upper_triangle_values(sim_l)
            if v1.size and v1.size == v2.size:
                rho, p = spearmanr(v1, v2)
                report["cross_model"][fmt] = {
                    "spearman_rho": float(rho) if not math.isnan(rho) else None,
                    "p_value": float(p) if not math.isnan(p) else None,
                    "n_pairs": int(v1.size),
                }
                print(f"[OK] Cross-model (B/32 vs L/14) for {fmt}: Spearman ρ={rho:.3f} (p={p:.2e})")
            else:
                report["cross_model"][fmt] = {
                    "spearman_rho": None,
                    "p_value": None,
                    "n_pairs": int(min(v1.size, v2.size)),
                    "warning": "vector sizes differ or empty upper-triangles",
                }
                print(f"[WARN] Cross-model comparison for {fmt} skipped (size mismatch).")
        else:
            print(f"[WARN] Missing one of the models for format: {fmt}")

    # save report
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            report, f, indent=2, ensure_ascii=False,
            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o
        )

    print(f"[OK] Saved report: {REPORT_JSON}")
    print(f"⏱ Total time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
