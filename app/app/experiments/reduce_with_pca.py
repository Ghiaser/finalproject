# app/app/experiments/reduce_with_pca.py
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

BASE_DIR = Path("app/app/experiments")
OUT_DIR = BASE_DIR
FORMATS = ["jpg", "png", "webp"]
MODELS = {
    "ViT-B/32": {"tag": "vit_b32", "dim": 512},
    "ViT-L/14": {"tag": "vit_l14", "dim": 768},
}
TARGET_DIM = 128

def load_embeddings(path: Path):
    """Load dict saved as {'embeddings': ndarray, 'filenames' or 'image_names': list}."""
    d = np.load(str(path), allow_pickle=True).item()
    X = np.asarray(d["embeddings"])
    names = d.get("filenames", d.get("image_names"))
    if names is None:
        raise KeyError(f"'{path.name}' is missing 'filenames'/'image_names'")
    return X, list(names)

def save_embeddings(path: Path, X: np.ndarray, names, meta: dict):
    out = {
        "embeddings": X,
        "filenames": list(names),  # always write as 'filenames'
        "meta": meta,
    }
    np.save(str(path), out)

def main():
    wrote = 0
    for model_name, meta in MODELS.items():
        tag = meta["tag"]

        # collect base files for this model (skip already reduced *_d*.npy)
        base_files = []
        for fmt in FORMATS:
            p = BASE_DIR / f"{tag}_{fmt}.npy"
            if p.exists():
                base_files.append((fmt, p))

        if not base_files:
            continue

        # fit PCA on all formats together (if possible)
        X_all = []
        for _, p in base_files:
            X, _ = load_embeddings(p)
            X_all.append(X)
        X_all = np.concatenate(X_all, axis=0)

        # choose n_components safely
        n_samples, n_features = X_all.shape
        n_comp = min(TARGET_DIM, n_features, max(1, n_samples - 1))
        if n_comp < 2:
            print(f"[WARN] Not enough samples to fit PCA for {model_name} "
                  f"(samples={n_samples}). Skipping reduction.")
            continue

        print(f"[INFO] Fitting PCA({n_comp}) for {model_name} on {n_samples} vectors of dim {n_features}...")
        pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
        pca.fit(X_all)

        # transform each format separately and save
        for fmt, p in base_files:
            X, names = load_embeddings(p)
            Xr = pca.transform(X)
            out_path = OUT_DIR / f"{tag}_{fmt}_d{n_comp}.npy"
            save_embeddings(
                out_path,
                Xr,
                names,
                meta={
                    "model": model_name,
                    "model_tag": tag,
                    "format": fmt,
                    "original_dim": X.shape[1],
                    "reduced_dim": n_comp,
                    "method": "PCA(full)",
                },
            )
            wrote += 1
            print(f"[OK] Wrote: {out_path}  (N={len(names)}, d={n_comp})")

    if wrote == 0:
        print("[INFO] No reduced files were written.")
    else:
        print(f"[DONE] Wrote {wrote} reduced embedding files.")

if __name__ == "__main__":
    main()
