# app/app/experiments/make_results_table.py
from pathlib import Path
import json
import csv
import matplotlib.pyplot as plt

BASE = Path("app/app/experiments")
SUMMARY_DIR = BASE / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_report():
    with open(BASE / "results_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)

def write_cross_model(report):
    rows = []
    cm = report.get("cross_model", {})
    for fmt, vals in cm.items():
        rows.append({
            "format": fmt.upper(),
            "spearman_rho": vals.get("spearman_rho", ""),
            "p_value": vals.get("p_value", "")
        })

    out_csv = SUMMARY_DIR / "cross_model_spearman.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["format","spearman_rho","p_value"])
        w.writeheader(); w.writerows(rows)

    # quick bar plot
    if rows:
        labels = [r["format"] for r in rows]
        values = [r["spearman_rho"] for r in rows]
        plt.figure()
        plt.bar(labels, values)
        plt.ylim(0.0, 1.0)
        plt.title("Cross-model agreement (ViT-B/32 vs ViT-L/14) by format")
        plt.ylabel("Spearman ρ")
        plt.tight_layout()
        plt.savefig(SUMMARY_DIR / "cross_model_bar.png", dpi=200)
        plt.close()
    print(f"[OK] wrote {out_csv}")

def write_cross_dim(report):
    """
    Expects report['dim_correlations'][model][format] = {
       'd128': {'spearman_rho': ..., 'p_value': ...}, ...
    }
    """
    rows = []
    dimc = report.get("dim_correlations", {})
    for model, per_fmt in dimc.items():
        for fmt, dims in per_fmt.items():
            for dkey, vals in dims.items():
                rows.append({
                    "model": model,
                    "format": fmt.upper(),
                    "reduced_dim": dkey,               # e.g. 'd128'
                    "spearman_rho": vals.get("spearman_rho",""),
                    "p_value": vals.get("p_value",""),
                })

    out_csv = SUMMARY_DIR / "cross_dim_spearman.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","format","reduced_dim","spearman_rho","p_value"])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] wrote {out_csv}")

def write_heatmap_index(report):
    # optional: simple index of heatmap files per model/format/dim
    rows = []
    pf = report.get("per_format", {})
    for model, per_fmt in pf.items():
        for fmt, dims in per_fmt.items():
            for dim, info in dims.items():
                rows.append({
                    "model": model,
                    "format": fmt.upper(),
                    "dim": dim,
                    "heatmap_path": info.get("heatmap","")
                })

    out_csv = SUMMARY_DIR / "per_format_heatmaps.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","format","dim","heatmap_path"])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] wrote {out_csv}")

def main():
    rep = load_report()
    write_cross_model(rep)
    write_cross_dim(rep)
    write_heatmap_index(rep)
    print("[DONE] tables built in:", SUMMARY_DIR)

if __name__ == "__main__":
    main()
