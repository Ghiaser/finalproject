import subprocess
import time
import sys
from pathlib import Path

class Colors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'

def run_command(command, step_name):
    print(f"{Colors.HEADER}=== {step_name} ==={Colors.ENDC}")
    start = time.time()
    try:
        subprocess.run(command, check=True)
        print(f"{Colors.OKGREEN}✔ Completed: {step_name}{Colors.ENDC} (Time: {time.time()-start:.2f}s)\n")
    except subprocess.CalledProcessError:
        print(f"{Colors.FAIL}✖ Failed: {step_name}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    # *** THIS matches your tree: app/app/experiments/images ***
    base_dir = Path("app/app/app/experiments/images").resolve()

    # 1) download
    run_command([sys.executable, "app/app/experiments/download_images.py"], "Downloading images")

    # 2) make variants
    run_command([sys.executable, "app/app/experiments/make_variants.py"], "Creating image variants (PNG, WEBP)")

    # 3) embeddings
    models = [("ViT-B/32", "vit_b32"), ("ViT-L/14", "vit_l14")]
    formats = ["jpg", "png", "webp"]

    for model_name, tag in models:
        for fmt in formats:
            input_dir = base_dir / fmt      # e.g., app/app/experiments/images/jpg
            output_file = Path("app/app/experiments") / f"{tag}_{fmt}.npy"
            run_command(
                [sys.executable, "app/app/experiments/clip_embeddings.py", model_name, str(input_dir), str(output_file)],
                f"Embedding {fmt.upper()} images with {model_name}"
            )

    run_command([sys.executable, "app/app/experiments/reduce_with_pca.py"], "Reducing embeddings to d=256 with PCA")

    # 4) compare
    run_command([sys.executable, "app/app/experiments/compare_embeddings.py"], "Comparing embeddings & generating report")

    # Make results table + summary charts
    run_command([sys.executable, "app/app/experiments/make_results_table.py"],
                "Building results tables & summary charts")

    print(f"{Colors.OKBLUE}🎯 All experiments completed successfully!{Colors.ENDC}")
