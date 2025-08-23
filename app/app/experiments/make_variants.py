import os
from PIL import Image

# Base JPG directory (with subfolders for categories)
BASE_JPG_DIR = r"app/app/app/experiments/images/jpg"

# Output base directories
BASE_PNG_DIR = BASE_JPG_DIR.replace("/jpg", "/png").replace("\\jpg", "\\png")
BASE_WEBP_DIR = BASE_JPG_DIR.replace("/jpg", "/webp").replace("\\jpg", "\\webp")

def make_variants(input_dir, output_dir, fmt):
    """
    Convert all .jpg images in input_dir to the specified format (png or webp).
    Maintains subfolder structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        # Determine relative path for nested folders
        rel_path = os.path.relpath(root, input_dir)
        target_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(target_subdir, exist_ok=True)

        for file in files:
            if file.lower().endswith(".jpg"):
                src_path = os.path.join(root, file)
                dst_file = os.path.splitext(file)[0] + f".{fmt}"
                dst_path = os.path.join(target_subdir, dst_file)
                try:
                    img = Image.open(src_path).convert("RGB")
                    img.save(dst_path, format=fmt.upper())
                    print(f"✔ Converted {src_path} → {dst_path}")
                except Exception as e:
                    print(f"⚠ Error converting {src_path}: {e}")

if __name__ == "__main__":
    print("📂 Converting JPG → PNG")
    make_variants(BASE_JPG_DIR, BASE_PNG_DIR, "png")

    print("\n📂 Converting JPG → WEBP")
    make_variants(BASE_JPG_DIR, BASE_WEBP_DIR, "webp")

    print("\n✅ All variants created successfully.")
