import os
import sys
import torch
import clip
from PIL import Image
import numpy as np
import time

if len(sys.argv) != 4:
    print("Usage: python clip_embeddings.py <model> <input_dir> <output_file>")
    sys.exit(1)

model_name = sys.argv[1]      # "ViT-B/32" or "ViT-L/14"
input_dir = sys.argv[2]       # path to images folder
output_file = sys.argv[3]     # .npy file

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Model: {model_name}")
print(f"Input dir: {input_dir}")

model, preprocess = clip.load(model_name, device=device)

embeddings = []
image_names = []

start_time = time.time()
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    if img_name.lower().endswith((".jpg", ".png", ".webp")):
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
            image_names.append(img_name)
        print(f"✔ Processed {img_name}")

embeddings = np.vstack(embeddings)
np.save(output_file, {"embeddings": embeddings, "image_names": image_names})

elapsed = time.time() - start_time
print(f"\n✅ Saved to: {output_file}")
print(f"⏱ Time taken: {elapsed:.2f} seconds")
