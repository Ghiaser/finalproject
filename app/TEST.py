
import os
from getpass import getpass
from main import CLIPSecureEncryptor

# Ask user for password securely
password = getpass("🔐 Enter your encryption password: ")

# Path to your data folder
folder = "/home/danielbes/Desktop/BETA/DATA"
files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(('.jpg', '.png', '.txt'))]

# Initialize encryptor
encryptor = CLIPSecureEncryptor()

# Build index if missing, otherwise load
if not os.path.exists("my_index.pkl"):
    print("🔨 Index not found. Building new one...")
    encryptor.build_index_from_files(files, password=password)
    encryptor.save_index("my_index.pkl")
    print("✅ Index built and saved.")
else:
    try:
        encryptor.load_index("my_index.pkl")
        print("✅ Index loaded.")
    except ValueError as e:
        print(str(e))
        exit()

# Perform semantic text query
query = input("💬 Enter search query: ")
try:
    results = encryptor.query_text(query, password=password)
    print("\n🔍 Results for query:")
    for res in results:
        print("  →", res)
except ValueError as e:
    print(str(e))

# Optional image query
image_query_path = input("\n🖼️ Enter image filename to search (or press Enter to skip): ").strip()
if image_query_path:
    full_path = os.path.join(folder, image_query_path)
    if os.path.exists(full_path):
        try:
            image_results = encryptor.query_image(full_path, password=password)
            print("\n🖼️ Image query results:")
            for res in image_results:
                print("  →", res)
        except ValueError as e:
            print(str(e))
    else:
        print(f"⚠️ Image not found: {full_path}")