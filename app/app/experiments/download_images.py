import os
import requests

# Base directory to save images
BASE_DIR = r"app/app/app/experiments/images/jpg"

# Categories and example image URLs (from Unsplash - free for demo use)
categories = {
    "animals": [
        "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d",
        "https://images.unsplash.com/photo-1507149833265-60c372daea22",
        "https://images.unsplash.com/photo-1592194996308-7b43878e84a6",
        "https://images.unsplash.com/photo-1601758124094-1c30b14d773d",
        "https://images.unsplash.com/photo-1546182990-dffeafbe841d",
        "https://images.unsplash.com/photo-1583337130417-3346a1afdd31",
        "https://images.unsplash.com/photo-1517849845537-4d257902454a",
        "https://images.unsplash.com/photo-1537151625747-768eb6cf92b6",
        "https://images.unsplash.com/photo-1518717758536-85ae29035b6d",
        "https://images.unsplash.com/photo-1508672019048-805c876b67e2",
    ],
    "people": [
        "https://images.unsplash.com/photo-1494790108377-be9c29b29330",
        "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e",
        "https://images.unsplash.com/photo-1524504388940-b1c1722653e1",
        "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91",
        "https://images.unsplash.com/photo-1520813792240-56fc4a3765a7",
        "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df",
        "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d",
        "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce",
        "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df",
        "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d",
    ],
    "nature": [
        "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee",
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
        "https://images.unsplash.com/photo-1470770841072-f978cf4d019e",
        "https://images.unsplash.com/photo-1465101162946-4377e57745c3",
        "https://images.unsplash.com/photo-1501785888041-af3ef285b470",
        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
        "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0",
        "https://images.unsplash.com/photo-1431794062232-2a99a5431c6c",
        "https://images.unsplash.com/photo-1472214103451-9374bd1c798e",
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
    ],
    "food": [
        "https://images.unsplash.com/photo-1516117172878-fd2c41f4a759",
        "https://images.unsplash.com/photo-1546069901-eacef0df6022",
        "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
        "https://images.unsplash.com/photo-1543352634-99a5d50ae78d",
        "https://images.unsplash.com/photo-1504754524776-8f4f37790ca0",
        "https://images.unsplash.com/photo-1512621776951-a57141f2eefd",
        "https://images.unsplash.com/photo-1506086679525-9d53a902c37f",
        "https://images.unsplash.com/photo-1478145046317-39f10e56b5e9",
        "https://images.unsplash.com/photo-1447078806655-40579c2520d6",
        "https://images.unsplash.com/photo-1512058564366-c9e2d86e0e12",
    ],
    "objects": [
        "https://images.unsplash.com/photo-1512820790803-83ca734da794",
        "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f",
        "https://images.unsplash.com/photo-1503602642458-232111445657",
        "https://images.unsplash.com/photo-1505691938895-1758d7feb511",
        "https://images.unsplash.com/photo-1512499617640-c2f999098c01",
        "https://images.unsplash.com/photo-1501386761578-eac5c94b800a",
        "https://images.unsplash.com/photo-1515165562835-c402adbbe1f5",
        "https://images.unsplash.com/photo-1491553895911-0055eca6402d",
        "https://images.unsplash.com/photo-1534723452862-4c874018d66d",
        "https://images.unsplash.com/photo-1519682337058-a94d519337bc",
    ],
}

os.makedirs(BASE_DIR, exist_ok=True)

for category, urls in categories.items():
    category_dir = os.path.join(BASE_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    print(f"\n📂 Category: {category}")
    for i, url in enumerate(urls, start=1):
        img_path = os.path.join(category_dir, f"{category}_{i}.jpg")
        try:
            r = requests.get(url + "?w=800", stream=True)
            if r.status_code == 200:
                with open(img_path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                print(f"✔ Downloaded: {img_path}")
            else:
                print(f"✖ Error downloading {url}")
        except Exception as e:
            print(f"⚠ Failed: {url} — {e}")
