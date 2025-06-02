# tests/conftest.py

import os
import shutil
import json
import tempfile
import pytest

from app.celery_app import celery
from app import user_manager

# ---------------------------------------------------------------------------------------------------
# Fixture: מפעיל Celery במצב eager (כל משימה מתבצעת מיד בתוך התהליך הנוכחי)
# ---------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def celery_eager(monkeypatch):
    """
    מגדיר את Celery כך שיריץ את המשימות מקומית (אין קשר ל־Redis ומצב ריצה).
    """
    celery.conf.update(task_always_eager=True)
    celery.conf.update(task_eager_propagates=True)
    yield
    # לא נדרש לנקות פה – Pytest עושה זאת בסוף הסשן.

# ---------------------------------------------------------------------------------------------------
# Fixture: יוצר תיקיית משתמש זמנית ומשתיל אותה ב־user_manager.BASE_USER_DATA
# ---------------------------------------------------------------------------------------------------
@pytest.fixture
def temp_user_dir(tmp_path, monkeypatch):
    """
    מגדיר תיקיית 'user_data' זמנית כך שכל קריאה לשמירת אינדקס/קבצים
    תיעשה לתיקייה זמנית, ולא תתערב בתיקיות הראשיות.
    """
    root = tmp_path / "user_data"
    root.mkdir()
    # המשתנה BASE_USER_DATA (מוגדר ב־user_manager.py) מצביע על המיקום של תיקיית user_data.
    monkeypatch.setattr(user_manager, "BASE_USER_DATA", str(root))

    yield str(root)
    # בסיום הבדיקה, tmp_path ימחק אוטומטית.

# ---------------------------------------------------------------------------------------------------
# Fixture: יוצר קובץ טקסט זמני וקובץ תמונה (1×1) לתיקייה זמנית
# ---------------------------------------------------------------------------------------------------
@pytest.fixture
def temp_files(tmp_path):
    """
    יוצר:
      - example.txt  (UTF-8) עם תוכן ידוע
      - example.jpg  (תמונה לבנה 1×1)
    ישתמשו בכך באינדוקס רב־ממדי ובבדיקתשאילתות.
    """
    # קובץ טקסט
    text_path = tmp_path / "example.txt"
    text_content = "This is a sample text for multimodal indexing."
    text_path.write_text(text_content, encoding="utf-8")

    # קובץ תמונה 1×1 (RGB לבן)
    from PIL import Image
    image_path = tmp_path / "example.jpg"
    img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    img.save(str(image_path), format="JPEG")

    return {
        "text_path": str(text_path),
        "image_path": str(image_path),
        "text_content": text_content
    }
