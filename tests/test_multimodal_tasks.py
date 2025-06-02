# tests/test_multimodal_tasks.py

import os
import shutil
import numpy as np
import pytest

# Import the standalone functions and the Celery tasks
from app.tasks import embed_text, embed_image, index_multimodal, search_multimodal


# ---------------------------------------------------------------------------------------------------
# 1. בדיקת embed_text() ו־embed_image()
# ---------------------------------------------------------------------------------------------------
def test_embed_text_and_image(temp_files):
    """
    Ensure that embed_text() and embed_image() both return
    a normalized, one‐dimensional NumPy array of size 512.
    """
    # 1.1 Test embed_text
    text = temp_files["text_content"]
    txt_vec = embed_text(text)

    assert isinstance(txt_vec, np.ndarray), "embed_text should return a NumPy array"
    assert txt_vec.ndim == 1, "embed_text output must be a 1-dimensional vector"
    assert txt_vec.size == 512, "embed_text output must have exactly 512 elements"
    norm_txt = np.linalg.norm(txt_vec)
    assert pytest.approx(norm_txt, rel=1e-5) == 1.0, "embed_text output must be normalized to unit length"

    # 1.2 Test embed_image
    img_path = temp_files["image_path"]
    img_vec = embed_image(img_path)

    assert isinstance(img_vec, np.ndarray), "embed_image should return a NumPy array"
    assert img_vec.ndim == 1, "embed_image output must be a 1-dimensional vector"
    assert img_vec.size == 512, "embed_image output must have exactly 512 elements"
    norm_img = np.linalg.norm(img_vec)
    assert pytest.approx(norm_img, rel=1e-5) == 1.0, "embed_image output must be normalized to unit length"


# ---------------------------------------------------------------------------------------------------
# 2. בדיקת index_multimodal() ו־search_multimodal()
# ---------------------------------------------------------------------------------------------------
def test_index_and_search_multimodal(tmp_path, temp_files, temp_user_dir):
    """
    2.1 Index a combined (text + image) document under a temporary user directory.
    2.2 Verify that both FAISS index and ID‐map files appear on disk.
    2.3 Run search_multimodal() by text only, image only, and text+image, and check that doc_id comes back.
    """
    username = "tester"
    doc_id = "doc0"

    # 2.1 Prepare a temporary "user_data/<username>/mm_index" folder
    user_folder = os.path.join(temp_user_dir, username)
    mm_index_folder = os.path.join(user_folder, "mm_index")
    os.makedirs(mm_index_folder, exist_ok=True)

    # 2.2 Call index_multimodal with text + image (Eager Celery -> immediate execution)
    text = temp_files["text_content"]
    img_path = temp_files["image_path"]

    # Use .delay().get() so that Celery’s eager mode runs it immediately
    res_idx = index_multimodal.delay(username, doc_id, text, img_path).get()
    assert isinstance(res_idx, dict), "index_multimodal should return a dict"
    assert res_idx["status"] == "indexed_multimodal"
    assert res_idx["doc_id"] == doc_id

    # 2.3 Confirm FAISS index files were created in <user_folder>/mm_index/
    faiss_file = os.path.join(mm_index_folder, "faiss_index.bin")
    map_file   = os.path.join(mm_index_folder, "id_map.pkl")
    assert os.path.exists(faiss_file), "FAISS index file was not created"
    assert os.path.exists(map_file),   "ID‐map file was not created"

    # 2.4 Search by text only
    res_search_txt = search_multimodal.delay(
        username,
        query_text=text,
        query_image_path=None,
        top_k=1
    ).get()
    assert "results" in res_search_txt, "search_multimodal result must contain 'results' key"
    results_txt = res_search_txt["results"]
    assert isinstance(results_txt, list) and results_txt, "No results returned for text search"
    assert results_txt[0][0] == doc_id, "Text search should return the correct doc_id"

    # 2.5 Search by image only
    res_search_img = search_multimodal.delay(
        username,
        query_text=None,
        query_image_path=img_path,
        top_k=1
    ).get()
    assert "results" in res_search_img, "search_multimodal result must contain 'results' key"
    results_img = res_search_img["results"]
    assert isinstance(results_img, list) and results_img, "No results returned for image search"
    assert results_img[0][0] == doc_id, "Image search should return the correct doc_id"

    # 2.6 Search by combined text + image
    res_search_mm = search_multimodal.delay(
        username,
        query_text=text,
        query_image_path=img_path,
        top_k=1
    ).get()
    assert "results" in res_search_mm, "search_multimodal result must contain 'results' key"
    results_mm = res_search_mm["results"]
    assert isinstance(results_mm, list) and results_mm, "No results returned for multimodal search"
    assert results_mm[0][0] == doc_id, "Multimodal search should return the correct doc_id"

    # 2.7 Clean up the temporary user folder
    shutil.rmtree(user_folder)
