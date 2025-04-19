import os
import streamlit as st
import tempfile
from main import CLIPSecureEncryptor

# הגדרות ממשק
st.set_page_config(page_title="🔐 חיפוש סמנטי מאובטח", layout="centered")
st.title("🔍 חיפוש סמנטי בכל סוגי הקבצים")

# נתיבים
index_path = "/home/danielbes/Desktop/BETA/app/app/my_index.pkl"
data_folder = "/home/danielbes/Desktop/BETA/DATA"

# קלט סיסמה
password = st.text_input("הכנס סיסמה", type="password")

# אתחול סטייטים
if "encryptor" not in st.session_state:
    st.session_state.encryptor = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# טעינת אינדקס אוטומטית
if password and os.path.exists(index_path) and not st.session_state.index_ready:
    try:
        encryptor = CLIPSecureEncryptor(password)
        encryptor.load_index(index_path)
        st.session_state.encryptor = encryptor
        st.session_state.index_ready = True
        st.success("✅ אינדקס נטען אוטומטית.")
    except Exception as e:
        st.error(f"שגיאה בטעינה: {e}")

# בניית אינדקס חדש
if st.button("🔨 בנה אינדקס חדש") and password:
    try:
        encryptor = CLIPSecureEncryptor(password)
        files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
        encryptor.build_index_from_files(files)
        encryptor.save_index(index_path)
        st.session_state.encryptor = encryptor
        st.session_state.index_ready = True
        st.success("✅ אינדקס נבנה ונשמר.")
    except Exception as e:
        st.error(f"שגיאה בבניית אינדקס: {e}")

# חיפוש טקסטואלי
if st.session_state.index_ready:
    st.subheader("💬 חיפוש לפי טקסט")
    query = st.text_input("טקסט לחיפוש:")
    if st.button("🔍 חפש טקסט"):
        try:
            results = st.session_state.encryptor.query_text(query)
            st.markdown("### 📁 תוצאות:")
            for path in results:
                st.write(f"📄 {os.path.basename(path)}")
        except Exception as e:
            st.error(f"שגיאה בחיפוש טקסט: {e}")

    st.subheader("📁 העלאת קובץ לחיפוש")
    uploaded = st.file_uploader("בחר קובץ", type=None)
    if uploaded and st.button("🔍 חפש לפי קובץ"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            results = st.session_state.encryptor.query_file(tmp_path)
            st.markdown("### 📁 תוצאות:")
            for path in results:
                st.write(f"📄 {os.path.basename(path)}")
        except Exception as e:
            st.error(f"שגיאה בחיפוש לפי קובץ: {e}")
        finally:
            os.remove(tmp_path)
