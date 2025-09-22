# app.py
# Minimal main that uses the pages/ scripts (Streamlit multi-page style).
import streamlit as st

st.set_page_config(page_title="VocalPrep", layout="wide")
st.title("VocalPrep — Your AI Interview Coach")

st.write("""
Use the left-hand menu (Streamlit pages) to:
- Read Instructions
- Record / Upload answers (Record Answers)
- Analyze Feedback
""")
st.write("Open the three pages under the Streamlit pages sidebar (if using streamlit run app.py).")
