# pages/1_Instructions.py
import streamlit as st
import os
import subprocess
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="VocalPrep — Instructions", layout="wide")
st.title("VocalPrep — Instructions & Excel → CSV helper")

st.markdown(
    """
    ## Quick overview
    VocalPrep uses two feature tables:
    - **Audio-derived features** (prosodic features) — expected at `data/audio_features.csv`
    - **Transcription/text-derived features** — expected at `data/transcription_features.csv`
    
    You mentioned you prefer to maintain these as **Excel** files (`.xlsx`).  
    To avoid changing the model and training code, this page provides a simple **Excel → CSV converter** that:
    1. reads `data/audio_features.xlsx` and writes `data/audio_features.csv`
    2. reads `data/transcription_features.xlsx` and writes `data/transcription_features.csv`
    
    After conversion you can run the training script `train_model.py` (a button is provided below).
    """
)

st.header("1) Where to place your Excel feature files")
st.markdown("""
- Put your Excel feature files here:
  - `data/audio_features.xlsx`
  - `data/transcription_features.xlsx`
- Each file should have a `filename` column that exactly matches the audio filenames saved in `data/raw_audios/`.
- Column names should match the features used in training (for example: `duration`, `zcr_mean`, `pitch_mean`, `pitch_std`, `word_count`, `filler_count`, etc.)
""")

st.header("2) Quick checks (files & folders)")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_audios"
A_XLSX = DATA_DIR / "audio_features.xlsx"
T_XLSX = DATA_DIR / "transcription_features.xlsx"
A_CSV = DATA_DIR / "audio_features.csv"
T_CSV = DATA_DIR / "transcription_features.csv"
MODEL_PATH = Path("models/voting_confidence_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")

col1, col2, col3 = st.columns(3)
with col1:
    st.write("data/raw_audios exists?")
    st.write("✅" if RAW_DIR.exists() else "❌ — create and save sample wavs here")

with col2:
    st.write("audio_features.xlsx exists?")
    st.write("✅" if A_XLSX.exists() else "❌ — place file at data/audio_features.xlsx")

with col3:
    st.write("transcription_features.xlsx exists?")
    st.write("✅" if T_XLSX.exists() else "❌ — place file at data/transcription_features.xlsx")

st.header("3) Convert Excel → CSV (one-click)")
st.markdown("""
If you want to keep maintaining the `.xlsx` files, click **Convert Excel → CSV** below.  
This will:
- Read `data/audio_features.xlsx` → write `data/audio_features.csv`
- Read `data/transcription_features.xlsx` → write `data/transcription_features.csv`

The rest of the pipeline (`train_model.py`, Streamlit pages) will continue to use the CSV files unchanged.
""")

def convert_excel_to_csv():
    DATA_DIR.mkdir(exist_ok=True)
    msgs = []
    # audio
    if A_XLSX.exists():
        try:
            df_a = pd.read_excel(A_XLSX)
            df_a.to_csv(A_CSV, index=False)
            msgs.append(f"Converted {A_XLSX} → {A_CSV} ({len(df_a)} rows).")
        except Exception as e:
            msgs.append(f"Failed converting {A_XLSX}: {e}")
    else:
        msgs.append(f"{A_XLSX} not found.")

    # transcription
    if T_XLSX.exists():
        try:
            df_t = pd.read_excel(T_XLSX)
            df_t.to_csv(T_CSV, index=False)
            msgs.append(f"Converted {T_XLSX} → {T_CSV} ({len(df_t)} rows).")
        except Exception as e:
            msgs.append(f"Failed converting {T_XLSX}: {e}")
    else:
        msgs.append(f"{T_XLSX} not found.")

    return msgs

if st.button("Convert Excel → CSV"):
    with st.spinner("Converting..."):
        output_msgs = convert_excel_to_csv()
    for m in output_msgs:
        st.write("-", m)

st.header("4) Train the model (use CSVs)")
st.markdown("""
After conversion, click the button below to run the training script **train_model.py** which:
- merges `data/audio_features.csv` and `data/transcription_features.csv` on `filename`,
- computes the rule-based `rule_confidence_score` and labels,
- trains the ensemble model and scaler,
- writes `models/voting_confidence_model.joblib` and `models/scaler.joblib`.
""")

def run_training():
    # Runs the training synchronously. This will block the Streamlit app during training.
    # Make sure your environment has `python` pointing to the Python interpreter with dependencies installed.
    cmd = ["python", "train_model.py", "--audio_csv", str(A_CSV), "--trans_csv", str(T_CSV)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

if st.button("Run training now (runs synchronously)"):
    # quick pre-check
    if not A_CSV.exists() or not T_CSV.exists():
        st.error("CSV files not found. Please convert Excel to CSV first (use the Convert button above).")
    else:
        st.info("Training started — this runs synchronously and will block until finished. Expect some time depending on dataset size.")
        code, out, err = run_training()
        if code == 0:
            st.success("Training completed successfully.")
            st.text(out)
        else:
            st.error("Training failed. See stderr below.")
            st.text(err)

st.header("5) After training")
if MODEL_PATH.exists() and SCALER_PATH.exists():
    st.success(f"Model found: {MODEL_PATH}")
    st.success(f"Scaler found: {SCALER_PATH}")
else:
    st.warning("Model &/or scaler not found. Run training above to create them.")

st.markdown("---")
st.header("6) Best practices & tips")
st.markdown("""
- Keep `filename` consistent between `data/raw_audios/` and the feature sheets.
- If you collect new user audio via the app, use the **Record Answers** page to upload/save: it will append features and (optionally) retrain.
- Retraining runs synchronously in the app — for large datasets this can be slow. For production, consider offloading retraining to a worker/queue.
- If you prefer direct Excel support in `train_model.py` we can adapt the training script to read `.xlsx` directly; currently we convert to CSV to avoid editing the training script.
""")

st.markdown("If you'd like, I can also: **(A)** add Excel-reading support in `train_model.py` (so conversion step is not needed), or **(B)** add a native in-browser recorder (Start/Stop) on the Record page. Tell me which and I will implement immediately.")
