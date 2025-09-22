# pages/3_Analyze_Feedback.py
import streamlit as st
import os
import re
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from train_model import FEATURES, ensure_subjective_columns, compute_rule_confidence, train_and_save

# Paths
DATA_AUDIO_CSV = "data/audio_features.csv"
DATA_TRANS_CSV = "data/transcription_features.csv"
RAW_DIR = "data/raw_audios"
MODEL_PATH = "models/voting_confidence_model.joblib"
SCALER_PATH = "models/scaler.joblib"

st.set_page_config(page_title="Analyze Feedback", layout="wide")
st.title("📊 Personalized Feedback — Auto-detected (Name & Question inferred)")

# -------------------------
# Helpers for detection & inference
# -------------------------
def list_emails_from_raw(raw_dir=RAW_DIR, limit=20):
    p = Path(raw_dir)
    if not p.exists() or not any(p.iterdir()):
        return []
    files = [f for f in p.iterdir() if f.is_file()]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    emails = []
    for f in files:
        name_no_ext = os.path.splitext(f.name)[0]
        parts = name_no_ext.split('_')
        ts_idx = None
        for i, tok in enumerate(parts):
            if tok.isdigit():
                ts_idx = i
                break
        if ts_idx is None:
            continue
        email_token = "_".join(parts[:ts_idx])
        reconstructed = email_token.replace("_", "@", 1) if "_" in email_token else email_token
        if reconstructed not in emails:
            emails.append(reconstructed)
        if len(emails) >= limit:
            break
    return emails

def auto_detect_email():
    # prefer session value
    if "email" in st.session_state and st.session_state.get("email"):
        return st.session_state.get("email")
    emails = list_emails_from_raw()
    return emails[0] if emails else None

def friendly_name_from_email(email):
    """Derive a readable name from an email local-part if session name not present."""
    if not email or "@" not in email:
        return None
    local = email.split("@", 1)[0]
    # local might contain dots/underscores; pick first token and title-case
    token = re.split[r"[._\-]", local][0] if False else None
    # simpler: replace underscores/dots, take first piece
    token = re.split(r"[._\-]", local)[0]
    return token.replace("_", " ").replace(".", " ").title()

def extract_question_from_filename(filename):
    """
    Tries to extract question index or text from filename.
    Expected pattern: ..._q{index}_... e.g. 'meghna_169xxx_q3_ab12.wav'
    Returns a string like 'Q3' or None if not found.
    """
    name = os.path.basename(filename)
    m = re.search(r"_q(\d+)_", name, flags=re.IGNORECASE)
    if m:
        return f"Q{int(m.group(1))}"
    # fallback: maybe pattern without trailing underscore, try _q{n} or -q{n}
    m2 = re.search(r"[._\-]q(\d+)(?:[._\-]|$)", name, flags=re.IGNORECASE)
    if m2:
        return f"Q{int(m2.group(1))}"
    return None

def friendly_actionable_feedback(row):
    """
    Build human-friendly suggestions from a merged feature row (audio+transcription merged).
    Uses numeric thresholds; returns list of suggestion strings.
    """
    suggestions = []
    def val(col, default=np.nan):
        return float(row.get(col, default)) if pd.notna(row.get(col, default)) else np.nan

    conf_label = row.get("rule_confidence_label", None)
    if conf_label == "High":
        suggestions.append("Good confidence — keep up the energy and clarity.")
    elif conf_label == "Medium":
        suggestions.append("Medium confidence — project your voice a bit more and reduce filler words.")
    else:
        suggestions.append("Low confidence — work on projecting your voice and structuring your answers.")

    filler_rate = val("filler_rate", 0)
    if filler_rate > 0.15:
        suggestions.append(f"High filler usage (≈{filler_rate:.2f}). Pause silently instead of saying 'um' or 'like'.")
    elif filler_rate > 0.05:
        suggestions.append(f"Some filler words (≈{filler_rate:.2f}). Aim to reduce pauses filled with 'um'.")

    pause_count = val("pause_count", 0)
    speaking_rate = val("speaking_rate", 0)
    if pause_count > 10:
        suggestions.append(f"Many pauses ({int(pause_count)}). Practice chaining ideas into short sentences.")
    if speaking_rate < 0.06:
        suggestions.append("Speaking rate slow — try speaking a bit more steadily.")
    elif speaking_rate > 0.20:
        suggestions.append("Speaking rate fast — slow down slightly to improve clarity.")

    pitch_std = val("pitch_std", np.nan)
    if not np.isnan(pitch_std):
        if pitch_std > 50:
            suggestions.append("Pitch varies a lot — practice keeping a steadier tone.")
        elif pitch_std < 15:
            suggestions.append("Pitch is flat — add slight variation to keep listener engaged.")

    articulation = row.get("articulation", None)
    if pd.notna(articulation):
        if articulation <= 2:
            suggestions.append("Work on articulation — pronounce key words clearly.")
        elif articulation == 3:
            suggestions.append("Articulation okay; aim for crisper consonants.")
        else:
            suggestions.append("Good articulation — keep it up.")

    # final fallback tip
    if len(suggestions) == 0:
        suggestions.append("No specific issues detected — practice mock answers to maintain consistency.")
    return suggestions

# -------------------------
# Main UI + logic
# -------------------------
emails = list_emails_from_raw()
detected_email = auto_detect_email()

# allow choosing from detected emails if multiple present
if emails:
    options = [detected_email] + [e for e in emails if e != detected_email]
    chosen = st.selectbox("Detected users (most recent first) — pick one to view", options=options)
    if chosen:
        detected_email = chosen

if not detected_email:
    st.warning("No user detected automatically. Please go to Record Answers and save at least one audio with your email.")
    st.stop()

st.markdown(f"**Showing feedback for:** `{detected_email}`")

# check model exists (we will produce feedback using last trained model)
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.error("Trained model not found. Please train the model first (use Record Answers -> retrain flow).")
    st.stop()

if not (os.path.exists(DATA_AUDIO_CSV) and os.path.exists(DATA_TRANS_CSV)):
    st.error("Feature CSVs missing. Please run the Record Answers flow and ensure files were saved.")
    st.stop()

# load CSVs
audio_df = pd.read_csv(DATA_AUDIO_CSV)
trans_df = pd.read_csv(DATA_TRANS_CSV)

# find rows for this user by tokenizing email to filename token
token = detected_email.replace("@", "_", 1)
user_audio = audio_df[audio_df['filename'].str.contains(token, na=False)]
user_trans = trans_df[trans_df['filename'].str.contains(token, na=False)]

if user_audio.empty or user_trans.empty:
    st.warning("No feature rows were found for this user. Make sure you saved answers on the Record page.")
    st.stop()

# merge on filename
merged = pd.merge(user_audio, user_trans, on="filename", how="inner")
merged = merged.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)

# ensure subjective columns are present (audio side provides them; transcription no longer has them)
merged = ensure_subjective_columns(merged)
merged = compute_rule_confidence(merged)

# load model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

feat_cols = [f for f in FEATURES if f in merged.columns]
if len(feat_cols) == 0:
    st.error("No model features found in merged data. Check CSV column names.")
    st.stop()

X = merged[feat_cols].fillna(0)
try:
    X_scaled = scaler.transform(X)
except Exception as e:
    st.error(f"Failed to scale features for prediction: {e}")
    st.stop()

preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)

# determine a friendly user name
user_name_display = st.session_state.get("user_name")
if not user_name_display:
    # fallback to email local-part
    try:
        user_name_display = detected_email.split("@", 1)[0].replace(".", " ").replace("_", " ").title()
    except Exception:
        user_name_display = None

if user_name_display:
    st.markdown(f"**Name:** {user_name_display}  &nbsp;&nbsp;&nbsp; **Email:** {detected_email}")
else:
    st.markdown(f"**Email:** {detected_email}")

st.subheader("Per-question feedback (based on the last trained model)")

for i, (_, row) in enumerate(merged.iterrows()):
    pred = preds[i]
    pvec = probs[i]
    filename = row.get("filename", "")
    q_text = extract_question_from_filename(filename) or f"Answer {i+1}"
    st.markdown(f"### {q_text} — `{filename}`")
    st.write(f"- Predicted confidence: **{pred}**  —  Probabilities: {dict(zip(model.classes_, [round(float(x),3) for x in pvec]))}")
    # show user-friendly suggestions
    tips = friendly_actionable_feedback(row)
    st.write("**Suggestions:**")
    for t in tips:
        st.write("-", t)
    # show a few readable metrics
    st.write("**Quick metrics:**")
    display_metrics = {
        "Filler rate": row.get("filler_rate", None),
        "Pauses": int(row.get("pause_count", 0)) if pd.notna(row.get("pause_count", None)) else None,
        "Speaking rate": round(float(row.get("speaking_rate", 0)), 3) if pd.notna(row.get("speaking_rate", None)) else None,
        "Articulation": int(row.get("articulation", 3)) if pd.notna(row.get("articulation", None)) else None,
        "Confidence (audio proxy)": int(row.get("confidence", 3)) if pd.notna(row.get("confidence", None)) else None
    }
    for k, v in display_metrics.items():
        st.write(f"- **{k}:** {v}")
    st.markdown("---")

# overall summary
st.subheader("Overall summary")
overall = pd.Series(preds).mode()[0]
st.write(f"**Overall Confidence:** {overall}")
if overall == "High":
    st.success("Excellent overall — keep practicing to maintain clarity & energy.")
elif overall == "Medium":
    st.info("Average — focus on reducing fillers and controlling pauses.")
else:
    st.error("Low — work on voice projection, structure, and pause control.")

# After showing feedback, retrain automatically so future predictions include these answers
st.info("The model will now retrain automatically to include these newly saved answers for future feedback.")
with st.spinner("Retraining model now (this will block the UI until finished)..."):
    try:
        model_out, scaler_out = train_and_save(audio_csv=DATA_AUDIO_CSV, trans_csv=DATA_TRANS_CSV)
        st.success("Retraining completed and model updated.")
        st.write(f"New model saved to: {model_out}")
    except Exception as e:
        st.error(f"Retraining failed: {e}")
