# pages/2_Record_Answers.py
import streamlit as st
import os, time, uuid
from question_generator import generate_questions
from utils.audio_feature_extractor import append_to_csv as append_audio_csv
from utils.transcription_feature_extractor import append_to_csv as append_trans_csv

DATA_AUDIO_CSV = "data/audio_features.csv"
DATA_TRANS_CSV = "data/transcription_features.csv"
RAW_DIR = "data/raw_audios"

st.title("🎙️ Record / Save Answers — one audio per question (auto-save on upload)")

# User metadata (kept in session only)
if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""
if "email" not in st.session_state:
    st.session_state["email"] = ""

name = st.text_input("Name", key="name_input", value=st.session_state.get("user_name", ""))
email = st.text_input("Email", key="email_input", value=st.session_state.get("email", ""))

# keep session values in sync
st.session_state["user_name"] = name
st.session_state["email"] = email

domain = st.selectbox("Domain", ["Software", "Data Science", "Product", "Other"])
level = st.selectbox("Expertise level", ["Beginner", "Intermediate", "Expert"])

if "vocalprep_questions" not in st.session_state:
    st.session_state["vocalprep_questions"] = []

# generate questions (stores them in session)
if st.button("Generate 10 Questions"):
    with st.spinner("Generating questions..."):
        qs = generate_questions(domain=domain, level=level, n=10)
        st.session_state["vocalprep_questions"] = qs

questions = st.session_state.get("vocalprep_questions", [])
if not questions:
    st.info("Click 'Generate 10 Questions' to get started.")
    st.stop()

st.markdown("Upload one audio per question. Files are saved automatically as soon as you upload them.")

# Ensure raw dir exists
os.makedirs(RAW_DIR, exist_ok=True)

# track saved questions to prevent duplicate saving during reruns
if "saved_questions" not in st.session_state:
    st.session_state["saved_questions"] = set()

# For each question show a file uploader; when uploaded, save immediately (once)
for idx, q in enumerate(questions, start=1):
    st.markdown(f"### Q{idx}. {q}")
    file_key = f"file_q{idx}"

    # Use a file_uploader widget (single file)
    uploaded = st.file_uploader(f"Upload answer for Q{idx} (auto-saves)", type=['wav','mp3','m4a'], key=file_key, accept_multiple_files=False)

    # if user already uploaded & we previously saved it, show saved info
    saved_key = f"saved_q{idx}"
    already_saved = saved_key in st.session_state and st.session_state[saved_key] is True

    if uploaded is not None and not already_saved:
        # Save immediately
        try:
            ts = int(time.time())
            safe_email = (st.session_state.get("email") or "unknown").replace("@", "_", 1)
            fname = f"{safe_email}_{ts}_q{idx}_{uuid.uuid4().hex[:6]}{os.path.splitext(uploaded.name)[1]}"
            out_path = os.path.join(RAW_DIR, fname)
            with open(out_path, "wb") as wf:
                wf.write(uploaded.getbuffer())

            # Append audio features (audio CSV)
            try:
                audio_feats = append_audio_csv(out_path, DATA_AUDIO_CSV)
            except Exception as e:
                st.error(f"Audio feature extraction failed for Q{idx}: {e}")
                # still mark as saved to avoid repeated failing attempts; you can inspect logs
                st.session_state[saved_key] = True
                continue

            # Append transcription features (transcription CSV)
            try:
                trans_feats = append_trans_csv(out_path, DATA_TRANS_CSV)
            except Exception as e:
                st.error(f"Transcription feature extraction failed for Q{idx}: {e}")
                # we still consider file saved even if transcription failed
                st.session_state[saved_key] = True
                continue

            # mark saved in session so we don't re-save on reruns
            st.session_state[saved_key] = True
            # add to saved_questions set for reference
            st.session_state["saved_questions"].add(idx)

            st.success(f"Saved answer for Q{idx} as `{fname}`")
            # show quick playback for confirmation
            st.audio(out_path)

        except Exception as e:
            st.error(f"Failed to save uploaded file for Q{idx}: {e}")
            # do not set saved flag so user can retry

    elif already_saved:
        st.info("This answer has already been saved in this session.")
        # attempt to display the most recent saved filename for this question if we can find it
        # Search raw dir for files matching pattern safe_email_*_q{idx}_
        try:
            safe_email = (st.session_state.get("email") or "unknown").replace("@", "_", 1)
            files = [f for f in os.listdir(RAW_DIR) if f.startswith(safe_email) and f"_q{idx}_" in f]
            files.sort(reverse=True)
            if files:
                st.write(f"Saved file: `{files[0]}`")
                # offer playback of the saved file if exists
                try:
                    st.audio(os.path.join(RAW_DIR, files[0]))
                except Exception:
                    pass
        except Exception:
            pass

# Summary area
st.markdown("---")
st.write(f"Answers saved this session: {sorted(list(st.session_state.get('saved_questions', [])))}")

