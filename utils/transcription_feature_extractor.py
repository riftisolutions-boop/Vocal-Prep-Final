# utils/transcription_feature_extractor.py
"""
Transcription + text feature extractor using SpeechRecognition (Google Web Speech API).
This version DOES NOT write subjective columns (satisfaction, perform_confidently,
articulation, nervousness) nor metadata (user_name, question).
It only writes technical/textual features used by the model.
"""

import os
import re
import tempfile
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

analyzer = SentimentIntensityAnalyzer()

FILLER_WORDS = ['um', 'uh', 'er', 'ah', 'hmm','like', 'you know', 'i mean', 'sort of', 'kind of', 'i guess',
                'maybe', 'probably', 'i think', 'i suppose']
HEDGE_WORDS = ['maybe', 'i guess', 'probably', 'sort of', 'kind of', 'i think', 'i suppose']
PRONOUNS = ['i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'she', 'they', 'them', 'their']
MODAL_WORDS = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'ought']

def _convert_to_wav(src_path):
    # convert to 16k mono WAV using pydub
    sound = AudioSegment.from_file(src_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
        tmp_path = tmpf.name
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(tmp_path, format="wav")
    return tmp_path

def _transcribe_with_speechrecognition(wav_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""
    except Exception:
        return ""

def _safe_tokens(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[\w']+\b", text.lower())

def transcribe_and_extract(file_path):
    """
    Transcribe file_path and compute text-derived technical features (no subjective scores).
    Returns a dict suitable for appending to transcription_features.csv.
    """
    tmp_wav = None
    try:
        tmp_wav = _convert_to_wav(file_path)
        transcription = _transcribe_with_speechrecognition(tmp_wav)
    except Exception:
        transcription = ""
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

    transcription = (transcription or "").strip()
    clean_text = re.sub(r'[^\w\s]', '', transcription.lower())
    tokens = _safe_tokens(clean_text)
    word_count = len(tokens)
    filler_count = sum(clean_text.count(w) for w in FILLER_WORDS)
    filler_rate = filler_count / word_count if word_count else 0.0
    hedge_count = sum(clean_text.count(w) for w in HEDGE_WORDS)
    pronoun_count = sum(tokens.count(w) for w in PRONOUNS)
    pronoun_rate = pronoun_count / word_count if word_count else 0.0

    sentence_count = max(1, len(re.findall(r'[.!?]', transcription))) if transcription else max(1, int(max(1, word_count / 12)))
    avg_sentence_length = word_count / sentence_count if sentence_count else float(word_count)

    try:
        flesch = float(textstat.flesch_reading_ease(transcription)) if transcription else 0.0
    except Exception:
        flesch = 0.0
    try:
        smog = float(textstat.smog_index(transcription)) if len(tokens) > 3 else 0.0
    except Exception:
        smog = 0.0

    try:
        polarity = float(TextBlob(clean_text).sentiment.polarity) if clean_text else 0.0
        subjectivity = float(TextBlob(clean_text).sentiment.subjectivity) if clean_text else 0.0
    except Exception:
        polarity = 0.0
        subjectivity = 0.0

    try:
        vader_score = float(analyzer.polarity_scores(clean_text)['compound']) if clean_text else 0.0
    except Exception:
        vader_score = 0.0

    modal_count = sum(1 for tok in tokens if tok in MODAL_WORDS)
    modal_pct = modal_count / (len(tokens) or 1)
    adverb_count = sum(1 for tok in tokens if tok.endswith('ly') or tok in ('very','really','quite','extremely','slightly'))
    adverb_pct = adverb_count / (len(tokens) or 1)

    # verb_count via TextBlob tags if available, fallback heuristics
    verb_count = 0
    try:
        tb = TextBlob(clean_text)
        tags = tb.tags
        verb_count = sum(1 for w,tag in tags if tag.startswith('VB'))
    except Exception:
        VERB_SUFFIXES = ('ed','ing','ize','ise')
        common_verbs = set(['be','is','are','am','have','has','had','do','does','did','say','says','go','goes','make','made','get','got','take','took','see','seen','know','knew','think','thought','come','came','want','wanted','use','used'])
        verb_count = sum(1 for tok in tokens if tok in common_verbs or tok.endswith(VERB_SUFFIXES))
    verb_pct = verb_count / (len(tokens) or 1)

    feats = {
        'filename': os.path.basename(file_path),
        'transcription': transcription,
        'clean_text': clean_text,
        'word_count': int(word_count),
        'filler_count': int(filler_count),
        'filler_rate': float(filler_rate),
        'hedge_count': int(hedge_count),
        'pronoun_count': int(pronoun_count),
        'pronoun_rate': float(pronoun_rate),
        'sentence_count': int(sentence_count),
        'avg_sentence_length': float(avg_sentence_length),
        'flesch_reading_ease': float(flesch),
        'smog_index': float(smog),
        'polarity': float(polarity),
        'subjectivity': float(subjectivity),
        'vader_score': float(vader_score),
        'modal_pct': float(modal_pct),
        'adverb_pct': float(adverb_pct),
        'verb_pct': float(verb_pct)
        # NOTE: no subjective columns and no metadata columns here
    }

    return feats

def append_to_csv(file_path, out_csv):
    """
    Extract transcription features and append to a CSV (create if missing).
    Does NOT write subjective or metadata columns.
    """
    feats = transcribe_and_extract(file_path)
    df_row = pd.DataFrame([feats])
    if os.path.exists(out_csv):
        try:
            df_prev = pd.read_csv(out_csv)
            df_new = pd.concat([df_prev, df_row], ignore_index=True)
            df_new.to_csv(out_csv, index=False)
        except Exception:
            df_row.to_csv(out_csv, index=False)
    else:
        df_row.to_csv(out_csv, index=False)
    return feats
