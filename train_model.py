# train_model.py
"""
Train / retrain pipeline for VocalPrep.

- Expects:
    data/audio_features.csv
    data/transcription_features.csv

- Produces:
    models/voting_confidence_model.joblib
    models/scaler.joblib

This version fixes categorical conversion issues (safe handling of NaNs and categories).
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---- FEATURE NAMES (expected by model logic) ----
FEATURES = [
    'duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean',
    'pitch_mean', 'pitch_std', 'speaking_rate', 'pause_count', 'pause_duration',
    'confidence','articulation','nervousness','perform_confidently','satisfaction',
    'word_count','filler_count','filler_rate','hedge_count',
    'pronoun_count','pronoun_rate','sentence_count','avg_sentence_length',
    'flesch_reading_ease','smog_index','polarity','subjectivity','vader_score',
    'modal_pct','adverb_pct','verb_pct','rule_confidence_score'
]

# -------------------------
def categorize_feature(series, q1, q3, reverse=False):
    """
    Safely categorize a numeric pandas Series into 1/2/3 using cut with q1/q3 boundaries.
    Missing values in `series` are mapped to neutral label 2.
    """
    import pandas as pd
    import numpy as np

    # If series is entirely NaN or empty return neutral series
    if series.isna().all():
        return pd.Series([2] * len(series), index=series.index)

    # If q1==q3, return neutral to avoid zero-width bins
    if q1 == q3:
        return pd.Series([2] * len(series), index=series.index)

    bins = [-np.inf, q1, q3, np.inf]
    if reverse:
        labels = [3, 2, 1]
    else:
        labels = [1, 2, 3]

    # Use pd.cut to create categorical; labels are ints
    cat = pd.cut(series, bins=bins, labels=labels)

    # Ensure category '2' exists before filling; add only if missing
    try:
        # cat.cat.categories may be of numeric dtype or object; to be safe compare by value
        existing = list(cat.cat.categories)
        if not any((int(x) == 2) for x in existing):
            # add category 2
            cat = cat.cat.add_categories([2])
    except Exception:
        # fallback: attempt to add but ignore if it fails
        try:
            cat = cat.cat.add_categories([2])
        except Exception:
            pass

    # Fill NaNs (missing values) with neutral 2 and convert to int
    filled = cat.fillna(2)

    # Convert to int safely: filled may still be categorical; converting via astype(int) is safe now
    try:
        return filled.astype(int)
    except Exception:
        # Final fallback: map categories to ints manually
        mapped = filled.astype(object).map(lambda x: int(x) if pd.notna(x) else 2)
        return pd.Series(mapped.tolist(), index=series.index).astype(int)

def pitch_conf_score(mean, std):
    if pd.isna(mean) or pd.isna(std):
        return 2
    if 128 <= mean <= 164 and std <= 40:
        return 3
    elif mean < 128 or mean > 164 or std > 50:
        return 1
    else:
        return 2

def energy_conf_score(mean, std):
    if pd.isna(mean) or pd.isna(std):
        return 2
    if 0.000011 <= mean <= 0.000272 and std <= 0.000885:
        return 3
    elif mean < 0.000011 or mean > 0.000272 or std > 0.000885:
        return 1
    else:
        return 2

def speaking_pause_conf_score(rate, pause_count):
    if pd.isna(rate) or pd.isna(pause_count):
        return 2
    if 0.084 <= rate <= 0.146 and pause_count <= 8:
        return 3
    elif rate < 0.084 or rate > 0.146 or pause_count > 12:
        return 1
    else:
        return 2

# -------------------------
def compute_rule_confidence(merged_df):
    """
    Adds 'rule_confidence_score' (numeric) and 'rule_confidence_label' (Low/Medium/High)
    to merged_df. Uses a set of prosodic/textual small rules and percentile-based categorizations.
    """
    df = merged_df.copy()
    rules_df = pd.DataFrame(index=df.index)

    # 1) direct rule functions where we have both features
    if 'pitch_mean' in df.columns and 'pitch_std' in df.columns:
        rules_df['pitch_conf'] = df.apply(lambda x: pitch_conf_score(x.get('pitch_mean', np.nan), x.get('pitch_std', np.nan)), axis=1)
    if 'energy_mean' in df.columns and 'energy_std' in df.columns:
        rules_df['energy_conf'] = df.apply(lambda x: energy_conf_score(x.get('energy_mean', np.nan), x.get('energy_std', np.nan)), axis=1)
    if 'speaking_rate' in df.columns and 'pause_count' in df.columns:
        rules_df['speaking_pause_conf'] = df.apply(lambda x: speaking_pause_conf_score(x.get('speaking_rate', np.nan), x.get('pause_count', np.nan)), axis=1)

    # 2) percentile-based categorizations. Use safe_apply wrapper to handle missing columns/NaNs.
    def safe_apply(feat, q1, q3, reverse=False):
        if feat in df.columns:
            # ensure numeric series
            s = pd.to_numeric(df[feat], errors='coerce')
            return categorize_feature(s, q1, q3, reverse=reverse)
        else:
            return pd.Series([2]*len(df), index=df.index)  # neutral if missing

    rules_df['pitch_var_conf'] = safe_apply('pitch_std', 24.96, 48.92, reverse=True)
    rules_df['energy_var_conf'] = safe_apply('energy_std', 0.000061, 0.000885, reverse=True)
    rules_df['zcr_conf'] = safe_apply('zcr_mean', 0.106, 0.147, reverse=False)
    rules_df['pause_dur_conf'] = safe_apply('pause_duration', 0, 5.94, reverse=True)
    rules_df['duration_conf'] = safe_apply('duration', 16.29, 26.70, reverse=False)
    rules_df['filler_conf'] = safe_apply('filler_rate', 0, 0.119, reverse=True)
    rules_df['hedge_conf'] = safe_apply('hedge_count', 0, 0, reverse=True)
    rules_df['word_count_conf'] = safe_apply('word_count', 9, 56.75, reverse=False)
    rules_df['sentence_count_conf'] = safe_apply('sentence_count', 2, 4, reverse=False)
    rules_df['avg_sent_len_conf'] = safe_apply('avg_sentence_length', 3, 17, reverse=False)
    rules_df['flesch_conf'] = safe_apply('flesch_reading_ease', 40.44, 66.02, reverse=False)
    rules_df['smog_conf'] = safe_apply('smog_index', 10.79, 13.23, reverse=True)
    rules_df['articulation_conf'] = safe_apply('articulation', 1, 4, reverse=False)
    rules_df['polarity_conf'] = safe_apply('polarity', -0.217, 0.225, reverse=False)
    rules_df['vader_conf'] = safe_apply('vader_score', -0.13, 0.834, reverse=False)
    rules_df['modal_conf'] = safe_apply('modal_pct', 0, 0.08, reverse=True)
    rules_df['adverb_conf'] = safe_apply('adverb_pct', 0.059, 0.15, reverse=True)
    rules_df['verb_conf'] = safe_apply('verb_pct', 0.128, 0.2, reverse=False)
    rules_df['pronoun_conf'] = safe_apply('pronoun_rate', 0.054, 0.216, reverse=True)

    # average across rules -> numeric rule score
    # ensure numeric then fill any accidental NaNs
    df['rule_confidence_score'] = rules_df.mean(axis=1)
    df['rule_confidence_score'] = pd.to_numeric(df['rule_confidence_score'], errors='coerce')
    if df['rule_confidence_score'].isna().all():
        df['rule_confidence_score'] = 2.0
    else:
        df['rule_confidence_score'] = df['rule_confidence_score'].fillna(df['rule_confidence_score'].median())

    # quantile mapping -> Low/Medium/High
    q_low = df['rule_confidence_score'].quantile(0.33)
    q_med = df['rule_confidence_score'].quantile(0.66)

    def map_overall_quant(score):
        if score <= q_low:
            return 'Low'
        elif score <= q_med:
            return 'Medium'
        else:
            return 'High'

    df['rule_confidence_label'] = df['rule_confidence_score'].apply(map_overall_quant)

    return df

# -------------------------
def ensure_subjective_columns(merged_df):
    """
    Ensure the merged dataframe has the five subjective columns:
      'confidence', 'articulation', 'nervousness', 'perform_confidently', 'satisfaction'

    Preference order:
      1) transcription-derived column (exact name)
      2) audio-derived proxy with suffix '_audio_proxy' or a known proxy name
      3) default neutral value (3)
    """
    df = merged_df
    fallback_map = {
        'confidence': ['confidence', 'confidence_audio_proxy', 'confidence_audio'],
        'articulation': ['articulation', 'articulation_audio_proxy', 'articulation_audio'],
        'nervousness': ['nervousness', 'nervousness_audio_proxy', 'nervousness_audio'],
        'perform_confidently': ['perform_confidently', 'perform_confidently_audio_proxy', 'perform_confidently_audio'],
        'satisfaction': ['satisfaction', 'satisfaction_audio_proxy', 'satisfaction_audio']
    }

    for target_col, candidates in fallback_map.items():
        if target_col in df.columns:
            if df[target_col].isna().any():
                for candidate in candidates[1:]:
                    if candidate in df.columns:
                        df[target_col] = df[target_col].fillna(df[candidate])
                        break
                df[target_col] = df[target_col].fillna(3)
        else:
            filled = False
            for candidate in candidates[1:]:
                if candidate in df.columns:
                    df[target_col] = df[candidate]
                    filled = True
                    break
            if not filled:
                df[target_col] = 3

    # Ensure integer type where appropriate, safely
    for col in ['confidence','articulation','nervousness','perform_confidently','satisfaction']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3).astype(int)
    return df

# -------------------------
def train_and_save(audio_csv='data/audio_features.csv',
                   trans_csv='data/transcription_features.csv',
                   out_model='models/voting_confidence_model.joblib',
                   out_scaler='models/scaler.joblib'):
    """
    Main entrypoint: trains model and scaler, saves them.
    """
    start_time = time.time()
    os.makedirs(os.path.dirname(out_model), exist_ok=True)

    if not os.path.exists(audio_csv) or not os.path.exists(trans_csv):
        raise FileNotFoundError(f"Both audio and transcription CSVs must exist: {audio_csv}, {trans_csv}")

    # Read CSVs
    a = pd.read_csv(audio_csv)
    t = pd.read_csv(trans_csv)

    # Merge on filename (inner join ensures only rows with both audio+text are used)
    merged = pd.merge(a, t, on='filename', how='inner')
    merged = merged.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)

    # Ensure subjective columns exist (using proxies or neutral defaults)
    merged = ensure_subjective_columns(merged)

    # Compute rule_confidence_score & label (adds columns to dataframe)
    merged = compute_rule_confidence(merged)

    # Ensure the rule_confidence_score/label are present
    if 'rule_confidence_score' not in merged.columns or 'rule_confidence_label' not in merged.columns:
        raise RuntimeError("rule_confidence_score or rule_confidence_label missing after compute_rule_confidence()")

    # Prepare X and y (keep features that are available from FEATURES list)
    feat_cols = [f for f in FEATURES if f in merged.columns]
    if len(feat_cols) == 0:
        raise RuntimeError("No features found in merged dataframe. Check CSV column names.")

    X = merged[feat_cols].fillna(0)
    y = merged['rule_confidence_label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    # SMOTE oversampling (train only)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Models
    lr = LogisticRegression(max_iter=500, random_state=42)
    rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)

    ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)], voting='soft', n_jobs=-1)
    ensemble.fit(X_train_scaled, y_train_res)

    # Evaluate
    y_pred = ensemble.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, digits=4)
    print("Validation classification report:\n", report)

    # Persist
    joblib.dump(ensemble, out_model)
    joblib.dump(scaler, out_scaler)
    elapsed = time.time() - start_time
    print(f"Saved model -> {out_model} and scaler -> {out_scaler} (elapsed {elapsed:.1f}s)")

    return out_model, out_scaler

# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_csv", default="data/audio_features.csv")
    parser.add_argument("--trans_csv", default="data/transcription_features.csv")
    parser.add_argument("--model_out", default="models/voting_confidence_model.joblib")
    parser.add_argument("--scaler_out", default="models/scaler.joblib")
    args = parser.parse_args()
    train_and_save(audio_csv=args.audio_csv, trans_csv=args.trans_csv,
                   out_model=args.model_out, out_scaler=args.scaler_out)
