# utils/audio_feature_extractor.py
import os
import numpy as np
import pandas as pd
import librosa

def _scale_to_1_5(x, lo, hi, reverse=False):
    """Scale numeric x to integer 1..5 by clipping then linear mapping."""
    try:
        if np.isnan(x):
            return 3
    except Exception:
        pass
    try:
        x = float(x)
    except Exception:
        return 3
    rng = float(hi - lo) if (hi is not None and lo is not None) else 1.0
    if rng == 0:
        return 3
    if reverse:
        clipped = max(min(x, hi), lo)
        frac = (hi - clipped) / (rng + 1e-9)
    else:
        clipped = max(min(x, hi), lo)
        frac = (clipped - lo) / (rng + 1e-9)
    score = 1 + int(round(frac * 4))
    return int(min(max(score, 1), 5))

def extract_features(file_path):
    """
    Extract audio prosodic features and derive 1..5 subjective columns.
    Returns a dict of features.
    """
    y, sr = librosa.load(file_path, sr=16000)
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    pause_count = max(len(non_silent_intervals) - 1, 0)
    total_silence = 0.0
    for i in range(1, len(non_silent_intervals)):
        prev_end = non_silent_intervals[i - 1][1]
        curr_start = non_silent_intervals[i][0]
        total_silence += (curr_start - prev_end) / sr

    duration = librosa.get_duration(y=y, sr=sr)
    energy = np.square(y) if y.size else np.array([0.0])
    energy_mean = float(np.mean(energy)) if energy.size else 0.0
    energy_std = float(np.std(energy)) if energy.size else 0.0
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)[0])) if y.size else 0.0
    rms_mean = float(np.mean(librosa.feature.rms(y=y))) if y.size else 0.0
    speaking_rate = (np.sum(energy > (0.01 * np.max(energy))) / sr) / duration if duration > 0 else 0.0

    try:
        pitches, _, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
        pitch_mean = float(np.nanmean(pitches)) if np.any(~np.isnan(pitches)) else np.nan
        pitch_std = float(np.nanstd(pitches)) if np.any(~np.isnan(pitches)) else np.nan
    except Exception:
        pitch_mean, pitch_std = np.nan, np.nan

    feats = {
        'filename': os.path.basename(file_path),
        'duration': float(duration),
        'zcr_mean': float(zcr_mean),
        'energy_mean': float(energy_mean),
        'energy_std': float(energy_std),
        'rms_mean': float(rms_mean),
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'speaking_rate': float(speaking_rate),
        'pause_count': int(pause_count),
        'pause_duration': float(total_silence),
        # subjective proxies
        'confidence': _scale_to_1_5(
            (energy_mean * 1e3) + (0 if np.isnan(pitch_mean) else pitch_mean/200.0) + (0 if np.isnan(pitch_std) else (1.0 / (1.0 + pitch_std))),
            lo=0.0, hi=max(energy_mean*5, 0.5)
        ),
        'articulation': _scale_to_1_5(rms_mean, lo=0.0005, hi=0.02),
        'nervousness': _scale_to_1_5(((0 if np.isnan(pitch_std) else pitch_std) * 0.6 + pause_count*0.2 + energy_std*0.2),
                                     lo=0.0, hi=max(1.0, energy_std+1.0)),
        'perform_confidently': 3,  # fallback (text will refine)
        'satisfaction': 3          # fallback (text will refine)
    }

    return feats

def append_to_csv(file_path, out_csv):
    feats = extract_features(file_path)
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
