#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, r2_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.cluster import KMeans

import joblib


# In[2]:


audio_df = pd.read_excel(r"C:\Users\Admin\Downloads\audios_features.xlsx")

transcription_df = pd.read_excel(r"C:\Users\Admin\Downloads\transcriptions_features.xlsx")

df = pd.merge(audio_df, transcription_df, on="filename", how="inner")

merged_df = df.drop_duplicates(subset=["filename", "clean_text", "transcription"], keep="first")
print(merged_df.head())


# In[3]:


print(merged_df.columns)


# In[4]:


pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.float_format', '{:.6f}'.format)  # Format floats nicely

# Full descriptive statistics
stats = merged_df.describe(include='all')
print(stats)


# In[5]:


merged_df[["energy_mean","energy_std"]].describe()


# In[6]:


import pandas as pd

# -----------------------------
# Helper function to categorize feature using Q1/Q3
# -----------------------------
def categorize_feature(series, q1, q3, reverse=False):
    def map_value(x):
        if not reverse:
            if x <= q1:
                return 1
            elif x <= q3:
                return 2
            else:
                return 3
        else:
            if x <= q1:
                return 3
            elif x <= q3:
                return 2
            else:
                return 1
    return series.apply(map_value)

# -----------------------------
# Mean-std tradeoff functions for key prosodic features
# -----------------------------
def pitch_conf_score(mean, std):
    if 128 <= mean <= 164 and std <= 40:
        return 3
    elif mean < 128 or mean > 164 or std > 50:
        return 1
    else:
        return 2

def energy_conf_score(mean, std):
    if 0.000011 <= mean <= 0.000272 and std <= 0.000885:
        return 3  # High confidence
    elif mean < 0.000011 or mean > 0.000272 or std > 0.000885:
        return 1  # Low confidence
    else:
        return 2  # Medium confidence

def speaking_pause_conf_score(rate, pause_count):
    if 0.084 <= rate <= 0.146 and pause_count <= 8:
        return 3
    elif rate < 0.084 or rate > 0.146 or pause_count > 12:
        return 1
    else:
        return 2

# -----------------------------
# Build rules dataframe
# -----------------------------
rules_df = pd.DataFrame()

# Prosodic features with mean-std tradeoff
rules_df['pitch_conf'] = merged_df.apply(
    lambda x: pitch_conf_score(x['pitch_mean'], x['pitch_std']), axis=1)
rules_df['energy_conf'] = merged_df.apply(
    lambda x: energy_conf_score(x['energy_mean'], x['energy_std']), axis=1)
rules_df['speaking_pause_conf'] = merged_df.apply(
    lambda x: speaking_pause_conf_score(x['speaking_rate'], x['pause_count']), axis=1)

# Other features using percentiles
rules_df['pitch_var_conf'] = categorize_feature(merged_df['pitch_std'], q1=24.96, q3=48.92, reverse=True)
rules_df['energy_var_conf'] = categorize_feature(merged_df['energy_std'], q1=0.000061, q3=0.000885, reverse=True)
rules_df['zcr_conf'] = categorize_feature(merged_df['zcr_mean'], q1=0.106, q3=0.147, reverse=False)
rules_df['pause_dur_conf'] = categorize_feature(merged_df['pause_duration'], q1=0, q3=5.94, reverse=True)
rules_df['duration_conf'] = categorize_feature(merged_df['duration'], q1=16.29, q3=26.70, reverse=False)
rules_df['filler_conf'] = categorize_feature(merged_df['filler_rate'], q1=0, q3=0.119, reverse=True)
rules_df['hedge_conf'] = categorize_feature(merged_df['hedge_count'], q1=0, q3=0, reverse=True)
rules_df['word_count_conf'] = categorize_feature(merged_df['word_count'], q1=9, q3=56.75, reverse=False)
rules_df['sentence_count_conf'] = categorize_feature(merged_df['sentence_count'], q1=2, q3=4, reverse=False)
rules_df['avg_sent_len_conf'] = categorize_feature(merged_df['avg_sentence_length'], q1=3, q3=17, reverse=False)
rules_df['flesch_conf'] = categorize_feature(merged_df['flesch_reading_ease'], q1=40.44, q3=66.02, reverse=False)
rules_df['smog_conf'] = categorize_feature(merged_df['smog_index'], q1=10.79, q3=13.23, reverse=True)
rules_df['articulation_conf'] = categorize_feature(merged_df['articulation'], q1=1, q3=4, reverse=False)
rules_df['polarity_conf'] = categorize_feature(merged_df['polarity'], q1=-0.217, q3=0.225, reverse=False)
rules_df['vader_conf'] = categorize_feature(merged_df['vader_score'], q1=-0.13, q3=0.834, reverse=False)
rules_df['modal_conf'] = categorize_feature(merged_df['modal_pct'], q1=0, q3=0.08, reverse=True)
rules_df['adverb_conf'] = categorize_feature(merged_df['adverb_pct'], q1=0.059, q3=0.15, reverse=True)
rules_df['verb_conf'] = categorize_feature(merged_df['verb_pct'], q1=0.128, q3=0.2, reverse=False)
rules_df['pronoun_conf'] = categorize_feature(merged_df['pronoun_rate'], q1=0.054, q3=0.216, reverse=True)

# -----------------------------
# Aggregate overall confidence score (average of all features)
# -----------------------------
merged_df['rule_confidence_score'] = rules_df.mean(axis=1)

# Map to Low / Medium / High using quantiles
q_low = merged_df['rule_confidence_score'].quantile(0.33)
q_med = merged_df['rule_confidence_score'].quantile(0.66)

def map_overall_quant(score):
    if score <= q_low:
        return 'Low'
    elif score <= q_med:
        return 'Medium'
    else:
        return 'High'

merged_df['rule_confidence_label'] = merged_df['rule_confidence_score'].apply(map_overall_quant)

# -----------------------------
# Preview results
# -----------------------------
print(merged_df[['filename', 'rule_confidence_score', 'rule_confidence_label']].head())


# In[7]:


print(merged_df['rule_confidence_label'].value_counts())


# In[8]:


rules_df.corrwith(merged_df['rule_confidence_score'])


# # Correlation analysis on scaled raw features 

# In[9]:


features = ['duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean',
            'pitch_mean', 'pitch_std', 'speaking_rate', 'pause_count',
            'pause_duration', 'articulation', 'nervousness', 'perform_confidently',
            'satisfaction', 'word_count', 'filler_count', 'filler_rate',
            'hedge_count', 'pronoun_count', 'pronoun_rate', 'sentence_count',
            'avg_sentence_length', 'flesch_reading_ease', 'smog_index', 'polarity',
            'subjectivity', 'vader_score', 'modal_pct', 'adverb_pct',
            'verb_pct']

data_corr = merged_df[features]


# In[10]:


scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_corr), columns=features)


# In[11]:


corr_matrix = data_scaled.corr()
print(corr_matrix)


# In[12]:


plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Features")
plt.show()


# # Rule Base After Correlation Analysis

# In[13]:


import pandas as pd
import numpy as np

# -----------------------------
# Helper function: assign Low(1), Medium(2), High(3)
# -----------------------------
def categorize(series, q1, q3, reverse=False):
    """
    Categorize values into Low(1), Medium(2), High(3).
    Handles cases where q1 == q3 by assigning Medium (2) to all values.
    """
    if q1 == q3:
        # All values same → assign 2 (Medium) to everything
        return pd.Series([2] * len(series), index=series.index)
    
    if reverse:
        bins = [-np.inf, q1, q3, np.inf]
        labels = [3, 2, 1]
    else:
        bins = [-np.inf, q1, q3, np.inf]
        labels = [1, 2, 3]
    
    return pd.cut(series, bins=bins, labels=labels).astype(int)

# -----------------------------
# Define quantiles from stats
# -----------------------------
quantiles = {
    'duration': (16.29, 26.70),
    'zcr_mean': (0.106, 0.147),
    'energy_mean': (0.106, 0.147),  
    'energy_std': (0.000061, 0.000885),
    'rms_mean': (0.002123, 0.012591),
    'pitch_mean': (128.35, 163.94),
    'pitch_std': (24.96, 48.92),
    'speaking_rate': (0.084, 0.146),
    'pause_count': (0, 8),
    'pause_duration': (0, 5.94),
    'articulation': (1, 4),
    'nervousness': (4, 5),
    'perform_confidently': (1, 4),
    'satisfaction': (2, 5),
    'word_count': (9, 56.75),
    'filler_count': (0, 5.75),
    'filler_rate': (0, 0.119),
    'hedge_count': (0, 0),
    'pronoun_count': (1, 3),
    'pronoun_rate': (0.054, 0.216),
    'sentence_count': (2, 4),
    'avg_sentence_length': (3, 17),
    'flesch_reading_ease': (40.44, 66.02),
    'smog_index': (10.79, 13.23),
    'polarity': (-0.217, 0.225),
    'subjectivity': (0.51, 0.64),
    'vader_score': (-0.13, 0.834),
    'modal_pct': (0, 0.08),
    'adverb_pct': (0.059, 0.15),
    'verb_pct': (0.128, 0.2)
}

# -----------------------------
# Features where higher value → lower confidence
# -----------------------------
reverse_feats = [
    'pause_count', 'pause_duration', 'filler_count', 'filler_rate', 
    'hedge_count', 'pronoun_count', 'pronoun_rate', 'smog_index', 
    'modal_pct', 'adverb_pct'
]

# -----------------------------
# Apply rules to create confidence per feature
# -----------------------------
rules_df = pd.DataFrame(index=merged_df.index)

for feat, (q1, q3) in quantiles.items():
    if feat in merged_df.columns:
        reverse = feat in reverse_feats
        rules_df[f'{feat}_conf'] = categorize(merged_df[feat], q1, q3, reverse=reverse)
    else:
        print(f"Warning: Feature '{feat}' not found in merged_df. Skipping.")

# -----------------------------
# Aggregate overall confidence score
# -----------------------------
merged_df['rule_confidence_score'] = rules_df.mean(axis=1)

# -----------------------------
# Map numeric score to Low / Medium / High
# -----------------------------
q_low = merged_df['rule_confidence_score'].quantile(0.33)
q_med = merged_df['rule_confidence_score'].quantile(0.66)

def map_label(score):
    if score <= q_low:
        return 'Low'
    elif score <= q_med:
        return 'Medium'
    else:
        return 'High'

merged_df['rule_confidence_label'] = merged_df['rule_confidence_score'].apply(map_label)

# -----------------------------
# Preview results
# -----------------------------
print(merged_df[['rule_confidence_score', 'rule_confidence_label']].head())


# In[14]:


merged_df['rule_confidence_label'].value_counts()


# In[15]:


# Convert numeric confidence to categorical labels
q_low = merged_df['confidence'].quantile(0.33)
q_med = merged_df['confidence'].quantile(0.66)

def map_conf_label(score):
    if score <= q_low:
        return 'Low'
    elif score <= q_med:
        return 'Medium'
    else:
        return 'High'

merged_df['confidence_label'] = merged_df['confidence'].apply(map_conf_label)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(merged_df['confidence_label'], merged_df['rule_confidence_label']))
print(classification_report(merged_df['confidence_label'], merged_df['rule_confidence_label']))


# In[16]:


merged_df.columns


# # Beginning with Forming Clusters based on the Rule base system

# In[17]:


from sklearn.preprocessing import StandardScaler

# Select features
features = ['duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean', 'pitch_mean', 'pitch_std', 'speaking_rate', 
            'pause_count','pause_duration', 'confidence', 'articulation', 'nervousness','perform_confidently', 
            'satisfaction', 'word_count', 'filler_count', 'filler_rate', 'hedge_count','pronoun_count', 'pronoun_rate', 
            'sentence_count','avg_sentence_length', 'flesch_reading_ease', 'smog_index', 'polarity','subjectivity',
            'vader_score','modal_pct', 'adverb_pct','verb_pct', 'rule_confidence_score']

X = merged_df[features].copy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[18]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

merged_df['cluster'] = clusters


# In[19]:


# Check cluster sizes
print(merged_df['cluster'].value_counts())

# Map clusters to confidence levels based on avg rule_confidence_score per cluster
cluster_means = merged_df.groupby('cluster')['rule_confidence_score'].mean().sort_values()

cluster_label_map = {cluster_means.index[0]: 'Low',
                     cluster_means.index[1]: 'Medium',
                     cluster_means.index[2]: 'High'}

merged_df['cluster_conf_label'] = merged_df['cluster'].map(cluster_label_map)

# Preview
print(merged_df[['cluster','cluster_conf_label','rule_confidence_label','rule_confidence_score']].head())


# In[20]:


# Crosstab
ct = pd.crosstab(merged_df['cluster_conf_label'], merged_df['rule_confidence_label'])
print(ct)


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

features_to_check = ['pitch_mean','energy_mean','speaking_rate','pause_count','confidence']

for f in features_to_check:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='cluster_conf_label', y=f, data=merged_df)
    plt.title(f'Feature distribution for {f}')
    plt.show()


# In[22]:


# ==========================================================================================


# In[23]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = merged_df[features]

# Optional: scale the features
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# In[24]:


print(K)


# In[25]:


print(inertia)


# In[26]:


from sklearn.metrics import silhouette_score

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, silhouette score={score:.3f}")


# In[27]:


# Checking clusters for 5 classes


# In[28]:


features = [
    'duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean', 'pitch_mean', 'pitch_std', 
    'speaking_rate', 'pause_count','pause_duration', 'confidence', 'articulation', 'nervousness',
    'perform_confidently', 'satisfaction', 'word_count', 'filler_count', 'filler_rate', 'hedge_count',
    'pronoun_count', 'pronoun_rate', 'sentence_count','avg_sentence_length', 'flesch_reading_ease', 
    'smog_index', 'polarity','subjectivity','vader_score','modal_pct', 'adverb_pct','verb_pct', 
    'rule_confidence_score'
]
X = merged_df[features]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
merged_df['cluster_5'] = kmeans.fit_predict(X_scaled)
print(merged_df['cluster_5'].value_counts())
cluster_summary = merged_df.groupby('cluster_5')[features + ['rule_confidence_score']].mean()
print(cluster_summary)
cross_tab = pd.crosstab(merged_df['cluster_5'], merged_df['rule_confidence_label'])
print(cross_tab)


# ## Mapping 5 clusters to 3 clusters

# In[29]:


from sklearn.cluster import KMeans

features = ['duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean', 
            'pitch_mean', 'pitch_std', 'speaking_rate', 'pause_count','pause_duration', 
            'confidence', 'articulation', 'nervousness','perform_confidently', 
            'satisfaction', 'word_count', 'filler_count', 'filler_rate', 'hedge_count',
            'pronoun_count', 'pronoun_rate', 'sentence_count','avg_sentence_length', 
            'flesch_reading_ease', 'smog_index', 'polarity','subjectivity',
            'vader_score','modal_pct', 'adverb_pct','verb_pct', 'rule_confidence_score']

kmeans = KMeans(n_clusters=5, random_state=42)
merged_df['cluster_5'] = kmeans.fit_predict(merged_df[features])

# Map each 5-cluster to Low/Medium/High
cluster_to_label = {
    0: 'High',    # Cluster 0 → High confidence
    1: 'Low',     # Cluster 1 → Low confidence
    2: 'Medium',  # Cluster 2 → Medium
    3: 'Medium',  # Cluster 3 → Medium
    4: 'Medium'   # Cluster 4 → Medium
}

merged_df['cluster_conf_label'] = merged_df['cluster_5'].map(cluster_to_label)

print(pd.crosstab(merged_df['cluster_conf_label'], merged_df['rule_confidence_label']))


# # MO + 5→3 cluster mapping

# In[30]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

# -----------------------------
# Parameters
# -----------------------------
n_outputations = 50  # number of MO resamples
n_clusters = 5      # original clusters
features = ['duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean',
            'pitch_mean', 'pitch_std', 'speaking_rate','pause_count','pause_duration',
            'confidence','articulation','nervousness','perform_confidently',
            'satisfaction','word_count','filler_count','filler_rate','hedge_count',
            'pronoun_count','pronoun_rate','sentence_count','avg_sentence_length',
            'flesch_reading_ease','smog_index','polarity','subjectivity','vader_score',
            'modal_pct','adverb_pct','verb_pct','rule_confidence_score']

# -----------------------------
# Step 1: MO resampling & clustering
# -----------------------------
cluster_assignments = []

for i in range(n_outputations):
    # Slightly perturb the features (simulate resampling)
    X_jittered = merged_df[features].copy()
    X_jittered += np.random.normal(0, 0.01, X_jittered.shape)  # small jitter
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=i)
    labels = kmeans.fit_predict(X_jittered)
    cluster_assignments.append(labels)

# -----------------------------
# Step 2: Aggregate cluster assignments
# -----------------------------
# Take majority vote across outputations
cluster_assignments = np.array(cluster_assignments)  # shape: (n_outputations, n_samples)
robust_clusters = []

for col in cluster_assignments.T:  # for each sample
    counts = Counter(col)
    robust_clusters.append(counts.most_common(1)[0][0])

merged_df['cluster_5_mo'] = robust_clusters

# -----------------------------
# Step 3: Silhouette score on robust clusters
# -----------------------------
X = merged_df[features]
score = silhouette_score(X, merged_df['cluster_5_mo'])
print("Silhouette score (5 clusters, MO robust):", score)

# -----------------------------
# Step 4: Map 5 clusters → 3 clusters based on rule_confidence_score
# -----------------------------
cluster_means = merged_df.groupby('cluster_5_mo')['rule_confidence_score'].mean().sort_values()
mapping_5_to_3 = {}

# Map lowest mean → 'Low', middle → 'Medium', highest → 'High'
mapping_5_to_3[cluster_means.index[0]] = 'Low'
mapping_5_to_3[cluster_means.index[1]] = 'Low'
mapping_5_to_3[cluster_means.index[2]] = 'Medium'
mapping_5_to_3[cluster_means.index[3]] = 'High'
mapping_5_to_3[cluster_means.index[4]] = 'High'

merged_df['cluster_conf_label'] = merged_df['cluster_5_mo'].map(mapping_5_to_3)

# -----------------------------
# Step 5: Cross-tab with rule-based labels
# -----------------------------
cross_tab = pd.crosstab(merged_df['cluster_conf_label'], merged_df['rule_confidence_label'])
print(cross_tab)


# In[31]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from collections import Counter

# -----------------------------
# Parameters
# -----------------------------
n_outputations = 50   # number of MO resamples
n_clusters = 5        # original clusters
features = ['duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean',
            'pitch_mean', 'pitch_std', 'speaking_rate','pause_count','pause_duration',
            'confidence','articulation','nervousness','perform_confidently',
            'satisfaction','word_count','filler_count','filler_rate','hedge_count',
            'pronoun_count','pronoun_rate','sentence_count','avg_sentence_length',
            'flesch_reading_ease','smog_index','polarity','subjectivity','vader_score',
            'modal_pct','adverb_pct','verb_pct','rule_confidence_score']

# -----------------------------
# Step 1: Feature weighting (domain knowledge)
# -----------------------------
weights = {
    'pitch_mean': 2.0,
    'energy_mean': 2.0,
    'pause_count': 2.0,
    'pause_duration': 2.0,
    'filler_rate': 2.0,
    'speaking_rate': 1.5
}

X_weighted = merged_df[features].copy()
for feat, w in weights.items():
    if feat in X_weighted.columns:
        X_weighted[feat] = X_weighted[feat] * w

# -----------------------------
# Step 2: MO resampling & clustering
# -----------------------------
cluster_assignments = []

for i in range(n_outputations):
    # Add jitter (simulate resampling noise)
    X_jittered = X_weighted.copy()
    X_jittered += np.random.normal(0, 0.01, X_jittered.shape)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
    labels = kmeans.fit_predict(X_jittered)
    cluster_assignments.append(labels)

# -----------------------------
# Step 3: Aggregate cluster assignments (majority vote + stability)
# -----------------------------
cluster_assignments = np.array(cluster_assignments)  # shape: (n_outputations, n_samples)
robust_clusters = []
stability_scores = []

for col in cluster_assignments.T:  # iterate over samples
    counts = Counter(col)
    cluster, count = counts.most_common(1)[0]
    robust_clusters.append(cluster)
    stability_scores.append(count / n_outputations)

merged_df['cluster_5_mo'] = robust_clusters
merged_df['cluster_stability'] = stability_scores

# -----------------------------
# Step 4: Evaluate clustering
# -----------------------------
X_eval = X_weighted  # use weighted features

sil_score = silhouette_score(X_eval, merged_df['cluster_5_mo'])
db_score = davies_bouldin_score(X_eval, merged_df['cluster_5_mo'])
ch_score = calinski_harabasz_score(X_eval, merged_df['cluster_5_mo'])

print("Silhouette score (5 clusters, MO robust):", sil_score)
print("Davies-Bouldin Index (lower=better):", db_score)
print("Calinski-Harabasz Score (higher=better):", ch_score)

# -----------------------------
# Step 5: Map 5 clusters → 3 labels
# -----------------------------
# Use both rule_confidence_score mean and majority rule_confidence_label
cluster_summary = merged_df.groupby('cluster_5_mo').agg({
    'rule_confidence_score': 'mean',
    'rule_confidence_label': lambda x: x.mode()[0]
}).sort_values('rule_confidence_score')

# Create mapping
mapping_5_to_3 = {}
ordered_clusters = cluster_summary.index.tolist()

# Lowest 2 clusters → Low, middle → Medium, highest 2 → High
mapping_5_to_3[ordered_clusters[0]] = 'Low'
mapping_5_to_3[ordered_clusters[1]] = 'Low'
mapping_5_to_3[ordered_clusters[2]] = 'Medium'
mapping_5_to_3[ordered_clusters[3]] = 'High'
mapping_5_to_3[ordered_clusters[4]] = 'High'

# Apply mapping
merged_df['cluster_conf_label'] = merged_df['cluster_5_mo'].map(mapping_5_to_3)

# -----------------------------
# Step 6: Cross-tab with rule-based labels
# -----------------------------
cross_tab = pd.crosstab(merged_df['cluster_conf_label'], merged_df['rule_confidence_label'])
print("\nCross-tab of Cluster vs Rule-base:\n", cross_tab)

# -----------------------------
# Step 7: Check stability
# -----------------------------
print("\nAverage cluster stability:", merged_df['cluster_stability'].mean())
print("Samples with low stability (<0.6):", (merged_df['cluster_stability'] < 0.6).sum())


# In[32]:


# Full clustering + consensus + evaluation pipeline
# Requirements: scikit-learn, pandas, numpy, scipy
# Run in a notebook where `merged_df` is loaded and contains the features + 'rule_confidence_label'.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
features = ['duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean',
            'pitch_mean', 'pitch_std', 'speaking_rate','pause_count','pause_duration',
            'confidence','articulation','nervousness','perform_confidently',
            'satisfaction','word_count','filler_count','filler_rate','hedge_count',
            'pronoun_count','pronoun_rate','sentence_count','avg_sentence_length',
            'flesch_reading_eease','flesch_reading_ease','smog_index','polarity','subjectivity','vader_score',
            'modal_pct','adverb_pct','verb_pct','rule_confidence_score']

# Note: you may have 'flesch_reading_eease' or 'flesch_reading_ease' etc. Clean duplicates:
features = [f for f in features if f in merged_df.columns]
print("Using features (count={}):".format(len(features)))
print(features)

# target label for evaluation (rule-based labels)
if 'rule_confidence_label' not in merged_df.columns:
    raise ValueError("merged_df must have 'rule_confidence_label' column for evaluation")

y_rule = merged_df['rule_confidence_label'].values

# -----------------------------
# Preprocessing: scale and optional PCA
# -----------------------------
X = merged_df[features].copy()
# Fill any NaN (if present) with median of column (safe default)
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: keep 95% variance (optional). If dims small, PCA will keep most dims.
pca = PCA(n_components=0.95, random_state=0)
X_pca = pca.fit_transform(X_scaled)
print("Original dim:", X_scaled.shape[1], "-> PCA dim:", X_pca.shape[1])

# We will use X_pca for clustering
X_use = X_pca

# -----------------------------
# Helper: evaluate clustering labels
# -----------------------------
def evaluate_labels(X, labels, y_true=None):
    # labels must be integer cluster IDs
    out = {}
    n_clusters = len(np.unique(labels))
    out['n_clusters'] = n_clusters
    out['silhouette'] = silhouette_score(X, labels) if n_clusters > 1 else np.nan
    out['calinski_harabasz'] = calinski_harabasz_score(X, labels) if n_clusters > 1 else np.nan
    out['davies_bouldin'] = davies_bouldin_score(X, labels) if n_clusters > 1 else np.nan
    if y_true is not None:
        # map string y_true -> ints
        y_true_enc = pd.factorize(y_true)[0]
        out['ARI'] = adjusted_rand_score(y_true_enc, labels)
        out['NMI'] = normalized_mutual_info_score(y_true_enc, labels)
    return out

# -----------------------------
# 1) Try baseline clustering (k=5)
# -----------------------------
k = 5
results = {}

# KMeans baseline
km = KMeans(n_clusters=k, random_state=0, n_init=20)
labels_km = km.fit_predict(X_use)
results['KMeans'] = evaluate_labels(X_use, labels_km, y_rule)

# Gaussian Mixture
gmm = GaussianMixture(n_components=k, random_state=0, n_init=3)
labels_gmm = gmm.fit_predict(X_use)
results['GMM'] = evaluate_labels(X_use, labels_gmm, y_rule)

# Agglomerative (ward)
agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
labels_agg = agg.fit_predict(X_use)
results['Agglomerative'] = evaluate_labels(X_use, labels_agg, y_rule)

print("\nBaseline clustering results (k=5):")
for name, metrics in results.items():
    print(name, metrics)

# -----------------------------
# 2) MO / Consensus clustering (using base KMeans runs) 
#    Build co-association matrix then cluster it.
# -----------------------------
def consensus_cluster(X, base_algo='kmeans', n_clusters=5, n_iters=50, jitter_scale=0.02, random_state=0):
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    coassoc = np.zeros((n_samples, n_samples), dtype=float)
    labels_list = []
    for i in range(n_iters):
        # jitter to simulate resampling / robust runs
        Xj = X + rng.normal(scale=jitter_scale, size=X.shape)
        if base_algo == 'kmeans':
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state + i)
            labels = model.fit_predict(Xj)
        elif base_algo == 'gmm':
            model = GaussianMixture(n_components=n_clusters, n_init=1, random_state=random_state + i)
            labels = model.fit_predict(Xj)
        else:
            # Agglomerative on jittered will be deterministic given linkage, but use random_state for jitter
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state + i)
            labels = model.fit_predict(Xj)
        labels_list.append(labels)
        # Update co-association matrix
        for cl in np.unique(labels):
            idx = np.where(labels == cl)[0]
            if len(idx) > 1:
                coassoc[np.ix_(idx, idx)] += 1
    # normalize
    coassoc = coassoc / n_iters
    # Convert to distance matrix for hierarchical clustering
    dist = 1 - coassoc
    # Hierarchical clustering on the distance matrix
    # We need condensed distance matrix for linkage; use upper triangle flattening
    from scipy.spatial.distance import squareform
    # Ensure symmetric and zero-diagonal
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1  # make 0-based
    return cluster_labels, coassoc, labels_list, Z

print("\nRunning consensus clustering (KMeans base)...")
cons_labels, coassoc, labels_runs, Z = consensus_cluster(X_use, base_algo='kmeans', n_clusters=k, n_iters=60, jitter_scale=0.01, random_state=42)
cons_metrics = evaluate_labels(X_use, cons_labels, y_rule)
print("Consensus (KMeans base) metrics:", cons_metrics)

# Stability: average pairwise agreement of clusterings across runs per sample (fraction times same as majority)
def compute_sample_stability(labels_runs):
    # labels_runs: list of arrays (n_iters x n_samples)
    arr = np.vstack(labels_runs)  # shape (n_iters, n_samples)
    n_iters = arr.shape[0]
    n_samples = arr.shape[1]
    stability = np.zeros(n_samples)
    for j in range(n_samples):
        col = arr[:, j]
        most = Counter(col).most_common(1)[0][1]
        stability[j] = most / n_iters
    return stability

stability = compute_sample_stability(labels_runs)
print("Average cluster stability:", float(np.mean(stability)))
low_stab_count = np.sum(stability < 0.6)
print("Samples with low stability (<0.6):", int(low_stab_count))

# -----------------------------
# 3) Map 5 clusters -> 3 clusters (two approaches)
# -----------------------------
merged = merged_df.copy()
merged['cons_cluster_5'] = cons_labels
merged['km_cluster_5'] = labels_km
merged['gmm_cluster_5'] = labels_gmm
merged['agg_cluster_5'] = labels_agg

# Approach A: rule_confidence_score mean mapping (simple / interpretable)
cluster_order = merged.groupby('cons_cluster_5')['rule_confidence_score'].mean().sort_values()
print("\nCluster means (rule_confidence_score) for consensus clusters:\n", cluster_order)
ordered = list(cluster_order.index)

# map lowest two -> Low, middle -> Medium, top two -> High
map_5_to_3_A = {}
map_5_to_3_A[ordered[0]] = 'Low'
map_5_to_3_A[ordered[1]] = 'Low' if len(ordered) >= 5 else 'Medium'
map_5_to_3_A[ordered[2]] = 'Medium'
map_5_to_3_A[ordered[3]] = 'High' if len(ordered) >= 4 else 'Medium'
if len(ordered) >= 5:
    map_5_to_3_A[ordered[4]] = 'High'

merged['cluster3_map_rule'] = merged['cons_cluster_5'].map(map_5_to_3_A)

# Approach B: hierarchical clustering of the 5 cluster centroids in feature-space (automated)
centroids = merged.groupby('cons_cluster_5')[features].mean().values
# cluster centroids into 3 groups
link = linkage(centroids, method='ward')
centroid_labels = fcluster(link, t=3, criterion='maxclust') - 1  # 0,1,2
# Map centroid group -> 'Low','Medium','High' by ordering their mean rule_confidence_score
centroid_df = pd.DataFrame({
    'centroid_idx': np.arange(len(centroid_labels)),
    'group': centroid_labels,
})
# mapping groups to labels by average rule score
centroid_to_group = {}
centroid_rule_mean = []
for i in range(len(centroid_labels)):
    centroid_rule_mean.append(merged[merged['cons_cluster_5'] == i]['rule_confidence_score'].mean())
centroid_rule_mean = np.array(centroid_rule_mean)
# map group id -> label by group mean
group_means = {}
for g in np.unique(centroid_labels):
    idxs = np.where(centroid_labels == g)[0]
    group_means[g] = centroid_rule_mean[idxs].mean()

# sort groups by mean and assign Low/Medium/High
sorted_groups = sorted(group_means.items(), key=lambda x: x[1])
group_label_map = {}
labels_3 = ['Low', 'Medium', 'High']
for i, (g, _) in enumerate(sorted_groups):
    group_label_map[g] = labels_3[i]

# Now map each sample by its centroid's group label
centroid_label_map_per_cluster = {i: group_label_map[centroid_labels[i]] for i in range(len(centroid_labels))}
merged['cluster3_map_centroid'] = merged['cons_cluster_5'].map(centroid_label_map_per_cluster)

# -----------------------------
# 4) Evaluation & cross-tabs
# -----------------------------
def print_cross_tab(colname):
    print("\nCross-tab:", colname)
    print(pd.crosstab(merged[colname], merged['rule_confidence_label']))

# Evaluate mapping approaches
for col in ['cluster3_map_rule', 'cluster3_map_centroid']:
    # compute ARI / NMI by encoding the mapped labels same as rule labels
    mapped = merged[col].values
    # encode both to ints
    enc_true = pd.factorize(merged['rule_confidence_label'])[0]
    enc_pred = pd.factorize(mapped)[0]
    ari = adjusted_rand_score(enc_true, enc_pred)
    nmi = normalized_mutual_info_score(enc_true, enc_pred)
    print(f"\nMapping method: {col}  -- ARI: {ari:.4f}, NMI: {nmi:.4f}")
    print_cross_tab(col)

# Print consensus cluster evaluation
print("\nConsensus cluster metrics:", cons_metrics)

# Print baseline cluster -> rule crosstabs for comparison
print_cross_tab('km_cluster_5')
print_cross_tab('gmm_cluster_5')
print_cross_tab('agg_cluster_5')

# -----------------------------
# 5) Save some outputs to merged_df for downstream modelling
# -----------------------------
merged_df['cons_cluster_5'] = merged['cons_cluster_5'].values
merged_df['cluster3_map_rule'] = merged['cluster3_map_rule'].values
merged_df['cluster3_map_centroid'] = merged['cluster3_map_centroid'].values

print("\nDone. Summary:")
print(" - Stored 'cons_cluster_5', 'cluster3_map_rule', 'cluster3_map_centroid' in merged_df")


# In[33]:


from sklearn.model_selection import train_test_split

# Features
X = merged_df[features]  # your 32 features
# Target: cluster-mapped label
y = merged_df['cluster3_map_rule']  # or 'cluster3_map_centroid'

# 1️⃣ Split into train+val (80%) and test (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2️⃣ Split train+val into train (75%) and val (25% of train+val → 60% train, 20% val)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
)

print("Train size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])
print("Test size:", X_test.shape[0])

# Optional: check distribution of classes
print("\nClass distribution in train:")
print(y_train.value_counts())
print("\nClass distribution in val:")
print(y_val.value_counts())
print("\nClass distribution in test:")
print(y_test.value_counts())


# # Random Forest Model

# In[35]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1️⃣ Select features and target
# -----------------------------
features = ['duration', 'zcr_mean', 'energy_mean', 'energy_std', 'rms_mean',
            'pitch_mean', 'pitch_std', 'speaking_rate','pause_count','pause_duration',
            'confidence','articulation','nervousness','perform_confidently',
            'satisfaction','word_count','filler_count','filler_rate','hedge_count',
            'pronoun_count','pronoun_rate','sentence_count','avg_sentence_length',
            'flesch_reading_ease','smog_index','polarity','subjectivity','vader_score',
            'modal_pct','adverb_pct','verb_pct','rule_confidence_score']

X = merged_df[features].fillna(0)
y = merged_df['cluster3_map_rule']  # or 'cluster3_map_centroid'

# -----------------------------
# 2️⃣ Train/Validation/Test split
# -----------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)  # 0.25*0.8 = 0.2 -> final 60/20/20 split

print("Train size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])
print("Test size:", X_test.shape[0])

print("\nClass distribution in train:\n", y_train.value_counts())
print("\nClass distribution in val:\n", y_val.value_counts())
print("\nClass distribution in test:\n", y_test.value_counts())

# -----------------------------
# 3️⃣ Handle class imbalance using SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:\n", pd.Series(y_train_res).value_counts())

# -----------------------------
# 4️⃣ Scale features
# -----------------------------
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5️⃣ Train Random Forest + Hyperparameter tuning
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')

grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train_res_scaled, y_train_res)

best_rf = grid.best_estimator_
print("\nBest RF params:", grid.best_params_)

# -----------------------------
# Evaluate on train set
# -----------------------------
y_train_pred = best_rf.predict(X_train_res_scaled)
print("\nTrain Classification Report:")
print(classification_report(y_train_res, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train_res, y_train_pred))

# -----------------------------
# Evaluate on validation set
# -----------------------------
y_val_pred = best_rf.predict(X_val_scaled)
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# -----------------------------
# 7️⃣ Evaluate on test set
# -----------------------------
y_test_pred = best_rf.predict(X_test_scaled)
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# -----------------------------
# 8️⃣ Feature importance
# -----------------------------
import matplotlib.pyplot as plt

importances = best_rf.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print(feat_imp)
# plt.figure(figsize=(10,6))
# feat_imp.plot(kind='bar')
# plt.title("Random Forest Feature Importance")
# plt.ylabel("Importance")
# plt.show()


# # XGBoost

# In[45]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# -----------------------------
# 1️⃣ Encode labels
# -----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)  # train labels
y_val_enc   = le.transform(y_val)        # val labels
y_test_enc  = le.transform(y_test)       # test labels

# -----------------------------
# 2️⃣ Apply SMOTE on training set only
# -----------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal_enc = smote.fit_resample(X_train, y_train_enc)

# -----------------------------
# 3️⃣ Scale features
# -----------------------------
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_val_scaled       = scaler.transform(X_val)
X_test_scaled      = scaler.transform(X_test)

# -----------------------------
# 4️⃣ Define XGBoost classifier
# -----------------------------
xgb_clf = XGBClassifier(
    objective='multi:softprob', 
    eval_metric='mlogloss', 
    use_label_encoder=False, 
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 5️⃣ Hyperparameter grid
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

# -----------------------------
# 6️⃣ Stratified K-Fold CV (3 folds) for grid search
# -----------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='f1_macro',   # better for imbalanced multi-class
    cv=cv,
    verbose=1
)

# -----------------------------
# 7️⃣ Fit grid search on scaled training data
# -----------------------------
grid.fit(X_train_bal_scaled, y_train_bal_enc)

# -----------------------------
# 8️⃣ Best estimator & predictions
# -----------------------------
best_xgb = grid.best_estimator_
print("Best XGBoost params:", grid.best_params_)

y_val_pred_enc = best_xgb.predict(X_val_scaled)
y_val_pred = le.inverse_transform(y_val_pred_enc)

y_test_pred_enc = best_xgb.predict(X_test_scaled)
y_test_pred = le.inverse_transform(y_test_pred_enc)

# -----------------------------
# 9️⃣ Evaluation
# -----------------------------
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

print("Train size (after SMOTE):", X_train_bal.shape[0])
print("Validation size:", X_val.shape[0])
print("Test size:", X_test.shape[0])

print("\nClass distribution in train (after SMOTE):")
print(Counter(y_train_bal_enc))  # encoded labels

print("\nClass distribution in validation:")
print(Counter(y_val_enc))

print("\nClass distribution in test:")
print(Counter(y_test_enc))

# -----------------------------
# Optional: decode train labels for readability
# -----------------------------
y_train_bal_labels = le.inverse_transform(y_train_bal_enc)
print("\nDecoded class counts in train (after SMOTE):")
print(Counter(y_train_bal_labels))


# In[37]:


from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1️⃣ Encode labels
# -----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)  # train labels
y_val_enc   = le.transform(y_val)        # val labels
y_test_enc  = le.transform(y_test)       # test labels

# -----------------------------
# 2️⃣ Apply SMOTE on training set only
# -----------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal_enc = smote.fit_resample(X_train, y_train_enc)

# -----------------------------
# 3️⃣ Define XGBoost classifier
# -----------------------------
xgb_clf = XGBClassifier(
    objective='multi:softprob', 
    eval_metric='mlogloss', 
    use_label_encoder=False, 
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 4️⃣ Hyperparameter grid
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

# -----------------------------
# 5️⃣ Stratified K-Fold CV (3 folds) for grid search
# -----------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='f1_macro',   # better for imbalanced multi-class
    cv=cv,
    verbose=1
)

# -----------------------------
# 6️⃣ Fit grid search
# -----------------------------
grid.fit(X_train_bal, y_train_bal_enc)

# -----------------------------
# 7️⃣ Best estimator & predictions
# -----------------------------
best_xgb = grid.best_estimator_
print("Best XGBoost params:", grid.best_params_)

y_val_pred_enc = best_xgb.predict(X_val)
y_val_pred = le.inverse_transform(y_val_pred_enc)

y_test_pred_enc = best_xgb.predict(X_test)
y_test_pred = le.inverse_transform(y_test_pred_enc)

# -----------------------------
# 8️⃣ Evaluation
# -----------------------------
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

print("Train size (after SMOTE):", X_train_bal.shape[0])
print("Validation size:", X_val.shape[0])
print("Test size:", X_test.shape[0])

print("\nClass distribution in train (after SMOTE):")
print(Counter(y_train_bal_enc))  # encoded labels

print("\nClass distribution in validation:")
print(Counter(y_val_enc))

print("\nClass distribution in test:")
print(Counter(y_test_enc))

# -----------------------------
# Optional: decode train labels for readability
# -----------------------------
y_train_bal_labels = le.inverse_transform(y_train_bal_enc)
print("\nDecoded class counts in train (after SMOTE):")
print(Counter(y_train_bal_labels))


# # Logistic Regression

# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
from collections import Counter

# -------------------------------
# 1️⃣ Encode labels
# -------------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)  # 0/1/2
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

# -------------------------------
# 2️⃣ Apply SMOTE on training set
# -------------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_enc)

print("Train size (after SMOTE):", X_train_bal.shape[0])
print("Class distribution in train (after SMOTE):", Counter(y_train_bal))
print("Class distribution in validation:", Counter(y_val_enc))
print("Class distribution in test:", Counter(y_test_enc))

# -------------------------------
# 3️⃣ Logistic Regression + Grid Search
# -------------------------------
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],          # regularization strength
    'penalty': ['l2'],                      # l1 can be used with solver='liblinear'
    'solver': ['lbfgs', 'saga'],            # solvers supporting multinomial
    'max_iter': [200, 500]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

log_reg = LogisticRegression(multi_class='multinomial', random_state=42)

grid = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

# -------------------------------
# 4️⃣ Fit model
# -------------------------------
grid.fit(X_train_bal, y_train_bal)
best_log = grid.best_estimator_
print("\nBest Logistic Regression params:", grid.best_params_)

# -------------------------------
# 5️⃣ Predictions
# -------------------------------
y_train_pred = best_log.predict(X_train_bal)
y_val_pred   = best_log.predict(X_val)
y_test_pred  = best_log.predict(X_test)

# Decode back to original string labels
y_train_pred_labels = le.inverse_transform(y_train_pred)
y_val_pred_labels   = le.inverse_transform(y_val_pred)
y_test_pred_labels  = le.inverse_transform(y_test_pred)

# -------------------------------
# 6️⃣ Metrics
# -------------------------------
print("\nTrain Accuracy:", accuracy_score(y_train_bal, y_train_pred))
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred_labels))
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred_labels))

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred_labels))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_labels))


# # Stacking LR, XGB, RF

# In[39]:


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import numpy as np
from collections import Counter

# ==============================
# 1️⃣ Encode labels
# ==============================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)  # original train before SMOTE
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

# ==============================
# 2️⃣ Balance train with SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_enc)

print("Train size (after SMOTE):", len(y_train_bal))
print("Class distribution in train (after SMOTE):", Counter(y_train_bal))
print("Class distribution in validation:", Counter(y_val_enc))
print("Class distribution in test:", Counter(y_test_enc))
print("Decoded class counts in train (after SMOTE):", Counter(le.inverse_transform(y_train_bal)))

# ==============================
# 3️⃣ Define base learners
# ==============================
log_clf = LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=200, random_state=42)
rf_clf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y_train_bal)),
    learning_rate=0.05,
    max_depth=5,
    n_estimators=100,
    subsample=1.0,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)

# ==============================
# 4️⃣ Define StackingClassifier
# ==============================
estimators = [
    ('log_reg', log_clf),
    ('rf', rf_clf),
    ('xgb', xgb_clf)
]

stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500, random_state=42),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    stack_method="predict_proba"
)

# ==============================
# 5️⃣ Fit stacking model
# ==============================
stack_clf.fit(X_train_bal, y_train_bal)

# ==============================
# 6️⃣ Predictions
# ==============================
y_train_pred = stack_clf.predict(X_train_bal)
y_val_pred   = stack_clf.predict(X_val)
y_test_pred  = stack_clf.predict(X_test)

# Decode labels
y_train_pred_labels = le.inverse_transform(y_train_pred)
y_val_pred_labels   = le.inverse_transform(y_val_pred)
y_test_pred_labels  = le.inverse_transform(y_test_pred)

# ==============================
# 7️⃣ Reports
# ==============================
print("\nTrain Classification Report:")
print(classification_report(le.inverse_transform(y_train_bal), y_train_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(le.inverse_transform(y_train_bal), y_train_pred_labels))

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred_labels))

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_labels))


# # Soft Voting Ensemble

# In[40]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# ✅ 1. Base models (best params from before)
log_clf = LogisticRegression(C=10, max_iter=200, penalty='l2', solver='lbfgs', random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    colsample_bytree=0.7,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=100,
    subsample=1.0,
    random_state=42,
    n_jobs=-1
)

# ✅ 2. Calibrate probabilities for better confidence estimation
log_cal = CalibratedClassifierCV(log_clf, cv=3, method="isotonic")
rf_cal  = CalibratedClassifierCV(rf_clf, cv=3, method="isotonic")
xgb_cal = CalibratedClassifierCV(xgb_clf, cv=3, method="isotonic")

# ✅ 3. Soft Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[('lr', log_cal), ('rf', rf_cal), ('xgb', xgb_cal)],
    voting='soft',   # probability averaging
    n_jobs=-1
)

# ✅ 4. Train on SMOTE-balanced data
voting_clf.fit(X_train_bal, y_train_bal)

# ✅ 5. Predictions
y_train_pred = voting_clf.predict(X_train_bal)
y_val_pred   = voting_clf.predict(X_val)
y_test_pred  = voting_clf.predict(X_test)

# ✅ 6. Reports
print("\nTrain Classification Report:")
print(classification_report(y_train_bal, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train_bal, y_train_pred))

print("\nValidation Classification Report:")
print(classification_report(y_val_enc, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val_enc, y_val_pred))

print("\nTest Classification Report:")
print(classification_report(y_test_enc, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_enc, y_test_pred))

# ✅ 7. Error Analysis for 'Medium'
import numpy as np
medium_idx = np.where(y_test_enc == 2)[0]   # true Medium samples
misclassified_idx = medium_idx[y_test_pred[medium_idx] != 2]

print("\n🔎 Error Analysis for 'Medium':")
print(f"Total Medium samples in test: {len(medium_idx)}")
print(f"Misclassified as: {y_test_pred[misclassified_idx]}")
print(f"Indices: {misclassified_idx}")


# # Saving the Model

# In[42]:


from joblib import dump, load

# Save model and scaler
dump(voting_clf, "voting_confidence_model.joblib")
dump(scaler, "scaler.joblib")


# In[ ]:




