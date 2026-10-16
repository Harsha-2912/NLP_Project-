"""
Script to generate the comprehensive, end-to-end Jupyter Notebook:
`notebooks/toxic_comment_classification.ipynb`
Covers all 10 project requirements, theoretical discussions, equations, code, and visualizations.
"""

import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Title & Metadata
add_md("""# 🛡️ Toxic Comment Detection & Content Moderation System
### Natural Language Processing Project | Assigned Teams: 9 & 11
**Dataset:** Jigsaw Toxic Comment Classification Challenge  
**Target Labels:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

---

## 📌 Problem Statement & Objectives
Online discussion platforms (Wikipedia, social media, community forums, gaming chats) receive immense volumes of user-generated content. A small fraction of these interactions contain personal attacks, obscenities, hate speech, and physical threats. 
Automated classification helps moderation teams prioritize toxic content for human review, shielding users and moderators from abuse while preserving platform safety and freedom of expression.

### Core Objectives:
1. **Content Moderation NLP:** Understand multi-channel text classification workflows for online moderation.
2. **Extreme Class Imbalance:** Quantify and mitigate severe skew (up to 300:1 ratio for rare harm classes like `threat`).
3. **Binary vs. Multilabel Classification:** Understand the distinction between coarse binary triage and fine-grained multilabel severity profiling.
4. **Moderation Trade-offs:** Analyze precision vs. recall trade-offs and evaluate false positive / false negative ethical risks.
5. **Algorithmic Fairness:** Unpack dataset bias, identity term disparities, colloquial cursing, and sarcasm.""")

# Requirement 1 & 2: Data Loading & Imbalance Inspection
add_md("""---
## 1. Load Dataset & Inspect Labels and Class Imbalance
We begin by loading the Jigsaw Toxic Comment Classification dataset (`data/train.csv`).
We inspect:
- Dataset dimensions and column types
- Missing values
- Distribution of individual harm categories
- Total clean vs. toxic ratio
- Label cardinality and co-occurrence patterns""")

add_code("""import os
import re
import html
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization aesthetics
sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load dataset
data_path = os.path.join('..', 'data', 'train.csv')
if not os.path.exists(data_path):
    data_path = os.path.join('data', 'train.csv')

df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.head(5)""")

add_md("""### 1.1 Class Frequencies and Imbalance Calculation
In content moderation datasets, toxic comments represent a small minority of user behavior.
We compute the positive frequencies and the **Imbalance Ratio** ($IR = N_{clean} / N_{class}$) for each label.""")

add_code("""# Create overall toxicity flag
df['is_toxic'] = (df[LABELS].sum(axis=1) > 0).astype(int)

n_total = len(df)
n_clean = int((df['is_toxic'] == 0).sum())
n_toxic = int((df['is_toxic'] == 1).sum())

print("=== OVERALL DATASET DISTRIBUTION ===")
print(f"Total Comments : {n_total:,}")
print(f"Clean Comments : {n_clean:,} ({n_clean/n_total*100:.2f}%)")
print(f"Toxic Comments : {n_toxic:,} ({n_toxic/n_total*100:.2f}%)")

imbalance_rows = []
for lbl in LABELS:
    cnt = int(df[lbl].sum())
    pct = (cnt / n_total) * 100
    ratio = n_clean / cnt if cnt > 0 else 0
    imbalance_rows.append({
        'Label': lbl,
        'Positive Samples': cnt,
        'Percentage (%)': round(pct, 2),
        'Imbalance Ratio (Clean : Class)': f"{ratio:.1f} : 1"
    })

imbalance_df = pd.DataFrame(imbalance_rows)
display(imbalance_df)""")

add_md("""### 1.2 Visualization of Class Imbalance
Notice the extreme long-tail distribution: `threat` appears in only 0.30% of comments (a 300:1 ratio), while general `toxic` appears in 9.58%.""")

add_code("""fig, ax = plt.subplots(figsize=(10, 4.5))
sns.barplot(data=imbalance_df, x='Label', y='Percentage (%)', palette='magma', ax=ax)
ax.set_title('Frequency of Toxic Labels in Jigsaw Dataset (%)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Percentage of Total Comments (%)')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.2),
                ha='center', va='baseline', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()""")

add_md("""### 1.3 Multilabel Properties: Cardinality, Density, and Co-occurrence
- **Label Cardinality ($LC$):** The average number of positive labels per comment:
  $$LC = \\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=1}^{L} y_{ij}$$
- **Label Density ($LD$):** Label cardinality normalized by total number of labels $L$:
  $$LD = \\frac{LC}{L}$$
- **Label Co-occurrence:** How frequently multiple labels trigger simultaneously (e.g. `obscene` and `insult` frequently overlap).""")

add_code("""label_counts = df[LABELS].sum(axis=1)
lc = label_counts.mean()
ld = lc / len(LABELS)

print(f"Label Cardinality: {lc:.4f} labels/comment")
print(f"Label Density    : {ld:.4f}")

# Distribution of number of labels per comment
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(x=label_counts, palette='viridis', ax=ax1)
ax1.set_title('Number of Toxic Labels Per Comment', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Active Labels (0 = Clean)')
ax1.set_ylabel('Comment Count (Log Scale)')
ax1.set_yscale('log')

# Correlation heatmap
corr = df[LABELS].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, ax=ax2)
ax2.set_title('Label Correlation (Co-occurrence Matrix)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()""")

# Requirement 3: Preprocessing
add_md("""---
## 2. Text Preprocessing & Linguistic Normalization
Online user comments contain heavy internet slang, intentional obfuscations, Wikipedia edit stamps, HTML entities, and character elongations.

### Key Moderation Preprocessing Strategies:
1. **HTML Entity Decoding:** Replaces `&quot;`, `&amp;`, `&lt;` with clean characters.
2. **Wikipedia & Web Metadata Removal:** Strips IP addresses, talk-page signatures, URLs, and section headers.
3. **Masked Profanity Decoding:** Users frequently censor words (`f**k`, `b*tch`, `a$$hole`) to bypass naïve keyword blocklists. We normalize these back to standard forms.
4. **Contraction Expansion:** Expands `can't` $\\rightarrow$ `cannot`, `you're` $\\rightarrow$ `you are`, `u` $\\rightarrow$ `you`.
5. **Character Elongation Reduction:** Users stretch words (`fuuuuck` $\\rightarrow$ `fuuck`, `sooooo` $\\rightarrow$ `soo`) to convey emotion. Reducing sequences of 3+ identical characters preserves root meaning without vocabulary explosion.
6. **Contextual Stopword Preservation:** Standard stopword lists discard pronouns (`you`, `your`, `he`, `she`, `they`) and negations (`not`, `never`). In toxic speech, **directed personal attack** is indicated by second-person pronouns (*"You are..."* vs *"This is..."*). Therefore, we preserve pronouns and negations.""")

add_code("""from src.preprocessing import clean_text

sample_raw_comments = [
    "YOU ARE AN IDIOT!!! GO F**K YOURSELF!! http://spam.com/123",
    "Hello 192.168.1.5, I can't believe u reverted my edit. == vandalism ==",
    "I loooove this movie sooooo much! It is f***ing fantastic!",
    "&quot;She said: you are a worthless b*tch&quot;",
    "I'm a proud gay Muslim student, why do people hate on us?"
]

print("Demonstration of Preprocessing Pipeline:\\n")
for raw in sample_raw_comments:
    cleaned = clean_text(raw)
    print(f"RAW    : {raw}")
    print(f"CLEANED: {cleaned}\\n" + "-"*65)""")

# Requirement 4: TF-IDF
add_md("""---
## 3. Feature Engineering: Dual-Channel TF-IDF Representation
We transform the preprocessed text into numerical vectors using a **dual-channel TF-IDF** strategy:
1. **Word-level N-Grams (1 to 2 words):**
   - Captures meaningful phrases: *"shut up"*, *"hate you"*, *"go away"*.
   - Uses `sublinear_tf=True` ($1 + \\log(tf)$) to prevent spammed words from dominating the feature weights.
2. **Character-level N-Grams (2 to 5 characters):**
   - Captures obfuscated words, misspellings, and morphological variants (*"fuk"*, *"shyt"*, *"nooob"*).
   - Character n-grams are immune to evasion tactics like spacing or minor substitutions.
3. **Feature Union:**
   - Concatenates word TF-IDF (25,000 features) and character TF-IDF (30,000 features) into a high-dimensional sparse matrix.""")

add_code("""from src.features import build_feature_pipeline
from sklearn.model_selection import train_test_split

# Subsample or use full dataset
SAMPLE_FOR_NOTEBOOK = 35000  # Adjust as needed for execution speed
if len(df) > SAMPLE_FOR_NOTEBOOK:
    print(f"Using a stratified sample of {SAMPLE_FOR_NOTEBOOK:,} rows for rapid notebook demonstration...")
    sub_df = df.sample(n=SAMPLE_FOR_NOTEBOOK, random_state=42, weights=df['is_toxic']*4 + 1).reset_index(drop=True)
else:
    sub_df = df.copy()

print("Preprocessing text comments...")
sub_df['cleaned_text'] = sub_df['comment_text'].apply(lambda x: clean_text(x))

train_df, test_df = train_test_split(
    sub_df, test_size=0.20, random_state=42, stratify=sub_df['is_toxic']
)

print(f"Train split: {len(train_df):,} | Test split: {len(test_df):,}")

# Fit FeatureUnion
print("Extracting TF-IDF Word + Character features...")
feature_pipeline = build_feature_pipeline(
    max_word_features=20000,
    max_char_features=25000,
    word_ngram_range=(1, 2),
    char_ngram_range=(2, 5)
)

X_train = feature_pipeline.fit_transform(train_df['cleaned_text'])
X_test = feature_pipeline.transform(test_df['cleaned_text'])

print(f"Feature Matrix Shape: {X_train.shape[0]:,} samples x {X_train.shape[1]:,} n-gram features")""")

# Requirement 5 & 6: Binary vs Multilabel Models
add_md("""---
## 4. Model Architectures: Binary vs. Multilabel Classification
### 4.1 Theoretical Formulation
- **Binary Classification:**
  Maps comment $x \\rightarrow \\{0, 1\\}$ (*toxic* vs. *non-toxic*).
  $$P(y=1|x) = \\sigma(w^T x + b)$$
  *Limitation:* A mild rude insult is assigned the exact same label as a violent death threat.
  
- **Multilabel Classification:**
  Maps comment $x \\rightarrow \\{0, 1\\}^L$ across $L=6$ harm categories.
  Each label is not mutually exclusive: a comment can be simultaneously `toxic`, `obscene`, and `insult`.
  
### Multilabel Strategies Evaluated:
1. **Binary Relevance (One-vs-Rest):**
   Decomposes the $L$-label problem into $L$ independent binary classifiers:
   $$h_j(x) \\rightarrow y_j \\in \\{0, 1\\}, \\quad j = 1, \\dots, L$$
   To overcome the severe class imbalance (up to 300:1), each classifier utilizes **inverse class weights**:
   $$w_1 = \\frac{N_{total}}{2 \\cdot N_{pos}}$$
   
2. **Classifier Chains:**
   Arranges classifiers in a sequence where each successive classifier receives the predictions of all preceding classifiers as augmented features:
   $$P(y_j | x, y_1, \\dots, y_{j-1}) = \\sigma(w_j^T [x, y_1, \\dots, y_{j-1}] + b_j)$$
   This explicitly models label correlation (e.g. `severe_toxic` conditioned on `toxic`).""")

add_code("""from src.models import MultilabelToxicClassifier, BinaryToxicClassifier

Y_train_multi = train_df[LABELS].values
Y_test_multi = test_df[LABELS].values

y_train_binary = train_df['is_toxic'].values
y_test_binary = test_df['is_toxic'].values

# 1. Train Multilabel Binary Relevance Classifier
print("Training Multilabel Classifier (Binary Relevance + Balanced Class Weights)...")
multi_clf = MultilabelToxicClassifier(
    strategy='binary_relevance',
    model_type='logistic_regression',
    labels=LABELS
)
multi_clf.fit(X_train, Y_train_multi)
print("Multilabel Model Trained Successfully.")

# 2. Train Binary Toxic Triage Classifier
print("Training Binary Toxic Classifier...")
bin_clf = BinaryToxicClassifier(model_type='logistic_regression')
bin_clf.fit(X_train, y_train_binary)
print("Binary Model Trained Successfully.")""")

# Requirement 7: Evaluation
add_md("""---
## 5. Comprehensive Model Evaluation
Evaluating multilabel models requires both per-label and aggregate metrics:
- **Mean ROC-AUC:** The primary Kaggle competition benchmark (area under receiver operating characteristic curve).
- **Hamming Loss:** Fraction of incorrectly predicted labels:
  $$\\text{Hamming Loss} = \\frac{1}{N \\cdot L} \\sum_{i=1}^N \\sum_{j=1}^L \\mathbf{1}(y_{ij} \\neq \\hat{y}_{ij})$$
- **Subset Accuracy (Exact Match):** Percentage of comments where all 6 labels are predicted perfectly.
- **Micro vs. Macro F1:**
  - Micro-average pools all true positives and false positives globally (dominated by frequent labels).
  - Macro-average weights all 6 categories equally (giving equal voice to rare classes like `threat`).""")

add_code("""from src.evaluation import evaluate_multilabel, evaluate_binary
from sklearn.metrics import classification_report, roc_curve, auc

# Multilabel Predictions
Y_prob_multi = multi_clf.predict_proba(X_test)
Y_pred_multi = multi_clf.predict(X_test)

multi_eval = evaluate_multilabel(Y_test_multi, Y_pred_multi, Y_prob_multi, label_names=LABELS)

print("=== MULTILABEL AGGREGATE METRICS ===")
for k, v in multi_eval['aggregate'].items():
    print(f"  {k:20s}: {v:.4f}")

# Per-label performance DataFrame
per_lbl_df = pd.DataFrame([
    {
        'Label': lbl,
        'Precision': m['precision'],
        'Recall': m['recall'],
        'F1-Score': m['f1_score'],
        'ROC-AUC': m['roc_auc'],
        'PR-AUC': m['pr_auc'],
        'Support': m['support']
    }
    for lbl, m in multi_eval['per_label'].items()
])
display(per_lbl_df)""")

add_md("""### 5.1 ROC Curves Across All 6 Harm Categories""")

add_code("""plt.figure(figsize=(9, 6))
colors = ['#e11d48', '#9333ea', '#f59e0b', '#06b6d4', '#ea580c', '#10b981']

for i, lbl in enumerate(LABELS):
    fpr, tpr, _ = roc_curve(Y_test_multi[:, i], Y_prob_multi[:, i])
    roc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{lbl} (AUC = {roc_score:.3f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('Multilabel ROC Curves (Jigsaw Test Set)', fontsize=14, fontweight='bold', pad=12)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()""")

# Requirement 8: Error Analysis & Sarcasm / Bias Discussion
add_md("""---
## 6. False Positive & False Negative Error Analysis
Content moderation decisions carry profound real-world consequences:
- **False Positives (Type I Error - Over-censorship):** An acceptable comment is flagged as toxic. This can silence marginalized users, chill debate, and generate moderator fatigue.
- **False Negatives (Type II Error - Under-censorship):** A harmful comment is missed as clean. This exposes users to threats, harassment, and hate speech.

We examine misclassified instances from the test set to diagnose their root linguistic causes.""")

add_code("""from src.error_analysis import perform_error_analysis, evaluate_threshold_tradeoffs

y_prob_bin = bin_clf.predict_proba(X_test)
y_pred_bin = bin_clf.predict(X_test)

error_results = perform_error_analysis(
    texts=test_df['comment_text'].tolist(),
    y_true_binary=y_test_binary,
    y_pred_binary=y_pred_bin,
    y_prob_binary=y_prob_bin,
    top_n=5
)

print(f"Total False Positives in Test Set: {error_results['total_false_positives']:,}")
print(f"Total False Negatives in Test Set: {error_results['total_false_negatives']:,}\\n")

print("=== TOP FALSE POSITIVES (Clean Comments Flagged as Toxic) ===")
for i, fp in enumerate(error_results['fp_examples'][:3]):
    print(f"[{i+1}] Confidence: {fp['confidence']*100:.1f}% | Cause: {fp['root_cause']}")
    print(f"    Text: {fp['comment'][:150]}...\\n")

print("=== TOP FALSE NEGATIVES (Toxic Comments Missed as Clean) ===")
for i, fn in enumerate(error_results['fn_examples'][:3]):
    print(f"[{i+1}] Confidence: {fn['confidence']*100:.1f}% | Cause: {fn['root_cause']}")
    print(f"    Text: {fn['comment'][:150]}...\\n")""")

add_md("""### 6.1 The Decision Threshold Trade-off
How does changing the classification threshold alter the balance between False Positives and False Negatives?
We plot the trade-off curve across thresholds from 0.1 to 0.9.""")

add_code("""tradeoffs = evaluate_threshold_tradeoffs(y_test_binary, y_prob_bin)
th_df = pd.DataFrame(tradeoffs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Precision, Recall, F1 Curve
ax1.plot(th_df['threshold'], th_df['precision'], marker='o', label='Precision', color='#3b82f6')
ax1.plot(th_df['threshold'], th_df['recall'], marker='s', label='Recall', color='#10b981')
ax1.plot(th_df['threshold'], th_df['f1_score'], marker='^', label='F1-Score', color='#f59e0b')
ax1.set_title('Precision, Recall, and F1 vs. Threshold', fontsize=12, fontweight='bold')
ax1.set_xlabel('Decision Threshold')
ax1.set_ylabel('Score')
ax1.legend()

# Error Counts Curve
ax2.plot(th_df['threshold'], th_df['false_positives'], marker='o', label='False Positives (Over-blocking)', color='#ef4444')
ax2.plot(th_df['threshold'], th_df['false_negatives'], marker='s', label='False Negatives (Missed Abuse)', color='#8b5cf6')
ax2.set_title('Error Count Trade-off vs. Threshold', fontsize=12, fontweight='bold')
ax2.set_xlabel('Decision Threshold')
ax2.set_ylabel('Absolute Error Count in Test Split')
ax2.legend()

plt.tight_layout()
plt.show()""")

# Discussion on Bias, Sarcasm & Ethics
add_md("""---
## 7. Critical Discussion: Bias, Sarcasm & Ethical Moderation
### 7.1 Dataset Bias & Identity Disparities
In Wikipedia talk pages and public web scrapes, slurs are disproportionately targeted at demographic groups (LGBTQ+, religious minorities, racial minorities).
Consequently, basic n-gram TF-IDF models assign high positive feature weights to identity keywords:
- Example: *"I am a proud gay Muslim woman"* can receive a high toxicity probability purely because the tokens `gay` and `muslim` frequently co-occur with hate speech in the training corpus.
- **Remedy:** Automated systems must never ban accounts based on simple keyword matches. Platforms must maintain adversarial fairness benchmarks to test whether swapping identity tokens changes predicted toxicity.

### 7.2 Contextual Language & Colloquial Swearing
Profanity does not equal toxicity:
- *"This concert was fucking phenomenal!"* $\\rightarrow$ High enthusiasm, zero harm.
- *"You are fucking pathetic."* $\\rightarrow$ Directed second-person personal attack.
Our dual TF-IDF and pronoun retention helps distinguish target pronouns from neutral predicates.

### 7.3 Sarcasm & Surface Politeness
Sarcasm is a known blind spot for bag-of-words and linear models.
- *"Oh wow, you really must have a PhD in stupidity to come up with that masterpiece."*
The presence of words like `masterpiece`, `PhD`, and `genius` artificially suppresses the toxicity score. Detecting subtle passive-aggression requires deeper transformer contextual representations (e.g. RoBERTa) and conversational thread history.

### 7.4 Ethical Framework: Predictions Must Support, Not Sentence
Predictions must serve as **decision support for human moderators** rather than autonomous punishment systems:
1. **Tiered Workflow:**
   - Score < 0.30: Automatic approval.
   - Score 0.30 - 0.70: Routed to human review queue.
   - Score > 0.70: Temporarily hidden pending human review; user notified with right to appeal.
   - Score (`threat`) > 0.35: Immediate priority safety dispatch.
2. **Right to Appeal:** All automated moderation decisions must be contestable by community members.""")

# Requirement 9: Saving Models
add_md("""---
## 8. Serializing Trained Models & Artifacts
We save the trained models, TF-IDF vectorizer, and evaluation metadata using `joblib` so they can be loaded instantly by the Streamlit application.""")

add_code("""import joblib

save_dir = os.path.join('..', 'models_saved')
if not os.path.exists(save_dir):
    save_dir = 'models_saved'
os.makedirs(save_dir, exist_ok=True)

# Save vectorizer and models
joblib.dump(feature_pipeline, os.path.join(save_dir, 'tfidf_vectorizer.joblib'))
joblib.dump(multi_clf, os.path.join(save_dir, 'multilabel_model.joblib'))
joblib.dump(bin_clf, os.path.join(save_dir, 'binary_model.joblib'))

print(f"Artifacts successfully saved to: {save_dir}")
for f in os.listdir(save_dir):
    print(f"  - {f} ({os.path.getsize(os.path.join(save_dir, f)):,} bytes)")""")

add_md("""---
## 9. Streamlit Application
A complete interactive web application has been implemented in `app.py`.
To launch the interactive dashboard:
```bash
streamlit run app.py
```
It provides:
- Live comment risk scoring with radar charts & gauge meters
- Dynamic threshold adjustment slider (tuning precision vs. recall)
- Batch comment processing with CSV export
- Interactive Bias & Sarcasm exploration lab
- Moderator queue triage simulation""")

# Build notebook json
notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = 'notebooks/toxic_comment_classification.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2)

print(f"Notebook successfully created at: {output_path}")
