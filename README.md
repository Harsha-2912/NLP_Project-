# 🛡️ Toxic Comment Detection & Content Moderation System
**Assigned Teams:** 9 and 11  
**Domain:** Natural Language Processing (NLP) & Content Moderation  
**Dataset:** Jigsaw Toxic Comment Classification Dataset  
**Target Labels:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

---

## 📋 Executive Summary
Online platforms (social media, community forums, Wikipedia, gaming networks) process millions of user comments daily. While the vast majority of interactions are constructive, abusive content (hate speech, personal insults, obscenity, and violent threats) disrupts civil discourse and inflicts real psychological harm.

This project delivers an end-to-end NLP-based decision support system that automates the triage of potentially harmful online comments. By combining **linguistic preprocessing**, **dual-channel TF-IDF n-grams** (word + character sub-word representations), and **cost-sensitive multilabel calibrated classifiers**, the system flags abusive content and provides multidimensional risk scores to help human moderators prioritize interventions.

Crucially, the system is designed under a **human-in-the-loop paradigm**, treating machine learning predictions as assistive intelligence for human review rather than an autonomous censorship or penalization mechanism.

---

## 🎯 Project Objectives & Implementation Checklist

| # | Requirement | Implementation Details | Status |
|---|---|---|:---:|
| 1 | **Dataset Loading & Inspection** | Analyzed all 159,571 comments, column types, text lengths, and missing values. | ✅ Completed |
| 2 | **Class Imbalance Examination** | Quantified label skew (from 9.58% down to 0.30% for `threat`), calculated imbalance ratios (up to 300:1), label cardinality, and co-occurrence. | ✅ Completed |
| 3 | **Text Preprocessing** | Built custom pipeline handling HTML unescaping, contraction expansion, character de-elongation, masked profanity de-masking, and pronoun-preserving tokenization. | ✅ Completed |
| 4 | **TF-IDF Feature Extraction** | Implemented dual-channel `FeatureUnion` combining word n-grams (1, 2) and character n-grams (2, 5) with sublinear term-frequency scaling. | ✅ Completed |
| 5 | **Binary vs. Multilabel Design** | Implemented both: (1) coarse binary triage (*toxic* vs. *clean*) and (2) fine-grained 6-label multilabel classification. | ✅ Completed |
| 6 | **Model Training & Strategies** | Trained Binary Relevance with calibrated cost-sensitive logistic regression and classifier chains. | ✅ Completed |
| 7 | **Comprehensive Evaluation** | Evaluated per-label and aggregate metrics: Mean ROC-AUC, Hamming Loss, Subset Accuracy, Macro/Micro F1, Precision, and Recall. | ✅ Completed |
| 8 | **Error Analysis (FP & FN)** | Deep-dived into False Positives (identity term bias, colloquial cursing, quoting abuse) and False Negatives (sarcasm, veiled threats, obfuscation). | ✅ Completed |
| 9 | **Model Serialization** | Serialized all models, vectorizers, and metadata using `joblib` into `models_saved/`. | ✅ Completed |
| 10 | **Interactive Streamlit App** | Built a modern web application featuring real-time analysis, gauge meters, radar charts, batch CSV processing, bias labs, and threshold sliders. | ✅ Completed |

---

## 🏗️ System Architecture

```
                                 [ User Comment Input ]
                                           │
                                           ▼
             ┌───────────────────────────────────────────────────────────┐
             │            Linguistic Preprocessing Pipeline              │
             │  - HTML decoding & entity cleanup                         │
             │  - Masked profanity normalization (f**k -> fuck)          │
             │  - Contraction expansion (can't -> cannot, u -> you)      │
             │  - Character de-elongation (fuuuuck -> fuuck)             │
             │  - Pronoun & negation preservation for moderation context │
             └─────────────────────────────┬─────────────────────────────┘
                                           │
                                           ▼
             ┌───────────────────────────────────────────────────────────┐
             │           Dual-Channel TF-IDF Feature Extractor           │
             │  ├── Word N-Grams (1-2 words, 25k features, sublinear tf) │
             │  └── Char N-Grams (2-5 chars, 30k features, sublinear tf) │
             └─────────────────────────────┬─────────────────────────────┘
                                           │  (55,000 sparse features)
                                           ▼
             ┌───────────────────────────────────────────────────────────┐
             │                Classification Ensembles                   │
             │  ├── Multilabel Classifier: Binary Relevance + Weights    │
             │  │   [Toxic, Severe Toxic, Obscene, Threat, Insult, Hate] │
             │  └── Binary Classifier: Overall Toxicity Triage           │
             └─────────────────────────────┬─────────────────────────────┘
                                           │
                                           ▼
             ┌───────────────────────────────────────────────────────────┐
             │            Streamlit Decision-Support Dashboard           │
             │  - Risk meter gauge & multidimensional radar chart        │
             │  - Dynamic decision threshold tuning (0.10 to 0.90)       │
             │  - Moderator queue triage (Approve / Escalate / Redact)   │
             │  - Interactive Bias, Sarcasm & Ethical Moderation Lab     │
             └───────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Class Imbalance Analysis

### 1. Label Frequencies & Imbalance Ratios (159,571 Comments)
| Label | Positive Count | Frequency (%) | Clean : Class Imbalance Ratio | Primary Harm Type |
|---|:---:|:---:|:---:|---|
| **Clean (Non-toxic)** | 143,346 | 89.83% | 1.0 : 1 | Acceptable discourse |
| **Toxic (Any)** | 16,225 | 10.17% | 8.8 : 1 | General hostility / abuse |
| `toxic` | 15,294 | 9.58% | 9.4 : 1 | Rude, disrespectful, or hostile tone |
| `obscene` | 8,449 | 5.29% | 17.0 : 1 | Vulgarities, sexualized expletives |
| `insult` | 7,877 | 4.94% | 18.2 : 1 | Disparaging or demeaning personal attacks |
| `severe_toxic` | 1,595 | 1.00% | 89.9 : 1 | Extremely aggressive / vicious attacks |
| `identity_hate` | 1,405 | 0.88% | 102.0 : 1 | Slurs targeting race, religion, sexual orientation |
| `threat` | 478 | 0.30% | **299.9 : 1** | Physical violence, death threats |

### 2. Multilabel Properties
- **Label Cardinality:** `0.2200` (on average, each comment contains 0.22 positive labels).
- **Label Density:** `0.0367` ($0.22 / 6$).
- **Co-occurrence Insight:** Over 85% of comments labeled `severe_toxic` also co-occur with `toxic` and `obscene`. Physical `threat` and `identity_hate` form distinct, sparser clusters requiring specialized sensitivity.

---

## ⚙️ Text Preprocessing & Linguistic Engineering

Standard text classification pipelines (e.g., lowercase, remove all punctuation, remove all stopwords) severely hurt toxic comment detection. Our custom preprocessing pipeline incorporates specialized moderation logic:

1. **Masked Profanity Decoding:**
   Abusive commenters deliberately mask characters to evade standard regex filters (`f**k`, `b*tch`, `a$$hole`, `sh!t`). We normalize these tokens to standard forms so the model can learn their true semantic weight.
2. **Character De-Elongation:**
   Trolls stretch vowels and consonants for emphasis (`fuuuuuck`, `stooopid`, `whyyyyy`). We reduce 3+ repeated characters to 2 (`fuuck`, `stoopid`), preserving emotional intensity while preventing vocabulary explosion.
3. **Contraction & Slang Expansion:**
   Expanded standard English contractions and internet abbreviations (`can't` $\rightarrow$ `cannot`, `u` $\rightarrow$ `you`, `wtf` $\rightarrow$ `what the fuck`, `ur` $\rightarrow$ `your`).
4. **Contextual Stopword Preservation:**
   Standard stopword lists remove pronouns (`you`, `he`, `she`, `they`) and negations (`not`, `no`, `never`). In toxic speech:
   - Second-person pronouns (`you`, `your`) are the primary signal of **directed personal insult** (*"You are an idiot"* vs *"That idea is an idiot mistake"*).
   - Negations invert toxic meaning (*"Not bad"* vs *"Bad"*).
   Therefore, we retain all pronouns and negations.

---

## 🔡 Feature Extraction: Dual-Channel TF-IDF

A pure bag-of-words model fails when users disguise swear words (e.g., `fuk`, `shyt`, `b!tch`). Conversely, pure character models lose higher-level multi-word context. We use a **FeatureUnion** of two complementary spaces:

1. **Word-Level TF-IDF:**
   - N-Gram Range: `(1, 2)` (unigrams and bigrams like *"shut up"*, *"hate you"*)
   - Max Features: `25,000`
   - Sublinear Scaling: $tf' = 1 + \log(tf)$ to dampen troll comments repeating the same word 50 times.
2. **Character-Level TF-IDF:**
   - Analyzer: `char_wb` (character n-grams strictly within word boundaries)
   - N-Gram Range: `(2, 5)`
   - Max Features: `30,000`
   - Captures sub-word morphological patterns regardless of spelling tricks.

Total feature dimensionality: **55,000 sparse n-gram features**.

---

## 🤖 Model Architectures: Binary vs. Multilabel

### Why Binary Classification is Insufficient Alone:
A binary classifier outputs a single score: *Is this comment toxic?*
While useful for preliminary queue routing, binary classification collapses all harms into a single bucket. A comment saying *"Your argument is dumb"* is treated identically to *"I will find where you live and burn your house down"*. 

### Multilabel Architecture:
We trained a **Binary Relevance (One-vs-Rest)** ensemble with calibrated logistic regression and balanced class weights:
$$\mathcal{L} = - \sum_{i=1}^N \left[ w_1 y_i \log(\hat{p}_i) + w_0 (1 - y_i) \log(1 - \hat{p}_i) \right]$$
Where $w_1 = \frac{N}{2 \cdot N_{pos}}$, forcing the optimization to heavily penalize misses on rare classes like `threat` (300:1 ratio) and `identity_hate` (102:1 ratio).

---

## 📈 Evaluation & Performance Results

### Multilabel Benchmark Metrics
- **Mean ROC-AUC (Kaggle Official Metric):** `0.978`+
- **Hamming Loss:** `0.021` (97.9% of all label decisions are correct)
- **Subset Accuracy (Exact Match):** `88.5%`+
- **Micro F1-Score:** `0.73`+
- **Macro F1-Score:** `0.59`+ (balanced across severe imbalance)

### Per-Harm Category Performance Table
| Category | Precision | Recall | F1-Score | ROC-AUC | Test Support |
|---|:---:|:---:|:---:|:---:|:---:|
| `toxic` | 0.72 | 0.88 | 0.79 | 0.981 | ~3,059 |
| `severe_toxic` | 0.38 | 0.86 | 0.53 | 0.988 | ~319 |
| `obscene` | 0.71 | 0.87 | 0.78 | 0.989 | ~1,690 |
| `threat` | 0.32 | 0.78 | 0.45 | 0.987 | ~96 |
| `insult` | 0.67 | 0.83 | 0.74 | 0.979 | ~1,575 |
| `identity_hate` | 0.44 | 0.79 | 0.56 | 0.976 | ~281 |

> **Key Observation:** Because we applied cost-sensitive class balancing, **Recall is prioritized across all classes (78% to 88%)**, ensuring dangerous content (`threat`, `severe_toxic`) is not silently overlooked. Precision can be further tuned using the application's dynamic threshold slider.

---

## 🔬 False Positive & False Negative Error Analysis

### 1. False Positives (Acceptable Comments Flagged as Toxic)
- **Identity Term Mention (Dataset Bias):**
  - *Example:* *"I am a proud gay Muslim student speaking out on human rights."*
  - *Root Cause:* Words like `gay` and `muslim` frequently co-occur with hate speech in training data. Linear models learn positive correlation between the identity token and toxicity.
- **Colloquial / Enthusiastic Swear Words:**
  - *Example:* *"Holy shit, that new guitar solo was fucking incredible!"*
  - *Root Cause:* The presence of `shit` and `fucking` triggers obscenity models despite the positive sentiment and absence of a human target.
- **Quoting Harassment / Reporting to Admins:**
  - *Example:* *"User42 posted: 'You are an ugly moron' before being blocked."*
  - *Root Cause:* The model parses the quoted phrase as primary text without recognizing meta-discourse.

### 2. False Negatives (Toxic Comments Missed as Clean)
- **Sarcasm & Surface Politeness:**
  - *Example:* *"Oh wow, you must have a PhD in stupidity to come up with that masterpiece."*
  - *Root Cause:* Words like `PhD`, `masterpiece`, and `congratulations` carry positive weights that counteract the insulting intent.
- **Coded / Veiled Threats:**
  - *Example:* *"I know which route you walk home at night. Be very careful."*
  - *Root Cause:* Zero explicit profanity or violent verbs (`kill`, `shoot`); relies entirely on situational menace.
- **Creative Leetspeak & Spacing:**
  - *Example:* *"k y s", "g0 to h3ll"*
  - *Root Cause:* Spaced characters break standard tokenizers unless caught by character n-grams.

---

## ⚖️ Moderation Trade-offs & The Threshold Spectrum

| Threshold | Policy Stance | Precision | Recall | False Positives | False Negatives | Operational Impact |
|:---:|---|:---:|:---:|:---:|:---:|---|
| **0.20** | Safety-First / High Catch | Moderate | Very High | High | Low | Catches >95% of threats; high moderator workload from false alarms. |
| **0.50** | Balanced Triage | High | High | Balanced | Balanced | Default threshold for standard moderation queue routing. |
| **0.80** | Free Expression Priority | Very High | Moderate | Very Low | High | Minimal censorship; subtle toxicity slips into public threads. |

---

## 🌐 Streamlit Web Application Features

The interactive web dashboard (`app.py`) provides:

1. **🔍 Real-Time Comment Analyzer:**
   - Real-time text scoring with risk status banner (🟢 Safe, 🟡 Review, 🔴 Harmful, 🚨 Critical Threat).
   - Interactive gauge meters & 6-dimensional radar chart.
   - Pre-loaded test cases (hostility, threats, positive swearing, identity statements, sarcasm).
   - Human-in-the-loop action buttons (Approve, Escalate, Issue Warning, Redact).
2. **📋 Moderator Queue & Batch Processing:**
   - Upload any CSV or load the pre-packaged 80-comment sample test set.
   - Filter by harm type (`threat`, `hate`, `obscene`) and confidence score.
   - Export reviewed moderation triage reports as CSV.
3. **⚖️ Bias, Sarcasm & Context Lab:**
   - Interactive identity term sensitivity tests (swapping `gay`, `black`, `muslim`, `straight` in neutral templates).
   - Directed abuse vs. colloquial swear comparisons.
   - Sarcasm vulnerability demonstrations.
4. **📊 Evaluation & Metrics Dashboard:**
   - Per-label performance charts, class imbalance graphs, and ROC curves.
5. **🔬 False Positive & Negative Explorer:**
   - Inspect misclassified instances categorized by root linguistic cause.
   - Interactive threshold trade-off curve slider.

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Environment Setup
Clone or navigate to the project directory:
```bash
cd d:\NLP-Project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline (Optional - Pre-trained Models Included)
To retrain the vectorizer, multilabel models, and regenerate evaluation metrics:
```bash
python train_pipeline.py
```
*Note: You can pass `--sample_size 40000` for rapid training or run without arguments to train on all 159,571 comments.*

### 3. Launch Streamlit Application
Start the interactive moderation web app:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Jupyter Notebook
To inspect the step-by-step interactive analysis and visualizations:
```bash
jupyter notebook notebooks/toxic_comment_classification.ipynb
```

---

## 📜 Ethical Moderation Guidelines & Conclusions

1. **AI Supports, Humans Decide:** Automated toxicity scores should **prioritize content for human review** rather than imposing autonomous bans, account suspensions, or automated censorship.
2. **Context Matters:** Moderation systems must distinguish between **directed personal harassment** and **enthusiastic casual swearing**, **academic discussions**, or **victims quoting abuse**.
3. **Protecting Marginalized Voices:** Keyword blocklists and uncalibrated models disproportionately censor LGBTQ+ and minority discussions due to dataset bias. Regular adversarial auditing is mandatory.
4. **Transparency & Due Process:** Users must be informed when their content is flagged and provided with a clear, accessible human appeal mechanism.
