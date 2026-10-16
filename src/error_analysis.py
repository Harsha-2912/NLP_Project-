"""
False Positive & False Negative Error Analysis Module.
Examines misclassified comments, identity term bias, contextual cursing,
sarcasm, and content moderation trade-offs.
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any

IDENTITY_TERMS = [
    'gay', 'lesbian', 'homosexual', 'queer', 'transgender', 'trans',
    'muslim', 'islam', 'jewish', 'jew', 'christian', 'hindu',
    'black', 'white', 'african', 'asian', 'latino', 'mexican',
    'woman', 'women', 'female', 'male', 'disabled'
]

CASUAL_SWEAR_TERMS = [
    'fuck', 'fucking', 'shit', 'damn', 'hell', 'badass', 'crap'
]

def categorize_false_positive(comment: str) -> str:
    """Categorizes the apparent cause of a False Positive flag."""
    c_lower = comment.lower()
    
    # 1. Identity term mention without toxic intent
    for term in IDENTITY_TERMS:
        if re.search(rf'\b{term}\b', c_lower):
            return "Identity Term Mention (Potential Dataset / Lexical Bias)"
            
    # 2. Colloquial / positive cursing
    for swear in CASUAL_SWEAR_TERMS:
        if re.search(rf'\b{swear}\b', c_lower):
            if any(pos in c_lower for pos in ['love', 'awesome', 'amazing', 'great', 'cool', 'good', 'best', 'incredible']):
                return "Colloquial / Enthusiastic Swear Word (Non-toxic Context)"
            return "Swear Word Used in Meta or Non-directed Context"
            
    # 3. Quoting abuse or policy debate
    if any(q in c_lower for q in ['he said', 'she said', 'quoted', 'banned', 'vandalism', 'rule', 'policy', 'talk page']):
        return "Quoting Policy / Meta-Discussion of Toxicity"
        
    return "Linguistic Ambiguity / Heated Debate"


def categorize_false_negative(comment: str) -> str:
    """Categorizes the apparent cause of a False Negative miss."""
    c_lower = comment.lower()
    
    # 1. Sarcasm / faux politeness
    if any(s in c_lower for s in ['genius', 'brilliant', 'congratulations', 'oh wow', 'surely', 'thanks a lot', 'good job']):
        return "Sarcasm / Passive Aggression (Surface Politeness)"
        
    # 2. Implicit / veiled threat or harassment
    if any(w in c_lower for w in ['watch out', 'be careful', 'where you live', 'regret', 'consequences', 'sleep with one eye']):
        return "Implicit / Veiled Threat (No Profanity)"
        
    # 3. Obfuscation / leetspeak / spaced letters
    if re.search(r'\b[a-z]\s[a-z]\s[a-z]\b', c_lower) or any(c in c_lower for c in ['1d10t', 'k y s', 'kys', 'stfu', 'fck']):
        return "Obfuscation / Acronyms / Evasion Tactics"
        
    return "Subtle Hostility / Low-intensity Abuse"


def perform_error_analysis(
    texts: List[str],
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    y_prob_binary: np.ndarray,
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Identifies and categorizes False Positives and False Negatives.
    """
    df_eval = pd.DataFrame({
        'comment_text': texts,
        'true_toxic': y_true_binary,
        'pred_toxic': y_pred_binary,
        'confidence': y_prob_binary
    })

    # False Positives: Ground truth = 0 (clean), Predicted = 1 (toxic)
    fp_df = df_eval[(df_eval['true_toxic'] == 0) & (df_eval['pred_toxic'] == 1)].copy()
    fp_df = fp_df.sort_values(by='confidence', ascending=False)
    
    false_positives = []
    for _, row in fp_df.head(top_n).iterrows():
        comment = str(row['comment_text'])
        category = categorize_false_positive(comment)
        false_positives.append({
            'comment': comment[:300] + ('...' if len(comment) > 300 else ''),
            'confidence': round(float(row['confidence']), 4),
            'root_cause': category
        })

    # False Negatives: Ground truth = 1 (toxic), Predicted = 0 (clean)
    fn_df = df_eval[(df_eval['true_toxic'] == 1) & (df_eval['pred_toxic'] == 0)].copy()
    fn_df = fn_df.sort_values(by='confidence', ascending=True)

    false_negatives = []
    for _, row in fn_df.head(top_n).iterrows():
        comment = str(row['comment_text'])
        category = categorize_false_negative(comment)
        false_negatives.append({
            'comment': comment[:300] + ('...' if len(comment) > 300 else ''),
            'confidence': round(float(row['confidence']), 4),
            'root_cause': category
        })

    # Identity bias quantification
    identity_stats = {}
    for term in IDENTITY_TERMS:
        mask = df_eval['comment_text'].str.lower().str.contains(rf'\b{term}\b', regex=True, na=False)
        count = int(mask.sum())
        if count >= 10:
            term_clean_mask = mask & (df_eval['true_toxic'] == 0)
            if term_clean_mask.sum() > 0:
                fp_rate = float(df_eval.loc[term_clean_mask, 'pred_toxic'].mean())
                identity_stats[term] = {
                    'total_mentions': count,
                    'clean_comments': int(term_clean_mask.sum()),
                    'false_positive_rate': round(fp_rate * 100, 2)
                }

    return {
        'total_false_positives': len(fp_df),
        'total_false_negatives': len(fn_df),
        'fp_examples': false_positives,
        'fn_examples': false_negatives,
        'identity_bias_stats': identity_stats
    }


def evaluate_threshold_tradeoffs(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Computes precision, recall, false positive count, and false negative count
    across a spectrum of practical decision thresholds.
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tradeoffs = []

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        tp = int(np.sum((y_true == 1) & (preds == 1)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        fn = int(np.sum((y_true == 1) & (preds == 0)))
        tn = int(np.sum((y_true == 0) & (preds == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        tradeoffs.append({
            'threshold': th,
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'false_positives': fp,
            'false_negatives': fn,
            'moderation_impact': (
                "High Catch Rate (Strict / Safety First): High moderator workload due to false alarms"
                if th <= 0.3 else
                "Balanced Triage: Standard threshold for general queue routing"
                if th <= 0.6 else
                "Conservative Flagging (Free Expression First): Low false alarms, but subtle toxicity missed"
            )
        })

    return tradeoffs
