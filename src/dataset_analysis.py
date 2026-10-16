"""
Dataset Exploration and Class Imbalance Analysis for Toxic Comment Detection.
Analyzes label frequencies, cardinality, co-occurrence, and linguistic distributions.
"""

import os
import json
import pandas as pd
import numpy as np

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def analyze_dataset(df: pd.DataFrame, output_dir: str = 'models_saved') -> dict:
    """
    Computes comprehensive dataset statistics:
    - Label counts and percentages
    - Class imbalance ratios relative to dominant clean class
    - Multilabel cardinality and density
    - Multilabel co-occurrence / correlation matrix
    - Text length statistics for clean vs. toxic comments
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = len(df)
    
    # Calculate row-level toxicity flags
    label_sum = df[LABELS].sum(axis=1)
    df_clean_mask = (label_sum == 0)
    df_toxic_mask = (label_sum > 0)
    
    clean_count = int(df_clean_mask.sum())
    any_toxic_count = int(df_toxic_mask.sum())
    
    # 1. Label Frequencies and Imbalance Ratios
    label_stats = {}
    for col in LABELS:
        count = int(df[col].sum())
        pct = float((count / n_samples) * 100)
        imbalance_ratio = float(clean_count / count) if count > 0 else 0.0
        label_stats[col] = {
            'count': count,
            'percentage': round(pct, 2),
            'imbalance_ratio': round(imbalance_ratio, 2)
        }
    
    # 2. Multilabel Metrics
    # Label Cardinality: Average number of labels per comment
    label_cardinality = float(label_sum.mean())
    # Label Density: Label Cardinality divided by number of labels
    label_density = float(label_cardinality / len(LABELS))
    
    # Label count breakdown (0, 1, 2, 3, 4, 5, 6 labels)
    label_distribution = df[LABELS].sum(axis=1).value_counts().sort_index().to_dict()
    label_distribution = {str(k): int(v) for k, v in label_distribution.items()}
    
    # 3. Label Correlation Matrix (Pearson)
    corr_matrix = df[LABELS].corr().round(3).to_dict()
    
    # 4. Text Length Distributions
    df['char_length'] = df['comment_text'].fillna('').str.len()
    df['word_count'] = df['comment_text'].fillna('').str.split().str.len()
    
    length_stats = {
        'clean': {
            'mean_chars': round(float(df.loc[df_clean_mask, 'char_length'].mean()), 1),
            'median_chars': round(float(df.loc[df_clean_mask, 'char_length'].median()), 1),
            'mean_words': round(float(df.loc[df_clean_mask, 'word_count'].mean()), 1),
            'median_words': round(float(df.loc[df_clean_mask, 'word_count'].median()), 1),
        },
        'toxic': {
            'mean_chars': round(float(df.loc[df_toxic_mask, 'char_length'].mean()), 1),
            'median_chars': round(float(df.loc[df_toxic_mask, 'char_length'].median()), 1),
            'mean_words': round(float(df.loc[df_toxic_mask, 'word_count'].mean()), 1),
            'median_words': round(float(df.loc[df_toxic_mask, 'word_count'].median()), 1),
        }
    }
    
    analysis_results = {
        'total_samples': n_samples,
        'clean_count': clean_count,
        'clean_pct': round((clean_count / n_samples) * 100, 2),
        'any_toxic_count': any_toxic_count,
        'any_toxic_pct': round((any_toxic_count / n_samples) * 100, 2),
        'label_stats': label_stats,
        'multilabel': {
            'label_cardinality': round(label_cardinality, 4),
            'label_density': round(label_density, 4),
            'label_count_distribution': label_distribution,
        },
        'correlation_matrix': corr_matrix,
        'length_stats': length_stats
    }
    
    # Save to disk for app and report reuse
    summary_path = os.path.join(output_dir, 'eda_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
        
    return analysis_results

if __name__ == '__main__':
    train_path = os.path.join('data', 'train.csv')
    if os.path.exists(train_path):
        print(f"Loading {train_path}...")
        df = pd.read_csv(train_path)
        stats = analyze_dataset(df)
        print("\n=== DATASET EXPLORATION & CLASS IMBALANCE SUMMARY ===")
        print(f"Total Comments       : {stats['total_samples']:,}")
        print(f"Clean Comments       : {stats['clean_count']:,} ({stats['clean_pct']}%)")
        print(f"Toxic Comments       : {stats['any_toxic_count']:,} ({stats['any_toxic_pct']}%)")
        print("\nLabel Breakdown:")
        for lbl, s in stats['label_stats'].items():
            print(f"  {lbl:15s}: {s['count']:6,d} ({s['percentage']:5.2f}%) | Imbalance Ratio: {s['imbalance_ratio']:6.1f} : 1")
        print(f"\nLabel Cardinality   : {stats['multilabel']['label_cardinality']}")
        print(f"Label Density       : {stats['multilabel']['label_density']}")
        print("\nSaved summary to models_saved/eda_summary.json")
    else:
        print(f"File {train_path} not found.")
