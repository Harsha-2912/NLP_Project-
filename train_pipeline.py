"""
Complete Training and Evaluation Pipeline for Toxic Comment Detection System.
Trains:
1. Dual-channel TF-IDF Vectorizer (Word + Character n-grams).
2. Multilabel Calibrated Classifier (6 Jigsaw labels).
3. Binary Toxic Classifier (overall triage).
Evaluates, generates error analysis, and serializes all models and metadata.
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from src.preprocessing import clean_text
from src.features import build_feature_pipeline
from src.models import MultilabelToxicClassifier, BinaryToxicClassifier, LABELS
from src.evaluation import evaluate_multilabel, evaluate_binary, find_optimal_threshold
from src.error_analysis import perform_error_analysis, evaluate_threshold_tradeoffs

def run_training_pipeline(
    data_path: str = 'data/train.csv',
    output_dir: str = 'models_saved',
    sample_size: int = None,
    random_state: int = 42
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(" TOXIC COMMENT DETECTION SYSTEM - TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/7] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        print(f"      File {data_path} not found locally. Downloading from Hugging Face repository...")
        from huggingface_hub import hf_hub_download
        import shutil
        os.makedirs(os.path.dirname(data_path) or '.', exist_ok=True)
        cached_file = hf_hub_download(
            repo_id='thesofakillers/jigsaw-toxic-comment-classification-challenge',
            filename='train.csv',
            repo_type='dataset'
        )
        shutil.copyfile(cached_file, data_path)
        print(f"      Downloaded successfully to {data_path}")

    df = pd.read_csv(data_path)
    df['comment_text'] = df['comment_text'].fillna('')
    print(f"      Total records loaded: {len(df):,}")

    # Subsample if requested (e.g. for rapid experimentation)
    if sample_size and sample_size < len(df):
        print(f"      Subsampling {sample_size:,} records for fast execution...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # Create binary overall toxicity target
    df['is_toxic'] = (df[LABELS].sum(axis=1) > 0).astype(int)

    # 2. Text Preprocessing
    print("\n[2/7] Preprocessing comment texts...")
    t0 = time.time()
    df['cleaned_text'] = df['comment_text'].apply(lambda x: clean_text(x, remove_stopwords=False, lemmatize=False))
    print(f"      Cleaned {len(df):,} comments in {time.time() - t0:.1f} seconds.")

    # 3. Train-Test Split (Stratified on overall toxicity)
    print("\n[3/7] Performing 80/20 train/test split...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=random_state,
        stratify=df['is_toxic']
    )
    print(f"      Train set: {len(train_df):,} samples ({train_df['is_toxic'].mean()*100:.1f}% toxic)")
    print(f"      Test set : {len(test_df):,} samples ({test_df['is_toxic'].mean()*100:.1f}% toxic)")

    # 4. Feature Extraction (TF-IDF Word + Char N-grams)
    print("\n[4/7] Fitting dual TF-IDF FeatureUnion (Word n-grams + Char n-grams)...")
    t0 = time.time()
    feature_union = build_feature_pipeline(
        max_word_features=25000,
        max_char_features=30000,
        word_ngram_range=(1, 2),
        char_ngram_range=(2, 5)
    )
    X_train = feature_union.fit_transform(train_df['cleaned_text'])
    X_test = feature_union.transform(test_df['cleaned_text'])
    print(f"      Feature matrix shape: {X_train.shape} in {time.time() - t0:.1f} seconds.")

    # Targets
    Y_train_multi = train_df[LABELS].values
    Y_test_multi = test_df[LABELS].values
    y_train_binary = train_df['is_toxic'].values
    y_test_binary = test_df['is_toxic'].values

    # 5. Model Training
    print("\n[5/7] Training models...")
    
    # A. Multilabel Classifier (Binary Relevance)
    print("      -> Training Multilabel Classifier (Binary Relevance with balanced weights)...")
    t0 = time.time()
    multilabel_model = MultilabelToxicClassifier(
        strategy='binary_relevance',
        model_type='logistic_regression',
        labels=LABELS
    )
    multilabel_model.fit(X_train, Y_train_multi)
    print(f"         Multilabel model trained in {time.time() - t0:.1f} seconds.")

    # B. Binary Toxic Classifier
    print("      -> Training Binary Toxic Triage Classifier...")
    t0 = time.time()
    binary_model = BinaryToxicClassifier(
        model_type='logistic_regression',
        threshold=0.5
    )
    binary_model.fit(X_train, y_train_binary)
    print(f"         Binary classifier trained in {time.time() - t0:.1f} seconds.")

    # 6. Evaluation
    print("\n[6/7] Evaluating model performance on held-out test set...")
    
    # Multilabel evaluation
    y_prob_multi = multilabel_model.predict_proba(X_test)
    y_pred_multi = multilabel_model.predict(X_test)
    multi_eval = evaluate_multilabel(Y_test_multi, y_pred_multi, y_prob_multi, label_names=LABELS)

    print("\n      === MULTILABEL EVALUATION RESULTS ===")
    print(f"      Mean ROC-AUC     : {multi_eval['aggregate']['mean_roc_auc']:.4f}")
    print(f"      Hamming Loss     : {multi_eval['aggregate']['hamming_loss']:.4f} (lower is better)")
    print(f"      Subset Accuracy  : {multi_eval['aggregate']['subset_accuracy']:.4f} (exact match)")
    print(f"      Micro F1-Score   : {multi_eval['aggregate']['micro_f1']:.4f}")
    print(f"      Macro F1-Score   : {multi_eval['aggregate']['macro_f1']:.4f}")
    print(f"      Weighted F1-Score: {multi_eval['aggregate']['weighted_f1']:.4f}")
    print("\n      Per-Label Performance:")
    print("      " + "-" * 62)
    print(f"      {'Label':15s} | {'Prec':6s} | {'Recall':6s} | {'F1':6s} | {'ROC-AUC':7s} | {'Support':7s}")
    print("      " + "-" * 62)
    for lbl, m in multi_eval['per_label'].items():
        print(f"      {lbl:15s} | {m['precision']:6.4f} | {m['recall']:6.4f} | {m['f1_score']:6.4f} | {m['roc_auc']:7.4f} | {m['support']:7d}")
    print("      " + "-" * 62)

    # Binary evaluation
    y_prob_binary = binary_model.predict_proba(X_test)
    y_pred_binary = binary_model.predict(X_test)
    binary_eval = evaluate_binary(y_test_binary, y_pred_binary, y_prob_binary)
    optimal_th = find_optimal_threshold(y_test_binary, y_prob_binary)

    print("\n      === BINARY CLASSIFIER EVALUATION ===")
    print(f"      Accuracy    : {binary_eval['accuracy']:.4f}")
    print(f"      Precision   : {binary_eval['precision']:.4f}")
    print(f"      Recall      : {binary_eval['recall']:.4f}")
    print(f"      F1-Score    : {binary_eval['f1_score']:.4f}")
    print(f"      ROC-AUC     : {binary_eval['roc_auc']:.4f}")
    print(f"      Specificity : {binary_eval['specificity']:.4f}")
    print(f"      Optimal F1 Threshold: {optimal_th['optimal_threshold']} (F1: {optimal_th['best_score']:.4f})")

    # 7. Error Analysis & Moderation Trade-offs
    print("\n[7/7] Performing False Positive & False Negative Error Analysis...")
    error_analysis_data = perform_error_analysis(
        texts=test_df['comment_text'].tolist(),
        y_true_binary=y_test_binary,
        y_pred_binary=y_pred_binary,
        y_prob_binary=y_prob_binary,
        top_n=15
    )
    threshold_tradeoffs = evaluate_threshold_tradeoffs(y_test_binary, y_prob_binary)

    print(f"      Identified {error_analysis_data['total_false_positives']:,} False Positives and {error_analysis_data['total_false_negatives']:,} False Negatives.")

    # 8. Save Artifacts
    print(f"\n[Artifacts] Saving serialized models and evaluation results to '{output_dir}'...")
    joblib.dump(feature_union, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(multilabel_model, os.path.join(output_dir, 'multilabel_model.joblib'))
    joblib.dump(binary_model, os.path.join(output_dir, 'binary_model.joblib'))

    all_eval = {
        'multilabel': multi_eval,
        'binary': binary_eval,
        'optimal_threshold': optimal_th['optimal_threshold'],
        'sample_size': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'labels': LABELS
    }

    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(all_eval, f, indent=2)

    with open(os.path.join(output_dir, 'error_analysis.json'), 'w') as f:
        json.dump(error_analysis_data, f, indent=2)

    with open(os.path.join(output_dir, 'threshold_tradeoffs.json'), 'w') as f:
        json.dump(threshold_tradeoffs, f, indent=2)

    metadata = {
        'model_name': 'Toxic Comment Detection System (Jigsaw)',
        'architecture': 'Dual-channel TF-IDF (Word (1,2) + Char (2,5)) + Calibrated Logistic Regression',
        'strategy': 'Multilabel Binary Relevance & Binary Toxic Triage',
        'labels': LABELS,
        'training_samples': len(train_df),
        'test_samples': len(test_df),
        'elapsed_seconds': round(time.time() - start_time, 2)
    }

    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n Pipeline completed successfully in {time.time() - start_time:.1f} seconds!")
    print(f" Serialized files in {output_dir}:")
    for f in os.listdir(output_dir):
        print(f"   - {f} ({os.path.getsize(os.path.join(output_dir, f)):,} bytes)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/train.csv')
    parser.add_argument('--sample_size', type=int, default=None, help='Subset sample size for quick test')
    args = parser.parse_args()

    run_training_pipeline(data_path=args.data_path, sample_size=args.sample_size)
