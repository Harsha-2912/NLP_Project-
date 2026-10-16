"""
TF-IDF Feature Extraction Pipeline for Toxic Comment Detection.
Combines word-level n-grams and character-level n-grams to capture
both semantic context and obfuscated / misspelled toxic terms.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import joblib

def build_feature_pipeline(
    max_word_features: int = 25000,
    max_char_features: int = 30000,
    word_ngram_range: tuple = (1, 2),
    char_ngram_range: tuple = (2, 5)
) -> FeatureUnion:
    """
    Constructs a dual-channel TF-IDF FeatureUnion:
    1. Word n-grams: Captures lexical phrases (e.g. "hate you", "shut up").
    2. Character n-grams: Captures intentional misspellings, character insertions,
       and obfuscations (e.g. "f*ck", "b!tch", "shyt", "fuuuuck").
    """
    word_vectorizer = TfidfVectorizer(
        ngram_range=word_ngram_range,
        max_features=max_word_features,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        min_df=2
    )

    char_vectorizer = TfidfVectorizer(
        ngram_range=char_ngram_range,
        max_features=max_char_features,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char_wb',  # char n-grams within word boundaries
        min_df=3
    )

    union = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])

    return union


class ToxicFeatureExtractor:
    """
    Wrapper for serializing and managing the TF-IDF feature extractor.
    """
    def __init__(self, max_word_features: int = 25000, max_char_features: int = 30000):
        self.pipeline = build_feature_pipeline(
            max_word_features=max_word_features,
            max_char_features=max_char_features
        )
        self.is_fitted = False

    def fit(self, texts):
        self.pipeline.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transforming.")
        return self.pipeline.transform(texts)

    def fit_transform(self, texts):
        transformed = self.pipeline.fit_transform(texts)
        self.is_fitted = True
        return transformed

    def save(self, filepath: str):
        joblib.dump(self.pipeline, filepath)

    @classmethod
    def load(cls, filepath: str):
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        instance.is_fitted = True
        return instance
