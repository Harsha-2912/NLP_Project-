"""
Text Preprocessing Pipeline for Toxic Comment Detection.
Handles Wikipedia markup, contractions, censored profanities, elongated characters,
and linguistic normalization while preserving crucial moderation context.
"""

import re
import html
from typing import List, Union
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure resources are available
try:
    _lemmatizer = WordNetLemmatizer()
except Exception:
    nltk.download('wordnet')
    _lemmatizer = WordNetLemmatizer()

try:
    _nltk_stopwords = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords')
    _nltk_stopwords = set(stopwords.words('english'))

# In toxic comment detection, pronouns and negations carry significant signal:
# - "you" / "your" indicates direct target of insult/threat.
# - "not" / "no" / "never" flips polarity and meaning.
# - "he" / "she" / "they" / "them" provides targets in identity hate.
RETAIN_WORDS = {
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'they', 'them', 'their', 'theirs', 'themselves',
    'not', 'no', 'nor', 'never', 'against', 'why', 'who', 'what', 'where'
}

CUSTOM_STOPWORDS = _nltk_stopwords - RETAIN_WORDS

# Common contractions dictionary with word boundaries
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bn't\b": " not",
    r"'re\b": " are",
    r"'s\b": " is",
    r"'d\b": " would",
    r"'ll\b": " will",
    r"'t\b": " not",
    r"'ve\b": " have",
    r"'m\b": " am",
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\br\b": "are",
    r"\bw/\b": "with",
    r"\bw/o\b": "without",
    r"\bb/c\b": "because",
    r"\bfyi\b": "for your information",
    r"\bwtf\b": "what the fuck",
    r"\bwth\b": "what the hell",
    r"\bomg\b": "oh my god",
    r"\bidk\b": "i do not know",
    r"\bimo\b": "in my opinion",
    r"\btbh\b": "to be honest",
}

# Regex for common masked profanities (evasion tactics)
MASKED_PATTERNS = [
    (re.compile(r'\bf[\*\@\#\$\%\!]+k\b', re.IGNORECASE), 'fuck'),
    (re.compile(r'\bf[\*\@\#\$\%\!]+ing\b', re.IGNORECASE), 'fucking'),
    (re.compile(r'\bb[\*\@\#\$\%\!]+(?:i|1)?tch\b', re.IGNORECASE), 'bitch'),
    (re.compile(r'\bb[\*\@\#\$\%\!]+h\b', re.IGNORECASE), 'bitch'),
    (re.compile(r'\ba[\$\*\@\#]+hole\b', re.IGNORECASE), 'asshole'),
    (re.compile(r'\ba[\$\*\@\#]{2,}\b', re.IGNORECASE), 'ass'),
    (re.compile(r'\bsh[\*\@\#\$\%!]+t\b', re.IGNORECASE), 'shit'),
    (re.compile(r'\bd[\*\@\#\$\%\!]+ck\b', re.IGNORECASE), 'dick'),
    (re.compile(r'\bc[\*\@\#\$\%\!]+nt\b', re.IGNORECASE), 'cunt'),
    (re.compile(r'\bn[\*\@\#\$\%\!]+gg[a-z]*\b', re.IGNORECASE), 'nigger'),
]

# Character elongation regex (e.g., "fuuuuck" -> "fuuck", "stooopp" -> "stopp")
ELONGATION_REGEX = re.compile(r'(.)\1{2,}')

# Wikipedia / Web artifacts
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_REGEX = re.compile(r'<.*?>')
IP_REGEX = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
WIKI_TALK_REGEX = re.compile(r'==.*?==|\[\[User:.*?\]\]|\d{2}:\d{2}, \d{1,2} [A-Za-z]+ \d{4} \(UTC\)')


def expand_contractions(text: str) -> str:
    """Expands standard English contractions and internet slang."""
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_censored_words(text: str) -> str:
    """Decodes common masked abusive tokens designed to bypass basic filters."""
    for pattern, replacement in MASKED_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def clean_text(
    text: Union[str, float],
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    lower: bool = True
) -> str:
    """
    Cleans raw comment text:
    1. Unescapes HTML entities and removes tags.
    2. Strips URLs, IP addresses, Wikipedia edit stamps.
    3. Normalizes masked profanity (e.g. f**k -> fuck).
    4. Expands contractions and internet slang abbreviations.
    5. Reduces character elongation (e.g. nooooo -> noo).
    6. Normalizes punctuation and spaces.
    7. Optionally filters non-essential stopwords.
    8. Optionally applies lemmatization.
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities (&amp;, &lt;, &gt;, &quot;, etc.)
    text = html.unescape(text)

    # Remove HTML tags
    text = HTML_TAG_REGEX.sub(' ', text)

    # Remove URLs and Wikipedia metadata
    text = URL_REGEX.sub(' ', text)
    text = WIKI_TALK_REGEX.sub(' ', text)
    text = IP_REGEX.sub(' ', text)

    # Normalize masked swear words
    text = normalize_censored_words(text)

    # Expand contractions
    text = expand_contractions(text)

    # Lowercase
    if lower:
        text = text.lower()

    # Reduce elongated characters: 'sooooo' -> 'soo', 'fuuuuck' -> 'fuuck'
    text = ELONGATION_REGEX.sub(r'\1\1', text)

    # Replace special symbols/newlines with space, keep alphanumeric and apostrophes
    text = re.sub(r'[^a-zA-Z0-9\s\']', ' ', text)

    # Tokenize by whitespace
    tokens = text.split()

    if remove_stopwords:
        tokens = [w for w in tokens if w not in CUSTOM_STOPWORDS]

    if lemmatize:
        tokens = [_lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


def batch_clean(texts: List[str], remove_stopwords: bool = False, lemmatize: bool = False) -> List[str]:
    """Applies clean_text across a list or series of texts."""
    return [clean_text(t, remove_stopwords=remove_stopwords, lemmatize=lemmatize) for t in texts]


if __name__ == '__main__':
    # Test suite for preprocessing
    raw_samples = [
        "YOU ARE AN IDIOT!!! GO F**K YOURSELF!! http://spam.com/123",
        "Hello 192.168.1.5, I can't believe u reverted my edit. == vandalism ==",
        "I loooove this movie sooooo much! It is f***ing fantastic!",
        "&quot;She said: you are a worthless b*tch&quot;",
        "I'm a proud gay Muslim student, why do people hate on us?"
    ]

    print("Preprocessing demonstration:")
    for raw in raw_samples:
        cleaned = clean_text(raw)
        print(f"RAW    : {raw}")
        print(f"CLEANED: {cleaned}")
        print("-" * 60)
