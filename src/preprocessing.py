"""
Text preprocessing and tokenization utilities for ScamShield.
Preserves security-relevant signals like currency symbols, numbers, and URLs.
"""

import re

# Regex patterns for normalization while preserving security indicators
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
CURRENCY_PATTERN = re.compile(r'[\$£€₹]')
NUMBER_PATTERN = re.compile(r'\b\d+\b')
SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z0-9\s\$£€₹]')
MULTI_SPACE_PATTERN = re.compile(r'\s+')

def preprocess_text(text: str, preserve_indicators: bool = True) -> str:
    """
    Cleans and normalizes text for TF-IDF vectorization.
    
    If preserve_indicators is True:
    - Replaces raw URLs with 'http_token'
    - Retains currency markers
    - Standardizes whitespace
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()

    if preserve_indicators:
        # Normalize URLs to a uniform token so TF-IDF learns URL presence
        cleaned = URL_PATTERN.sub(' url_token ', cleaned)
        # Normalize phone numbers
        cleaned = PHONE_PATTERN.sub(' phone_token ', cleaned)

    # Convert to lowercase
    cleaned = cleaned.lower()

    # Remove non-alphanumeric except currency and whitespace
    cleaned = SPECIAL_CHAR_PATTERN.sub(' ', cleaned)

    # Normalize excessive whitespace
    cleaned = MULTI_SPACE_PATTERN.sub(' ', cleaned).strip()

    return cleaned
