"""
Data ingestion, validation, and stratified train/test split for ScamShield.
Dataset: UCI SMS Spam Collection (Almeida & Gomez Hidalgo)
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/SMSSpamCollection")
PROCESSED_DIR = Path("data/processed")

def prepare_dataset():
    if not RAW_DATA_PATH.exists():
        print(f"Error: {RAW_DATA_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading raw data from: {RAW_DATA_PATH}")
    # Read tab-separated file without header
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep='\t',
        names=['label', 'message'],
        quoting=3, # csv.QUOTE_NONE
        encoding='utf-8',
        on_bad_lines='skip'
    )

    print(f"Raw record count: {len(df)}")
    
    # Drop rows with null message or label
    df = df.dropna(subset=['label', 'message'])
    
    # Strip whitespace
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['message'] = df['message'].astype(str).str.strip()

    # Filter only expected binary labels
    df = df[df['label'].isin(['ham', 'spam'])].copy()

    # Encode label: ham=0 (legitimate), spam=1 (scam/phishing)
    df['target'] = (df['label'] == 'spam').astype(int)

    # Inspect class balance before deduplication
    print("\n--- Distribution Before Deduplication ---")
    print(df['label'].value_counts())
    print(df['label'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))

    # Inspect duplicates
    duplicate_count = df.duplicated(subset=['message']).sum()
    print(f"\nDuplicate messages found: {duplicate_count}")

    # Remove duplicate messages to prevent identical messages leaking between train and test sets
    df = df.drop_duplicates(subset=['message']).reset_index(drop=True)
    
    print("\n--- Distribution After Deduplication ---")
    print(df['label'].value_counts())
    print(df['label'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    print(f"Total unique records: {len(df)}")

    # Stratified Train/Test Split (80% train, 20% held-out test)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['target']
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n--- Split Verification ---")
    print(f"Train split saved to {train_path}: {len(train_df)} rows")
    print("Train class balance:")
    print(train_df['label'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    
    print(f"\nHeld-out Test split saved to {test_path}: {len(test_df)} rows")
    print("Test class balance:")
    print(test_df['label'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))

    print("\nDataset preparation completed successfully without data leakage.")

if __name__ == "__main__":
    prepare_dataset()
