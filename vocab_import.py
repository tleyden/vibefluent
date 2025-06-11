import pandas as pd
import requests
from io import StringIO
from typing import List, Tuple, Optional
from models import VocabWord, OnboardingData
from database import get_database
import logfire


def extract_sheet_id_from_url(url: str) -> str:
    """Extract Google Sheets ID from various URL formats."""
    # Handle different Google Sheets URL formats
    if "/spreadsheets/d/" in url:
        sheet_id = url.split("/spreadsheets/d/")[1].split("/")[0]
    elif "docs.google.com" in url:
        # Try to extract from other formats
        parts = url.split("/")
        for i, part in enumerate(parts):
            if part == "spreadsheets" and i + 2 < len(parts):
                sheet_id = parts[i + 2]
                break
        else:
            raise ValueError("Could not extract sheet ID from URL")
    else:
        # Assume it's already a sheet ID
        sheet_id = url

    return sheet_id


def download_google_sheet(url: str) -> pd.DataFrame:
    """Download Google Sheet as CSV and return as DataFrame."""
    try:
        sheet_id = extract_sheet_id_from_url(url)
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

        response = requests.get(csv_url)
        response.raise_for_status()

        # Parse CSV content
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)

        return df
    except Exception as e:
        raise Exception(f"Failed to download Google Sheet: {str(e)}")


def find_language_columns(
    df: pd.DataFrame, native_language: str, target_language: str
) -> Tuple[Optional[str], Optional[str]]:
    """Find columns that match the native and target languages."""
    columns = df.columns.tolist()

    # Convert language names to lowercase for matching
    native_lower = native_language.lower()
    target_lower = target_language.lower()

    native_col = None
    target_col = None

    # Look for exact matches first
    for col in columns:
        col_lower = col.lower()
        if native_lower in col_lower or col_lower in native_lower:
            native_col = col
        if target_lower in col_lower or col_lower in target_lower:
            target_col = col

    # If no exact matches, look for common language name variations
    language_mappings = {
        "english": ["english", "en", "eng"],
        "spanish": ["spanish", "es", "esp", "español"],
        "portuguese": ["portuguese", "pt", "port", "português"],
        "french": ["french", "fr", "français"],
        "german": ["german", "de", "deutsch"],
        "italian": ["italian", "it", "italiano"],
        "chinese": ["chinese", "zh", "mandarin", "中文"],
        "japanese": ["japanese", "ja", "日本語"],
    }

    if not native_col or not target_col:
        for col in columns:
            col_lower = col.lower()

            # Check native language mappings
            if not native_col:
                for lang, variations in language_mappings.items():
                    if native_lower in variations or lang == native_lower:
                        if any(var in col_lower for var in variations):
                            native_col = col
                            break

            # Check target language mappings
            if not target_col:
                for lang, variations in language_mappings.items():
                    if target_lower in variations or lang == target_lower:
                        if any(var in col_lower for var in variations):
                            target_col = col
                            break

    return native_col, target_col


def extract_vocab_words_from_sheet(
    df: pd.DataFrame, native_col: str, target_col: str
) -> List[VocabWord]:
    """Extract vocabulary words from the specified columns."""
    vocab_words = []

    for _, row in df.iterrows():
        # Get values and handle potential NaN/None values
        native_word = row[native_col]
        target_word = row[target_col]

        # Convert to string and check for empty/invalid values
        native_word_str = str(native_word).strip() if pd.notna(native_word) else ""
        target_word_str = str(target_word).strip() if pd.notna(target_word) else ""

        # Skip empty or invalid entries
        if (
            not native_word_str
            or not target_word_str
            or native_word_str.lower() in ["nan", "none", ""]
            or target_word_str.lower() in ["nan", "none", ""]
        ):
            continue

        vocab_word = VocabWord(
            word_in_native_language=native_word_str,
            word_in_target_language=target_word_str,
        )
        vocab_words.append(vocab_word)

    return vocab_words


def import_vocab_from_google_sheet(
    sheet_url: str, onboarding_data: OnboardingData
) -> int:
    """Import vocabulary words from Google Sheet for the given user.

    Returns the number of words imported.
    """
    try:
        logfire.info(
            "Starting vocab import from Google Sheet",
            sheet_url=sheet_url,
            user_id=onboarding_data.id,
        )

        # Download the sheet
        df = download_google_sheet(sheet_url)
        logfire.info(
            f"Downloaded sheet with {len(df)} rows and columns: {list(df.columns)}"
        )

        # Find the appropriate columns
        native_col, target_col = find_language_columns(
            df, onboarding_data.native_language, onboarding_data.target_language
        )

        if not native_col or not target_col:
            available_columns = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Could not find columns for {onboarding_data.native_language} and {onboarding_data.target_language}. "
                f"Available columns: {available_columns}"
            )

        logfire.info(f"Using columns: {native_col} (native) and {target_col} (target)")

        # Extract vocabulary words
        vocab_words = extract_vocab_words_from_sheet(df, native_col, target_col)

        logfire.info(
            f"Extracted {len(vocab_words)} vocabulary words from the sheet",
            vocab_words=vocab_words[:50],  # Log first 10 words for brevity
        )

        if not vocab_words:
            raise ValueError("No valid vocabulary words found in the sheet")

        raise RuntimeError("Dont save to db yet.")

        # Save to database
        database = get_database()
        database.save_vocab_words_for_user(vocab_words, onboarding_data)

        logfire.info(
            f"Successfully imported {len(vocab_words)} vocabulary words",
            user_id=onboarding_data.id,
            vocab_count=len(vocab_words),
        )

        return len(vocab_words)

    except Exception as e:
        logfire.error(
            f"Failed to import vocabulary from Google Sheet: {str(e)}",
            sheet_url=sheet_url,
            user_id=onboarding_data.id,
        )
        raise
