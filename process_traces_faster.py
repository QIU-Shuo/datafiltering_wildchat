#!/usr/bin/env python3
"""
Fast Trace Processing - Quality Filtering, Deduplication, and Diverse Sampling

A lightweight pipeline that applies quality filters, MinHash deduplication,
and optionally selects diverse samples using Farthest First Traversal (FFT).

Input: Parquet file with conversation data
Output: Parquet file with filter_passed, filter_reason, and optionally is_sampled columns

Output Schema (New Columns):
- filter_passed: bool        - True if passed all quality + dedup checks
- filter_reason: str | null  - Reason for filtering (null if passed)
- is_sampled: bool           - True if selected by FFT sampling (only if --sample-n specified)

FFT Sampling:
  Uses MinHash signatures to compute Jaccard distance between conversations.
  Iteratively selects the point farthest from all previously selected points,
  ensuring maximum diversity in the sample set.
"""

import json
import argparse
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from datasketch import MinHash, MinHashLSH

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Tokenization Functions
# =============================================================================

# Unicode ranges for scripts that don't use spaces between words
CHAR_SPLIT_RANGES = [
    # CJK
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x3000, 0x303F),   # CJK Punctuation
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Korean Hangul
    # Southeast Asian
    (0x0E00, 0x0E7F),   # Thai
    (0x0E80, 0x0EFF),   # Lao
    (0x1780, 0x17FF),   # Khmer
    (0x1000, 0x109F),   # Myanmar
    # Tibetan
    (0x0F00, 0x0FFF),   # Tibetan
]


def is_char_split_script(char):
    """Check if a character is from a script that needs character-level splitting."""
    cp = ord(char)
    return any(start <= cp <= end for start, end in CHAR_SPLIT_RANGES)


def tokenize_text(text):
    """
    Tokenize text for similarity comparison.

    - Lowercases text
    - Removes punctuation
    - Splits CJK characters individually
    - Splits alphabetic words by whitespace

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    if not text:
        return []

    text = text.lower()
    # Remove punctuation, keep alphanumeric and CJK characters
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)

    tokens = []
    for segment in text.split():
        if any(is_char_split_script(c) for c in segment):
            # Split CJK into individual characters, keep alphanumeric together
            current_alnum = []
            for c in segment:
                if is_char_split_script(c):
                    if current_alnum:
                        tokens.append(''.join(current_alnum))
                        current_alnum = []
                    tokens.append(c)
                elif c.isalnum():
                    current_alnum.append(c)
            if current_alnum:
                tokens.append(''.join(current_alnum))
        else:
            if segment:
                tokens.append(segment)

    return tokens


# =============================================================================
# Text Extraction Functions
# =============================================================================

def get_user_input_text(conversation):
    """Extract all user input text from a conversation."""
    if not conversation:
        return ""

    text_parts = []
    for message in conversation:
        if isinstance(message, dict) and message.get('role') == 'user':
            content = message.get('content', '')
            if isinstance(content, str):
                text_parts.append(content)

    return " ".join(text_parts)


def get_conversation_text_for_dedup(record):
    """
    Extract text for deduplication.
    Includes all messages except the last assistant response.
    """
    conversation = record.get('conversation', [])

    messages = []
    for msg in conversation:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            messages.append({'role': role, 'content': content})

    text_parts = []
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']

        if role == 'user':
            text_parts.append(content)
        elif role == 'assistant':
            is_last_message = (i == len(messages) - 1)
            if not is_last_message:
                text_parts.append(content)

    return " ".join(text_parts)


def get_user_input_for_dedup(record):
    """
    Extract only user input text for deduplication.
    Concatenates all user messages in the conversation.
    """
    conversation = record.get('conversation', [])

    user_parts = []
    for msg in conversation:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            content = msg.get('content', '').strip()
            if content:
                user_parts.append(content)

    return " ".join(user_parts)


# =============================================================================
# Quality Filtering Functions
# =============================================================================

def is_spam_pattern(text):
    """Detect common spam patterns while preserving code/technical content."""
    if not text:
        return False

    lower_text = text.lower().strip()

    # Pattern 1: Excessive repetition of same character
    # Only flag truly spammy patterns, not technical content
    # Skip: whitespace, code separators, error pointers (^), alphanumeric (paths/tokens)
    for match in re.finditer(r'(.)\1{20,}', lower_text):  # Increased threshold to 20
        char = match.group(1)
        # Skip common technical patterns
        if char.isalnum():  # Skip repeated letters/numbers (API tokens, paths)
            continue
        if char in ' \t\n\r-=_#*^~/\\|':  # Skip formatting chars
            continue
        return True

    # Pattern 2: Excessive repetition of same word (very aggressive threshold)
    words = tokenize_text(lower_text)
    if len(words) > 20:  # Only check longer texts
        word_counts = {}
        for word in words:
            if len(word) > 4:  # Only longer words
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] / len(words) > 0.7:  # 70% threshold
                    return True

    # Pattern 3: Only special characters (but not code)
    if re.match(r'^[^\w]+$', lower_text, re.UNICODE) and len(lower_text) > 20:
        if not re.search(r'[{}\[\]();:,.<>]', text):  # More code chars
            return True

    # Pattern 4: Excessive repeated punctuation (only truly spammy)
    if re.search(r'[!?]{15,}', text):  # Increased threshold
        return True

    return False


def check_quality(record) -> Tuple[bool, Optional[str]]:
    """
    Check if a conversation record passes quality filters.

    Returns:
        Tuple of (passed, reason) where reason is None if passed
    """
    conversation = record.get('conversation', [])

    # Check if conversation is empty
    if not conversation:
        return False, "empty"

    # Check for toxic messages
    for message in conversation:
        if isinstance(message, dict) and message.get('toxic', False):
            return False, "toxic"

    # Get user input text
    user_text = get_user_input_text(conversation)

    # Check if user input is empty
    if not user_text or not user_text.strip():
        return False, "empty_user_input"

    # Check if user input is too short (less than 1 word or less than 5 characters)
    stripped_text = user_text.strip()
    word_count = len(tokenize_text(stripped_text))
    if word_count < 1 or len(stripped_text) < 5:
        return False, "too_short_user_input"

    # Check for spam patterns
    if is_spam_pattern(user_text):
        return False, "spam_pattern"

    return True, None


def apply_quality_filters(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply quality filters to all records (non-destructive).
    Adds filter_passed and filter_reason to each record.

    Returns:
        Same list with added fields (no records removed)
    """
    print("Phase 1: Applying quality filters...")

    stats = defaultdict(int)

    for record in tqdm(data, desc="Quality filtering"):
        passed, reason = check_quality(record)
        record['filter_passed'] = passed
        record['filter_reason'] = reason

        if passed:
            stats['passed'] += 1
        else:
            stats[reason] += 1

    # Print statistics
    print(f"  Passed quality filters: {stats['passed']:,}")
    print(f"  Failed:")
    for reason in ['empty', 'empty_user_input', 'too_short_user_input', 'toxic', 'spam_pattern']:
        if stats[reason] > 0:
            print(f"    - {reason}: {stats[reason]:,}")

    return data


# =============================================================================
# Deduplication Functions
# =============================================================================

def create_minhash(text, num_perm=128):
    """Create MinHash signature for text."""
    m = MinHash(num_perm=num_perm)

    words = tokenize_text(text)
    ngrams = []
    for word in words:
        if len(word) >= 3:
            for i in range(len(word) - 2):
                ngrams.append(word[i:i+3])

    tokens = words + ngrams
    for token in tokens:
        m.update(token.encode('utf-8'))

    return m


def apply_deduplication(
    data: List[Dict[str, Any]],
    threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Apply two-phase MinHash LSH deduplication (non-destructive):
    1. Deduplicate based on user input (since inputs are usually small)
    2. Deduplicate based on full conversation trace

    Updates filter_passed and filter_reason for duplicates.

    Args:
        data: List of records with filter_passed field
        threshold: Jaccard similarity threshold (0.0-1.0, default 0.8)

    Returns:
        Same list with updated fields (no records removed)
    """
    print(f"Phase 2: Applying two-phase MinHash deduplication (threshold={threshold})...")

    # Phase 1: Deduplicate based on user input
    print("  Phase 2a: Deduplicating by user input...")
    user_input_duplicates = 0
    lsh_user = MinHashLSH(threshold=threshold, num_perm=128)

    for i, record in enumerate(tqdm(data, desc="User input dedup")):
        if not record['filter_passed']:
            continue

        user_text = get_user_input_for_dedup(record)
        if len(user_text.strip()) < 5:  # Skip very short user inputs
            continue

        minhash = create_minhash(user_text, num_perm=128)
        key = f"user_{i}"
        similar = lsh_user.query(minhash)

        if similar:
            record['filter_passed'] = False
            record['filter_reason'] = 'duplicate_user_input'
            user_input_duplicates += 1
        else:
            lsh_user.insert(key, minhash)

    print(f"    User input duplicates: {user_input_duplicates:,}")

    # Phase 2: Deduplicate based on full conversation (for records that passed Phase 1)
    print("  Phase 2b: Deduplicating by full conversation...")
    conversation_duplicates = 0
    lsh_conv = MinHashLSH(threshold=threshold, num_perm=128)

    for i, record in enumerate(tqdm(data, desc="Conversation dedup")):
        if not record['filter_passed']:
            continue

        conv_text = get_conversation_text_for_dedup(record)
        if len(conv_text.strip()) < 10:
            continue

        minhash = create_minhash(conv_text, num_perm=128)
        key = f"conv_{i}"
        similar = lsh_conv.query(minhash)

        if similar:
            record['filter_passed'] = False
            record['filter_reason'] = 'duplicate_conversation'
            conversation_duplicates += 1
        else:
            lsh_conv.insert(key, minhash)

    print(f"    Conversation duplicates: {conversation_duplicates:,}")

    total_duplicates = user_input_duplicates + conversation_duplicates
    passed = sum(1 for r in data if r['filter_passed'])
    print(f"  Total duplicates found: {total_duplicates:,} ({user_input_duplicates:,} user input, {conversation_duplicates:,} conversation)")
    print(f"  Records passing all filters: {passed:,}")

    return data


# =============================================================================
# Farthest First Traversal Sampling
# =============================================================================

def minhash_distance(mh1: MinHash, mh2: MinHash) -> float:
    """
    Compute distance between two MinHash signatures.
    Distance = 1 - Jaccard similarity estimate.
    """
    return 1.0 - mh1.jaccard(mh2)


def farthest_first_traversal(
    data: List[Dict[str, Any]],
    n: int,
    num_perm: int = 128,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample n records using Farthest First Traversal based on MinHash signatures.

    This method iteratively selects points that are maximally distant from
    all previously selected points, ensuring diverse coverage of the dataset.

    Args:
        data: List of records with filter_passed field
        n: Target number of samples to select
        num_perm: Number of permutations for MinHash (default 128)
        seed: Random seed for initial point selection

    Returns:
        Same list with added 'is_sampled' field
    """
    print(f"Phase 3: Farthest First Traversal sampling (target={n})...")

    # Get indices of records that passed all filters
    passed_indices = [i for i, r in enumerate(data) if r.get('filter_passed', False)]

    if len(passed_indices) == 0:
        print("  Warning: No records passed filters, skipping sampling")
        for record in data:
            record['is_sampled'] = False
        return data

    print(f"  Records available for sampling: {len(passed_indices):,}")

    # If we want more samples than available, select all
    if n >= len(passed_indices):
        print(f"  Target ({n}) >= available ({len(passed_indices)}), selecting all")
        for i, record in enumerate(data):
            record['is_sampled'] = record.get('filter_passed', False)
        return data

    # Generate MinHash signatures for all passed records
    print("  Generating MinHash signatures...")
    minhashes = []
    for idx in tqdm(passed_indices, desc="Computing MinHash"):
        record = data[idx]
        text = get_conversation_text_for_dedup(record)
        mh = create_minhash(text, num_perm=num_perm)
        minhashes.append(mh)

    # Initialize FFT
    np.random.seed(seed)

    # Start with a random point
    first_idx = np.random.randint(0, len(passed_indices))
    selected_local = [first_idx]  # Indices into passed_indices/minhashes

    # Track minimum distance from each point to the selected set
    min_distances = np.full(len(passed_indices), np.inf)

    print(f"  Running Farthest First Traversal...")
    for _ in tqdm(range(n - 1), desc="FFT sampling"):
        last_selected = selected_local[-1]
        last_mh = minhashes[last_selected]

        # Update min distances with distance to the newly selected point
        for i in range(len(passed_indices)):
            if min_distances[i] > 0:  # Skip already selected (distance = 0)
                dist = minhash_distance(last_mh, minhashes[i])
                if dist < min_distances[i]:
                    min_distances[i] = dist

        # Mark selected point with distance 0 so it won't be picked again
        min_distances[last_selected] = -1

        # Find the point with maximum minimum distance
        farthest_idx = np.argmax(min_distances)
        selected_local.append(farthest_idx)

    # Convert local indices back to global indices
    selected_global = set(passed_indices[i] for i in selected_local)

    # Mark sampled records
    for i, record in enumerate(data):
        record['is_sampled'] = i in selected_global

    sampled_count = sum(1 for r in data if r['is_sampled'])
    print(f"  Sampled {sampled_count:,} records using FFT")

    return data


# =============================================================================
# I/O Functions
# =============================================================================

def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data from Parquet file or directory."""
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading data from {input_file}...")

    if input_path.is_dir():
        dataset = load_dataset('parquet', data_dir=input_file)['train']
    else:
        dataset = load_dataset('parquet', data_files=input_file)['train']

    data = [dict(record) for record in dataset]

    print(f"Loaded {len(data):,} records")
    return data


def save_data(data: List[Dict[str, Any]], output_file: str):
    """Save data to Parquet file."""
    print(f"\nSaving data to {output_file}...")

    dataset = Dataset.from_list(data)
    dataset.to_parquet(output_file)

    print(f"Successfully saved {len(data):,} records to {output_file}")


def save_statistics(data: List[Dict[str, Any]], output_file: str):
    """Save pipeline statistics to JSON file."""
    output_path = Path(output_file)
    stats_file = output_path.parent / f"{output_path.stem}_stats.json"

    # Compute filter statistics
    filter_stats = defaultdict(int)
    for record in data:
        if record['filter_passed']:
            filter_stats['passed'] += 1
        else:
            filter_stats[record['filter_reason']] += 1

    stats = {
        'total_records': len(data),
        'filter_statistics': dict(filter_stats),
        'pass_rate': filter_stats['passed'] / len(data) * 100 if data else 0
    }

    # Add sampling statistics if available
    if any('is_sampled' in r for r in data):
        sampled_count = sum(1 for r in data if r.get('is_sampled', False))
        stats['sampling_statistics'] = {
            'sampled_count': sampled_count,
            'sample_rate': sampled_count / filter_stats['passed'] * 100 if filter_stats['passed'] > 0 else 0
        }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics to {stats_file}")


# =============================================================================
# Main Pipeline Function
# =============================================================================

def process_traces(
    input_file: str,
    output_file: str,
    dedup_threshold: float = 0.8,
    sample_n: Optional[int] = None,
    seed: int = 42
):
    """
    Run the fast processing pipeline.

    Phases:
    1. Load data
    2. Quality filtering (adds filter_passed, filter_reason)
    3. MinHash deduplication (updates filter_passed, filter_reason for duplicates)
    4. (Optional) FFT sampling (adds is_sampled if sample_n is specified)
    5. Save complete dataset
    """
    print("=" * 60)
    print("FAST TRACE PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Dedup threshold: {dedup_threshold}")
    if sample_n is not None:
        print(f"Sample target: {sample_n}")
    print("=" * 60)

    # Phase 1: Load data
    data = load_data(input_file)
    initial_count = len(data)

    # Phase 2: Quality filtering
    data = apply_quality_filters(data)

    # Phase 3: MinHash Deduplication
    data = apply_deduplication(data, dedup_threshold)

    # Phase 4: FFT Sampling (optional)
    if sample_n is not None:
        data = farthest_first_traversal(data, n=sample_n, seed=seed)

    # Phase 5: Save
    save_data(data, output_file)
    save_statistics(data, output_file)

    # Final summary
    passed = sum(1 for r in data if r['filter_passed'])
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total records: {len(data):,}")
    print(f"Filter passed: {passed:,} ({passed / len(data) * 100:.1f}%)")
    print(f"Filter failed: {len(data) - passed:,} ({(len(data) - passed) / len(data) * 100:.1f}%)")
    if sample_n is not None:
        sampled = sum(1 for r in data if r.get('is_sampled', False))
        print(f"Sampled (FFT): {sampled:,}")
    print(f"Output: {output_file}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fast trace processing: quality filter, deduplicate, and sample',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (filter + dedup only)
  python process_traces_faster.py -i input.parquet -o output.parquet

  # Custom deduplication threshold
  python process_traces_faster.py -i input.parquet -o output.parquet --dedup-threshold 0.85

  # With FFT sampling to select exactly 1000 diverse samples
  python process_traces_faster.py -i input.parquet -o output.parquet --sample-n 1000

  # Full pipeline with custom parameters
  python process_traces_faster.py -i input.parquet -o output.parquet \\
      --dedup-threshold 0.85 --sample-n 500 --seed 123
        """
    )

    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input parquet file or directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Output parquet file')

    # Filtering arguments
    parser.add_argument('--dedup-threshold', type=float, default=0.8,
                        help='MinHash deduplication threshold (Jaccard similarity 0.0-1.0, default: 0.8)')

    # Sampling arguments
    parser.add_argument('--sample-n', type=int, default=None,
                        help='Target number of samples to select using Farthest First Traversal. '
                             'If not specified, no sampling is performed.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for FFT initial point selection (default: 42)')

    args = parser.parse_args()

    process_traces(
        input_file=args.input,
        output_file=args.output,
        dedup_threshold=args.dedup_threshold,
        sample_n=args.sample_n,
        seed=args.seed
    )
