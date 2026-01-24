import json
from pathlib import Path
import re
import hashlib
import argparse
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from datasets import load_dataset, Dataset
from tqdm import tqdm
import datetime

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types."""
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return super().default(obj)

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

def get_conversation_text(conversation):
    """Extract all text from a conversation."""
    if not conversation:
        return ""

    text_parts = []
    for message in conversation:
        if isinstance(message, dict) and 'content' in message:
            content = message['content']
            if isinstance(content, str):
                text_parts.append(content)

    return " ".join(text_parts)

def is_spam_pattern(text):
    """Detect common spam patterns."""
    if not text:
        return False

    # Convert to lowercase for pattern matching
    lower_text = text.lower().strip()

    # Pattern 1: Excessive repetition of same character (e.g., "aaaaaaa", "????")
    if re.search(r'(.)\1{10,}', lower_text):
        return True

    # Pattern 2: Excessive repetition of same word (more strict)
    words = lower_text.split()
    if len(words) > 10:  # Only check longer texts
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count words longer than 3 chars
                word_counts[word] = word_counts.get(word, 0) + 1
                # If any word appears more than 60% of the time in long text
                if word_counts[word] / len(words) > 0.6:
                    return True

    # Pattern 3: Only special characters or numbers (no alphanumeric at all)
    # But allow non-Latin scripts (Chinese, Japanese, Korean, Cyrillic, etc.)
    # Only filter if it's purely punctuation/symbols
    if re.match(r'^[^\w]+$', lower_text, re.UNICODE) and len(lower_text) > 10:
        return True

    # Pattern 4: Excessive use of same punctuation (e.g., "!!!!!!!!!", "?????????")
    if re.search(r'[!?.,]{10,}', text):
        return True

    # Pattern 5: Very long repetitive instructions (likely prompt injection attempts)
    # Check if text is very long and has many repetitive lines
    lines = text.split('\n')
    if len(text) > 2000 and len(lines) > 20:
        # Count similar lines
        similar_lines = 0
        for i in range(len(lines) - 1):
            for j in range(i + 1, min(i + 10, len(lines))):  # Check next 10 lines
                if lines[i].strip() and len(lines[i]) > 20:
                    # Check if lines are very similar (share many words)
                    words_i = set(lines[i].lower().split())
                    words_j = set(lines[j].lower().split())
                    if len(words_i) > 0 and len(words_j) > 0:
                        similarity = len(words_i & words_j) / max(len(words_i), len(words_j))
                        if similarity > 0.7:
                            similar_lines += 1

        # If more than 30% of line pairs are similar, likely spam/template
        if similar_lines > len(lines) * 0.3:
            return True

    return False

def get_conversation_fingerprint(record):
    """
    Generate a fingerprint for exact deduplication.

    Fingerprint includes:
    - All user messages (in order)
    - All assistant messages EXCEPT the last one (in order)
    - Conversation length (number of turns)

    Two conversations are duplicates if they have the same context leading up to
    the final assistant response - i.e., same evaluation scenario.
    """
    conversation = record.get('conversation', [])

    messages = []
    for msg in conversation:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            messages.append({'role': role, 'content': content})

    # Build fingerprint components
    fingerprint_parts = []

    # Add all messages except the last assistant message
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']

        # Always include user messages
        if role == 'user':
            fingerprint_parts.append(f"USER:{content}")
        # Include assistant messages except the last one
        elif role == 'assistant':
            # Check if this is the last message
            is_last_message = (i == len(messages) - 1)
            if not is_last_message:
                fingerprint_parts.append(f"ASSISTANT:{content}")

    # Add conversation length
    fingerprint_parts.append(f"LENGTH:{len(messages)}")

    # Create hash
    fingerprint_string = "|||".join(fingerprint_parts)
    return hashlib.md5(fingerprint_string.encode('utf-8')).hexdigest()

def get_conversation_text_for_dedup(record):
    """
    Extract text for near-deduplication (MinHash/SimHash).
    Same logic as exact dedup: all messages except the last assistant response.
    """
    conversation = record.get('conversation', [])

    messages = []
    for msg in conversation:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            messages.append({'role': role, 'content': content})

    # Collect text parts (same as fingerprint but return raw text)
    text_parts = []

    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']

        # Always include user messages
        if role == 'user':
            text_parts.append(content)
        # Include assistant messages except the last one
        elif role == 'assistant':
            is_last_message = (i == len(messages) - 1)
            if not is_last_message:
                text_parts.append(content)

    return " ".join(text_parts)

def create_minhash(text, num_perm=128):
    """Create MinHash signature for text."""
    m = MinHash(num_perm=num_perm)

    # Tokenize text into words
    words = text.lower().split()

    # Also create character n-grams for better similarity detection
    # Use 3-grams
    ngrams = []
    for word in words:
        if len(word) >= 3:
            for i in range(len(word) - 2):
                ngrams.append(word[i:i+3])

    # Add both words and n-grams
    tokens = words + ngrams

    for token in tokens:
        m.update(token.encode('utf-8'))

    return m

def simhash(text, hash_bits=64):
    """
    Create SimHash signature for text.

    SimHash creates a fingerprint where similar documents have similar hashes
    (measured by Hamming distance).
    """
    # Tokenize
    words = text.lower().split()

    # Create character n-grams
    ngrams = []
    for word in words:
        if len(word) >= 3:
            for i in range(len(word) - 2):
                ngrams.append(word[i:i+3])

    tokens = words + ngrams

    # Initialize vector
    vector = [0] * hash_bits

    # For each token, hash it and update the vector
    for token in tokens:
        # Get hash of token
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)

        # Update vector based on each bit
        for i in range(hash_bits):
            if h & (1 << i):
                vector[i] += 1
            else:
                vector[i] -= 1

    # Create final fingerprint
    fingerprint = 0
    for i in range(hash_bits):
        if vector[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hashes."""
    x = hash1 ^ hash2
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance

def deduplicate_exact(records):
    """Exact deduplication based on fingerprint."""
    seen_fingerprints = set()
    deduplicated = []
    duplicates_list = []
    duplicates = 0

    for record in tqdm(records, desc="Exact dedup"):
        fingerprint = get_conversation_fingerprint(record)

        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            deduplicated.append(record)
        else:
            duplicates += 1
            record_with_reason = record.copy()
            record_with_reason['filter_reason'] = 'duplicate_exact'
            duplicates_list.append(record_with_reason)

    return deduplicated, duplicates, duplicates_list

def deduplicate_minhash(records, threshold=0.8, num_perm=128):
    """
    Near-deduplication using MinHash LSH.

    Args:
        records: List of conversation records
        threshold: Jaccard similarity threshold (0.0-1.0)
        num_perm: Number of permutations for MinHash

    Returns:
        deduplicated records, number of duplicates, list of duplicate records
    """
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    deduplicated = []
    duplicates_list = []
    duplicates = 0

    for i, record in enumerate(tqdm(records, desc="MinHash dedup")):
        text = get_conversation_text_for_dedup(record)

        # Skip if text is too short
        if len(text.strip()) < 10:
            deduplicated.append(record)
            continue

        # Create MinHash
        minhash = create_minhash(text, num_perm=num_perm)

        # Query for similar items
        key = f"record_{i}"
        similar = lsh.query(minhash)

        if not similar:
            # No similar items found, add to index and keep
            lsh.insert(key, minhash)
            deduplicated.append(record)
        else:
            # Similar item found, mark as duplicate
            duplicates += 1
            record_with_reason = record.copy()
            record_with_reason['filter_reason'] = 'duplicate_minhash'
            duplicates_list.append(record_with_reason)

    return deduplicated, duplicates, duplicates_list

def deduplicate_simhash(records, threshold=3, hash_bits=64):
    """
    Near-deduplication using SimHash.

    Args:
        records: List of conversation records
        threshold: Maximum Hamming distance to consider duplicates
        hash_bits: Number of bits in hash

    Returns:
        deduplicated records, number of duplicates, list of duplicate records
    """
    deduplicated = []
    duplicates_list = []
    duplicates = 0
    seen_hashes = []  # List of (hash, index) tuples

    for i, record in enumerate(tqdm(records, desc="SimHash dedup")):
        text = get_conversation_text_for_dedup(record)

        # Skip if text is too short
        if len(text.strip()) < 10:
            deduplicated.append(record)
            continue

        # Create SimHash
        hash_value = simhash(text, hash_bits=hash_bits)

        # Check against existing hashes
        is_duplicate = False
        for existing_hash, _ in seen_hashes:
            distance = hamming_distance(hash_value, existing_hash)
            if distance <= threshold:
                is_duplicate = True
                duplicates += 1
                record_with_reason = record.copy()
                record_with_reason['filter_reason'] = 'duplicate_simhash'
                duplicates_list.append(record_with_reason)
                break

        if not is_duplicate:
            seen_hashes.append((hash_value, i))
            deduplicated.append(record)

    return deduplicated, duplicates, duplicates_list

def is_valid_conversation(record):
    """Check if a conversation record is valid (not empty, not too short, not toxic, not spam)."""
    # Get conversation
    conversation = record.get('conversation', [])

    # Check if conversation is empty
    if not conversation:
        return False, "empty"

    # Check if any message in the conversation is marked as toxic
    for message in conversation:
        if isinstance(message, dict):
            if message.get('toxic', False):
                return False, "toxic"

    # Get user input text specifically
    user_text = get_user_input_text(conversation)

    # Check if user input is empty
    if not user_text or not user_text.strip():
        return False, "empty_user_input"

    # Check if user input is less than 3 characters
    if len(user_text.strip()) < 3:
        return False, "too_short_user_input"

    # Check for spam patterns in user input
    if is_spam_pattern(user_text):
        return False, "spam_pattern"

    return True, "valid"

def filter_wildchat(input_file, output_file, dedup_method='minhash', dedup_threshold=0.8, input_format='auto', top_k=None):
    """
    Filter the WildChat dataset and report statistics.

    Args:
        input_file: Path to input JSON file or directory containing Parquet files
        output_file: Path to output JSON or Parquet file
        dedup_method: Deduplication method ('exact', 'minhash', or 'simhash')
        dedup_threshold: Threshold for near-deduplication
            - For 'minhash': Jaccard similarity threshold (0.0-1.0, default 0.8)
            - For 'simhash': Maximum Hamming distance (integer, default 3)
        input_format: Input format ('json', 'parquet', or 'auto' to detect)
        top_k: Only process the first k records from input (None = process all)
    """

    # Detect input format if auto
    input_path = Path(input_file)
    if input_format == 'auto':
        if input_path.is_dir():
            # Check if directory contains parquet files
            if list(input_path.glob('*.parquet')):
                input_format = 'parquet'
            else:
                raise ValueError(f"No parquet files found in directory: {input_file}")
        elif input_path.suffix == '.json':
            input_format = 'json'
        elif input_path.suffix == '.parquet':
            input_format = 'parquet'
        else:
            raise ValueError(f"Cannot detect format for: {input_file}. Use --input-format to specify.")

    # Load the data
    print(f"Loading data from {input_file} (format: {input_format})...")

    if input_format == 'json':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif input_format == 'parquet':
        if input_path.is_dir():
            dataset = load_dataset('parquet', data_dir=input_file)['train']
        else:
            dataset = load_dataset('parquet', data_files=input_file)['train']
        # Convert to list of dictionaries
        data = list(dataset)
        print(f"Loaded {len(data)} records from Parquet dataset")

    # Apply top_k limit if specified
    if top_k is not None and top_k > 0:
        original_count = len(data)
        data = data[:top_k]
        print(f"Limited to first {len(data)} records (from {original_count} total)")

    # Statistics tracking
    stats = {
        'total_before': len(data),
        'empty': 0,
        'empty_user_input': 0,
        'too_short_user_input': 0,
        'toxic': 0,
        'spam_pattern': 0,
        'duplicates': 0,
        'valid': 0
    }

    # Filtered out records by category
    filtered_records = {
        'empty': [],
        'empty_user_input': [],
        'too_short_user_input': [],
        'toxic': [],
        'spam_pattern': [],
        'duplicates': []
    }

    # Phase 1: Filter by quality criteria
    print("Phase 1: Filtering by quality criteria...")
    quality_filtered = []

    for record in tqdm(data, desc="Quality filtering"):
        is_valid, reason = is_valid_conversation(record)

        if is_valid:
            quality_filtered.append(record)
        else:
            # Add filter reason to the record
            record_with_reason = record.copy()
            record_with_reason['filter_reason'] = reason
            filtered_records[reason].append(record_with_reason)
            stats[reason] += 1

    print(f"  After quality filtering: {len(quality_filtered)} records")

    # Phase 2: Deduplication
    print(f"Phase 2: Deduplicating conversations using {dedup_method}...")

    if dedup_method == 'exact':
        deduplicated_data, num_duplicates, duplicate_records = deduplicate_exact(quality_filtered)
        print(f"  Using exact matching")
    elif dedup_method == 'minhash':
        deduplicated_data, num_duplicates, duplicate_records = deduplicate_minhash(
            quality_filtered,
            threshold=dedup_threshold,
            num_perm=128
        )
        print(f"  Using MinHash LSH with threshold={dedup_threshold}")
    elif dedup_method == 'simhash':
        # Convert threshold to int for simhash
        simhash_threshold = int(dedup_threshold) if dedup_method == 'simhash' else 3
        deduplicated_data, num_duplicates, duplicate_records = deduplicate_simhash(
            quality_filtered,
            threshold=simhash_threshold,
            hash_bits=64
        )
        print(f"  Using SimHash with Hamming distance threshold={simhash_threshold}")
    else:
        raise ValueError(f"Unknown dedup_method: {dedup_method}")

    # Add duplicate records to filtered_records
    filtered_records['duplicates'] = duplicate_records
    stats['duplicates'] = num_duplicates
    stats['valid'] = len(deduplicated_data)
    stats['total_after'] = len(deduplicated_data)

    # Calculate total filtered
    total_filtered = (stats['empty'] + stats['empty_user_input'] +
                     stats['too_short_user_input'] + stats['toxic'] +
                     stats['spam_pattern'] + stats['duplicates'])

    # Report statistics
    print("\n" + "="*60)
    print("FILTERING STATISTICS")
    print("="*60)
    print(f"Total records before filtering:  {stats['total_before']:,}")
    print(f"\nFiltered out:")
    print(f"  - Empty conversations:          {stats['empty']:,}")
    print(f"  - Empty user input:             {stats['empty_user_input']:,}")
    print(f"  - User input < 3 chars:         {stats['too_short_user_input']:,}")
    print(f"  - Toxic conversations:          {stats['toxic']:,}")
    print(f"  - Spam patterns detected:       {stats['spam_pattern']:,}")
    print(f"  - Duplicate conversations:      {stats['duplicates']:,} ({dedup_method})")
    print(f"\nTotal filtered out:              {total_filtered:,}")
    print(f"Total records after filtering:   {stats['total_after']:,}")
    print(f"\nRetention rate:                  {stats['total_after'] / stats['total_before'] * 100:.2f}%")
    print("="*60)

    # Save filtered data
    output_path = Path(output_file)
    output_format = 'json' if output_path.suffix == '.json' else 'parquet'

    print(f"\nSaving filtered data to {output_file} (format: {output_format})...")

    if output_format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_data, f, indent=2, ensure_ascii=False)
    else:
        # Save as Parquet using Hugging Face datasets
        dataset = Dataset.from_list(deduplicated_data)
        dataset.to_parquet(output_file)

    print(f"✓ Successfully saved {stats['total_after']:,} records to {output_file}")

    # Save filtered-out records to JSON files
    output_dir = output_path.parent
    output_stem = output_path.stem

    print(f"\nSaving filtered-out records to JSON files...")

    filtered_out_saved = 0
    for reason, records in filtered_records.items():
        if records:
            filtered_out_file = output_dir / f"{output_stem}_filtered_{reason}.json"
            with open(filtered_out_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False, cls=JSONEncoder)
            print(f"  ✓ Saved {len(records):,} {reason} records to {filtered_out_file}")
            filtered_out_saved += len(records)

    print(f"\n✓ Total filtered-out records saved: {filtered_out_saved:,}")

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Filter WildChat dataset by quality and deduplication',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full WildChat-1M Parquet dataset with default MinHash
  python filter_wildchat.py -i WildChat-1M/data -o wildchat_filtered.parquet

  # Use exact deduplication
  python filter_wildchat.py --dedup-method exact

  # Use MinHash with custom threshold
  python filter_wildchat.py --dedup-method minhash --dedup-threshold 0.9

  # Use SimHash with Hamming distance threshold of 5
  python filter_wildchat.py --dedup-method simhash --dedup-threshold 5

  # Process JSON file (legacy)
  python filter_wildchat.py -i input.json -o output.json
        """
    )

    parser.add_argument(
        '-i', '--input',
        default='/Users/qiushuo/wsp/datafiltering/WildChat-1M/data',
        help='Input file/directory path. Can be: JSON file, Parquet file, or directory with Parquet files (default: WildChat-1M/data)'
    )

    parser.add_argument(
        '-o', '--output',
        default='/Users/qiushuo/wsp/datafiltering/wildchat_filtered.parquet',
        help='Output file path. Extension determines format: .json or .parquet (default: wildchat_filtered.parquet)'
    )

    parser.add_argument(
        '--input-format',
        choices=['auto', 'json', 'parquet'],
        default='auto',
        help='Input format (default: auto-detect from file extension)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Only process the first k records from input (default: None, process all records)'
    )

    parser.add_argument(
        '--dedup-method',
        choices=['exact', 'minhash', 'simhash'],
        default='minhash',
        help='Deduplication method (default: minhash)'
    )

    parser.add_argument(
        '--dedup-threshold',
        type=float,
        default=0.8,
        help='Deduplication threshold. For minhash: Jaccard similarity 0.0-1.0 (default: 0.8). For simhash: Hamming distance (default: 3)'
    )

    args = parser.parse_args()

    filter_wildchat(
        input_file=args.input,
        output_file=args.output,
        dedup_method=args.dedup_method,
        dedup_threshold=args.dedup_threshold,
        input_format=args.input_format,
        top_k=args.top_k
    )
