#!/usr/bin/env python3
"""
Process Pre-Sampled WildChat Data

Takes data from sample_data.py (with embeddings and scores already computed)
and applies filtering, deduplication, and clustering phases.

Input: Parquet file with 'embedding' and 'score_*' columns from sample_data.py
Output: Parquet file with additional columns for filtering and clustering

Output Schema (New Columns):
- filter_passed: bool        - True if passed all quality + dedup checks
- filter_reason: str | null  - Reason for filtering (null if passed)
- cluster_id: int            - Cluster assignment (-1 for noise)
- cluster_probability: float - HDBSCAN probability
- is_sampled: bool           - True if selected (1 per cluster + noise to fill quota)
"""

import json
import argparse
import hashlib
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from datasketch import MinHash, MinHashLSH
import hdbscan

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


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
    Extract text for embedding/deduplication.
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


# =============================================================================
# Quality Filtering Functions (Non-destructive)
# =============================================================================

def is_spam_pattern(text):
    """Detect common spam patterns."""
    if not text:
        return False

    lower_text = text.lower().strip()

    # Pattern 1: Excessive repetition of same character
    if re.search(r'(.)\1{10,}', lower_text):
        return True

    # Pattern 2: Excessive repetition of same word
    words = lower_text.split()
    if len(words) > 10:
        word_counts = {}
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] / len(words) > 0.6:
                    return True

    # Pattern 3: Only special characters or numbers
    if re.match(r'^[^\w]+$', lower_text, re.UNICODE) and len(lower_text) > 10:
        return True

    # Pattern 4: Excessive use of same punctuation
    if re.search(r'[!?.,]{10,}', text):
        return True

    # Pattern 5: Very long repetitive instructions
    lines = text.split('\n')
    if len(text) > 2000 and len(lines) > 20:
        similar_lines = 0
        for i in range(len(lines) - 1):
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[i].strip() and len(lines[i]) > 20:
                    words_i = set(lines[i].lower().split())
                    words_j = set(lines[j].lower().split())
                    if len(words_i) > 0 and len(words_j) > 0:
                        similarity = len(words_i & words_j) / max(len(words_i), len(words_j))
                        if similarity > 0.7:
                            similar_lines += 1
        if similar_lines > len(lines) * 0.3:
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

    # Check if user input is too short
    if len(user_text.strip()) < 3:
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
# Deduplication Functions (Non-destructive)
# =============================================================================

def get_conversation_fingerprint(record):
    """Generate MD5 fingerprint for exact deduplication."""
    conversation = record.get('conversation', [])

    messages = []
    for msg in conversation:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            messages.append({'role': role, 'content': content})

    fingerprint_parts = []
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']

        if role == 'user':
            fingerprint_parts.append(f"USER:{content}")
        elif role == 'assistant':
            is_last_message = (i == len(messages) - 1)
            if not is_last_message:
                fingerprint_parts.append(f"ASSISTANT:{content}")

    fingerprint_parts.append(f"LENGTH:{len(messages)}")
    fingerprint_string = "|||".join(fingerprint_parts)
    return hashlib.md5(fingerprint_string.encode('utf-8')).hexdigest()


def create_minhash(text, num_perm=128):
    """Create MinHash signature for text."""
    m = MinHash(num_perm=num_perm)

    words = text.lower().split()
    ngrams = []
    for word in words:
        if len(word) >= 3:
            for i in range(len(word) - 2):
                ngrams.append(word[i:i+3])

    tokens = words + ngrams
    for token in tokens:
        m.update(token.encode('utf-8'))

    return m


def simhash(text, hash_bits=64):
    """Create SimHash signature for text."""
    words = text.lower().split()

    ngrams = []
    for word in words:
        if len(word) >= 3:
            for i in range(len(word) - 2):
                ngrams.append(word[i:i+3])

    tokens = words + ngrams
    vector = [0] * hash_bits

    for token in tokens:
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        for i in range(hash_bits):
            if h & (1 << i):
                vector[i] += 1
            else:
                vector[i] -= 1

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


def apply_deduplication(
    data: List[Dict[str, Any]],
    method: str = 'minhash',
    threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Apply deduplication to records that passed quality filters (non-destructive).
    Updates filter_passed and filter_reason for duplicates.

    Args:
        data: List of records with filter_passed field
        method: 'exact', 'minhash', or 'simhash'
        threshold: Similarity threshold (0.0-1.0 for minhash, int for simhash)

    Returns:
        Same list with updated fields (no records removed)
    """
    print(f"Phase 2: Applying {method} deduplication...")

    duplicates = 0

    if method == 'exact':
        seen_fingerprints = set()

        for record in tqdm(data, desc="Exact dedup"):
            if not record['filter_passed']:
                continue

            fingerprint = get_conversation_fingerprint(record)
            if fingerprint in seen_fingerprints:
                record['filter_passed'] = False
                record['filter_reason'] = 'duplicate_exact'
                duplicates += 1
            else:
                seen_fingerprints.add(fingerprint)

    elif method == 'minhash':
        lsh = MinHashLSH(threshold=threshold, num_perm=128)

        for i, record in enumerate(tqdm(data, desc="MinHash dedup")):
            if not record['filter_passed']:
                continue

            text = get_conversation_text_for_dedup(record)
            if len(text.strip()) < 10:
                continue

            minhash = create_minhash(text, num_perm=128)
            key = f"record_{i}"
            similar = lsh.query(minhash)

            if similar:
                record['filter_passed'] = False
                record['filter_reason'] = 'duplicate_minhash'
                duplicates += 1
            else:
                lsh.insert(key, minhash)

    elif method == 'simhash':
        simhash_threshold = int(threshold) if threshold > 1 else 3
        seen_hashes = []

        for record in tqdm(data, desc="SimHash dedup"):
            if not record['filter_passed']:
                continue

            text = get_conversation_text_for_dedup(record)
            if len(text.strip()) < 10:
                continue

            hash_value = simhash(text, hash_bits=64)
            is_duplicate = False

            for existing_hash in seen_hashes:
                distance = hamming_distance(hash_value, existing_hash)
                if distance <= simhash_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                record['filter_passed'] = False
                record['filter_reason'] = 'duplicate_simhash'
                duplicates += 1
            else:
                seen_hashes.append(hash_value)

    else:
        raise ValueError(f"Unknown dedup method: {method}")

    passed = sum(1 for r in data if r['filter_passed'])
    print(f"  Duplicates found: {duplicates:,}")
    print(f"  Records passing all filters: {passed:,}")

    return data


# =============================================================================
# Clustering Functions
# =============================================================================

def compute_auto_cluster_params(
    n_records: int,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None
) -> Tuple[int, int]:
    """
    Automatically determine HDBSCAN parameters based on dataset size.

    Heuristics:
    - min_cluster_size: log(n) + 2, clamped to [3, 20]
    - min_samples: log(n) + 1, clamped to [2, 10]

    Args:
        n_records: Number of records to cluster
        min_cluster_size: Override value (None for auto)
        min_samples: Override value (None for auto)

    Returns:
        Tuple of (min_cluster_size, min_samples)
    """
    log_n = np.log10(max(1, n_records))  # Avoid log(0)

    if min_cluster_size is None:
        # Use log(n) + 2 as default, clamped to [3, 20]
        auto_min_cluster_size = int(log_n + 2)
        auto_min_cluster_size = max(3, min(20, auto_min_cluster_size))
    else:
        auto_min_cluster_size = min_cluster_size

    if min_samples is None:
        # Use log(n) + 1 as default, clamped to [2, 10]
        auto_min_samples = int(log_n + 1)
        auto_min_samples = max(2, min(10, auto_min_samples))
    else:
        auto_min_samples = min_samples

    return auto_min_cluster_size, auto_min_samples


def apply_clustering(
    data: List[Dict[str, Any]],
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    metric: str = 'euclidean',
    cluster_selection_method: str = 'eom'
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply HDBSCAN clustering to ALL records using their embeddings.

    Args:
        data: List of records with embedding field
        min_cluster_size: Minimum cluster size (None for auto)
        min_samples: Minimum samples (None for auto)
        metric: Distance metric for HDBSCAN
        cluster_selection_method: 'eom' or 'leaf'

    Returns:
        Tuple of (data with cluster fields, clustering stats)
    """
    print("Phase 3: Clustering ALL records...")

    # Track if parameters were auto-computed
    auto_cluster_size = min_cluster_size is None
    auto_samples = min_samples is None

    # Auto-determine parameters if not specified
    min_cluster_size, min_samples = compute_auto_cluster_params(
        n_records=len(data),
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    print(f"  min_cluster_size: {min_cluster_size}{' (auto)' if auto_cluster_size else ''}")
    print(f"  min_samples: {min_samples}{' (auto)' if auto_samples else ''}")

    # Extract embeddings from records
    embeddings = np.array([record['embedding'] for record in data])

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1
    )

    cluster_labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    # Add cluster info to records
    for record, label, prob in zip(data, cluster_labels, probabilities):
        record['cluster_id'] = int(label)
        record['cluster_probability'] = float(prob)

    # Compute statistics
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels[unique_labels >= 0])
    num_noise = int(np.sum(cluster_labels == -1))

    try:
        dbcv_score = float(clusterer.relative_validity_)
    except AttributeError:
        dbcv_score = None

    stats = {
        'num_clusters': num_clusters,
        'num_noise': num_noise,
        'noise_percentage': num_noise / len(cluster_labels) * 100,
        'dbcv_score': dbcv_score
    }

    print(f"  Clusters: {num_clusters}")
    print(f"  Noise points: {num_noise:,} ({stats['noise_percentage']:.2f}%)")
    if dbcv_score is not None:
        print(f"  DBCV score: {dbcv_score:.4f}")

    return data, stats


# =============================================================================
# Sampling Functions
# =============================================================================

def compute_combined_score(record: Dict[str, Any]) -> float:
    """Compute combined score from individual score dimensions."""
    scores = []
    for dim in ['difficulty', 'creativity', 'realism']:
        score = record.get(f'score_{dim}')
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def apply_sampling(
    data: List[Dict[str, Any]],
    total_samples: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply sampling strategy after clustering:
    1. Pick 1 record with highest score from each cluster (excluding noise)
    2. Fill remaining quota from noise points (cluster_id == -1)

    Only considers records that passed filtering (filter_passed=True).

    Args:
        data: List of records with cluster_id and filter_passed fields
        total_samples: Total number of samples to select (None = 1 per cluster + all noise)

    Returns:
        Tuple of (data with is_sampled field, sampling stats)
    """
    print("Phase 4: Sampling records...")

    # Initialize is_sampled to False for all records
    for record in data:
        record['is_sampled'] = False

    # Group filtered records by cluster
    clusters = defaultdict(list)
    noise_records = []

    for i, record in enumerate(data):
        if not record.get('filter_passed', False):
            continue

        cluster_id = record.get('cluster_id', -1)
        if cluster_id == -1:
            noise_records.append(i)
        else:
            clusters[cluster_id].append(i)

    # Step 1: Pick 1 highest-scoring record from each cluster
    sampled_indices = []
    for cluster_id, indices in clusters.items():
        # Find record with highest combined score
        best_idx = max(indices, key=lambda i: compute_combined_score(data[i]))
        sampled_indices.append(best_idx)
        data[best_idx]['is_sampled'] = True

    num_from_clusters = len(sampled_indices)
    print(f"  Selected {num_from_clusters} records from {len(clusters)} clusters (1 per cluster)")

    # Step 2: Fill remaining quota from noise points
    if total_samples is not None:
        remaining = total_samples - num_from_clusters
        if remaining > 0:
            # Sort noise records by score (highest first)
            noise_records_sorted = sorted(
                noise_records,
                key=lambda i: compute_combined_score(data[i]),
                reverse=True
            )
            # Take top N from noise
            noise_to_sample = noise_records_sorted[:remaining]
            for idx in noise_to_sample:
                data[idx]['is_sampled'] = True
                sampled_indices.append(idx)
            print(f"  Selected {len(noise_to_sample)} additional records from noise points")
        elif remaining < 0:
            print(f"  Warning: total_samples ({total_samples}) < clusters ({num_from_clusters}), no noise sampling")
    else:
        # If no total_samples specified, include all noise points
        for idx in noise_records:
            data[idx]['is_sampled'] = True
            sampled_indices.append(idx)
        print(f"  Selected all {len(noise_records)} noise points")

    total_sampled = sum(1 for r in data if r['is_sampled'])

    stats = {
        'total_sampled': total_sampled,
        'from_clusters': num_from_clusters,
        'from_noise': total_sampled - num_from_clusters,
        'num_clusters': len(clusters),
        'total_noise_available': len(noise_records)
    }

    print(f"  Total sampled: {total_sampled}")

    return data, stats


# =============================================================================
# I/O Functions
# =============================================================================

def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data from Parquet file."""
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading data from {input_file}...")
    dataset = load_dataset('parquet', data_files=input_file)['train']
    data = [dict(record) for record in dataset]

    print(f"Loaded {len(data)} records")

    # Verify required columns exist
    if data:
        required_cols = ['embedding', 'score_difficulty', 'score_creativity', 'score_realism']
        missing = [col for col in required_cols if col not in data[0]]
        if missing:
            raise ValueError(f"Input file missing required columns: {missing}")

    return data


def save_data(data: List[Dict[str, Any]], output_file: str):
    """Save data to Parquet file."""
    output_path = Path(output_file)

    print(f"\nSaving data to {output_file}...")

    dataset = Dataset.from_list(data)
    dataset.to_parquet(output_file)

    print(f"Successfully saved {len(data):,} records to {output_file}")


def save_statistics(
    data: List[Dict[str, Any]],
    cluster_stats: Dict[str, Any],
    sampling_stats: Dict[str, Any],
    output_file: str
):
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

    # Compute score statistics (scores already exist from sample_data.py)
    score_stats = {}
    for dim in ['difficulty', 'creativity', 'realism']:
        scores = [r[f'score_{dim}'] for r in data if r.get(f'score_{dim}') is not None]
        if scores:
            score_stats[dim] = {
                'average': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'distribution': {i: scores.count(i) for i in range(1, 6)}
            }

    stats = {
        'total_records': len(data),
        'filter_statistics': dict(filter_stats),
        'clustering_statistics': cluster_stats,
        'sampling_statistics': sampling_stats,
        'score_statistics': score_stats
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics to {stats_file}")


# =============================================================================
# Main Pipeline Function
# =============================================================================

def process_sampled(
    input_file: str,
    output_file: str,
    # Clustering options
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    # Sampling options
    total_samples: Optional[int] = None,
    # Filtering options
    dedup_method: str = "minhash",
    dedup_threshold: float = 0.8,
    # Other options
    seed: int = 42
):
    """
    Run the processing pipeline on pre-sampled data.

    Phases:
    1. Load data (with embeddings and scores from sample_data.py)
    2. Quality filtering (adds filter_passed, filter_reason)
    3. Deduplication (updates filter_passed, filter_reason for duplicates)
    4. HDBSCAN clustering (adds cluster_id, cluster_probability)
    5. Sampling (adds is_sampled: 1 per cluster + noise to fill quota)
    6. Save complete dataset
    """
    np.random.seed(seed)

    print("="*60)
    print("PROCESS PRE-SAMPLED DATA PIPELINE")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Dedup method: {dedup_method}")
    if total_samples:
        print(f"Total samples target: {total_samples}")
    print("="*60)

    # Phase 1: Load data
    data = load_data(input_file)
    initial_count = len(data)

    # Phase 2: Quality filtering
    data = apply_quality_filters(data)

    # Phase 3: Deduplication
    data = apply_deduplication(data, dedup_method, dedup_threshold)

    # Phase 4: Clustering (using existing embeddings)
    data, cluster_stats = apply_clustering(
        data=data,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    # Phase 5: Sampling (1 per cluster + noise to fill quota)
    data, sampling_stats = apply_sampling(data, total_samples=total_samples)

    # Phase 6: Save
    save_data(data, output_file)
    save_statistics(data, cluster_stats, sampling_stats, output_file)

    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total records: {len(data):,} (same as input: {initial_count:,})")
    print(f"Filter passed: {sum(1 for r in data if r['filter_passed']):,}")
    print(f"Clusters: {cluster_stats['num_clusters']}")
    print(f"Sampled: {sampling_stats['total_sampled']} ({sampling_stats['from_clusters']} from clusters, {sampling_stats['from_noise']} from noise)")
    print(f"Output: {output_file}")
    print("="*60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process pre-sampled WildChat data: filter, deduplicate, and cluster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python process_sampled.py -i 2000.parquet -o 2000_processed.parquet

  # Custom clustering parameters
  python process_sampled.py -i 2000.parquet -o output.parquet \\
      --min-cluster-size 10 --min-samples 5

  # Use exact deduplication instead of minhash
  python process_sampled.py -i 2000.parquet -o output.parquet --dedup-method exact
        """
    )

    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input parquet file (from sample_data.py)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output parquet file')

    # Clustering arguments
    parser.add_argument('--min-cluster-size', type=int, default=None,
                        help='HDBSCAN min cluster size (default: auto, log(n)+2 clamped to [3,20])')
    parser.add_argument('--min-samples', type=int, default=None,
                        help='HDBSCAN min samples (default: auto, log(n)+1 clamped to [2,10])')

    # Sampling arguments
    parser.add_argument('--total-samples', type=int, default=None,
                        help='Total samples to select (default: 1 per cluster + all noise)')

    # Filtering arguments
    parser.add_argument('--dedup-method', choices=['exact', 'minhash', 'simhash'],
                        default='minhash', help='Deduplication method (default: minhash)')
    parser.add_argument('--dedup-threshold', type=float, default=0.8,
                        help='Deduplication threshold (default: 0.8)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    process_sampled(
        input_file=args.input,
        output_file=args.output,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        total_samples=args.total_samples,
        dedup_method=args.dedup_method,
        dedup_threshold=args.dedup_threshold,
        seed=args.seed
    )
